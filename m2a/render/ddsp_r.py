# m2a/render/ddsp_r.py
"""f0-contour renderer for microtonal leads (Step 1.10).

[v1.1] MIDI-DDSP's note encoder takes *integer* MIDI pitch, so feeding it a
19-EDO part loses the microtonality at the front door. This renderer
bypasses the note encoder entirely and drives the DDSP **synthesis** stage —
a harmonic oscillator bank plus filtered noise — with an explicit f0 contour
computed from the EDO lattice (`mode="f0_contour"`).

The learned MIDI-DDSP expression module (envelope/vibrato from a trained
model) is optional and not wired yet (see DECISIONS.md D017); without it the
control signals are a deterministic ADSR amplitude envelope, a 1/k harmonic
rolloff and a small fixed noise floor. Tuning is exact by construction, which
is what the M1.5 exit criterion measures; timbral quality is a later concern.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from m2a.contracts import Stem
from m2a.midi_io import track_notes
from m2a.render.fluidsynth_r import SAMPLE_RATE, TARGET_PEAK_DBFS, file_hash, measure_peak_dbfs
from m2a.render.tuning import TuningSpec, note_pitch, pitch_to_hz

ATTACK_SEC = 0.01
RELEASE_SEC = 0.08
MAX_HARMONICS = 24
NOISE_LEVEL = 0.01


def _envelope(n_samples: int, n_release: int, sr: int) -> np.ndarray:
    n_attack = max(1, int(ATTACK_SEC * sr))
    env = np.ones(n_samples + n_release)
    env[:n_attack] = np.linspace(0.0, 1.0, n_attack, endpoint=False)
    if n_release:
        env[n_samples:] = np.linspace(1.0, 0.0, n_release)
    return env


def harmonic_note(f0_hz: float, duration_sec: float, amplitude: float,
                  sr: int = SAMPLE_RATE, seed: int = 0) -> np.ndarray:
    """One note of harmonic-plus-noise synthesis at a constant f0."""
    n = max(1, int(duration_sec * sr))
    n_release = int(RELEASE_SEC * sr)
    t = np.arange(n + n_release) / sr
    out = np.zeros_like(t)
    for k in range(1, MAX_HARMONICS + 1):
        if k * f0_hz >= sr / 2:
            break
        out += np.sin(2 * np.pi * k * f0_hz * t) / k
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(len(t))
    noise = np.convolve(noise, np.ones(8) / 8, mode="same")   # gentle low-pass
    out = out + NOISE_LEVEL * noise
    return amplitude * out * _envelope(n, n_release, sr)


class MidiDdspRenderer:
    name = "midi_ddsp"

    def __init__(self, config=None, mode: str = "f0_contour"):
        if mode != "f0_contour":
            raise ValueError("only mode='f0_contour' is tuning-exact; see module docstring")
        self.config = config
        self.mode = mode

    def supports(self, role: str, microtonal: bool) -> bool:
        return role != "drums"

    def available(self) -> bool:
        return True

    def render(self, midi_path: str, track_idx: int, track_name: str, role: str,
               soundfont: str, out_path: str, tuning: TuningSpec | None = None) -> Stem:
        tuning = tuning or TuningSpec()
        notes = track_notes(midi_path, track_idx)
        end = max((n.end_sec for n in notes), default=0.0) + RELEASE_SEC + 0.5
        audio = np.zeros(int(end * SAMPLE_RATE) + 1)
        for i, n in enumerate(notes):
            f0 = pitch_to_hz(note_pitch(n.pitch, n.bend, tuning))
            seg = harmonic_note(f0, n.end_sec - n.start_sec, n.velocity / 127.0, seed=i)
            start = int(round(n.start_sec * SAMPLE_RATE))
            stop = min(len(audio), start + len(seg))
            audio[start:stop] += seg[:stop - start]
        peak = float(np.max(np.abs(audio))) if audio.size else 0.0
        if peak > 0:
            audio *= 10.0 ** (TARGET_PEAK_DBFS / 20.0) / peak
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, audio, SAMPLE_RATE)
        return Stem(track=track_name, role=role, path=out_path,
                    microtonal=not tuning.is_12_edo,
                    peak_dbfs=measure_peak_dbfs(out_path), content_hash=file_hash(out_path))
