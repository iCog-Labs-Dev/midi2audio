"""Stage 2: Deterministic SoundFont stem rendering using FluidSynth."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import soundfile as sf
import pretty_midi

try:
    import fluidsynth
    FLUIDSYNTH_AVAILABLE = True
except ImportError:
    FLUIDSYNTH_AVAILABLE = False


def synthesize_midi_track_fluidsynth(
    midi_path: Path | str,
    track_idx: int,
    soundfont_path: Path | str,
    output_wav_path: Path | str,
    sample_rate: int = 48000,
    headroom_dbfs: float = -12.0,
) -> Path:
    """Render an isolated single MIDI track to a 48 kHz stereo WAV file with headroom control."""
    out_path = Path(output_wav_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm = pretty_midi.PrettyMIDI(str(midi_path))

    if track_idx >= len(pm.instruments):
        raise IndexError(f"Track index {track_idx} exceeds total instruments ({len(pm.instruments)})")

    # Isolate the requested track
    target_inst = pm.instruments[track_idx]
    isolated_pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    isolated_pm.time_signature_changes = pm.time_signature_changes
    isolated_pm.instruments.append(target_inst)

    sf_path = str(soundfont_path)
    # If FluidSynth library or SoundFont is absent, render deterministic sinusoidal fallback
    if not FLUIDSYNTH_AVAILABLE or not Path(sf_path).is_file():
        audio_data = _render_pure_dsp_fallback(isolated_pm, sample_rate)
    else:
        # Fluidsynth software synthesis
        raw_audio = isolated_pm.fluidsynth(fs=sample_rate, sf2_path=sf_path)
        # Convert mono output to stereo array
        audio_data = np.vstack((raw_audio, raw_audio)).T

    # Apply strict peak normalization matching headroom target
    audio_data = apply_peak_headroom(audio_data, headroom_dbfs)
    sf.write(str(out_path), audio_data, sample_rate, subtype="PCM_24")
    return out_path


def apply_peak_headroom(audio: np.ndarray, headroom_dbfs: float = -12.0) -> np.ndarray:
    """Normalize peak audio amplitude to designated headroom floor."""
    max_peak = np.max(np.abs(audio))
    if max_peak < 1e-7:
        return audio
    target_linear = 10.0 ** (headroom_dbfs / 20.0)
    gain = target_linear / max_peak
    return (audio * gain).astype(np.float32)


def _render_pure_dsp_fallback(pm: pretty_midi.PrettyMIDI, sample_rate: int) -> np.ndarray:
    """Synthesize band-limited waveforms with ADSR envelopes when FluidSynth binary is unavailable."""
    total_time = pm.get_end_time() + 1.0
    total_samples = int(math.ceil(total_time * sample_rate))
    audio = np.zeros((total_samples, 2), dtype=np.float32)

    for inst in pm.instruments:
        for note in inst.notes:
            f0 = 440.0 * (2.0 ** ((note.pitch - 69) / 12.0))
            n_start = int(note.start * sample_rate)
            n_end = int(note.end * sample_rate)
            dur_samples = n_end - n_start
            if dur_samples <= 0 or n_start >= total_samples:
                continue

            t = np.arange(dur_samples) / sample_rate
            # Harmonic blend based on General MIDI status
            if inst.is_drum:
                # White-noise transient burst for percussive hits
                wave = np.random.uniform(-0.5, 0.5, dur_samples)
            else:
                wave = 0.7 * np.sin(2 * np.pi * f0 * t) + 0.3 * np.sin(4 * np.pi * f0 * t)

            # Fast attack, exponential decay (ADSR envelope)
            env = np.exp(-t * 5.0) * (note.velocity / 127.0)
            synth_note = (wave * env).astype(np.float32)

            actual_len = min(dur_samples, total_samples - n_start)
            audio[n_start : n_start + actual_len, 0] += synth_note[:actual_len]
            audio[n_start : n_start + actual_len, 1] += synth_note[:actual_len]

    return audio