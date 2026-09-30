# m2a/render/fluidsynth_r.py
"""Stage 2: render one MIDI track to a dry WAV stem via the FluidSynth CLI.

[v1.1] The deterministic spine renders dry (`--reverb=0 --chorus=0`): any
reverb/chorus baked in here would be re-processed by STAGE 6's mix/master
pass and produce the "low-end mud / smeared room" failure mode. `-g` on the
fluidsynth CLI is an input gain, not a peak guarantee, so peak headroom is
normalised explicitly after render.
"""
from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import mido
import numpy as np
import soundfile as sf

from m2a.contracts import Stem

SAMPLE_RATE = 48_000
TARGET_PEAK_DBFS = -12.0


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _extract_single_track_midi(mid: "mido.MidiFile", track_idx: int) -> "mido.MidiFile":
    """Build a standalone MIDI file containing only `track_idx`'s note/program
    events, preceded by every set_tempo/time_signature meta message found
    anywhere in the source file (merged in chronological order) so the
    extracted track renders at the correct tempo regardless of which
    original track carried the global meta events."""
    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)

    metas: list[tuple[int, mido.MetaMessage]] = []
    for track in mid.tracks:
        clock = 0
        for msg in track:
            clock += msg.time
            if msg.type in ("set_tempo", "time_signature"):
                metas.append((clock, msg.copy()))
    metas.sort(key=lambda pair: pair[0])

    meta_track = mido.MidiTrack()
    clock = 0
    for t, msg in metas:
        meta_track.append(msg.copy(time=t - clock))
        clock = t
    out.tracks.append(meta_track)

    note_track = mido.MidiTrack()
    carry = 0
    for msg in mid.tracks[track_idx]:
        if msg.type in ("set_tempo", "time_signature"):
            carry += msg.time
            continue
        note_track.append(msg.copy(time=msg.time + carry))
        carry = 0
    out.tracks.append(note_track)
    return out


def measure_peak_dbfs(path: str) -> float:
    audio, _ = sf.read(path)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return -120.0
    return 20.0 * np.log10(peak)


def _normalise_peak(path: str, target_dbfs: float) -> None:
    audio, sr = sf.read(path)
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak <= 0.0:
        return
    target_linear = 10.0 ** (target_dbfs / 20.0)
    gain = target_linear / peak
    sf.write(path, audio * gain, sr)


class FluidSynthRenderer:
    """Renders one MIDI track to a dry 48 kHz WAV stem via the FluidSynth CLI."""

    def supports(self, role: str, microtonal: bool) -> bool:
        return not microtonal  # microtonal routing added in M1.5

    def render(self, midi_path: str, track_idx: int, track_name: str, role: str,
               soundfont: str, out_path: str) -> Stem:
        if shutil.which("fluidsynth") is None:
            raise RuntimeError("fluidsynth CLI not found on PATH")

        mid = mido.MidiFile(midi_path)
        single_track_mid = _extract_single_track_midi(mid, track_idx)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tmp_midi = str(Path(out_path).with_suffix(".track.mid"))
        single_track_mid.save(tmp_midi)

        # [v1.1] fluidsynth 2.x requires `[options] [soundfonts] [midifiles]` —
        # options (including -F) must precede the positional soundfont/MIDI args.
        subprocess.run([
            "fluidsynth", "-ni",
            "-F", out_path, "-r", str(SAMPLE_RATE), "-g", "0.5",
            "--reverb=0", "--chorus=0",
            soundfont, tmp_midi,
        ], check=True, capture_output=True)

        _normalise_peak(out_path, target_dbfs=TARGET_PEAK_DBFS)
        peak = measure_peak_dbfs(out_path)

        return Stem(
            track=track_name, role=role, path=out_path, microtonal=False,
            peak_dbfs=peak, content_hash=file_hash(out_path),
        )
