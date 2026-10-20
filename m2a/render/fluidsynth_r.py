# m2a/render/fluidsynth_r.py
"""Stage 2: render one MIDI track to a dry WAV stem via the FluidSynth CLI.

[v1.1] The deterministic spine renders dry (`--reverb=0 --chorus=0`): any
reverb/chorus baked in here would be re-processed by STAGE 6's mix/master
pass and produce the "low-end mud / smeared room" failure mode. `-g` on the
fluidsynth CLI is an input gain, not a peak guarantee, so peak headroom is
normalised explicitly after render.

`FluidSynthMpeRenderer` (M1.5) renders microtonal pitched tracks: each note
is re-voiced onto its own MPE member channel with a pitch bend carrying the
cents offset from 12-EDO (`tuning.channel_rotation_plan`), so any GM
SoundFont plays any EDO exactly.
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
from m2a.midi_io import global_meta_track, read_track_notes
from m2a.render.tuning import TuningSpec, channel_rotation_plan, note_pitch

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
    out.tracks.append(global_meta_track(mid))

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


def _render_with_fluidsynth(mid: "mido.MidiFile", soundfont: str, out_path: str) -> None:
    if shutil.which("fluidsynth") is None:
        raise RuntimeError("fluidsynth CLI not found on PATH")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_midi = str(Path(out_path).with_suffix(".track.mid"))
    mid.save(tmp_midi)
    # [v1.1] fluidsynth 2.x requires `[options] [soundfonts] [midifiles]` —
    # options (including -F) must precede the positional soundfont/MIDI args.
    subprocess.run([
        "fluidsynth", "-ni",
        "-F", out_path, "-r", str(SAMPLE_RATE), "-g", "0.5",
        "--reverb=0", "--chorus=0",
        soundfont, tmp_midi,
    ], check=True, capture_output=True)
    _normalise_peak(out_path, target_dbfs=TARGET_PEAK_DBFS)


def _stem(out_path: str, track_name: str, role: str, microtonal: bool) -> Stem:
    return Stem(
        track=track_name, role=role, path=out_path, microtonal=microtonal,
        peak_dbfs=measure_peak_dbfs(out_path), content_hash=file_hash(out_path),
    )


class FluidSynthRenderer:
    """Renders one MIDI track to a dry 48 kHz WAV stem via the FluidSynth CLI."""

    name = "fluidsynth"

    def supports(self, role: str, microtonal: bool) -> bool:
        # Unpitched percussion is safe at any EDO; pitched microtonal tracks
        # go to FluidSynthMpeRenderer / MidiDdspRenderer.
        return not microtonal or role == "drums"

    def available(self) -> bool:
        return shutil.which("fluidsynth") is not None

    def render(self, midi_path: str, track_idx: int, track_name: str, role: str,
               soundfont: str, out_path: str, tuning: TuningSpec | None = None) -> Stem:
        mid = mido.MidiFile(midi_path)
        _render_with_fluidsynth(_extract_single_track_midi(mid, track_idx), soundfont, out_path)
        return _stem(out_path, track_name, role, microtonal=False)


def mpe_track_midi(mid: "mido.MidiFile", track_idx: int, tuning: TuningSpec) -> "mido.MidiFile":
    """Re-voice one track onto rotating MPE member channels with per-note bends."""
    notes = read_track_notes(mid, track_idx)
    pitches = [note_pitch(n.pitch, n.bend, tuning) for n in notes]
    plan = channel_rotation_plan([(n.start_tick, n.end_tick, p) for n, p in zip(notes, pitches)],
                                 tuning.bend_range_semitones)
    program = notes[0].program if notes else 0
    bend_semis = int(tuning.bend_range_semitones)
    bend_cents = int(round((tuning.bend_range_semitones - bend_semis) * 100))

    events: list[tuple[int, int, mido.Message]] = []   # (tick, order, msg)
    for ch in sorted({c for c, _, _ in plan.values()}):
        # RPN 0 (pitch-bend sensitivity) + program on every member channel.
        for order, msg in enumerate([
            mido.Message("control_change", channel=ch, control=101, value=0),
            mido.Message("control_change", channel=ch, control=100, value=0),
            mido.Message("control_change", channel=ch, control=6, value=bend_semis),
            mido.Message("control_change", channel=ch, control=38, value=bend_cents),
            mido.Message("program_change", channel=ch, program=program),
        ]):
            events.append((0, order, msg))
    for i, n in enumerate(notes):
        ch, key, bend = plan[i]
        # note_offs sort before pitchwheels before note_ons at the same tick.
        events.append((n.start_tick, 20, mido.Message("pitchwheel", channel=ch, pitch=bend)))
        events.append((n.start_tick, 30, mido.Message("note_on", channel=ch, note=key,
                                                      velocity=n.velocity)))
        events.append((n.end_tick, 10, mido.Message("note_off", channel=ch, note=key, velocity=0)))
    events.sort(key=lambda e: (e[0], e[1]))

    out = mido.MidiFile(ticks_per_beat=mid.ticks_per_beat)
    out.tracks.append(global_meta_track(mid))
    track = mido.MidiTrack()
    clock = 0
    for tick, _, msg in events:
        track.append(msg.copy(time=tick - clock))
        clock = tick
    out.tracks.append(track)
    return out


class FluidSynthMpeRenderer:
    """Tuning-exact rendering of pitched microtonal tracks through any GM SoundFont."""

    name = "fluidsynth_mpe"

    def supports(self, role: str, microtonal: bool) -> bool:
        return role != "drums"

    def available(self) -> bool:
        return shutil.which("fluidsynth") is not None

    def render(self, midi_path: str, track_idx: int, track_name: str, role: str,
               soundfont: str, out_path: str, tuning: TuningSpec | None = None) -> Stem:
        tuning = tuning or TuningSpec()
        mid = mido.MidiFile(midi_path)
        _render_with_fluidsynth(mpe_track_midi(mid, track_idx, tuning), soundfont, out_path)
        return _stem(out_path, track_name, role, microtonal=not tuning.is_12_edo)
