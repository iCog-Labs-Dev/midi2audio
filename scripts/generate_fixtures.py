"""scripts/generate_fixtures.py — produce deterministic test fixture files.

Run once from the repo root:

    python scripts/generate_fixtures.py

Outputs (written to tests/fixtures/):
  eight_bar.mid          4-track MIDI, 8 bars, 120 BPM, 4/4, fully quantised
  five_sec_stem.wav      5 s of silence at 48 kHz (placeholder; replaced in M1)
  synthetic_host.wav     5 s of kick+bass clicks at 120 BPM (single-ref testing)
"""
from __future__ import annotations

import json
import pathlib

import mido
import numpy as np
import soundfile as sf


def make_eight_bar_midi(out: str, bpm: float = 120.0, tpb: int = 480) -> None:
    """Write a 4-track, 8-bar, 120 BPM, 4/4, fully-quantised MIDI file.

    [v1.1] BUG FIX — v1.0 emitted note_on(time=0) + note_off(time=tpb-10) per
    beat, so the delta between consecutive onsets was 470 ticks, not 480: the
    fixture was 2 % sharp and drifted ~150 ms over 8 bars. The 10-tick gap is
    now charged to the *next* note_on so every onset falls exactly on tpb
    boundaries. ``test_fixture_onsets_are_exactly_one_beat_apart`` guards this.

    [v1.1] tempo and time-signature meta messages are emitted *once*, on track 0
    only. Writing set_tempo into all four tracks is legal MIDI but causes
    pretty_midi to report four coincident tempo changes and complicates assertions.
    """
    mid = mido.MidiFile(ticks_per_beat=tpb)
    us_per_beat = int(60_000_000 / bpm)

    # (track name, MIDI channel, GM program or None for drums)
    tracks_spec = [
        ("drums",   9, None),
        ("bass",    0, 33),
        ("comping", 1, 4),
        ("lead",    2, 81),
    ]
    pitches = {"drums": 36, "bass": 40, "comping": 60, "lead": 72}

    for i, (name, ch, prog) in enumerate(tracks_spec):
        t = mido.MidiTrack()
        t.name = name
        mid.tracks.append(t)

        # Global meta messages on track 0 only
        if i == 0:
            t.append(mido.MetaMessage("set_tempo", tempo=us_per_beat, time=0))
            t.append(mido.MetaMessage("time_signature",
                                      numerator=4, denominator=4, time=0))

        if prog is not None:
            t.append(mido.Message("program_change", channel=ch,
                                  program=prog, time=0))

        # 8 bars × 4 beats = 32 notes, one per beat
        for beat in range(32):
            # note_on: first beat has time=0; subsequent beats carry the 10-tick
            # gap from the previous note_off, keeping onset spacing exactly tpb.
            note_on_time = 0 if beat == 0 else 10
            t.append(mido.Message("note_on",  channel=ch,
                                  note=pitches[name], velocity=64,
                                  time=note_on_time))
            t.append(mido.Message("note_off", channel=ch,
                                  note=pitches[name], velocity=0,
                                  time=tpb - 10))

    mid.save(out)


def make_silent_wav(out: str, seconds: float = 5.0, sr: int = 48000) -> None:
    """Write a mono WAV file containing ``seconds`` of silence at ``sr`` Hz."""
    sf.write(out, np.zeros(int(seconds * sr), dtype=np.float32), sr)


def make_synthetic_host_wav(out: str, bpm: float = 120.0,
                             seconds: float = 5.0, sr: int = 48000) -> None:
    """Write a synthetic kick+bass click track at ``bpm`` for single-ref testing.

    Each beat gets a short raised-cosine click (10 ms, 1 kHz) so onset
    detectors have something real to lock onto.
    """
    n = int(seconds * sr)
    audio = np.zeros(n, dtype=np.float32)
    beat_samples = int(sr * 60.0 / bpm)
    click_len = int(0.010 * sr)          # 10 ms click
    click_t = np.linspace(0, np.pi, click_len)
    click = (0.5 * (1.0 - np.cos(click_t)) * np.sin(2 * np.pi * 1000 * np.arange(click_len) / sr)).astype(np.float32)

    onset = 0
    while onset + click_len < n:
        audio[onset:onset + click_len] += click
        onset += beat_samples

    sf.write(out, audio, sr)


if __name__ == "__main__":
    fx = pathlib.Path("tests/fixtures")
    fx.mkdir(parents=True, exist_ok=True)

    print("Generating eight_bar.mid …")
    make_eight_bar_midi(str(fx / "eight_bar.mid"))

    print("Generating five_sec_stem.wav …")
    make_silent_wav(str(fx / "five_sec_stem.wav"))

    print("Generating synthetic_host.wav …")
    make_synthetic_host_wav(str(fx / "synthetic_host.wav"))

    print("Done. Hand-authored JSON fixtures must be present in tests/fixtures/")
    print("  eight_bar_structure.json")
    print("  eight_bar_tuning.json")
    print("  eight_bar_manifest.json")
