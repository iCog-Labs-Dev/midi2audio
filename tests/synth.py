"""tests/synth.py — tiny deterministic audio + MIDI builders for scorer tests.

Independent of the pipeline's renderers on purpose: scorer tests must not
pass merely because the renderer and the scorer share a bug.
"""
from __future__ import annotations

import mido
import numpy as np
import soundfile as sf

SR = 48_000


def midi_hz(pitch: float) -> float:
    return 440.0 * 2.0 ** ((pitch - 69.0) / 12.0)


def tone(pitch: float, dur: float, sr: int = SR, harmonics: int = 6, amp: float = 0.3) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    f = midi_hz(pitch)
    y = sum(np.sin(2 * np.pi * k * f * t) / k for k in range(1, harmonics + 1) if k * f < sr / 2)
    # 5 ms attack, exponential decay, 20 ms release (a hard cut reads as an onset)
    env = np.minimum(1.0, t / 0.005) * np.exp(-3.0 * t) * np.minimum(1.0, (dur - t) / 0.02)
    return amp * y * env


def click(dur: float = 0.06, sr: int = SR, seed: int = 0) -> np.ndarray:
    t = np.arange(int(dur * sr)) / sr
    rng = np.random.default_rng(seed)
    return 0.5 * rng.standard_normal(len(t)) * np.exp(-60.0 * t)


def render_events(path: str, events: list[tuple[float, float, float]], total: float,
                  sr: int = SR, percussive: bool = False) -> str:
    """events: (start_sec, dur_sec, pitch). Writes a mono WAV and returns its path."""
    y = np.zeros(int(total * sr))
    for i, (start, dur, pitch) in enumerate(events):
        seg = click(sr=sr, seed=i) if percussive else tone(pitch, dur, sr)
        a = int(round(start * sr))
        b = min(len(y), a + len(seg))
        if a < len(y):
            y[a:b] += seg[:b - a]
    sf.write(path, y, sr)
    return path


def write_midi(path: str, tracks: list[list[tuple[float, float, int]]], bpm: float = 120.0,
               tpb: int = 480, channels: list[int] | None = None,
               programs: list[int] | None = None) -> str:
    """tracks: per track, (start_beat, dur_beats, pitch). Track 0 carries tempo."""
    mid = mido.MidiFile(ticks_per_beat=tpb)
    for ti, notes in enumerate(tracks):
        ch = channels[ti] if channels else ti
        tr = mido.MidiTrack()
        if ti == 0:
            tr.append(mido.MetaMessage("set_tempo", tempo=int(60_000_000 / bpm), time=0))
            tr.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        if programs:
            tr.append(mido.Message("program_change", channel=ch, program=programs[ti], time=0))
        evs = []
        for start, dur, pitch in notes:
            evs.append((int(round(start * tpb)), 1, mido.Message("note_on", channel=ch, note=pitch, velocity=90)))
            evs.append((int(round((start + dur) * tpb)), 0, mido.Message("note_off", channel=ch, note=pitch, velocity=0)))
        evs.sort(key=lambda e: (e[0], e[1]))
        clock = 0
        for tick, _, msg in evs:
            tr.append(msg.copy(time=tick - clock))
            clock = tick
        mid.tracks.append(tr)
    mid.save(path)
    return path
