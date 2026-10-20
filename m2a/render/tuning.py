# m2a/render/tuning.py
"""EDO tuning utilities (Step 1.8). Pure functions, no I/O except `load_tuning`.

How a MIDI note encodes a pitch depends on `tuning.json.method`:

* ``direct_12`` — plain 12-EDO: key number is the pitch, pitch bend is ignored.
* ``scala`` / ``mts_esp`` — key-mapped: key `k` is EDO step `k - base_tuning`
  above the base pitch (MIDI note `base_tuning`). This is the layout a Scala
  keyboard mapping or an MTS-ESP master gives a linear keyboard.
* ``mpe`` — key number + per-note-channel pitch bend gives the exact pitch;
  `bend_range_semitones` is the receiver's pitch-bend range.

`note_pitch` turns (key, bend) into a fractional MIDI pitch under any of
these, so every renderer and scorer agrees on what a note *means*.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

SAFE_MICROTONAL_METHODS = ("mpe", "mts_esp", "scala")
METHODS = ("direct_12",) + SAFE_MICROTONAL_METHODS
PERCUSSION_CHANNEL = 9


@dataclass(frozen=True)
class TuningSpec:
    edo: int = 12
    base_tuning: float = 60.0
    method: str = "direct_12"
    bend_range_semitones: float = 2.0

    def __post_init__(self):
        if self.edo < 1:
            raise ValueError(f"edo must be >= 1, got {self.edo}")
        if self.method not in METHODS:
            raise ValueError(f"unknown tuning method {self.method!r}; expected one of {METHODS}")
        if self.bend_range_semitones <= 0:
            raise ValueError("bend_range_semitones must be positive")

    @property
    def is_12_edo(self) -> bool:
        return self.edo == 12 and self.method == "direct_12"


def load_tuning(path: str) -> TuningSpec:
    data = json.loads(Path(path).read_text())
    return TuningSpec(
        edo=int(data["edo"]),
        base_tuning=float(data.get("base_tuning", 60.0)),
        method=data.get("method", "direct_12"),
        bend_range_semitones=float(data.get("bend_range_semitones", 2.0)),
    )


def to_scala(edo: int, base_hz: float = 440.0) -> str:
    """Generate a Scala .scl file string for the given EDO.

    `base_hz` is not part of the .scl format (it belongs in the .kbm keyboard
    mapping); it is accepted so callers can pass one tuning object around.
    """
    lines = [f"! {edo}-edo.scl",
             f"{edo}-equal temperament",
             f" {edo}"]
    for step in range(1, edo + 1):
        lines.append(f" {step * 1200.0 / edo:.6f}")
    return "\n".join(lines)


def step_to_cents(step: int, edo: int) -> float:
    return step * (1200.0 / edo)


def note_pitch(key: int, bend: int, tuning: TuningSpec) -> float:
    """Fractional MIDI pitch (60.0 = middle C, 12-EDO semitone units)."""
    if tuning.method == "direct_12":
        return float(key)
    if tuning.method == "mpe":
        return key + bend / 8192.0 * tuning.bend_range_semitones
    # scala / mts_esp: key-mapped EDO steps above base_tuning
    step = key - round(tuning.base_tuning)
    return tuning.base_tuning + step_to_cents(step, tuning.edo) / 100.0 + bend / 8192.0 * tuning.bend_range_semitones


def pitch_to_hz(pitch: float) -> float:
    return 440.0 * 2.0 ** ((pitch - 69.0) / 12.0)


def split_pitch(pitch: float, bend_range_semitones: float) -> tuple[int, int]:
    """Nearest 12-EDO key plus the pitchwheel value carrying the remainder."""
    key = int(round(pitch))
    bend = int(round((pitch - key) / bend_range_semitones * 8192.0))
    return max(0, min(127, key)), max(-8192, min(8191, bend))


def mts_tuning_table(edo: int, base_tuning_midi: int = 60, program: int = 0,
                     device_id: int = 0x7F, name: str = "") -> bytes:
    """MTS non-real-time bulk tuning dump (F0 7E dev 08 01 ...) for the given EDO.

    Key `k` sounds EDO step `k - base_tuning_midi` above MIDI note
    `base_tuning_midi`. Each key is encoded as (semitone, 14-bit fraction of
    100 cents); keys whose pitch falls outside MIDI 0..127.99 get the
    spec's "no change" marker 7F 7F 7F.
    """
    label = (name or f"{edo}-EDO")[:16].ljust(16).encode("ascii", "replace")
    body = bytearray([0x7E, device_id & 0x7F, 0x08, 0x01, program & 0x7F])
    body += bytes(b & 0x7F for b in label)
    for key in range(128):
        pitch = base_tuning_midi + step_to_cents(key - base_tuning_midi, edo) / 100.0
        if not 0.0 <= pitch < 128.0:
            body += bytes([0x7F, 0x7F, 0x7F])
            continue
        semitone = int(math.floor(pitch))
        frac = int(round((pitch - semitone) * (1 << 14)))
        if frac >= 1 << 14:            # rounding carried into the next semitone
            semitone, frac = semitone + 1, 0
        if semitone > 127:
            body += bytes([0x7F, 0x7F, 0x7F])
            continue
        body += bytes([semitone, (frac >> 7) & 0x7F, frac & 0x7F])
    checksum = 0
    for b in body:
        checksum ^= b
    return bytes([0xF0]) + bytes(body) + bytes([checksum & 0x7F, 0xF7])


def channel_rotation_plan(notes: list, bend_range_semitones: float,
                          channels: tuple[int, ...] | None = None) -> dict[int, tuple[int, int, int]]:
    """Assign MPE member channels so that no two sounding notes with different
    pitch bends share a channel.

    `notes` is a list of `(start, end, pitch)` with fractional `pitch`. Returns
    `{note_index: (channel, key, bend)}`. Channels are reused least-recently-
    used first so a re-bent channel does not bend a still-ringing release
    tail more often than necessary. Two simultaneous notes needing the same
    bend may share a channel. Raises `ValueError` when polyphony exceeds the
    channel pool, rather than silently mistuning a note.
    """
    if channels is None:
        channels = tuple(c for c in range(16) if c != PERCUSSION_CHANNEL)
    order = sorted(range(len(notes)), key=lambda i: (notes[i][0], notes[i][2]))
    # channel -> (bend, latest end of notes on it, last-used order)
    state: dict[int, tuple[int, float, int]] = {c: (0, float("-inf"), -len(channels) + k)
                                                 for k, c in enumerate(channels)}
    plan: dict[int, tuple[int, int, int]] = {}
    counter = 0
    for i in order:
        start, end, pitch = notes[i]
        key, bend = split_pitch(pitch, bend_range_semitones)
        busy_same = [c for c, (b, e, _) in state.items() if e > start and b == bend]
        free = [c for c, (_, e, _) in state.items() if e <= start]
        if busy_same:
            ch = busy_same[0]
        elif free:
            ch = min(free, key=lambda c: (state[c][2], c))
        else:
            raise ValueError(f"polyphony at t={start} exceeds {len(channels)} MPE channels")
        state[ch] = (bend, max(end, state[ch][1]), counter)
        counter += 1
        plan[i] = (ch, key, bend)
    return plan
