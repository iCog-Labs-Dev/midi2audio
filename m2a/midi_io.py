# m2a/midi_io.py
"""Per-track note extraction shared by the renderers and the scorers.

Tracks are addressed by their zero-based index in score.mid (contract
invariant 1), never by pretty_midi instrument order: pretty_midi splits and
merges tracks by (program, channel), so its instrument list does not line up
with `structure.tracks[].idx` on real-world files. Times are computed against
the global tempo map merged from *every* track, because type-1 files usually
carry set_tempo on track 0 only.
"""
from __future__ import annotations

from dataclasses import dataclass

import mido


@dataclass(frozen=True)
class NoteEvent:
    start_tick: int
    end_tick: int
    start_sec: float
    end_sec: float
    channel: int
    pitch: int
    velocity: int
    bend: int = 0      # pitchwheel value (-8192..8191) on the note's channel at note-on
    program: int = 0   # program in force on the note's channel at note-on


class TempoMap:
    """Piecewise-constant tick -> seconds conversion from merged set_tempo events."""

    def __init__(self, mid: "mido.MidiFile"):
        self.tpb = mid.ticks_per_beat
        changes: list[tuple[int, int]] = []
        for track in mid.tracks:
            clock = 0
            for msg in track:
                clock += msg.time
                if msg.type == "set_tempo":
                    changes.append((clock, msg.tempo))
        changes.sort(key=lambda c: c[0])
        if not changes or changes[0][0] != 0:
            changes.insert(0, (0, 500_000))
        # (start_tick, start_sec, us_per_beat)
        self._segs: list[tuple[int, float, int]] = []
        sec = 0.0
        prev_tick, prev_tempo = changes[0]
        self._segs.append((prev_tick, 0.0, prev_tempo))
        for tick, tempo in changes[1:]:
            sec += (tick - prev_tick) * prev_tempo / 1e6 / self.tpb
            self._segs.append((tick, sec, tempo))
            prev_tick, prev_tempo = tick, tempo

    def seconds(self, tick: int) -> float:
        seg = self._segs[0]
        for s in self._segs:
            if s[0] <= tick:
                seg = s
            else:
                break
        start_tick, start_sec, tempo = seg
        return start_sec + (tick - start_tick) * tempo / 1e6 / self.tpb


def read_track_notes(mid: "mido.MidiFile", track_idx: int,
                     tempo: TempoMap | None = None) -> list[NoteEvent]:
    """Return the notes of one track, sorted by (start_tick, pitch)."""
    tempo = tempo or TempoMap(mid)
    bends = [0] * 16
    programs = [0] * 16
    open_notes: dict[tuple[int, int], list[tuple[int, int, int, int]]] = {}
    notes: list[NoteEvent] = []
    clock = 0
    for msg in mid.tracks[track_idx]:
        clock += msg.time
        if msg.type == "pitchwheel":
            bends[msg.channel] = msg.pitch
        elif msg.type == "program_change":
            programs[msg.channel] = msg.program
        elif msg.type == "note_on" and msg.velocity > 0:
            open_notes.setdefault((msg.channel, msg.note), []).append(
                (clock, msg.velocity, bends[msg.channel], programs[msg.channel]))
        elif msg.type in ("note_off", "note_on"):
            stack = open_notes.get((msg.channel, msg.note))
            if stack:
                start, vel, bend, prog = stack.pop(0)
                notes.append(NoteEvent(start, clock, tempo.seconds(start), tempo.seconds(clock),
                                       msg.channel, msg.note, vel, bend, prog))
    # Notes never closed run to the end of the track.
    for (channel, pitch), stack in open_notes.items():
        for start, vel, bend, prog in stack:
            notes.append(NoteEvent(start, clock, tempo.seconds(start), tempo.seconds(clock),
                                   channel, pitch, vel, bend, prog))
    notes.sort(key=lambda n: (n.start_tick, n.pitch))
    return notes


def track_notes(midi_path: str, track_idx: int) -> list[NoteEvent]:
    mid = mido.MidiFile(midi_path)
    return read_track_notes(mid, track_idx)


def global_meta_track(mid: "mido.MidiFile") -> "mido.MidiTrack":
    """Every set_tempo/time_signature event in the file, merged onto one track."""
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
    return meta_track
