# m2a/groove/apply.py
"""Stage 1: apply a validated GrooveSpec to a quantised MIDI file.

`apply_groove` is a pure function of (midi_path, spec, structure, rng_seed):
same inputs always produce byte-identical output. The only randomness is
jitter and ghost-note placement, each drawn from a `random.Random` seeded
per (track, transformation) so that editing one track's spec, or adding a
track, never perturbs the random draws made for any other track.

Implements, in order: swing, meter-aware microtiming, velocity accent map,
ghost notes, and articulation (duration scale / strum). Phrase arcs,
anticipation and tempo-curve rewriting are accepted on the schema but not
yet implemented in M1 — a spec that sets them raises `NotImplementedError`
rather than silently doing nothing with them.

D001 (tick rounding): ties round half-to-even (Python's built-in `round`),
and each transformation's own timing contribution is computed from the
note's *original* quantised metrical position, not from the position after
previously-applied transformations — the contributions are then summed and
rounded once, so displacement accumulates additively across stages without
successive transformations drifting off their intended grid slot.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from pathlib import Path

import mido

from m2a.contracts import StructureJSON
from m2a.groove.spec import GrooveSpec, positions_per_bar


@dataclass(frozen=True)
class DiffReport:
    applied_timing_offsets: tuple = ()      # (role, position, offset_ticks) per note
    applied_velocity_changes: tuple = ()    # (role, original, final) per note
    ghost_notes_inserted: int = 0
    bar_crossing_flags: tuple = ()          # (track_idx, orig_start_tick) per flagged note
    rule_violations: tuple = ()


@dataclass
class _Note:
    track_idx: int
    channel: int
    pitch: int
    velocity: int
    orig_start_tick: int
    orig_end_tick: int
    delta_ticks: float = 0.0
    final_velocity: int | None = None
    duration_scale: float = 1.0
    strum_offset_ticks: float = 0.0
    is_ghost: bool = False


# ── tick <-> seconds, tempo/meter lookups ────────────────────────────────────

def _tempo_segments_ticks(tempo_map: list[list[float]], tpb: int) -> list[tuple[float, float, float]]:
    """[(start_tick, start_sec, bpm), ...] piecewise-constant tempo segments."""
    segs: list[tuple[float, float, float]] = []
    for i, (t_sec, bpm) in enumerate(tempo_map):
        if i == 0:
            segs.append((0.0, 0.0, bpm))
            continue
        prev_tick, prev_sec, prev_bpm = segs[-1]
        dt = t_sec - prev_sec
        dticks = dt * (prev_bpm / 60.0) * tpb
        segs.append((prev_tick + dticks, t_sec, bpm))
    return segs or [(0.0, 0.0, 120.0)]


def _bpm_at_tick(tick: float, segs: list[tuple[float, float, float]]) -> float:
    bpm = segs[0][2]
    for start_tick, _, seg_bpm in segs:
        if start_tick <= tick + 1e-9:
            bpm = seg_bpm
        else:
            break
    return bpm


def _seconds_to_ticks(sec: float, segs: list[tuple[float, float, float]], tpb: int) -> float:
    seg = segs[0]
    for s in segs:
        if s[1] <= sec + 1e-9:
            seg = s
        else:
            break
    start_tick, start_sec, bpm = seg
    sec_per_tick = 60.0 / (bpm * tpb)
    return start_tick + (sec - start_sec) / sec_per_tick


def ms_to_ticks(ms: float, bpm: float, tpb: int) -> float:
    return ms / 1000.0 * (bpm / 60.0) * tpb


def _meter_for_bar(meter_map: list, bar_idx: int) -> str:
    cur = "4/4"
    for b, m in meter_map:
        if b <= bar_idx:
            cur = m
        else:
            break
    return cur


class _TimingContext:
    """Precomputes bar-tick boundaries and tempo/meter lookups for one run."""

    def __init__(self, structure: StructureJSON, tpb: int):
        self.tpb = tpb
        self.tempo_segs = _tempo_segments_ticks(structure.tempo_map, tpb)
        self.meter_map = structure.meter
        self.bar_start_ticks = [
            _seconds_to_ticks(t, self.tempo_segs, tpb) for _, t in structure.bar_table
        ]
        if len(self.bar_start_ticks) < 2:
            self.bar_start_ticks = self.bar_start_ticks + [self.bar_start_ticks[-1] + 4 * tpb]

    def bar_index_for_tick(self, tick: float) -> int:
        idx = 0
        for i, start in enumerate(self.bar_start_ticks):
            if start <= tick + 1e-6:
                idx = i
            else:
                break
        return idx

    def bar_bounds_ticks(self, bar_idx: int) -> tuple[float, float]:
        start = self.bar_start_ticks[min(bar_idx, len(self.bar_start_ticks) - 1)]
        if bar_idx + 1 < len(self.bar_start_ticks):
            end = self.bar_start_ticks[bar_idx + 1]
        else:
            prev_dur = (self.bar_start_ticks[-1] - self.bar_start_ticks[-2]
                        if len(self.bar_start_ticks) > 1 else 4 * self.tpb)
            end = start + prev_dur
        return start, end

    def bpm_at(self, tick: float) -> float:
        return _bpm_at_tick(tick, self.tempo_segs)

    def slot_for_tick(self, tick: float, grid: int) -> tuple[int, int]:
        """Return (bar_idx, grid_slot) for `tick` at the given grid resolution."""
        bar_idx = self.bar_index_for_tick(tick)
        start, end = self.bar_bounds_ticks(bar_idx)
        meter = _meter_for_bar(self.meter_map, bar_idx)
        n_positions = max(1, positions_per_bar(meter, grid))
        frac = (tick - start) / max(end - start, 1e-9)
        slot = int(round(frac * n_positions)) % n_positions
        return bar_idx, slot


# ── role resolution (D009) ───────────────────────────────────────────────────

def _build_role_map(structure: StructureJSON) -> dict[str, list[int]]:
    role_map: dict[str, list[int]] = {}
    for t in structure.tracks:
        role_map.setdefault(t.role, []).append(t.idx)
    return role_map


def _resolve_targets(key: str, role_map: dict[str, list[int]]) -> list[int]:
    if key.startswith("track:"):
        return [int(key.split(":", 1)[1])]
    return list(role_map.get(key, []))


def _role_of(track_idx: int, structure: StructureJSON) -> str:
    for t in structure.tracks:
        if t.idx == track_idx:
            return t.role
    return f"track:{track_idx}"


# ── seeded RNG per (track, transformation) ───────────────────────────────────

def _rng_for(rng_seed: int, track_idx: int, transformation: str) -> random.Random:
    key = f"{rng_seed}:{track_idx}:{transformation}".encode()
    seed_int = int(hashlib.sha256(key).hexdigest(), 16) % (2 ** 32)
    return random.Random(seed_int)


# ── MIDI parsing / re-serialization ──────────────────────────────────────────

def _parse_notes(mid: "mido.MidiFile") -> tuple[list[_Note], list[list[tuple[int, "mido.Message"]]]]:
    """Extract Note objects; return (notes, non-note messages per track with abs tick)."""
    notes: list[_Note] = []
    other_msgs: list[list[tuple[int, mido.Message]]] = []
    for track_idx, track in enumerate(mid.tracks):
        clock = 0
        pending: dict[tuple[int, int], list[int]] = {}
        track_others: list[tuple[int, mido.Message]] = []
        for msg in track:
            clock += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                pending.setdefault((msg.channel, msg.note), []).append(clock)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                starts = pending.get(key)
                if starts:
                    start_tick = starts.pop(0)
                    notes.append(_Note(
                        track_idx=track_idx, channel=msg.channel, pitch=msg.note,
                        velocity=64, orig_start_tick=start_tick, orig_end_tick=clock,
                    ))
            else:
                track_others.append((clock, msg.copy()))
        other_msgs.append(track_others)

    # velocity comes from the note_on message, fix it up in a second pass
    vel_lookup: dict[tuple[int, int, int, int], int] = {}
    for track_idx, track in enumerate(mid.tracks):
        clock = 0
        for msg in track:
            clock += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                vel_lookup.setdefault((track_idx, msg.channel, msg.note, clock), msg.velocity)
    for n in notes:
        n.velocity = vel_lookup.get((n.track_idx, n.channel, n.pitch, n.orig_start_tick), n.velocity)
    return notes, other_msgs


def _serialize(mid_in: "mido.MidiFile", notes: list[_Note],
               other_msgs: list[list[tuple[int, mido.Message]]]) -> "mido.MidiFile":
    mid_out = mido.MidiFile(ticks_per_beat=mid_in.ticks_per_beat)
    notes_by_track: dict[int, list[_Note]] = {}
    for n in notes:
        notes_by_track.setdefault(n.track_idx, []).append(n)

    for track_idx in range(len(mid_in.tracks)):
        events: list[tuple[float, int, mido.Message]] = []  # (tick, priority, msg)
        for tick, msg in other_msgs[track_idx]:
            events.append((tick, 0, msg))
        for n in notes_by_track.get(track_idx, []):
            start = n.orig_start_tick + n.delta_ticks + n.strum_offset_ticks
            duration = (n.orig_end_tick - n.orig_start_tick) * n.duration_scale
            end = start + duration
            vel = n.final_velocity if n.final_velocity is not None else n.velocity
            vel = max(1, min(127, vel))
            events.append((start, 1, mido.Message("note_on", channel=n.channel,
                                                    note=n.pitch, velocity=vel, time=0)))
            events.append((end, 2, mido.Message("note_off", channel=n.channel,
                                                  note=n.pitch, velocity=0, time=0)))

        events.sort(key=lambda e: (round(e[0]), e[1]))
        out_track = mido.MidiTrack()
        out_track.name = mid_in.tracks[track_idx].name
        clock = 0
        for tick, _, msg in events:
            abs_tick = max(0, round(tick))
            delta = max(0, abs_tick - clock)
            out_track.append(msg.copy(time=delta))
            clock = abs_tick
        mid_out.tracks.append(out_track)
    return mid_out


# ── transformations ──────────────────────────────────────────────────────────

def compute_swing_delay_ms(ratio: float, subdivision: int, bpm_local: float,
                            tempo_scaling: bool = True, reference_bpm: float = 120.0) -> float:
    """Pure swing-delay computation, exposed for direct unit testing.

    [v1.1] `delay_ms = (ratio_eff - 0.5) * pair_duration_ms` — no factor of
    two; `ratio` already is the position of the second note as a fraction
    of the pair. Tempo scaling (Friberg & Sundstrom 2002): the *short*
    off-beat note is held roughly constant in absolute duration, so the
    ratio collapses toward 0.5 as tempo rises above `reference_bpm`.
    """
    if subdivision == 8:
        pair_beats = 1.0
    elif subdivision == 16:
        pair_beats = 0.5
    else:
        raise ValueError(f"swing.subdivision must be 8 or 16, got {subdivision}")

    ratio_eff = ratio
    if tempo_scaling:
        beat_ms_ref = 60_000.0 / reference_bpm
        beat_ms_local = 60_000.0 / bpm_local
        short_ms = (1.0 - ratio) * beat_ms_ref
        ratio_eff = max(0.5, min(ratio, 1.0 - short_ms / beat_ms_local))

    pair_duration_ms = pair_beats * (60_000.0 / bpm_local)
    return (ratio_eff - 0.5) * pair_duration_ms


def _apply_swing(notes: list[_Note], spec: GrooveSpec, structure: StructureJSON,
                  ctx: _TimingContext, role_map: dict[str, list[int]],
                  rng_seed: int, timing_offsets: list, rule_violations: list) -> None:
    swing = spec.swing
    if swing is None:
        return
    targets: set[int] = set()
    for key in swing.applies_to:
        targets.update(_resolve_targets(key, role_map))

    for n in notes:
        if n.track_idx not in targets:
            continue
        bar_idx, slot16 = ctx.slot_for_tick(n.orig_start_tick, grid=16)
        if swing.subdivision == 8:
            is_offbeat = slot16 % 4 == 2
        elif swing.subdivision == 16:
            is_offbeat = slot16 % 2 == 1
        else:
            rule_violations.append(f"swing.subdivision must be 8 or 16, got {swing.subdivision}")
            continue
        if not is_offbeat:
            continue

        bpm_local = ctx.bpm_at(n.orig_start_tick)
        delay_ms = compute_swing_delay_ms(swing.ratio, swing.subdivision, bpm_local,
                                           swing.tempo_scaling, swing.reference_bpm)
        delay_ticks = ms_to_ticks(delay_ms, bpm_local, ctx.tpb)
        n.delta_ticks += delay_ticks
        role = _role_of(n.track_idx, structure)
        timing_offsets.append((role, slot16, round(delay_ticks)))


def _apply_microtiming(notes: list[_Note], spec: GrooveSpec, structure: StructureJSON,
                        ctx: _TimingContext, role_map: dict[str, list[int]],
                        rng_seed: int, timing_offsets: list, rule_violations: list) -> None:
    for key, mt in spec.microtiming_ms.items():
        targets = _resolve_targets(key, role_map)
        if not targets:
            continue
        grid = mt.grid

        if mt.by_position_16 is not None:
            for track_idx in targets:
                bar_idx = ctx.bar_index_for_tick(0)
                meter = _meter_for_bar(ctx.meter_map, bar_idx)
                expected_len = positions_per_bar(meter, grid)
                if len(mt.by_position_16) != expected_len:
                    raise ValueError(
                        f"microtiming_ms[{key!r}].by_position_16 has length "
                        f"{len(mt.by_position_16)}, expected {expected_len} for "
                        f"grid={grid} meter={meter}"
                    )

        for track_idx in targets:
            rng = _rng_for(rng_seed, track_idx, "microtiming")
            for n in [n for n in notes if n.track_idx == track_idx]:
                bar_idx, slot = ctx.slot_for_tick(n.orig_start_tick, grid=grid)
                offset_ms = mt.global_offset
                if mt.by_position_16 is not None:
                    offset_ms += mt.by_position_16[slot]
                jitter = rng.gauss(0.0, mt.jitter_sd) if mt.jitter_sd > 0 else 0.0
                offset_ms += jitter
                bpm_local = ctx.bpm_at(n.orig_start_tick)
                offset_ticks = ms_to_ticks(offset_ms, bpm_local, ctx.tpb)
                n.delta_ticks += offset_ticks
                role = _role_of(track_idx, structure)
                timing_offsets.append((role, slot, round(offset_ticks)))


def _apply_velocity(notes: list[_Note], spec: GrooveSpec, structure: StructureJSON,
                     ctx: _TimingContext, velocity_changes: list, rule_violations: list) -> None:
    velocity = spec.velocity
    if velocity is None:
        return
    if velocity.phrase_arc is not None:
        raise NotImplementedError("VelocitySpec.phrase_arc is not implemented in M1")

    if velocity.accent_map_16 is not None:
        for n in notes:
            _, slot = ctx.slot_for_tick(n.orig_start_tick, grid=16)
            multiplier = velocity.accent_map_16[slot % len(velocity.accent_map_16)]
            original = n.velocity
            final = max(1, min(127, round(original * multiplier)))
            if final != original:
                n.final_velocity = final
                role = _role_of(n.track_idx, structure)
                velocity_changes.append((role, original, final))


def _apply_ghost_notes(notes: list[_Note], spec: GrooveSpec, structure: StructureJSON,
                        ctx: _TimingContext, role_map: dict[str, list[int]],
                        rng_seed: int, rule_violations: list) -> list[_Note]:
    velocity = spec.velocity
    if velocity is None or velocity.ghost_notes is None:
        return []
    gspec = velocity.ghost_notes
    key = gspec.get("role") or (f"track:{gspec['track']}" if "track" in gspec else None)
    if key is None:
        rule_violations.append("velocity.ghost_notes: missing 'role' or 'track' key")
        return []
    targets = _resolve_targets(key, role_map)
    if not targets:
        rule_violations.append(f"velocity.ghost_notes: no tracks resolved for {key!r}")
        return []

    positions = gspec.get("positions", [])
    prob = float(gspec.get("prob", 1.0))
    vel_scale = float(gspec.get("vel_scale", 0.3))
    inserted: list[_Note] = []

    for track_idx in targets:
        track_notes = [n for n in notes if n.track_idx == track_idx]
        if not track_notes:
            continue
        ref_pitch = track_notes[0].pitch
        ref_channel = track_notes[0].channel
        base_vel = max(n.velocity for n in track_notes)
        durations = [n.orig_end_tick - n.orig_start_tick for n in track_notes]
        ghost_dur = max(1, min(durations) // 2)
        rng = _rng_for(rng_seed, track_idx, "ghost_notes")

        n_bars = len(ctx.bar_start_ticks) - 1
        for bar_idx in range(max(1, n_bars)):
            start, end = ctx.bar_bounds_ticks(bar_idx)
            meter = _meter_for_bar(ctx.meter_map, bar_idx)
            n_positions = max(1, positions_per_bar(meter, 16))
            for pos in positions:
                if pos >= n_positions:
                    continue
                if rng.random() > prob:
                    continue
                tick = start + (end - start) * (pos / n_positions)
                ghost = _Note(
                    track_idx=track_idx, channel=ref_channel, pitch=ref_pitch,
                    velocity=max(1, round(base_vel * vel_scale)),
                    orig_start_tick=round(tick), orig_end_tick=round(tick) + ghost_dur,
                    is_ghost=True,
                )
                inserted.append(ghost)
    return inserted


def _apply_articulation(notes: list[_Note], spec: GrooveSpec, structure: StructureJSON,
                         role_map: dict[str, list[int]]) -> None:
    for key, art in spec.articulation.items():
        targets = set(_resolve_targets(key, role_map))
        for n in notes:
            if n.track_idx not in targets:
                continue
            n.duration_scale = art.duration_scale

        if art.strum_ms > 0:
            by_onset: dict[tuple[int, int], list[_Note]] = {}
            for n in notes:
                if n.track_idx not in targets:
                    continue
                by_onset.setdefault((n.track_idx, n.orig_start_tick), []).append(n)
            for chord_notes in by_onset.values():
                if len(chord_notes) < 2:
                    continue
                chord_notes.sort(key=lambda n: n.pitch)
                for i, n in enumerate(chord_notes):
                    # accumulated in ms here; converted to ticks at each note's local
                    # tempo in the apply_groove loop below (bpm can vary note-to-note).
                    n.strum_offset_ticks += i * art.strum_ms


# ── public entry point ───────────────────────────────────────────────────────

def apply_groove(
    midi_path: str,
    spec: GrooveSpec,
    structure: StructureJSON,
    rng_seed: int,
    out_path: str | None = None,
) -> tuple[str, DiffReport]:
    if spec.anticipation is not None:
        raise NotImplementedError("GrooveSpec.anticipation is not implemented in M1")
    if spec.tempo_curve is not None:
        raise NotImplementedError("GrooveSpec.tempo_curve is not implemented in M1")

    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    ctx = _TimingContext(structure, tpb)
    role_map = _build_role_map(structure)

    notes, other_msgs = _parse_notes(mid)
    original_count = len(notes)

    timing_offsets: list = []
    velocity_changes: list = []
    rule_violations: list = []

    _apply_swing(notes, spec, structure, ctx, role_map, rng_seed, timing_offsets, rule_violations)
    _apply_microtiming(notes, spec, structure, ctx, role_map, rng_seed, timing_offsets, rule_violations)
    _apply_velocity(notes, spec, structure, ctx, velocity_changes, rule_violations)
    ghosts = _apply_ghost_notes(notes, spec, structure, ctx, role_map, rng_seed, rule_violations)
    notes.extend(ghosts)
    _apply_articulation(notes, spec, structure, role_map)

    # strum_offset_ticks was accumulated in ms above; convert to ticks at each note's local tempo
    for n in notes:
        if n.strum_offset_ticks:
            bpm_local = ctx.bpm_at(n.orig_start_tick)
            n.strum_offset_ticks = ms_to_ticks(n.strum_offset_ticks, bpm_local, tpb)

    bar_crossing_flags: list = []
    for n in notes:
        if n.is_ghost:
            continue
        orig_bar = ctx.bar_index_for_tick(n.orig_start_tick)
        new_bar = ctx.bar_index_for_tick(n.orig_start_tick + n.delta_ticks + n.strum_offset_ticks)
        if new_bar != orig_bar:
            bar_crossing_flags.append((n.track_idx, n.orig_start_tick))

    if spec.edo_safe:
        # M1 performs no pitch-table operations at all, so this invariant holds trivially.
        pass

    mid_out = _serialize(mid, notes, other_msgs)

    if out_path is None:
        out_path = str(Path(midi_path).with_suffix("")) + ".expressive.mid"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    mid_out.save(out_path)

    report = DiffReport(
        applied_timing_offsets=tuple(timing_offsets),
        applied_velocity_changes=tuple(velocity_changes),
        ghost_notes_inserted=len(ghosts),
        bar_crossing_flags=tuple(bar_crossing_flags),
        rule_violations=tuple(rule_violations),
    )
    assert len(notes) == original_count + len(ghosts)
    return out_path, report
