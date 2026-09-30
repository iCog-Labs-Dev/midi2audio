# m2a/analysis/from_midi.py
"""Stage 0: infer structure.json from a raw MIDI file.

Everything downstream (groove authoring, applicator, rendering) reads
structure.json rather than re-parsing MIDI, so this module is the only
place that owns tempo/meter/key/chord/section inference.

Per D016, `sections[].tension` is always null here — only the aimusic
planner (host-supplied structure.json) can populate it. `boundary_lvl` comes
from the novelty-curve peak height, quantised to 0-3.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pretty_midi

from m2a.contracts import StructureJSON

FLAT_VELOCITY_STD_THRESHOLD = 2.0
QUANTIZED_FRACTION_THRESHOLD = 0.85
QUANTIZED_TOLERANCE_MS = 10.0
BAR_SAFETY_LIMIT = 100_000

_MAJOR_TEMPLATE = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0])
_MINOR_TEMPLATE = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
_PITCH_CLASS_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def analyse_midi(midi_path: str, out_path: str, edo: int = 12, base_tuning: float = 60.0) -> StructureJSON:
    """Produce and write a validated `StructureJSON` from a MIDI file."""
    pm = pretty_midi.PrettyMIDI(midi_path)

    tempo_map = _extract_tempo_map(pm)
    bar_table, meter_map = _build_bar_table_and_meter(pm)
    key_map = _infer_key(pm)
    chords = _infer_chords(pm, bar_table)
    sections = _detect_sections(pm, bar_table)
    tracks = [_analyse_track(idx, inst, bar_table) for idx, inst in enumerate(pm.instruments)]

    data = {
        "schema": "midi2audio.structure/1",
        "provenance": "inferred",
        "source_hash": f"sha256:{_file_hash(midi_path)}",
        "edo": edo,
        "base_tuning": base_tuning,
        "tempo_map": tempo_map,
        "meter": meter_map,
        "key": key_map,
        "chords": chords,
        "sections": sections,
        "bar_table": bar_table,
        "tracks": tracks,
    }
    structure = StructureJSON.model_validate(data)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(structure.model_dump(by_alias=True), f, indent=2)
    return structure


def load_structure(path: str) -> StructureJSON:
    """Load and validate a structure.json file already on disk."""
    data = json.loads(Path(path).read_text())
    return StructureJSON.model_validate(data)


# ── helpers ──────────────────────────────────────────────────────────────────

def _file_hash(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _extract_tempo_map(pm: "pretty_midi.PrettyMIDI") -> list[list[float]]:
    times, tempi = pm.get_tempo_changes()
    if len(times) == 0:
        return [[0.0, 120.0]]
    return [[float(t), float(bpm)] for t, bpm in zip(times, tempi)]


def _tempo_at(times: np.ndarray, tempi: np.ndarray, t: float) -> float:
    bpm = float(tempi[0]) if len(tempi) else 120.0
    for tt, bp in zip(times, tempi):
        if tt <= t + 1e-9:
            bpm = float(bp)
        else:
            break
    return bpm


def _meter_at(ts_changes: list, t: float) -> tuple[int, int]:
    num, den = 4, 4
    for ts in ts_changes:
        if ts.time <= t + 1e-9:
            num, den = ts.numerator, ts.denominator
        else:
            break
    return num, den


def _build_bar_table_and_meter(pm: "pretty_midi.PrettyMIDI") -> tuple[list[list[float]], list[list]]:
    """Walk bar-by-bar from t=0, looking up the tempo/meter in force at each
    bar boundary. Stops once the walk passes the end of the piece."""
    tempo_times, tempi = pm.get_tempo_changes()
    ts_changes = sorted(pm.time_signature_changes, key=lambda ts: ts.time)
    end_time = pm.get_end_time()

    bar_table: list[list[float]] = []
    meter_map: list[list] = []
    t = 0.0
    bar_idx = 0
    last_meter: tuple[int, int] | None = None

    while t <= end_time + 1e-6:
        bar_table.append([bar_idx, t])
        num, den = _meter_at(ts_changes, t)
        if (num, den) != last_meter:
            meter_map.append([bar_idx, f"{num}/{den}"])
            last_meter = (num, den)
        bpm = _tempo_at(tempo_times, tempi, t)
        beat_sec = 60.0 / bpm
        bar_sec = beat_sec * num * (4.0 / den)
        t += bar_sec
        bar_idx += 1
        if bar_idx > BAR_SAFETY_LIMIT:
            break

    if not bar_table:
        bar_table = [[0, 0.0]]
        meter_map = [[0, "4/4"]]
    return bar_table, meter_map


def _infer_key(pm: "pretty_midi.PrettyMIDI") -> list[dict]:
    from music21 import note, stream

    s = stream.Stream()
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            m21n = note.Note()
            m21n.pitch.midi = n.pitch
            s.insert(n.start, m21n)

    if len(s.notes) == 0:
        return [{"start_bar": 0, "key": "C major"}]

    k = s.analyze("key")
    return [{"start_bar": 0, "key": f"{k.tonic.name} {k.mode}"}]


def _chord_templates() -> dict[str, np.ndarray]:
    templates: dict[str, np.ndarray] = {}
    for i, root in enumerate(_PITCH_CLASS_NAMES):
        templates[root] = np.roll(_MAJOR_TEMPLATE, i)
        templates[f"{root}m"] = np.roll(_MINOR_TEMPLATE, i)
    return templates


def _best_chord_template(vec: np.ndarray, templates: dict[str, np.ndarray]) -> str:
    best_name, best_score = "C", -1.0
    for name, tmpl in templates.items():
        score = float(np.dot(vec, tmpl))
        if score > best_score:
            best_score, best_name = score, name
    return best_name


def _infer_chords(pm: "pretty_midi.PrettyMIDI", bar_table: list[list[float]]) -> list[dict]:
    if pm.get_end_time() <= 0:
        return []
    chroma = pm.get_chroma(fs=100)
    if chroma.shape[1] == 0:
        return []
    templates = _chord_templates()
    chords: list[dict] = []
    last_chord = None
    for bar_idx, t in bar_table:
        frame = min(int(t * 100), chroma.shape[1] - 1)
        vec = chroma[:, frame]
        if vec.sum() == 0:
            continue
        name = _best_chord_template(vec, templates)
        if name != last_chord:
            chords.append({"bar": int(bar_idx), "beat": 0, "chord": name})
            last_chord = name
    return chords


def _detect_sections(pm: "pretty_midi.PrettyMIDI", bar_table: list[list[float]]) -> list[dict]:
    """Bar-level piano-roll self-similarity, novelty-curve peak detection.

    D016: `tension` is always null for inferred provenance; `boundary_lvl`
    is the novelty peak height quantised to 0-3.
    """
    n_bars = len(bar_table)
    if n_bars < 2:
        return [{"label": "A", "bars": [0, max(1, n_bars)], "energy": 0.5,
                  "tension": None, "boundary_lvl": 0}]

    fs = 10
    roll = pm.get_piano_roll(fs=fs)
    bar_times = [t for _, t in bar_table] + [pm.get_end_time()]

    bar_vectors = []
    for i in range(len(bar_times) - 1):
        start_f = int(bar_times[i] * fs)
        end_f = max(start_f + 1, int(bar_times[i + 1] * fs))
        start_f = min(start_f, roll.shape[1] - 1) if roll.shape[1] else 0
        end_f = min(end_f, roll.shape[1])
        if end_f <= start_f:
            bar_vectors.append(np.zeros(roll.shape[0]))
        else:
            bar_vectors.append(roll[:, start_f:end_f].mean(axis=1))
    bar_vectors = np.array(bar_vectors)

    norms = np.linalg.norm(bar_vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = bar_vectors / norms
    sim = normed @ normed.T

    novelty = np.zeros(len(bar_vectors))
    for i in range(1, len(bar_vectors)):
        novelty[i] = 1.0 - sim[i - 1, i]

    threshold = float(novelty.mean() + novelty.std())
    max_novelty = float(novelty.max())
    boundaries = [0]
    if max_novelty > 1e-9:
        for i in range(1, len(novelty)):
            if novelty[i] > threshold and novelty[i] > 1e-6:
                boundaries.append(i)
    boundaries = sorted(set(boundaries))

    sections = []
    for j, start in enumerate(boundaries):
        end = boundaries[j + 1] if j + 1 < len(boundaries) else n_bars
        if end <= start:
            continue
        seg_energy = float(bar_vectors[start:end].mean() / 127.0) if bar_vectors.size else 0.0
        seg_energy = min(max(seg_energy, 0.0), 1.0)
        boundary_novelty = float(novelty[start]) if start < len(novelty) else 0.0
        boundary_lvl = int(min(3, boundary_novelty / max(threshold, 1e-9))) if threshold > 0 else 0
        boundary_lvl = max(0, min(3, boundary_lvl))
        sections.append({
            "label": chr(ord("A") + j) if j < 26 else f"S{j}",
            "bars": [start, end],
            "energy": seg_energy,
            "tension": None,
            "boundary_lvl": boundary_lvl,
        })
    return sections


def _infer_role(inst: "pretty_midi.Instrument") -> str:
    if inst.is_drum:
        return "drums"
    prog = inst.program
    if 32 <= prog <= 39:
        return "bass"
    if 80 <= prog <= 87:
        return "lead"
    return "comping"


def _onset_histogram_16(onsets: list[float], bar_table: list[list[float]]) -> list[float]:
    bins = [0.0] * 16
    if not onsets or len(bar_table) < 1:
        return bins
    bar_starts = [t for _, t in bar_table]
    for onset in onsets:
        bar_i = _bar_index_for_time(onset, bar_starts)
        bar_start = bar_starts[bar_i]
        bar_end = bar_starts[bar_i + 1] if bar_i + 1 < len(bar_starts) else bar_start + (bar_starts[-1] - bar_starts[-2] if len(bar_starts) > 1 else 1.0)
        bar_dur = max(bar_end - bar_start, 1e-9)
        pos = (onset - bar_start) / bar_dur
        slot = int(pos * 16) % 16
        bins[slot] += 1.0
    peak = max(bins)
    if peak > 0:
        bins = [b / peak for b in bins]
    return bins


def _bar_index_for_time(t: float, bar_starts: list[float]) -> int:
    idx = 0
    for i, start in enumerate(bar_starts):
        if start <= t + 1e-9:
            idx = i
        else:
            break
    return idx


def _fraction_on_grid(onsets: list[float], bar_table: list[list[float]]) -> float:
    if not onsets:
        return 1.0
    bar_starts = [t for _, t in bar_table]
    on_grid = 0
    for onset in onsets:
        bar_i = _bar_index_for_time(onset, bar_starts)
        bar_start = bar_starts[bar_i]
        bar_end = bar_starts[bar_i + 1] if bar_i + 1 < len(bar_starts) else bar_start + (bar_starts[-1] - bar_starts[-2] if len(bar_starts) > 1 else 1.0)
        bar_dur = max(bar_end - bar_start, 1e-9)
        sixteenth = bar_dur / 16.0
        pos_in_bar = onset - bar_start
        nearest = round(pos_in_bar / sixteenth) * sixteenth
        distance_ms = abs(pos_in_bar - nearest) * 1000.0
        if distance_ms <= QUANTIZED_TOLERANCE_MS:
            on_grid += 1
    return on_grid / len(onsets)


def _analyse_track(idx: int, inst: "pretty_midi.Instrument", bar_table: list[list[float]]) -> dict:
    onsets = sorted(n.start for n in inst.notes)
    velocities = [n.velocity for n in inst.notes]
    pitch_range = [min(n.pitch for n in inst.notes), max(n.pitch for n in inst.notes)] if inst.notes else [0, 0]

    quantized = _fraction_on_grid(onsets, bar_table) >= QUANTIZED_FRACTION_THRESHOLD
    flat_velocity = (float(np.std(velocities)) < FLAT_VELOCITY_STD_THRESHOLD) if velocities else True

    return {
        "idx": idx,
        "name": inst.name or "",
        "role": _infer_role(inst),
        "program": None if inst.is_drum else inst.program,
        "percussion": inst.is_drum,
        "quantized": quantized,
        "flat_velocity": flat_velocity,
        "onset_profile_16": _onset_histogram_16(onsets, bar_table),
        "pitch_range": pitch_range,
        "microtonal": False,
    }
