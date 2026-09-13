"""Stage 0: Multitrack MIDI structural, harmonic, and rhythmic analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pretty_midi
from pydantic import BaseModel, Field
from m2a.manifest import compute_file_sha256


class TrackAnalysis(BaseModel):
    """Per-track symbolic characteristics and quantization diagnostics."""

    idx: int
    name: str
    program: int
    is_drum: bool
    role: str
    pitch_range: Tuple[int, int]
    polyphony_max: int
    note_density_per_bar: float
    quantized: bool
    flat_velocity: bool
    onset_profile_16: List[float]


class SectionAnalysis(BaseModel):
    """Section boundaries and relative structural energy."""

    label: str
    bars: Tuple[int, int]
    start_time: float
    end_time: float
    energy: float


class Stage0AnalysisReport(BaseModel):
    """Comprehensive Stage 0 analysis contract consumed by downstream stages."""

    source_midi_hash: str
    ppq: int
    total_duration_sec: float
    tempo_map: List[Tuple[float, float]]  # (time_sec, bpm)
    time_signatures: List[Tuple[float, int, int]]  # (time_sec, num, denom)
    estimated_key: str
    sections: List[SectionAnalysis]
    tracks: List[TrackAnalysis]


def infer_instrument_role(program: int, is_drum: bool, name: str) -> str:
    """Infer compositional role from General MIDI program numbers and channel flags."""
    if is_drum:
        return "drums"
    name_clean = name.lower()
    if any(k in name_clean for k in ["kick", "snare", "hat", "drum", "perc"]):
        return "drums"
    if 32 <= program <= 39 or "bass" in name_clean:
        return "bass"
    if 0 <= program <= 7:  # Piano
        return "comp"
    if 24 <= program <= 31:  # Guitar
        return "comp"
    if 48 <= program <= 55 or 88 <= program <= 95:  # Strings / Pads
        return "pads"
    if 40 <= program <= 43 or 56 <= program <= 79:  # Solo Strings, Brass, Reed
        return "lead"
    return "ornament"


def analyze_quantization_and_velocity(
    notes: List[pretty_midi.Note],
    bpm: float,
    threshold_ms: float = 12.0,
    flat_vel_std: float = 3.0,
) -> Tuple[bool, bool, List[float]]:
    """Determine if note onsets are tightly quantized and detect flat velocity distributions."""
    if not notes:
        return True, True, [0.0] * 16

    sec_per_beat = 60.0 / max(bpm, 1e-6)
    sixteenth_sec = sec_per_beat / 4.0
    diffs_ms: List[float] = []
    positions: List[int] = []
    velocities: List[int] = []

    for n in notes:
        velocities.append(n.velocity)
        pos_exact = n.start / sixteenth_sec
        pos_quantized = round(pos_exact)
        dev_ms = abs(n.start - (pos_quantized * sixteenth_sec)) * 1000.0
        diffs_ms.append(dev_ms)
        positions.append(int(pos_quantized % 16))

    quantized = bool(np.mean(diffs_ms) < threshold_ms)
    flat_velocity = bool(np.std(velocities) < flat_vel_std)

    histogram = np.zeros(16, dtype=np.float64)
    for pos in positions:
        histogram[pos] += 1.0
    total = np.sum(histogram)
    if total > 0:
        histogram /= total

    return quantized, flat_velocity, [float(x) for x in histogram]


def analyze_midi(midi_path: Path | str, config: Dict[str, Any]) -> Stage0AnalysisReport:
    """Analyze a multitrack MIDI file and generate an immutable structural description."""
    path = Path(midi_path)
    file_hash = compute_file_sha256(path)
    midi_data = pretty_midi.PrettyMIDI(str(path))

    # Base tempo extraction
    tempo_changes = midi_data.get_tempo_changes()
    tempo_map: List[Tuple[float, float]] = []
    for t, bpm in zip(tempo_changes[0], tempo_changes[1]):
        tempo_map.append((float(t), float(bpm)))
    if not tempo_map:
        tempo_map = [(0.0, 120.0)]
    primary_bpm = tempo_map[0][1]

    # Time Signatures
    time_sigs: List[Tuple[float, int, int]] = []
    for ts in midi_data.time_signature_changes:
        time_sigs.append((float(ts.time), int(ts.numerator), int(ts.denominator)))
    if not time_sigs:
        time_sigs = [(0.0, 4, 4)]

    # Estimate tonal center via pitch-class profile
    total_chroma = np.zeros(12)
    for inst in midi_data.instruments:
        if not inst.is_drum:
            total_chroma += np.sum(inst.get_chroma(), axis=1)
    pitch_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key_root = pitch_names[int(np.argmax(total_chroma))] if np.sum(total_chroma) > 0 else "C"
    estimated_key = f"{key_root} Major"

    # Track-level analysis
    tracks: List[TrackAnalysis] = []
    sec_per_bar = (60.0 / primary_bpm) * time_sigs[0][1]
    total_bars = max(1, math.ceil(midi_data.get_end_time() / sec_per_bar))

    for idx, inst in enumerate(midi_data.instruments):
        notes = inst.notes
        if not notes:
            continue
        pitches = [n.pitch for n in notes]
        p_range = (min(pitches), max(pitches))
        role = infer_instrument_role(inst.program, inst.is_drum, inst.name)
        q_pass, flat_vel, profile = analyze_quantization_and_velocity(
            notes,
            bpm=primary_bpm,
            threshold_ms=config.get("stage0_analysis", {}).get("quantization_threshold_ms", 12.0),
            flat_vel_std=config.get("stage0_analysis", {}).get("flat_velocity_threshold_std", 3.0),
        )

        # Measure local polyphony
        step_dt = 0.05
        timeline_len = math.ceil(midi_data.get_end_time() / step_dt) + 1
        poly_arr = np.zeros(timeline_len, dtype=int)
        for n in notes:
            s_idx = int(n.start / step_dt)
            e_idx = int(n.end / step_dt)
            poly_arr[s_idx : max(s_idx + 1, e_idx)] += 1
        max_poly = int(np.max(poly_arr)) if len(poly_arr) > 0 else 0

        tracks.append(
            TrackAnalysis(
                idx=idx,
                name=inst.name or f"Track_{idx}",
                program=inst.program,
                is_drum=inst.is_drum,
                role=role,
                pitch_range=p_range,
                polyphony_max=max_poly,
                note_density_per_bar=float(len(notes) / total_bars),
                quantized=q_pass,
                flat_velocity=flat_vel,
                onset_profile_16=profile,
            )
        )

    # Coarse Structural Segmentation (A/B)
    midpoint_bar = total_bars // 2
    sections = [
        SectionAnalysis(
            label="A",
            bars=(0, midpoint_bar),
            start_time=0.0,
            end_time=float(midpoint_bar * sec_per_bar),
            energy=0.45,
        ),
        SectionAnalysis(
            label="B",
            bars=(midpoint_bar, total_bars),
            start_time=float(midpoint_bar * sec_per_bar),
            end_time=float(midi_data.get_end_time()),
            energy=0.75,
        ),
    ]

    return Stage0AnalysisReport(
        source_midi_hash=file_hash,
        ppq=midi_data.resolution,
        total_duration_sec=float(midi_data.get_end_time()),
        tempo_map=tempo_map,
        time_signatures=time_sigs,
        estimated_key=estimated_key,
        sections=sections,
        tracks=tracks,
    )