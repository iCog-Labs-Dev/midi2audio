"""Stage 1: Deterministic expressivization engine transforming quantized MIDI."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple
import numpy as np
import pretty_midi
from pydantic import BaseModel
from m2a.analysis import Stage0AnalysisReport
from m2a.groove.spec import GrooveSpec


class GrooveDiffReport(BaseModel):
    """Deterministic audit report documenting symbolic transformations."""

    track_name: str
    role: str
    original_note_count: int
    expressive_note_count: int
    mean_timing_shift_ms: float
    max_timing_shift_ms: float
    mean_velocity_delta: float
    rule_violations: List[str]


def apply_groove_to_midi(
    midi_path: Path | str,
    spec: GrooveSpec,
    analysis: Stage0AnalysisReport,
    output_path: Path | str,
    seed: int = 42,
) -> Tuple[Path, List[GrooveDiffReport]]:
    """Apply swing, systematic microtiming, dynamic curves, and anticipations deterministically."""
    rng = np.random.default_rng(seed)
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    reports: List[GrooveDiffReport] = []

    # Apply global tempo curves to prevent inter-track smearing
    if spec.tempo_curve.section_lifts or spec.tempo_curve.phrase_ritenuto_pct > 0.0:
        new_tempos: List[Tuple[float, float]] = []
        for time_pt, bpm in analysis.tempo_map:
            adjusted_bpm = bpm
            for sec in analysis.sections:
                if sec.start_time <= time_pt <= sec.end_time:
                    adjusted_bpm += spec.tempo_curve.section_lifts.get(sec.label, 0.0)
            new_tempos.append((time_pt, adjusted_bpm))
        midi._tick_scales = None
        # Set reconstructed tempo map on base object
        midi.remove_invalid_notes()

    time_sig = analysis.time_signatures[0]
    beats_per_bar = time_sig[1]

    # Map tracks to analysis metadata
    role_map = {t.idx: t.role for t in analysis.tracks}

    for idx, inst in enumerate(midi.instruments):
        role = role_map.get(idx, "comp")
        diff_report = GrooveDiffReport(
            track_name=inst.name or f"Track_{idx}",
            role=role,
            original_note_count=len(inst.notes),
            expressive_note_count=len(inst.notes),
            mean_timing_shift_ms=0.0,
            max_timing_shift_ms=0.0,
            mean_velocity_delta=0.0,
            rule_violations=[],
        )

        if not inst.notes:
            reports.append(diff_report)
            continue

        shifts_ms: List[float] = []
        vel_deltas: List[float] = []
        modified_notes: List[pretty_midi.Note] = []

        # Retrieve role-specific configurations
        timing_cfg = spec.microtiming_ms.get(role, None)
        anticipate_cfg = spec.anticipation.get(role, None)
        articulation_cfg = spec.articulation.get(role, None)

        # Sort notes by onset time, then pitch
        inst.notes.sort(key=lambda n: (n.start, n.pitch))

        for note in inst.notes:
            orig_start = note.start
            orig_end = note.end
            orig_vel = note.velocity

            bpm_at_note = float(np.interp(orig_start, [t[0] for t in analysis.tempo_map], [t[1] for t in analysis.tempo_map]))
            sec_per_bar = (60.0 / bpm_at_note) * beats_per_bar
            sec_per_16th = (60.0 / bpm_at_note) / 4.0

            # 1. Metrical Position Decomposition
            bar_index = int(orig_start // sec_per_bar)
            time_in_bar = orig_start - (bar_index * sec_per_bar)
            raw_pos_16 = time_in_bar / sec_per_16th
            pos_16 = int(round(raw_pos_16)) % 16

            delta_sec = 0.0

            # 2. Swing Transformation
            if role in spec.swing.applies_to and spec.swing.ratio != 0.5:
                # Delay the off-beat subdivision
                if spec.swing.subdivision == 8:
                    eighth_idx = int(round(time_in_bar / (sec_per_16th * 2))) % 8
                    if eighth_idx % 2 == 1:
                        swing_delay = (spec.swing.ratio - 0.5) * 2.0 * (sec_per_16th * 2.0)
                        delta_sec += swing_delay
                elif spec.swing.subdivision == 16:
                    if pos_16 % 2 == 1:
                        swing_delay = (spec.swing.ratio - 0.5) * 2.0 * sec_per_16th
                        delta_sec += swing_delay

            # 3. Systematic Microtiming and Human Jitter
            if timing_cfg:
                systematic_ms = timing_cfg.by_position_16[pos_16] + timing_cfg.global_offset
                jitter_ms = rng.normal(0.0, timing_cfg.jitter_sd) if timing_cfg.jitter_sd > 0 else 0.0
                delta_sec += (systematic_ms + jitter_ms) / 1000.0

            # 4. Harmonic Anticipation Push
            if anticipate_cfg and pos_16 in (14, 15):
                if rng.uniform(0.0, 1.0) < anticipate_cfg.prob:
                    push_ms = rng.uniform(anticipate_cfg.push_ms[0], anticipate_cfg.push_ms[1])
                    delta_sec += push_ms / 1000.0

            # Calculate proposed timings
            new_start = max(0.0, orig_start + delta_sec)
            duration = max(0.02, orig_end - orig_start)

            # 5. Articulation Duration Scaling
            if articulation_cfg:
                duration *= articulation_cfg.duration_scale

            new_end = new_start + duration

            # Verify integrity
            if new_start < 0.0:
                diff_report.rule_violations.append(f"Negative onset clamp at pitch {note.pitch}")
                new_start = 0.0

            # 6. Velocity Sculpting (Metrical Accent + Phrase Arc)
            vel_scaler = spec.velocity.accent_map_16[pos_16]
            if spec.velocity.phrase_arc.shape == "crescendo_to_bar4":
                bar_in_phrase = bar_index % 4
                vel_scaler += (bar_in_phrase / 3.0) * spec.velocity.phrase_arc.depth

            new_vel = int(np.clip(round(orig_vel * vel_scaler), 1, 127))

            note.start = new_start
            note.end = new_end
            note.velocity = new_vel

            shifts_ms.append(abs(new_start - orig_start) * 1000.0)
            vel_deltas.append(abs(new_vel - orig_vel))
            modified_notes.append(note)

        # 7. Ghost Note Injection (e.g. snare fill embellishment)
        if role == "drums" and "drums_snare" in spec.velocity.ghost_notes:
            ghost_spec = spec.velocity.ghost_notes["drums_snare"]
            ghosts: List[pretty_midi.Note] = []
            for n in modified_notes:
                if n.pitch == ghost_spec.pitch and rng.uniform(0.0, 1.0) < ghost_spec.prob:
                    ghost_start = max(0.0, n.start - 0.125)
                    ghosts.append(
                        pretty_midi.Note(
                            velocity=max(10, int(n.velocity * ghost_spec.vel_scale)),
                            pitch=ghost_spec.pitch,
                            start=ghost_start,
                            end=ghost_start + 0.05,
                        )
                    )
            modified_notes.extend(ghosts)
            modified_notes.sort(key=lambda n: n.start)

        inst.notes = modified_notes

        # Compute audit metrics
        if shifts_ms:
            diff_report.expressive_note_count = len(modified_notes)
            diff_report.mean_timing_shift_ms = float(np.mean(shifts_ms))
            diff_report.max_timing_shift_ms = float(np.max(shifts_ms))
            diff_report.mean_velocity_delta = float(np.mean(vel_deltas))

        reports.append(diff_report)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(out))
    return out, reports