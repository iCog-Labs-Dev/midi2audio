# m2a/orchestrator.py
"""Stage DAG runner for the deterministic spine: S0 (analysis) -> S1
(expressivization) -> S2 (render). Each stage is skipped on a cache hit,
keyed by content hash of its actual inputs, so editing the groove config
alone re-runs S1+S2 without re-analysing the MIDI, and a config-only S2
change (e.g. a different soundfont) does not re-run S0 or S1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from m2a import __version__ as VERSION
from m2a.analysis.from_midi import analyse_midi, load_structure
from m2a.artifacts import ArtifactStore, content_hash
from m2a.config import AudioConfig
from m2a.contracts import RenderPackage, Stem, StructureJSON
from m2a.groove.author import author_groove_spec
from m2a.groove.apply import DiffReport, apply_groove
from m2a.render.fluidsynth_r import file_hash, measure_peak_dbfs
from m2a.render.registry import render_stems, stem_filename


@dataclass(frozen=True)
class SpineResult:
    structure: StructureJSON
    expressive_midi: str
    stems: tuple[Stem, ...]
    diff_report: DiffReport


class Orchestrator:
    def __init__(self, config: AudioConfig, work_dir: str):
        self.config = config
        self.store = ArtifactStore(work_dir)

    def run_spine(self, package: RenderPackage) -> SpineResult:
        structure, structure_path = self._run_stage0(package)
        expressive_midi, diff_report = self._run_stage1(package, structure)
        stems = self._run_stage2(structure, expressive_midi)
        return SpineResult(structure=structure, expressive_midi=expressive_midi,
                            stems=tuple(stems), diff_report=diff_report)

    # ── Stage 0: analysis ────────────────────────────────────────────────────

    def _run_stage0(self, package: RenderPackage) -> tuple[StructureJSON, str]:
        s0_hash = content_hash(package.midi_path, config={"stage": "s0"}, code_version=VERSION)
        s0_dir = self.store.stage_dir("s0", s0_hash)
        structure_path = str(s0_dir / "structure.json")

        if not self.store.is_complete("s0", s0_hash):
            structure = analyse_midi(package.midi_path, structure_path)
            self.store.mark_complete("s0", s0_hash)
        else:
            structure = load_structure(structure_path)
        return structure, structure_path

    # ── Stage 1: expressivization ────────────────────────────────────────────

    def _run_stage1(self, package: RenderPackage, structure: StructureJSON) -> tuple[str, DiffReport]:
        groove_cfg = self.config.groove
        s1_hash = content_hash(
            package.midi_path,
            config={"stage": "s1", "style_intent": groove_cfg.style_intent, "seed": groove_cfg.seed},
            code_version=VERSION,
        )
        s1_dir = self.store.stage_dir("s1", s1_hash)
        expressive_path = str(s1_dir / "expressive.mid")
        report_path = s1_dir / "diff_report.json"
        spec_path = s1_dir / "groove_spec.json"

        if not self.store.is_complete("s1", s1_hash):
            spec = author_groove_spec(
                structure.model_dump(by_alias=True), groove_cfg.style_intent, groove_cfg.llm_fn,
            )
            spec_path.write_text(json.dumps(spec.model_dump(), indent=2))
            _, diff_report = apply_groove(
                package.midi_path, spec, structure, groove_cfg.seed, out_path=expressive_path,
            )
            report_path.write_text(json.dumps({
                "applied_timing_offsets": diff_report.applied_timing_offsets,
                "applied_velocity_changes": diff_report.applied_velocity_changes,
                "ghost_notes_inserted": diff_report.ghost_notes_inserted,
                "bar_crossing_flags": diff_report.bar_crossing_flags,
                "rule_violations": diff_report.rule_violations,
            }, indent=2))
            self.store.mark_complete("s1", s1_hash)
        else:
            data = json.loads(report_path.read_text())
            diff_report = DiffReport(
                applied_timing_offsets=tuple(tuple(x) for x in data["applied_timing_offsets"]),
                applied_velocity_changes=tuple(tuple(x) for x in data["applied_velocity_changes"]),
                ghost_notes_inserted=data["ghost_notes_inserted"],
                bar_crossing_flags=tuple(tuple(x) for x in data["bar_crossing_flags"]),
                rule_violations=tuple(data["rule_violations"]),
            )
        return expressive_path, diff_report

    # ── Stage 2: render ───────────────────────────────────────────────────────

    def _run_stage2(self, structure: StructureJSON, expressive_midi: str) -> list[Stem]:
        s2_hash = content_hash(
            expressive_midi,
            config={"stage": "s2", "soundfont": self.config.soundfont},
            code_version=VERSION,
        )
        s2_dir = self.store.stage_dir("s2", s2_hash)

        if not self.store.is_complete("s2", s2_hash):
            stems = render_stems(expressive_midi, structure, str(s2_dir), self.config.soundfont)
            self.store.mark_complete("s2", s2_hash)
        else:
            stems = self._load_cached_stems(structure, s2_dir)
        return stems

    @staticmethod
    def _load_cached_stems(structure: StructureJSON, s2_dir: Path) -> list[Stem]:
        stems = []
        for track in structure.tracks:
            path = str(s2_dir / stem_filename(track))
            stems.append(Stem(
                track=track.name or f"track{track.idx:02d}_{track.role}", role=track.role,
                path=path, microtonal=track.microtonal,
                peak_dbfs=measure_peak_dbfs(path), content_hash=file_hash(path),
            ))
        return stems
