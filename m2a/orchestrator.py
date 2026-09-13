"""Content-addressable DAG pipeline orchestrator managing stages 0, 1, and 2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from m2a.analysis import analyze_midi
from m2a.groove.apply import apply_groove_to_midi
from m2a.groove.spec import GrooveSpec
from m2a.manifest import ArtifactManifest, compute_dict_sha256, compute_file_sha256
from m2a.render.registry import render_all_stems


class PipelineOrchestrator:
    """Manages deterministic pipeline DAG execution, artifact immutability, and state caching."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.artifacts_dir = Path(config.get("pipeline", {}).get("artifacts_dir", "artifacts"))
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

    def run_deterministic_spine(self, midi_path: Path | str) -> Dict[str, Any]:
        """Execute Stage 0 (Analysis), Stage 1 (Groove), and Stage 2 (Rendering)."""
        input_midi = Path(midi_path)
        source_hash = compute_file_sha256(input_midi)

        # -------------------------------------------------------------
        # STAGE 0: Analysis
        # -------------------------------------------------------------
        stage0_dir = self.artifacts_dir / "stage0_analysis"
        stage0_manifest_path = stage0_dir / "manifest.json"
        analysis_json_path = stage0_dir / "analysis.json"

        if stage0_manifest_path.is_file() and analysis_json_path.is_file():
            with open(analysis_json_path, "r", encoding="utf-8") as f:
                analysis_data = json.load(f)
            from m2a.analysis import Stage0AnalysisReport
            analysis = Stage0AnalysisReport(**analysis_data)
        else:
            analysis = analyze_midi(input_midi, self.config)
            stage0_dir.mkdir(parents=True, exist_ok=True)
            with open(analysis_json_path, "w", encoding="utf-8") as f:
                f.write(analysis.model_dump_json(indent=2))

            manifest0 = ArtifactManifest(
                stage_name="stage0_analysis",
                input_hashes={"source_midi": source_hash},
                output_hashes={"analysis_json": compute_file_sha256(analysis_json_path)},
                config_snapshot=self.config.get("stage0_analysis", {}),
            )
            manifest0.write_sidecar(stage0_manifest_path)

        # -------------------------------------------------------------
        # STAGE 1: Symbolic Expressivization
        # -------------------------------------------------------------
        stage1_dir = self.artifacts_dir / "stage1_groove"
        groove_midi_path = stage1_dir / "expressive.mid"
        diff_report_path = stage1_dir / "diff_report.json"
        stage1_manifest_path = stage1_dir / "manifest.json"

        groove_spec = GrooveSpec(**self.config.get("stage1_groove", {}))
        groove_spec_hash = compute_dict_sha256(groove_spec.model_dump())

        if not (groove_midi_path.is_file() and stage1_manifest_path.is_file()):
            _, diff_reports = apply_groove_to_midi(
                midi_path=input_midi,
                spec=groove_spec,
                analysis=analysis,
                output_path=groove_midi_path,
            )
            with open(diff_report_path, "w", encoding="utf-8") as f:
                json.dump([r.model_dump() for r in diff_reports], f, indent=2)

            manifest1 = ArtifactManifest(
                stage_name="stage1_groove",
                input_hashes={"source_midi": source_hash, "groove_spec": groove_spec_hash},
                output_hashes={
                    "expressive_midi": compute_file_sha256(groove_midi_path),
                    "diff_report": compute_file_sha256(diff_report_path),
                },
                config_snapshot=groove_spec.model_dump(),
            )
            manifest1.write_sidecar(stage1_manifest_path)

        # -------------------------------------------------------------
        # STAGE 2: Deterministic Stem Render
        # -------------------------------------------------------------
        stage2_dir = self.artifacts_dir / "stage2_render"
        stage2_manifest_path = stage2_dir / "manifest.json"

        render_session = render_all_stems(
            expressive_midi_path=groove_midi_path,
            analysis=analysis,
            output_dir=stage2_dir,
            config=self.config,
        )

        stem_hashes = {stem.stem_name: compute_file_sha256(stem.audio_path) for stem in render_session.stems}
        manifest2 = ArtifactManifest(
            stage_name="stage2_render",
            input_hashes={"expressive_midi": compute_file_sha256(groove_midi_path)},
            output_hashes=stem_hashes,
            config_snapshot=self.config.get("pipeline", {}),
            metadata={"bar_timestamps": render_session.bar_timestamps},
        )
        manifest2.write_sidecar(stage2_manifest_path)

        return {
            "analysis": analysis,
            "groove_midi_path": str(groove_midi_path),
            "stems": [s.model_dump() for s in render_session.stems],
            "bar_timestamps": render_session.bar_timestamps,
        }