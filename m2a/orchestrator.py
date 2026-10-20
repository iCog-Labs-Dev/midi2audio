# m2a/orchestrator.py
"""Stage DAG runner.

Deterministic spine: S0 (analysis) -> S1 (expressivization) -> S2 (render).
Each stage is skipped on a cache hit, keyed by content hash of its actual
inputs, so editing the groove config alone re-runs S1+S2 without
re-analysing the MIDI, and a config-only S2 change (e.g. a different
soundfont) does not re-run S0 or S1.

S0 honours a caller-supplied structure.json (D005) and inference runs only
when none exists. The package is validated against the five contract
invariants before anything else runs (P3).

`run()` adds S3 — per-stem restyle search through one endpoint, gated by
the M2 scorer and the budget guard — and writes the run outputs: selected
stems, a rough mix, a click track, bar_table.json, scores.json and the
provenance manifest (§8.1). S3 has no stage sentinel of its own: the
request-hash cache makes a re-run free, and a re-run must still re-score
and re-record its calls.
"""
from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

from m2a import __version__ as VERSION
from m2a.analysis.from_midi import analyse_midi, load_structure
from m2a.artifacts import ArtifactStore, content_hash, file_sha256
from m2a.config import AudioConfig, config_fingerprint
from m2a.contracts import RenderPackage, Stem, StructureJSON
from m2a.groove.apply import DiffReport, apply_groove
from m2a.groove.author import author_groove_spec
from m2a.manifest import RunManifest, StageRecord, StemOutcome
from m2a.package import validate_package
from m2a.prompts.generate import generate_prompts
from m2a.prompts.stylecard import make_style_card
from m2a.render.fluidsynth_r import SAMPLE_RATE, file_hash, measure_peak_dbfs
from m2a.render.registry import RendererRegistry, render_stems, stem_filename
from m2a.render.tuning import TuningSpec, load_tuning
from m2a.restyle.base import RestyleEndpoint
from m2a.scoring.clap_sim import default_clap_encoder
from m2a.scoring.combine import StemScorer
from m2a.scoring.fad import ReferenceSet
from m2a.scoring.notes import default_transcriber
from m2a.search.loop import BudgetGuard, CandidateCache, SearchLoop, reason_histogram


@dataclass(frozen=True)
class SpineResult:
    structure: StructureJSON
    expressive_midi: str
    stems: tuple[Stem, ...]
    diff_report: DiffReport


@dataclass(frozen=True)
class RunResult:
    spine: SpineResult
    out_dir: str
    stems: tuple[Stem, ...]          # the selected stem per track (restyled or dry)
    rough_mix: str
    manifest_path: str


def make_endpoint(config: AudioConfig) -> RestyleEndpoint:
    name = config.restyle.endpoint
    kwargs = {} if config.restyle.usd_per_call is None else {"usd_per_call": config.restyle.usd_per_call}
    if name == "offline":
        from m2a.restyle.offline import OfflineEndpoint
        return OfflineEndpoint(**kwargs)
    if name == "stable_audio":
        from m2a.restyle.stable_audio import StableAudioEndpoint
        return StableAudioEndpoint(**kwargs)
    raise ValueError(f"unknown restyle endpoint {name!r}")


class Orchestrator:
    def __init__(self, config: AudioConfig, work_dir: str):
        self.config = config
        self.store = ArtifactStore(work_dir)
        self.stage_records: list[StageRecord] = []

    def _record(self, stage: str, key: str, hit: bool, t0: float, outputs: list[str], cfg: dict) -> None:
        self.stage_records.append(StageRecord(stage=stage, key=key, cache_hit=hit,
                                              wall_clock_s=round(time.perf_counter() - t0, 4),
                                              outputs=outputs, config=cfg))

    def run_spine(self, package: RenderPackage) -> SpineResult:
        structure, structure_path, tuning = self._run_stage0(package)
        expressive_midi, diff_report = self._run_stage1(package, structure, structure_path)
        stems = self._run_stage2(structure, expressive_midi, tuning)
        return SpineResult(structure=structure, expressive_midi=expressive_midi,
                           stems=tuple(stems), diff_report=diff_report)

    # ── Stage 0: analysis ────────────────────────────────────────────────────

    def _load_tuning(self, package: RenderPackage, edo: int, base: float) -> TuningSpec:
        if package.tuning_path and Path(package.tuning_path).exists():
            return load_tuning(package.tuning_path)
        return TuningSpec(edo=edo, base_tuning=base)

    def _run_stage0(self, package: RenderPackage) -> tuple[StructureJSON, str, TuningSpec]:
        t0 = time.perf_counter()
        supplied = bool(package.structure_path) and Path(package.structure_path).exists()
        inputs = [package.midi_path] + ([package.structure_path] if supplied else [])
        s0_hash = content_hash(*inputs, config={"stage": "s0", "supplied": supplied}, code_version=VERSION)
        s0_dir = self.store.stage_dir("s0", s0_hash)
        structure_path = str(s0_dir / "structure.json")

        hit = self.store.is_complete("s0", s0_hash)
        if hit:
            structure = load_structure(structure_path)
        elif supplied:
            shutil.copyfile(package.structure_path, structure_path)
            structure = load_structure(structure_path)
        else:
            tuning = self._load_tuning(package, 12, 60.0)
            structure = analyse_midi(package.midi_path, structure_path, edo=tuning.edo,
                                     base_tuning=tuning.base_tuning)
        tuning = self._load_tuning(package, structure.edo, structure.base_tuning)
        validate_package(structure, tuning, package.midi_path)
        if not hit:
            self.store.mark_complete("s0", s0_hash)
        self._record("s0", s0_hash, hit, t0, [structure_path], {"supplied_structure": supplied})
        return structure, structure_path, tuning

    # ── Stage 1: expressivization ────────────────────────────────────────────

    def _run_stage1(self, package: RenderPackage, structure: StructureJSON,
                    structure_path: str) -> tuple[str, DiffReport]:
        t0 = time.perf_counter()
        groove_cfg = self.config.groove
        cfg = {"stage": "s1", "style_intent": groove_cfg.style_intent, "seed": groove_cfg.seed}
        s1_hash = content_hash(package.midi_path, structure_path, config=cfg, code_version=VERSION)
        s1_dir = self.store.stage_dir("s1", s1_hash)
        expressive_path = str(s1_dir / "expressive.mid")
        report_path = s1_dir / "diff_report.json"
        spec_path = s1_dir / "groove_spec.json"

        hit = self.store.is_complete("s1", s1_hash)
        if not hit:
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
        self._record("s1", s1_hash, hit, t0, [expressive_path, str(spec_path)], cfg)
        return expressive_path, diff_report

    # ── Stage 2: render ───────────────────────────────────────────────────────

    def _run_stage2(self, structure: StructureJSON, expressive_midi: str,
                    tuning: TuningSpec) -> list[Stem]:
        t0 = time.perf_counter()
        cfg = {"stage": "s2", "soundfont": self.config.soundfont,
               "sfz": self.config.sfz_instruments, "tuning": tuning.__dict__,
               "routing": [(t.idx, t.role, t.microtonal) for t in structure.tracks]}
        s2_hash = content_hash(expressive_midi, config=cfg, code_version=VERSION)
        s2_dir = self.store.stage_dir("s2", s2_hash)

        hit = self.store.is_complete("s2", s2_hash)
        if not hit:
            stems = render_stems(expressive_midi, structure, str(s2_dir), self.config.soundfont,
                                 RendererRegistry(self.config), tuning)
            self.store.mark_complete("s2", s2_hash)
        else:
            stems = self._load_cached_stems(structure, s2_dir)
        self._record("s2", s2_hash, hit, t0, [s.path for s in stems], {"soundfont": self.config.soundfont})
        return stems

    @staticmethod
    def _load_cached_stems(structure: StructureJSON, s2_dir: Path) -> list[Stem]:
        stems = []
        for track in structure.tracks:
            path = str(s2_dir / stem_filename(track))
            stems.append(Stem(
                track=track.label, role=track.role,
                path=path, microtonal=track.microtonal,
                peak_dbfs=measure_peak_dbfs(path), content_hash=file_hash(path),
            ))
        return stems

    # ── Full run: spine + S3 restyle search ──────────────────────────────────

    def run(self, package: RenderPackage, out_dir: str, refs_dir: str | None = None,
            endpoint: RestyleEndpoint | None = None) -> RunResult:
        t_start = time.perf_counter()
        endpoint = endpoint or make_endpoint(self.config)
        spine = self.run_spine(package)
        structure = spine.structure
        tuning = self._load_tuning(package, structure.edo, structure.base_tuning)
        out = Path(out_dir)
        (out / "stems").mkdir(parents=True, exist_ok=True)

        cfg_dict = config_fingerprint(self.config)
        manifest = RunManifest(
            run_id=Path(package.root).name or package.content_hash[:16],
            code_version=VERSION,
            config_hash=content_hash(config=cfg_dict, code_version=VERSION),
            inputs={
                "midi_sha256": file_sha256(package.midi_path),
                "structure_sha256": file_sha256(package.structure_path) if Path(package.structure_path).exists() else "",
                "tuning_sha256": file_sha256(package.tuning_path) if Path(package.tuning_path).exists() else "",
                "package_hash": package.content_hash,
                "reference_corpus": {},
            },
        )

        refs = None
        if refs_dir:
            refs = ReferenceSet.from_dir(refs_dir)
            manifest.inputs["reference_corpus"] = {Path(p).name: file_sha256(p) for p in refs.paths}
        clap = default_clap_encoder(self.config.scoring.clap)
        scorer = StemScorer(spine.expressive_midi, structure, list(spine.stems), tuning, refs, clap,
                            transcriber=default_transcriber(self.config.scoring.transcriber))
        style_card = make_style_card(refs_dir, user_notes=self.config.restyle.style_notes)
        budget = BudgetGuard(self.config.budget.max_calls, self.config.budget.max_usd,
                             self.config.budget.max_calls_per_stem)
        cache = CandidateCache(str(self.store.root / "restyle_cache"))
        loop = SearchLoop(endpoint, scorer, budget, cache, str(self.store.root / "restyle_work"), manifest)

        t3 = time.perf_counter()
        selected: list[Stem] = []
        scores_out = {}
        for track, stem in zip(structure.tracks, spine.stems):
            prompts = generate_prompts(structure, track, style_card,
                                       n_variants=max(1, self.config.search.n_prompt_variants))
            result = loop.search(stem, prompts, structure, self.config.search)
            best = result.best
            if best.endpoint == "dry":
                best = scorer.score(best)   # the shipped dry stem is scored too, for the record
            dest = out / "stems" / stem_filename(track)
            shutil.copyfile(best.audio_path, dest)
            selected.append(Stem(track=stem.track, role=stem.role, path=str(dest),
                                 microtonal=stem.microtonal, peak_dbfs=measure_peak_dbfs(str(dest)),
                                 content_hash=file_hash(str(dest))))
            outcome = StemOutcome(
                stem=stem.track, role=stem.role,
                source="dry" if result.best.endpoint == "dry" else best.endpoint,
                audio_path=str(dest), request_hash=best.request_hash, total_score=best.total_score,
                scores=dict(best.scores), candidates_tried=len(result.candidates),
                reject_reasons=dict(reason_histogram([c for c in result.candidates if not c.accepted])),
                fallback_reason=result.fallback_reason, stopped_by_budget=result.stopped_by_budget,
            )
            manifest.stems.append(outcome)
            scores_out[stem.track] = {
                "source": outcome.source, "divergence": best.strength, "seed": best.seed,
                "prompt": best.prompt, "total_score": best.total_score, "scores": outcome.scores,
                "fallback_reason": outcome.fallback_reason,
                "candidates": [{"divergence": c.strength, "seed": c.seed, "prompt": c.prompt,
                                "accepted": c.accepted, "reject_reason": c.reject_reason,
                                "total_score": c.total_score, "scores": dict(c.scores)}
                               for c in result.candidates],
            }
        manifest.stages = self.stage_records + [StageRecord(
            stage="s3", key=manifest.config_hash, cache_hit=False,
            wall_clock_s=round(time.perf_counter() - t3, 4),
            outputs=[s.path for s in selected],
            config={"endpoint": endpoint.name, "endpoint_version": endpoint.version},
        )]
        manifest.decisions = {
            "D001": "round-half-even, offsets summed then rounded once",
            "D002": "linear crossfade after cross-correlation alignment, 2-bar overlap",
            "D004": {"floor_p95_cents": 15.0},
            "D005": "supplied structure.json accepted; inferred when absent",
            "D008": {"midi_gpt": self.config.midi_gpt},
            "D009": "groove keys are roles; track:<idx> overrides",
            "D011": {"endpoint": endpoint.name, "strength_semantics": endpoint.strength_semantics},
            "D012": {"on_no_candidate": self.config.search.on_no_candidate},
            "D013": {"embedder": refs.embedder_name if refs else None, "tau": refs.tau if refs else None},
            "D016": "tension null for inferred provenance",
        }

        (out / "scores.json").write_text(json.dumps(scores_out, indent=2))
        (out / "bar_table.json").write_text(json.dumps(structure.bar_table))
        write_click_track(structure, str(out / "click.wav"),
                          duration=max(sf.info(s.path).duration for s in selected))
        rough = write_rough_mix(selected, str(out / "rough_mix.wav"))
        manifest.finalise(time.perf_counter() - t_start)
        manifest_path = manifest.write(str(out / "manifest.json"))
        return RunResult(spine=spine, out_dir=str(out), stems=tuple(selected), rough_mix=rough,
                         manifest_path=manifest_path)


def write_rough_mix(stems: list[Stem], out_path: str, peak_dbfs: float = -1.0) -> str:
    """Unprocessed stereo sum of the selected stems (the M4 mix replaces this)."""
    tracks = []
    for s in stems:
        audio, sr = sf.read(s.path, always_2d=True)
        if sr != SAMPLE_RATE:
            raise ValueError(f"{s.path} is {sr} Hz; stems must be {SAMPLE_RATE} Hz")
        tracks.append(audio if audio.shape[1] == 2 else np.repeat(audio[:, :1], 2, axis=1))
    n = max(len(t) for t in tracks)
    mix = np.zeros((n, 2))
    for t in tracks:
        mix[:len(t)] += t
    peak = float(np.max(np.abs(mix))) or 1.0
    sf.write(out_path, mix / peak * 10 ** (peak_dbfs / 20.0), SAMPLE_RATE)
    return out_path


def write_click_track(structure: StructureJSON, out_path: str, duration: float) -> str:
    """Downbeat (1.5 kHz) and beat (1 kHz) blips from the bar table and meter."""
    y = np.zeros(int(duration * SAMPLE_RATE) + 1)
    blip_t = np.arange(int(0.02 * SAMPLE_RATE)) / SAMPLE_RATE
    meter = {int(b): m for b, m in structure.meter}
    times = [t for _, t in structure.bar_table] + [duration]
    num = 4
    for i in range(len(times) - 1):
        num = int(str(meter.get(i, f"{num}/4")).split("/")[0])
        for beat in range(num):
            t = times[i] + beat * (times[i + 1] - times[i]) / num
            f = 1500.0 if beat == 0 else 1000.0
            a = int(t * SAMPLE_RATE)
            seg = 0.5 * np.sin(2 * np.pi * f * blip_t) * np.exp(-blip_t / 0.005)
            b = min(len(y), a + len(seg))
            if a < len(y):
                y[a:b] += seg[:b - a]
    sf.write(out_path, y, SAMPLE_RATE)
    return out_path
