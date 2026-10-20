"""tests/test_pipeline_run.py — M3 exit: the full pipeline runs end-to-end on the
fixture with the (offline) mocked endpoint, unattended, in under 60 s; the
manifest makes a re-run reproducible at zero cost."""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time

import pytest
import soundfile as sf

from m2a.config import default_config, load_config
from m2a.manifest import RunManifest
from m2a.orchestrator import Orchestrator
from m2a.package import build_render_package
from tests.conftest import FIXTURES
from tests.synth import render_events

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth CLI not installed")]

FIXTURE_MIDI = str(FIXTURES / "eight_bar.mid")


@pytest.fixture(scope="module")
def refs(tmp_path_factory):
    d = tmp_path_factory.mktemp("refs")
    for k, base in enumerate([40, 52, 64]):
        render_events(str(d / f"ref{k}.wav"), [(i * 0.5, 0.45, base + (i % 4)) for i in range(16)], 8.5)
    return str(d)


def _config():
    cfg = default_config()
    cfg.scoring.clap = "off"
    return cfg


def test_render_audio_cli_end_to_end_under_60s(tmp_path):
    t0 = time.perf_counter()
    result = subprocess.run([
        sys.executable, "scripts/run_pipeline.py", "render-audio", FIXTURE_MIDI,
        "--out", str(tmp_path / "out"),
    ], capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    assert result.returncode == 0, result.stderr
    assert elapsed < 60.0, elapsed
    out = tmp_path / "out"
    for name in ("manifest.json", "scores.json", "rough_mix.wav", "click.wav", "bar_table.json"):
        assert (out / name).exists(), name
    assert len(list((out / "stems").glob("*.wav"))) == 4


def test_run_with_reference_corpus(tmp_path, refs):
    pkg = build_render_package(FIXTURE_MIDI, str(tmp_path / "pkgs"))
    result = Orchestrator(_config(), str(tmp_path / "work")).run(pkg, str(tmp_path / "out"), refs_dir=refs)
    manifest = RunManifest.load(result.manifest_path)

    assert {s.role for s in result.stems} == {"drums", "bass", "comping", "lead"}
    for s in result.stems:
        info = sf.info(s.path)
        assert info.samplerate == 48_000 and info.frames > 0
    for outcome in manifest.stems:
        assert outcome.source in ("offline_mock", "dry")
        assert "fad" in outcome.scores                    # reference corpus wired into scoring
        assert 0.0 <= outcome.total_score <= 1.0
    assert manifest.decisions["D013"]["tau"] > 0
    assert set(manifest.inputs["reference_corpus"]) == {"ref0.wav", "ref1.wav", "ref2.wav"}
    assert [s.stage for s in manifest.stages] == ["s0", "s1", "s2", "s3"]
    assert manifest.totals["endpoint_calls"] == sum(c.n_calls for c in manifest.endpoint_calls)


def test_rerun_from_cache_reproduces_selection_at_zero_cost(tmp_path):
    """§8.1: a completed run, replayed with every endpoint response coming from the
    recorded request hashes, reproduces the same output hashes."""
    pkg = build_render_package(FIXTURE_MIDI, str(tmp_path / "pkgs"))
    first = Orchestrator(_config(), str(tmp_path / "work")).run(pkg, str(tmp_path / "out1"))

    cfg = _config()
    cfg.budget.max_calls = 0                                # replay must not buy anything
    second = Orchestrator(cfg, str(tmp_path / "work")).run(pkg, str(tmp_path / "out2"))

    assert [s.content_hash for s in first.stems] == [s.content_hash for s in second.stems]
    m2 = RunManifest.load(second.manifest_path)
    assert m2.totals["endpoint_calls"] == 0 and m2.totals["cache_hit_rate"] == 1.0
    assert all(st.cache_hit for st in m2.stages if st.stage != "s3")
    m1 = RunManifest.load(first.manifest_path)
    assert [s.request_hash for s in m1.stems] == [s.request_hash for s in m2.stems]


def test_impossible_floors_fall_back_to_dry_and_are_flagged(tmp_path, monkeypatch):
    from m2a.scoring import combine
    monkeypatch.setattr(combine, "ROLE_FLOORS",
                        {r: combine.ScoreFloors(rhythm_f=1.01) for r in combine.ROLE_FLOORS})
    pkg = build_render_package(FIXTURE_MIDI, str(tmp_path / "pkgs"))
    cfg = _config()
    cfg.search.seeds = 1
    cfg.search.n_prompt_variants = 1
    cfg.search.refine = False
    result = Orchestrator(cfg, str(tmp_path / "work")).run(pkg, str(tmp_path / "out"))
    manifest = json.loads(open(result.manifest_path).read())
    assert sorted(manifest["totals"]["stems_fallen_back"]) == ["bass", "comping", "drums", "lead"]
    assert all(s["fallback_reason"] == "all_candidates_below_floor" for s in manifest["stems"])


def test_strict_cli_exits_2_when_nothing_clears(tmp_path):
    profile = tmp_path / "strict.yaml"
    profile.write_text("search:\n  strength_grid: [0.95]\n  seeds: 1\n  n_prompt_variants: 1\n"
                       "  refine: false\nscoring:\n  clap: 'off'\n")
    result = subprocess.run([
        sys.executable, "scripts/run_pipeline.py", "render-audio", FIXTURE_MIDI,
        "--profile", str(profile), "--strict", "--out", str(tmp_path / "out"),
    ], capture_output=True, text=True)
    assert result.returncode == 2, (result.returncode, result.stderr[-2000:])


def test_contract_violation_exits_3(tmp_path):
    tuning = tmp_path / "t.json"
    tuning.write_text(json.dumps({"edo": 19, "base_tuning": 60.0, "method": "scala"}))
    result = subprocess.run([
        sys.executable, "scripts/run_pipeline.py", "render-audio", FIXTURE_MIDI,
        "--structure", str(FIXTURES / "eight_bar_structure.json"), "--tuning", str(tuning),
        "--out", str(tmp_path / "out"),
    ], capture_output=True, text=True)
    assert result.returncode == 3 and "edo" in result.stderr


def test_default_profile_loads():
    cfg = load_config("config/default.yaml")
    assert cfg.midi_gpt is False and cfg.restyle.endpoint == "offline"
