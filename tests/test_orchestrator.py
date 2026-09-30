"""tests/test_orchestrator.py — deterministic-spine DAG caching."""
from __future__ import annotations

import shutil
from unittest.mock import patch

import pytest

from m2a.config import AudioConfig, GrooveConfig, default_config
from m2a.orchestrator import Orchestrator
from m2a.render.fluidsynth_r import FluidSynthRenderer
from tests.conftest import fixture_render_package

pytestmark = pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth CLI not installed")

FIXTURE_PACKAGE = fixture_render_package()


@pytest.mark.slow
def test_spine_produces_stems(tmp_path):
    result = Orchestrator(default_config(), str(tmp_path)).run_spine(FIXTURE_PACKAGE)
    assert len(result.stems) == 4


@pytest.mark.slow
def test_cache_hit_skips_render(tmp_path):
    orch = Orchestrator(default_config(), str(tmp_path))
    orch.run_spine(FIXTURE_PACKAGE)
    with patch.object(FluidSynthRenderer, "render", wraps=FluidSynthRenderer.render) as spy:
        orch.run_spine(FIXTURE_PACKAGE)  # second run
        spy.assert_not_called()


@pytest.mark.slow
def test_config_change_invalidates_s1_but_not_s0(tmp_path):
    orch1 = Orchestrator(default_config(), str(tmp_path))
    orch1.run_spine(FIXTURE_PACKAGE)

    with patch("m2a.orchestrator.analyse_midi") as spy_s0:
        different_config = AudioConfig(groove=GrooveConfig(style_intent="funk", seed=99))
        orch2 = Orchestrator(different_config, str(tmp_path))
        orch2.run_spine(FIXTURE_PACKAGE)
        spy_s0.assert_not_called()


@pytest.mark.slow
def test_rerun_produces_identical_stems(tmp_path):
    orch = Orchestrator(default_config(), str(tmp_path))
    result1 = orch.run_spine(FIXTURE_PACKAGE)
    result2 = orch.run_spine(FIXTURE_PACKAGE)
    hashes1 = sorted(s.content_hash for s in result1.stems)
    hashes2 = sorted(s.content_hash for s in result2.stems)
    assert hashes1 == hashes2
