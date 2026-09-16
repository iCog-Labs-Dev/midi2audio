"""tests/test_render_fluidsynth.py — Stage 2 FluidSynth rendering."""
from __future__ import annotations

import shutil

import numpy as np
import pytest
import soundfile as sf

from m2a.render.registry import render_all_stems
from tests.conftest import fixture_render_package

pytestmark = pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth CLI not installed")

FIXTURE_PACKAGE = fixture_render_package()


@pytest.mark.slow
def test_renders_four_stems(tmp_path):
    stems = render_all_stems(FIXTURE_PACKAGE, str(tmp_path))
    assert len(stems) == 4
    for stem in stems:
        assert __import__("pathlib").Path(stem.path).stat().st_size > 0


@pytest.mark.slow
def test_stems_are_48khz(tmp_path):
    stems = render_all_stems(FIXTURE_PACKAGE, str(tmp_path))
    for stem in stems:
        _, sr = sf.read(stem.path)
        assert sr == 48000


@pytest.mark.slow
def test_peak_within_headroom(tmp_path):
    stems = render_all_stems(FIXTURE_PACKAGE, str(tmp_path))
    for stem in stems:
        assert -12.5 < stem.peak_dbfs < -11.5


@pytest.mark.slow
def test_render_is_deterministic(tmp_path):
    stems1 = render_all_stems(FIXTURE_PACKAGE, str(tmp_path / "r1"))
    stems2 = render_all_stems(FIXTURE_PACKAGE, str(tmp_path / "r2"))
    for s1, s2 in zip(stems1, stems2):
        a, _ = sf.read(s1.path)
        b, _ = sf.read(s2.path)
        np.testing.assert_array_equal(a, b)


@pytest.mark.slow
def test_stem_roles_match_structure(tmp_path):
    stems = render_all_stems(FIXTURE_PACKAGE, str(tmp_path))
    roles = {s.role for s in stems}
    assert roles == {"drums", "bass", "comping", "lead"}
