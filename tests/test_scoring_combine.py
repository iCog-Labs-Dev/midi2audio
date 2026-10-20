"""tests/test_scoring_combine.py — weighted sum + hard floors (Step 2.5)."""
from __future__ import annotations

import math
import shutil

import pytest

from m2a.contracts import Candidate
from m2a.scoring.combine import (
    ROLE_FLOORS, ROLE_WEIGHTS, ScoreWeights, StemScorer, combine, weights_for,
)


def _cand(stem="bass"):
    return Candidate(stem=stem, audio_path="/tmp/x.wav", prompt="p", negative_prompt="n",
                     strength=0.7, seed=1, endpoint="sa", endpoint_version="2.5",
                     scores=(), accepted=False, reject_reason=None)


GOOD = {"rhythm": 0.95, "notes": 0.9, "clap": 0.3, "fad": 5.0, "shift_sec": 0.01}


def test_below_rhythm_floor_rejected():
    c = combine(_cand(), {**GOOD, "rhythm": 0.4}, "bass", tau=10.0)
    assert not c.accepted and c.reject_reason == "rhythm_floor"
    assert dict(c.scores)["rhythm"] == 0.4


def test_below_notes_floor_rejected():
    c = combine(_cand(), {**GOOD, "notes": 0.5}, "bass", tau=10.0)
    assert c.reject_reason == "notes_floor"


def test_excessive_shift_rejected():
    c = combine(_cand(), {**GOOD, "shift_sec": 0.24}, "bass", tau=10.0)
    assert c.reject_reason == "shift_floor"


def test_microtonal_tuning_floor():
    c = combine(_cand("lead"), {**GOOD, "tuning": 22.0}, "lead", microtonal=True, tau=10.0)
    assert c.reject_reason == "tuning_floor"
    ok = combine(_cand("lead"), {**GOOD, "tuning": 4.0}, "lead", microtonal=True, tau=10.0)
    assert ok.accepted


def test_above_floors_accepted_with_total():
    c = combine(_cand(), GOOD, "bass", tau=10.0)
    assert c.accepted and c.reject_reason is None
    assert 0.0 < c.total_score <= 1.0


def test_pure():
    assert combine(_cand(), GOOD, "bass", tau=10.0) == combine(_cand(), GOOD, "bass", tau=10.0)


def test_lower_fad_scores_higher():
    """Regression for v1.0 bug A — the most important test in scoring/."""
    near = combine(_cand(), {**GOOD, "fad": 5.0}, "bass", tau=10.0)
    far = combine(_cand(), {**GOOD, "fad": 50.0}, "bass", tau=10.0)
    assert near.total_score > far.total_score


def test_role_weights_sum_to_one():
    for role, w in ROLE_WEIGHTS.items():
        assert abs(w.rhythm + w.notes + w.clap + w.fad + w.tuning - 1.0) < 1e-9, role
        mw = weights_for(role, microtonal=True)
        assert abs(mw.rhythm + mw.notes + mw.clap + mw.fad + mw.tuning - 1.0) < 1e-9, role


def test_bad_weights_rejected():
    with pytest.raises(ValueError):
        ScoreWeights(rhythm=0.5, notes=0.5, clap=0.5, fad=0.0)


@pytest.mark.parametrize("role", list(ROLE_WEIGHTS))
@pytest.mark.parametrize("raw", [
    {"rhythm": 1.0, "notes": 1.0, "clap": 2.0, "fad": 0.0},
    {"rhythm": 0.9, "notes": 0.8, "clap": -2.0, "fad": 1e6},
    {"rhythm": 0.95, "notes": None, "clap": None, "fad": None},
])
def test_total_score_bounded(role, raw):
    c = combine(_cand(role), raw, role, tau=10.0)
    assert 0.0 <= c.total_score <= 1.0


def test_drums_drop_notes_axis():
    raw = {"rhythm": 0.9, "notes": None, "clap": 0.2, "fad": 8.0}
    c = combine(_cand("drums"), raw, "drums", tau=10.0)
    assert c.accepted and 0.0 <= c.total_score <= 1.0
    assert "notes" not in dict(c.scores)
    # renormalised: only the rhythm + clap + fad weights count
    w = ROLE_WEIGHTS["drums"]
    expected = (w.rhythm * 0.9 + w.clap * (0.2 + 2.0) / 4.0 + w.fad * math.exp(-8.0 / 10.0))
    assert c.total_score == pytest.approx(expected / (w.rhythm + w.clap + w.fad))


def test_drum_floors_ignore_notes():
    assert ROLE_FLOORS["drums"].notes_f1 == 0.0


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth CLI not installed")
def test_dry_fixture_stems_clear_every_floor(tmp_path):
    """Integration: a perfect deterministic render must pass its own gate, or the
    search would fall back to dry on every stem forever."""
    from m2a.analysis.from_midi import load_structure
    from m2a.render.registry import render_all_stems
    from tests.conftest import fixture_render_package

    pkg = fixture_render_package()
    stems = render_all_stems(pkg, str(tmp_path))
    scorer = StemScorer(pkg.midi_path, load_structure(pkg.structure_path), stems)
    for stem in stems:
        c = scorer.score(Candidate(stem=stem.track, audio_path=stem.path, prompt="", negative_prompt="",
                                   strength=0.0, seed=0, endpoint="dry", endpoint_version="",
                                   scores=(), accepted=False, reject_reason=None))
        assert c.accepted, (stem.track, c.reject_reason, c.scores)
