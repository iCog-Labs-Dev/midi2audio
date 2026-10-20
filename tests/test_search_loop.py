"""tests/test_search_loop.py — budget guard, request cache, selection, fallback (Step 3.4)."""
from __future__ import annotations

import dataclasses
import shutil

import pytest

from m2a.analysis.from_midi import load_structure
from m2a.config import SearchConfig
from m2a.contracts import Candidate, Stem
from m2a.manifest import RunManifest
from m2a.restyle.base import candidate
from m2a.search.loop import (
    BudgetGuard, CandidateCache, NoAcceptableCandidateError, SearchLoop, dry_candidate,
)
from tests.conftest import FIXTURES

STRUCTURE = load_structure(str(FIXTURES / "eight_bar_structure.json"))
PROMPTS = [("funk bass, instrumental", "vocals"), ("fingerstyle bass, instrumental", "vocals")]


class MockEndpoint:
    name, version = "mock", "1"
    is_diffusion, follows_f0 = True, False
    max_input_seconds = 190.0
    strength_semantics = "divergence"

    def __init__(self, usd_per_call=0.05, fail_with=None):
        self.usd_per_call = usd_per_call
        self.fail_with = fail_with
        self.call_count = 0

    def raw_params(self, prompt, negative_prompt, strength, seed):
        return {"strength": strength, "seed": seed}

    def transform(self, audio_path, prompt, negative_prompt, strength, seed, extra):
        self.call_count += 1
        if self.fail_with:
            return candidate(self, extra.get("stem", ""), "", prompt, negative_prompt, strength, seed,
                             reject_reason=self.fail_with)
        shutil.copyfile(audio_path, extra["out_path"])
        return candidate(self, extra.get("stem", ""), extra["out_path"], prompt, negative_prompt,
                         strength, seed)


class MockScorer:
    """Deterministic: score rises with seed and falls with distance from divergence 0.65;
    anything at or above `floor_above` divergence fails the rhythm floor."""

    def __init__(self, floor_above=1.0):
        self.floor_above = floor_above
        self.scored: list[Candidate] = []

    def score(self, c: Candidate) -> Candidate:
        if c.strength >= self.floor_above:
            out = dataclasses.replace(c, accepted=False, reject_reason="rhythm_floor",
                                      scores=(("rhythm", 0.4),))
        else:
            total = 0.5 + 0.1 * c.seed - abs(c.strength - 0.65)
            out = dataclasses.replace(c, accepted=True, reject_reason=None, total_score=total,
                                      scores=(("rhythm", 0.95),))
        self.scored.append(out)
        return out


@pytest.fixture
def stem(tmp_path):
    path = tmp_path / "bass.wav"
    shutil.copyfile(FIXTURES / "five_sec_stem.wav", path)
    return Stem(track="bass", role="bass", path=str(path), microtonal=False, peak_dbfs=-12.0,
                content_hash="stemhash")


def _loop(tmp_path, endpoint, scorer, budget=None, manifest=None):
    return SearchLoop(endpoint, scorer, budget or BudgetGuard(), CandidateCache(str(tmp_path / "cache")),
                      str(tmp_path / "work"), manifest)


def test_budget_guard_stops_search(tmp_path, stem):
    ep = MockEndpoint()
    result = _loop(tmp_path, ep, MockScorer(), BudgetGuard(max_calls=5)).search(
        stem, PROMPTS, STRUCTURE, SearchConfig(seeds=3))
    assert ep.call_count == 5
    assert result.stopped_by_budget and result.best.accepted


def test_usd_and_per_stem_budgets():
    g = BudgetGuard(max_usd=0.10, max_calls_per_stem=3)
    assert g.can_spend(0.05, "a")
    g.record_call(0.05, "a")
    g.record_call(0.05, "a")
    assert not g.can_spend(0.05, "b")                 # global spend cap
    g = BudgetGuard(max_calls_per_stem=2)
    g.record_call(0.0, "a", n_calls=2)
    assert not g.can_spend(0.0, "a") and g.can_spend(0.0, "b")


def test_selects_highest_scoring_candidate(tmp_path, stem):
    scorer = MockScorer()
    result = _loop(tmp_path, MockEndpoint(), scorer).search(stem, PROMPTS, STRUCTURE, SearchConfig(seeds=3))
    assert result.best.total_score == max(c.total_score for c in scorer.scored)
    assert result.best.seed == 2


def test_refinement_samples_finer_divergence(tmp_path, stem):
    result = _loop(tmp_path, MockEndpoint(), MockScorer()).search(
        stem, PROMPTS, STRUCTURE, SearchConfig(seeds=1, n_prompt_variants=1))
    tried = sorted({c.strength for c in result.candidates})
    assert len(tried) > 3                                   # grid of 3 + refinement points


def test_cache_hits_are_free_and_reproducible(tmp_path, stem):
    ep = MockEndpoint()
    cfg = SearchConfig(seeds=2)
    first = _loop(tmp_path, ep, MockScorer()).search(stem, PROMPTS, STRUCTURE, cfg)
    calls = ep.call_count
    manifest = RunManifest(run_id="r", code_version="0", config_hash="c")
    budget = BudgetGuard(max_calls=0)                       # nothing may be bought
    second = _loop(tmp_path, ep, MockScorer(), budget, manifest).search(stem, PROMPTS, STRUCTURE, cfg)
    assert ep.call_count == calls
    assert second.best.request_hash == first.best.request_hash
    assert all(r.outcome == "cache_hit" and r.usd_estimate == 0 for r in manifest.endpoint_calls)


def test_no_candidate_falls_back_to_dry(tmp_path, stem):          # D012
    result = _loop(tmp_path, MockEndpoint(), MockScorer(floor_above=0.0)).search(
        stem, PROMPTS, STRUCTURE, SearchConfig(seeds=1))
    assert result.best == dry_candidate(stem)
    assert result.fallback_reason == "all_candidates_below_floor"
    assert all(c.reject_reason == "rhythm_floor" for c in result.candidates)


def test_strict_mode_raises(tmp_path, stem):
    with pytest.raises(NoAcceptableCandidateError):
        _loop(tmp_path, MockEndpoint(), MockScorer(floor_above=0.0)).search(
            stem, PROMPTS, STRUCTURE, SearchConfig(seeds=1, on_no_candidate="strict"))


def test_endpoint_failures_are_logged_not_raised(tmp_path, stem):   # P3
    manifest = RunManifest(run_id="r", code_version="0", config_hash="c")
    result = _loop(tmp_path, MockEndpoint(fail_with="fatal"), MockScorer(), manifest=manifest).search(
        stem, PROMPTS, STRUCTURE, SearchConfig(seeds=1))
    assert result.best.endpoint == "dry"
    assert {r.outcome for r in manifest.endpoint_calls} == {"fatal"}


def test_microtonal_stem_is_routed_not_restyled(tmp_path, stem):   # P6
    ep = MockEndpoint()
    micro = dataclasses.replace(stem, microtonal=True)
    result = _loop(tmp_path, ep, MockScorer()).search(micro, PROMPTS, STRUCTURE, SearchConfig())
    assert ep.call_count == 0
    assert result.best.endpoint == "dry" and result.fallback_reason == "microtonal_routed"


def test_manifest_records_every_paid_call(tmp_path, stem):
    ep = MockEndpoint(usd_per_call=0.06)
    manifest = RunManifest(run_id="r", code_version="0", config_hash="c")
    _loop(tmp_path, ep, MockScorer(), manifest=manifest).search(stem, PROMPTS, STRUCTURE, SearchConfig(seeds=1))
    assert len(manifest.endpoint_calls) == ep.call_count
    rec = manifest.endpoint_calls[0]
    assert rec.input_audio_hash == "stemhash" and rec.output_audio_hash and rec.raw_params
    manifest.finalise(1.0)
    assert manifest.totals["usd_estimate"] == pytest.approx(0.06 * ep.call_count)
