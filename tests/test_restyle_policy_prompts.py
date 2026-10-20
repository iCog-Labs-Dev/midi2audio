"""tests/test_restyle_policy_prompts.py — divergence schedule (§8.3) and prompt layer (Step 3.3)."""
from __future__ import annotations

import pytest

from m2a.analysis.from_midi import load_structure
from m2a.prompts.generate import generate_prompts
from m2a.prompts.stylecard import StyleCard, make_style_card
from m2a.restyle.policy import (
    BASE_GRID, MAX_DIVERGENCE, MIN_DIVERGENCE, divergence_grid, mean_section_energy, refine_grid,
)
from tests.conftest import FIXTURES
from tests.synth import render_events


def _structure(meter="4/4"):
    s = load_structure(str(FIXTURES / "eight_bar_structure.json"))
    s.meter = [[0, meter]]
    return s


# ── policy ───────────────────────────────────────────────────────────────────

def test_neutral_energy_returns_base_grid():
    for role, base in BASE_GRID.items():
        assert divergence_grid(role, False, 0.5) == base


def test_microtonal_is_routed_away():                      # P6
    assert divergence_grid("lead", True, 0.9) == []


def test_energy_raises_divergence_and_stays_clamped():
    lo, hi = divergence_grid("drums", False, 0.0), divergence_grid("drums", False, 1.0)
    assert all(h > l for h, l in zip(hi, lo))
    for g in (lo, hi):
        assert all(MIN_DIVERGENCE <= d <= MAX_DIVERGENCE for d in g)


def test_pitched_roles_start_lower_than_percussive():
    assert max(divergence_grid("comping", False, 0.5)) < max(divergence_grid("drums", False, 0.5))


def test_tension_is_optional():                            # D016
    assert divergence_grid("bass", False, 0.5, tension=None) == divergence_grid("bass", False, 0.5, 0.8)


def test_refine_grid_skips_tried_points():
    out = refine_grid([0.7], tried={0.675})
    assert 0.725 in out and 0.675 not in out


def test_mean_section_energy():
    assert mean_section_energy(_structure()) == pytest.approx(0.6)


# ── prompts ──────────────────────────────────────────────────────────────────

def test_odd_meter_is_stated_and_four_on_the_floor_negated():
    s = _structure("7/8")
    for prompt, neg in generate_prompts(s, s.tracks[0], StyleCard()):
        assert "7/8" in prompt
        assert "four-on-the-floor" in neg


def test_four_four_does_not_negate_four_on_the_floor():
    s = _structure()
    for prompt, neg in generate_prompts(s, s.tracks[0], StyleCard()):
        assert "four-on-the-floor" not in neg and "4/4" not in prompt


def test_failure_mode_defaults_always_present():           # §8.7
    s = _structure()
    card = StyleCard(descriptors=("warm",), negatives=("lo-fi",))
    for track in s.tracks:
        for prompt, neg in generate_prompts(s, track, card, n_variants=3):
            assert "instrumental" in prompt
            assert "vocals" in neg and "singing" in neg and "lo-fi" in neg
            assert "warm" in prompt


def test_variants_are_distinct_and_deterministic():
    s = _structure()
    a = generate_prompts(s, s.tracks[1], StyleCard(), n_variants=4)
    assert len({p for p, _ in a}) == 4
    assert a == generate_prompts(s, s.tracks[1], StyleCard(), n_variants=4)


def test_style_card_from_reference_audio(tmp_path):
    for k in range(2):
        render_events(str(tmp_path / f"r{k}.wav"), [(i * 0.5, 0.4, 40 + k) for i in range(12)], 6.5)
    card = make_style_card(str(tmp_path), user_notes="70s funk")
    for key in ("tempo_bpm", "centroid_hz_p50", "crest_db", "integrated_lufs", "band_ratio"):
        assert key in card.features
    assert card.negatives and card.user_notes == "70s funk"


def test_style_card_without_references_carries_notes():
    card = make_style_card(None, user_notes="dry and close")
    assert card.descriptors == () and card.user_notes == "dry and close"


def test_style_card_llm_output_is_validated(tmp_path):
    render_events(str(tmp_path / "r.wav"), [(i * 0.5, 0.4, 45) for i in range(12)], 6.5)
    card = make_style_card(str(tmp_path), llm_fn=lambda s, u: "not json")
    assert card.negatives                                   # rule-based card survives bad LLM output
    card = make_style_card(str(tmp_path), llm_fn=lambda s, u: '{"descriptors": ["groovy"]}')
    assert "groovy" in card.descriptors
