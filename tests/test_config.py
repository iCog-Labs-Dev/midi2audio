"""tests/test_config.py — pipeline configuration defaults."""
from __future__ import annotations

import pytest

from m2a.config import config_fingerprint, default_config, load_config


def test_default_config_disables_midi_gpt():
    """D008: MIDI-GPT is CC-BY-NC; the default config must load with it off."""
    assert default_config().midi_gpt is False


def test_default_config_has_neutral_groove_style():
    cfg = default_config()
    assert cfg.groove.style_intent == "neutral"
    assert cfg.groove.seed == 0


def test_yaml_profile_overlays_defaults(tmp_path):
    profile = tmp_path / "p.yaml"
    profile.write_text("search:\n  seeds: 5\nbudget:\n  max_usd: 1.5\n")
    cfg = load_config(str(profile))
    assert cfg.search.seeds == 5 and cfg.budget.max_usd == 1.5
    assert cfg.search.n_prompt_variants == default_config().search.n_prompt_variants


def test_unknown_profile_key_rejected(tmp_path):
    profile = tmp_path / "p.yaml"
    profile.write_text("search:\n  seedz: 5\n")
    with pytest.raises(ValueError, match="search.seedz"):
        load_config(str(profile))


def test_invalid_fallback_policy_rejected(tmp_path):
    profile = tmp_path / "p.yaml"
    profile.write_text("search:\n  on_no_candidate: shrug\n")
    with pytest.raises(ValueError):
        load_config(str(profile))


def test_fingerprint_is_json_safe():
    import json
    json.dumps(config_fingerprint(default_config()))
