"""tests/test_config.py — pipeline configuration defaults."""
from __future__ import annotations

from m2a.config import default_config


def test_default_config_disables_midi_gpt():
    """D008: MIDI-GPT is CC-BY-NC; the default config must load with it off."""
    assert default_config().midi_gpt is False


def test_default_config_has_neutral_groove_style():
    cfg = default_config()
    assert cfg.groove.style_intent == "neutral"
    assert cfg.groove.seed == 0
