# m2a/config.py
"""Pipeline configuration for the deterministic spine (M1) and beyond.

D008: `midi_gpt` gates the (post-M6, CC-BY-NC-licensed) MIDI-GPT branch and
defaults to False. `test_default_config_disables_midi_gpt` in
tests/test_config.py is the CI assertion named in DECISIONS.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from m2a.render.registry import DEFAULT_SOUNDFONT


def _stub_llm(system: str, user: str) -> str:
    """Neutral placeholder LLM: emits an empty (no-op) GrooveSpec.

    Real LLM wiring is out of scope for M1; callers inject their own
    `llm_fn` via `GrooveConfig.llm_fn` (the orchestrator never calls a
    network endpoint on its own).
    """
    return "{}"


@dataclass
class GrooveConfig:
    style_intent: str = "neutral"
    seed: int = 0
    llm_fn: Callable[[str, str], str] = field(default=_stub_llm)


@dataclass
class AudioConfig:
    groove: GrooveConfig = field(default_factory=GrooveConfig)
    soundfont: str = DEFAULT_SOUNDFONT
    midi_gpt: bool = False  # D008 — CC-BY-NC gated branch, off by default


def default_config() -> AudioConfig:
    return AudioConfig()
