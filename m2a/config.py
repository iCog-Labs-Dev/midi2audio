# m2a/config.py
"""Pipeline configuration: the AudioConfig tree (frozen-in-spirit dataclasses,
optionally overlaid from a YAML profile with `load_config`).

D008: `midi_gpt` gates the (post-M6, CC-BY-NC-licensed) MIDI-GPT branch and
defaults to False. `test_default_config_disables_midi_gpt` in
tests/test_config.py is the CI assertion named in DECISIONS.md.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import yaml

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
class ScoringConfig:
    clap: str = "auto"                  # auto | laion | off
    transcriber: str = "cqt"            # cqt | basic_pitch


@dataclass
class SearchConfig:
    seeds: int = 2
    n_prompt_variants: int = 2
    strength_grid: list[float] | None = None    # None: restyle.policy.divergence_grid
    refine: bool = True
    on_no_candidate: str = "fallback_dry"       # D012: fallback_dry | strict

    def __post_init__(self):
        if self.on_no_candidate not in ("fallback_dry", "strict"):
            raise ValueError(f"on_no_candidate must be fallback_dry or strict, got {self.on_no_candidate!r}")


@dataclass
class BudgetConfig:
    max_calls: int | None = 200
    max_usd: float | None = 20.0
    max_calls_per_stem: int | None = 40         # the design document's 20-40 per stem


@dataclass
class RestyleConfig:
    endpoint: str = "offline"           # offline | stable_audio
    usd_per_call: float | None = None   # None: the endpoint's own default
    style_notes: str = ""


@dataclass
class AudioConfig:
    groove: GrooveConfig = field(default_factory=GrooveConfig)
    soundfont: str = DEFAULT_SOUNDFONT
    sfz_instruments: dict[str, str] = field(default_factory=dict)   # role -> .sfz (opt-in sfizz)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    restyle: RestyleConfig = field(default_factory=RestyleConfig)
    midi_gpt: bool = False  # D008 — CC-BY-NC gated branch, off by default


def default_config() -> AudioConfig:
    return AudioConfig()


def _overlay(obj, data: dict, path: str = ""):
    names = {f.name for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in names:
            raise ValueError(f"unknown config key {path + key!r}")
        current = getattr(obj, key)
        if dataclasses.is_dataclass(current) and isinstance(value, dict):
            _overlay(current, value, f"{path}{key}.")
        else:
            setattr(obj, key, value)
    if hasattr(obj, "__post_init__"):
        obj.__post_init__()
    return obj


def load_config(path: str | None) -> AudioConfig:
    """Default config overlaid with a YAML profile; unknown keys are an error."""
    cfg = default_config()
    if path and Path(path).exists():
        data = yaml.safe_load(Path(path).read_text()) or {}
        _overlay(cfg, data)
    return cfg


def config_fingerprint(cfg: AudioConfig) -> dict:
    """JSON-safe view of the config for hashing / the manifest (callables by name)."""
    def conv(x):
        if dataclasses.is_dataclass(x):
            return {f.name: conv(getattr(x, f.name)) for f in dataclasses.fields(x)}
        if callable(x):
            return getattr(x, "__qualname__", repr(x))
        if isinstance(x, dict):
            return {k: conv(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [conv(v) for v in x]
        return x
    return conv(cfg)
