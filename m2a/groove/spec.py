# m2a/groove/spec.py
"""GrooveSpec — the LLM output contract for Stage 1 (expressivization).

Every groove spec produced by `groove/author.py` is validated against this
schema before being applied by `groove/apply.py`. Invalid specs are rejected
and regenerated; the applicator never patches an invalid spec.

D009: keys in `SwingSpec.applies_to` and `microtiming_ms` / `articulation`
are **roles** (drums | bass | comping | lead | pads), not track names. A
`track:<idx>` prefix (e.g. `"track:3"`) names an explicit per-track override
that bypasses the role map; the applicator expands roles to track indices
via `structure.tracks`.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class SwingSpec(BaseModel):
    ratio: float = Field(ge=0.5, le=0.75,
        description="0.5=straight, 0.66=triplet swing")
    subdivision: int = Field(default=8,
        description="8 or 16 — which subdivision swings")
    applies_to: list[str] = Field(default_factory=list,
        description="roles (drums, bass, comping, lead, pads) or 'track:<idx>' overrides")
    # [v1.1] tempo scaling (Friberg & Sundstrom 2002): the short (off-beat) note
    # is held roughly constant in absolute duration, so ratio collapses toward
    # 0.5 at fast tempi rather than staying constant.
    tempo_scaling: bool = Field(default=True)
    reference_bpm: float = Field(default=120.0, gt=0.0)


class MicrotimingSpec(BaseModel):
    by_position_16: list[float] | None = Field(default=None,
        description="ms offsets indexed by metrical position (length must match `grid`)")
    global_offset: float = Field(default=0.0,
        description="ms — positive = late, negative = ahead")
    jitter_sd: float = Field(default=2.0, ge=0.0, le=10.0,
        description="standard deviation of random jitter in ms")
    # [v1.1] meter-aware grid: 16 (sixteenth-note) or 24 (triplet/shuffle feel).
    # `by_position_16` length must equal positions_per_bar(meter, grid).
    grid: int = Field(default=16, description="16 or 24")


class VelocitySpec(BaseModel):
    accent_map_16: list[float] | None = Field(default=None,
        description="16 multipliers 0.0-1.0 for metrical accent")
    ghost_notes: dict | None = None
    phrase_arc: dict | None = None


class ArticulationSpec(BaseModel):
    duration_scale: float = Field(default=1.0, ge=0.1, le=2.0)
    strum_ms: float = Field(default=0.0, ge=0.0)


class GrooveSpec(BaseModel):
    swing: SwingSpec | None = None
    microtiming_ms: dict[str, MicrotimingSpec] = Field(default_factory=dict,
        description="keys are roles, or 'track:<idx>' overrides")
    velocity: VelocitySpec | None = None
    anticipation: dict | None = None
    tempo_curve: dict | None = None
    articulation: dict[str, ArticulationSpec] = Field(default_factory=dict)
    # Repo-specific extensions
    section_overrides: dict[str, "GrooveSpec"] | None = None
    tension_coupling: dict | None = None
    edo_safe: bool = Field(default=False,
        description="If true, applicator asserts no 12-EDO pitch operations are applied")
    groove_id_hint: int | None = None


GrooveSpec.model_rebuild()  # resolves forward ref in section_overrides


def positions_per_bar(meter: str, grid: int) -> int:
    """Number of metrical grid positions in one bar of `meter` ("N/D").

    `grid` counts subdivisions per quarter note scaled to the bar's actual
    beat unit: `positions = numerator * grid / denominator`. A 4/4 bar at
    grid=16 has 16 positions; a 7/8 bar at grid=16 has 14; a 4/4 bar at
    grid=24 (triplet/shuffle feel) has 24.
    """
    numerator, denominator = (int(x) for x in meter.split("/"))
    return int(round(numerator * grid / denominator))
