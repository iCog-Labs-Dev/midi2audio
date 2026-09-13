"""Pydantic schema definitions and validation for symbolic GrooveSpec contracts."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class SwingSpec(BaseModel):
    """Subdivision delay specification for swing feel."""

    ratio: float = Field(ge=0.5, le=0.75, default=0.5, description="0.5=Straight, 0.66=Triplet")
    subdivision: Literal[8, 16] = Field(default=8, description="Target swing division")
    applies_to: List[str] = Field(default_factory=lambda: ["drums", "bass", "comp"])


class TrackMicrotiming(BaseModel):
    """Systematic and stochastic timing offsets per instrument role."""

    by_position_16: List[float] = Field(
        default_factory=lambda: [0.0] * 16,
        description="Systematic offsets in ms per 16th-note metrical position within the bar.",
    )
    global_offset: float = Field(default=0.0, description="Global offset in ms applied to all notes.")
    jitter_sd: float = Field(default=0.0, ge=0.0, le=4.0, description="Standard deviation in ms for Gaussian jitter.")

    @field_validator("by_position_16")
    @classmethod
    def validate_len_16(cls, v: List[float]) -> List[float]:
        if len(v) != 16:
            raise ValueError(f"by_position_16 must have exactly 16 offsets, received {len(v)}")
        return v


class GhostNoteSpec(BaseModel):
    """Stochastic insertion of ghost notes for articulation richness."""

    pitch: int = Field(ge=0, le=127)
    prob: float = Field(ge=0.0, le=1.0, default=0.1)
    vel_scale: float = Field(ge=0.1, le=0.8, default=0.35)


class PhraseArcSpec(BaseModel):
    """Long-range dynamic shaping across multi-bar structures."""

    shape: Literal["crescendo_to_bar4", "flat", "arch"] = "flat"
    depth: float = Field(ge=0.0, le=0.5, default=0.0)


class VelocitySpec(BaseModel):
    """Metrical accentuation, dynamic phrase curves, and ghost notes."""

    accent_map_16: List[float] = Field(
        default_factory=lambda: [1.0] * 16,
        description="Multiplicative velocity scalers per 16th position.",
    )
    ghost_notes: Dict[str, GhostNoteSpec] = Field(default_factory=dict)
    phrase_arc: PhraseArcSpec = Field(default_factory=PhraseArcSpec)

    @field_validator("accent_map_16")
    @classmethod
    def validate_len_16(cls, v: List[float]) -> List[float]:
        if len(v) != 16:
            raise ValueError(f"accent_map_16 must have length 16, received {len(v)}")
        return v


class AnticipationSpec(BaseModel):
    """Early arrival of harmonic notes at boundary events."""

    prob: float = Field(ge=0.0, le=1.0, default=0.0)
    push_ms: Tuple[float, float] = Field(default=(-50.0, -20.0))
    at: Literal["chord_changes", "all_downbeats"] = "chord_changes"


class ArticulationSpec(BaseModel):
    """Note duration scaling and polyphonic chord strum timing."""

    duration_scale: float = Field(ge=0.1, le=2.0, default=1.0)
    strum_ms: float = Field(ge=0.0, le=50.0, default=0.0)


class TempoCurveSpec(BaseModel):
    """Global tempo adjustments at phrase and section boundaries."""

    section_lifts: Dict[str, float] = Field(default_factory=dict, description="BPM lift per section")
    phrase_ritenuto_pct: float = Field(default=0.0, ge=0.0, le=10.0)


class GrooveSpec(BaseModel):
    """Master parametric specification schema for Stage 1 symbolic expressivization."""

    swing: SwingSpec = Field(default_factory=SwingSpec)
    microtiming_ms: Dict[str, TrackMicrotiming] = Field(default_factory=dict)
    velocity: VelocitySpec = Field(default_factory=VelocitySpec)
    anticipation: Dict[str, AnticipationSpec] = Field(default_factory=dict)
    tempo_curve: TempoCurveSpec = Field(default_factory=TempoCurveSpec)
    articulation: Dict[str, ArticulationSpec] = Field(default_factory=dict)