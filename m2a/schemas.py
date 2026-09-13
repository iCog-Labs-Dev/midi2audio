from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrackRole(str, Enum):
    DRUMS = "drums"
    BASS = "bass"
    COMPING = "comping"
    LEAD = "lead"
    PADS = "pads"
    ORNAMENT = "ornament"
    UNKNOWN = "unknown"


class SourceInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str
    ppq: int = Field(gt=0)
    duration_seconds: float = Field(ge=0)


class TempoEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_ticks: int = Field(ge=0)
    bpm: float = Field(gt=0)


class MeterEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    position_ticks: int = Field(ge=0)
    numerator: int = Field(gt=0)
    denominator: int = Field(gt=0)

    @field_validator("denominator")
    @classmethod
    def denominator_power_of_two(cls, value: int) -> int:
        if value & (value - 1):
            raise ValueError("meter denominator must be a power of two")
        return value


class KeyEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_bar: int = Field(ge=0)
    key: str = Field(min_length=1)


class ChordEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start_tick: int = Field(ge=0)
    end_tick: int = Field(gt=0)
    chord: str = Field(min_length=1)

    @field_validator("end_tick")
    @classmethod
    def end_after_start(cls, value: int, info):
        start = info.data.get("start_tick")
        if start is not None and value <= start:
            raise ValueError("end_tick must be greater than start_tick")
        return value


class HarmonyAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution: Literal["beat", "half_bar"]
    events: list[ChordEvent] = Field(default_factory=list)


class Section(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(min_length=1)
    bars: tuple[int, int]
    energy: float = Field(ge=0, le=1)

    @field_validator("bars")
    @classmethod
    def valid_bar_range(cls, value: tuple[int, int]):
        start, end = value
        if start < 0 or end <= start:
            raise ValueError("bars must be [start, end) with end > start")
        return value


class PolyphonyStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean: float = Field(ge=0)
    max: int = Field(ge=0)


class QuantizationStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fraction_within_tolerance: float = Field(ge=0, le=1)
    tolerance_ms: float = Field(ge=0)
    quantized: bool


class VelocityStats(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mean: float = Field(ge=0, le=127)
    std: float = Field(ge=0)
    unique_values: int = Field(ge=0)
    flat: bool


class TrackAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    idx: int = Field(ge=0)
    role: TrackRole
    program: int | None = Field(default=None, ge=0, le=127)
    range: tuple[int, int] | None = None
    polyphony: PolyphonyStats
    note_density: list[int]
    onset_profile_16: list[float] | None = None
    quantization: QuantizationStats
    velocity: VelocityStats

    @field_validator("range")
    @classmethod
    def valid_pitch_range(cls, value):
        if value is None:
            return value
        lo, hi = value
        if not (0 <= lo <= hi <= 127):
            raise ValueError("range must contain MIDI pitches 0..127")
        return value

    @field_validator("note_density")
    @classmethod
    def valid_note_density(cls, value):
        if any(v < 0 for v in value):
            raise ValueError("note density cannot be negative")
        return value

    @field_validator("onset_profile_16")
    @classmethod
    def valid_onset_profile(cls, value):
        if value is None:
            return value
        if len(value) != 16:
            raise ValueError("onset_profile_16 must contain exactly 16 values")
        if any(v < 0 or v > 1 for v in value):
            raise ValueError("onset profile values must be in [0, 1]")
        return value


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str
    content_hash: str
    source: SourceInfo
    tempo_map: list[TempoEvent]
    meter: list[MeterEvent]
    key: list[KeyEvent]
    harmony: HarmonyAnalysis
    sections: list[Section]
    tracks: list[TrackAnalysis]
