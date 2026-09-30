# m2a/contracts.py
"""
Inter-stage data types for the midi2audio pipeline.

Frozen dataclasses hold Python-side objects (paths + hashes, never audio buffers).
Pydantic models validate external JSON input at the boundary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ── Python-side contracts (frozen dataclasses) ──────────────────────────────

@dataclass(frozen=True)
class RenderPackage:
    """The input contract. Everything downstream reads from this."""
    root: str            # run_<hash>/ directory
    midi_path: str       # score.mid inside root
    structure_path: str  # structure.json inside root
    tuning_path: str     # tuning.json inside root
    manifest_path: str   # manifest.json inside root
    content_hash: str    # sha256 of midi + structure + tuning


@dataclass(frozen=True)
class Stem:
    """One rendered audio stem."""
    track: str          # matches structure.tracks[].name (advisory; track{idx}_{role} when absent)
    role: str           # drums | bass | comping | lead | pads
    path: str           # absolute path to .wav
    microtonal: bool
    peak_dbfs: float
    content_hash: str


@dataclass(frozen=True)
class Candidate:
    """One restyled audio candidate for a single stem."""
    stem: str                   # track name
    audio_path: str
    prompt: str
    negative_prompt: str
    strength: float             # normalised divergence in [0, 1]
    seed: int
    endpoint: str
    endpoint_version: str
    # [v1.1] a dict inside a frozen dataclass is mutable and unhashable — the freeze is
    # cosmetic.  Store an ordered tuple of pairs; `dict(c.scores)` at the read site.
    scores: tuple[tuple[str, float], ...]
    accepted: bool
    reject_reason: Optional[str]
    total_score: float = 0.0
    # [v1.1] every field the manifest needs to reproduce this candidate:
    request_hash: str = ""      # sha256 of (audio hash, prompt, neg, divergence, seed, endpoint_version)
    usd_estimate: float = 0.0


# ── External-input schema (pydantic, validates structure.json) ───────────────

class TrackInfo(BaseModel):
    """Metadata for one MIDI track inside structure.json."""
    # [v1.1] tracks are identified by zero-based index, not by name.
    # name is advisory and may be empty or duplicated; canonical form is track{idx:02d}_{role}.
    idx: int = Field(ge=0, description="Zero-based track index in score.mid")
    name: str = ""
    role: str           # drums | bass | comping | lead | pads
    program: Optional[int] = None
    percussion: bool = False
    quantized: bool = True
    flat_velocity: bool = False
    onset_profile_16: list[float] = Field(default_factory=list)
    pitch_range: list[int] = Field(default_factory=list)
    microtonal: bool = False


class SectionInfo(BaseModel):
    """One structural section (e.g. intro, verse, chorus)."""
    label: str
    bars: list[int]                             # [start_bar, end_bar]
    energy: float = Field(ge=0.0, le=1.0)
    # [v1.1 / D016] tension is null when provenance is "inferred" — consumers must treat as optional
    tension: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    boundary_lvl: int = Field(ge=0)


class StructureJSON(BaseModel):
    """Validated representation of structure.json (schema v1)."""
    schema_version: str = Field(alias="schema", default="midi2audio.structure/1")
    provenance: str         # inferred | planner | host
    source_hash: str
    edo: int = Field(ge=1)
    base_tuning: float
    tempo_map: list[list[float]]
    meter: list[list]
    key: list[dict]
    chords: list[dict]
    sections: list[SectionInfo]
    bar_table: list[list[float]]
    tracks: list[TrackInfo]

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def check_invariants(self) -> "StructureJSON":
        # [v1.1] raise ValueError, never assert: bare asserts are stripped under `python -O`,
        # and pydantic only converts ValueError/AssertionError into ValidationError reliably
        # when the message is preserved. A validator that silently vanishes in an optimised
        # build is worse than no validator.
        times = [row[1] for row in self.bar_table]
        if times != sorted(times):
            raise ValueError("bar_table must be monotonically increasing")
        idxs = [t.idx for t in self.tracks]
        if len(set(idxs)) != len(idxs):
            raise ValueError(f"duplicate track idx in structure.tracks: {idxs}")
        # Invariant 5: microtonal tracks must have safe method in tuning.json
        # (cross-file check performed by orchestrator at run time)
        return self
