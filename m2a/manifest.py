# m2a/manifest.py
"""Run provenance manifest (§8.1).

"Every render carries a manifest of all parameters, prompts, seeds, and
endpoint versions"; "all requests and responses (minus audio payloads) are
logged to the manifest". The input package's manifest.json records who made
the MIDI; this records what the pipeline then did with it: every stage
(cache hit or not), every endpoint call (including cache hits, which cost
nothing), which candidate each stem shipped, and the resolved decisions.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class StageRecord:
    stage: str
    key: str
    cache_hit: bool
    wall_clock_s: float
    outputs: list[str] = field(default_factory=list)
    config: dict = field(default_factory=dict)


@dataclass
class CallRecord:
    stem: str
    endpoint: str
    endpoint_version: str
    request_hash: str
    prompt: str
    negative_prompt: str
    divergence: float            # normalised; raw_params carries the provider value sent
    raw_params: dict
    seed: int
    input_audio_hash: str
    output_audio_hash: str
    usd_estimate: float
    latency_s: float
    n_calls: int                 # >1 when the stem was chunked
    outcome: str                 # ok | retryable | fatal | content_policy | cache_hit
    watermark_declared: bool = False
    terms_version: str = ""


@dataclass
class StemOutcome:
    stem: str
    role: str
    source: str                  # endpoint name, or "dry"
    audio_path: str
    request_hash: str
    total_score: float
    scores: dict
    candidates_tried: int
    reject_reasons: dict         # reason -> count
    fallback_reason: str | None = None
    stopped_by_budget: bool = False


@dataclass
class RunManifest:
    run_id: str
    code_version: str
    config_hash: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    inputs: dict = field(default_factory=dict)
    stages: list[StageRecord] = field(default_factory=list)
    endpoint_calls: list[CallRecord] = field(default_factory=list)
    stems: list[StemOutcome] = field(default_factory=list)
    decisions: dict = field(default_factory=dict)
    totals: dict = field(default_factory=dict)

    def finalise(self, wall_clock_s: float) -> None:
        paid = [c for c in self.endpoint_calls if c.outcome != "cache_hit"]
        n_requests = len(self.endpoint_calls)
        self.totals = {
            "endpoint_calls": sum(c.n_calls for c in paid),
            "usd_estimate": round(sum(c.usd_estimate for c in paid), 6),
            "cache_hit_rate": (sum(c.outcome == "cache_hit" for c in self.endpoint_calls) / n_requests
                               if n_requests else 0.0),
            "stems_fallen_back": [s.stem for s in self.stems if s.source == "dry"],
            "wall_clock_s": round(wall_clock_s, 3),
        }

    def to_dict(self) -> dict:
        return asdict(self)

    def write(self, path: str) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str))
        return path

    @classmethod
    def load(cls, path: str) -> "RunManifest":
        data = json.loads(Path(path).read_text())
        return cls(
            run_id=data["run_id"], code_version=data["code_version"], config_hash=data["config_hash"],
            started_at=data["started_at"], inputs=data["inputs"],
            stages=[StageRecord(**s) for s in data["stages"]],
            endpoint_calls=[CallRecord(**c) for c in data["endpoint_calls"]],
            stems=[StemOutcome(**s) for s in data["stems"]],
            decisions=data["decisions"], totals=data["totals"],
        )
