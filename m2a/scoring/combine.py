# m2a/scoring/combine.py
"""Combined scorer with hard floors (Step 2.5).

`combine()` is pure: raw axis values in, scored `Candidate` out. Hard floors
are evaluated on the RAW axis values first (P4: a candidate that fails a
floor is rejected, never repaired); only survivors get a weighted total,
computed over axes that are each mapped to "higher is better, in [0, 1]".

[v1.1] bug A — FAD is a distance and enters the sum as `exp(-fad / tau)`, so
a lower FAD scores higher. Bug B — every role's weights sum to 1.0, and an
axis that is not applicable (notes on drums) or not available (no CLAP
model, no reference corpus) is dropped with its weight redistributed
proportionally, so totals stay comparable across roles and runs.

`StemScorer` computes the raw axes for one candidate from audio and hands
them to `combine()`.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from m2a.contracts import Candidate, Stem, StructureJSON
from m2a.render.tuning import TuningSpec
from m2a.scoring.clap_sim import ClapEncoder, clap_similarity
from m2a.scoring.fad import Embedder, MelStatsEmbedder, ReferenceSet, fad_score
from m2a.scoring.notes import note_fidelity
from m2a.scoring.rhythm import MAX_ACCEPTABLE_SHIFT_SEC, rhythm_report
from m2a.scoring.tuning_check import frame_times, intended_f0_contour, track_f0, tuning_check

AXES = ("rhythm", "notes", "clap", "fad", "tuning")


# ── Axis normalisation: every axis -> [0, 1], higher is better ───────────────

def norm_rhythm(f: float) -> float:
    return float(np.clip(f, 0.0, 1.0))


def norm_notes(f1: float) -> float:
    return float(np.clip(f1, 0.0, 1.0))


def norm_clap(delta: float) -> float:
    # cosine(pos) - cosine(neg) lives in [-2, 2]; squash to [0, 1]
    return float(np.clip((delta + 2.0) / 4.0, 0.0, 1.0))


def norm_fad(fad: float, tau: float) -> float:
    # tau = median leave-one-out FAD of the reference corpus: a candidate as
    # close to the corpus as the references are to each other scores ~0.37.
    return float(np.exp(-max(fad, 0.0) / tau))


def norm_tuning(p95_cents: float, floor: float) -> float:
    return float(np.clip(1.0 - p95_cents / floor, 0.0, 1.0))


@dataclass(frozen=True)
class ScoreWeights:
    rhythm: float
    notes: float
    clap: float
    fad: float
    tuning: float = 0.0        # non-zero only for microtonal stems

    def __post_init__(self):
        total = self.rhythm + self.notes + self.clap + self.fad + self.tuning
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total}")


@dataclass(frozen=True)
class ScoreFloors:
    rhythm_f: float = 0.85
    notes_f1: float = 0.70
    tuning_cents_p95: float = 15.0   # used only when microtonal=True (D004)
    max_shift_sec: float = MAX_ACCEPTABLE_SHIFT_SEC


# Per role, per the design document: drums weight rhythm, pads weight timbre
# and reference proximity, leads weight note fidelity.
ROLE_WEIGHTS = {
    "drums":   ScoreWeights(rhythm=0.45, notes=0.05, clap=0.20, fad=0.30),
    "bass":    ScoreWeights(rhythm=0.30, notes=0.30, clap=0.15, fad=0.25),
    "comping": ScoreWeights(rhythm=0.25, notes=0.30, clap=0.20, fad=0.25),
    "lead":    ScoreWeights(rhythm=0.20, notes=0.45, clap=0.15, fad=0.20),
    "pads":    ScoreWeights(rhythm=0.10, notes=0.20, clap=0.30, fad=0.40),
}

# A 25 ms drum window is a harder test than a 50 ms lead window, so floors
# are per role too.
ROLE_FLOORS = {
    "drums":   ScoreFloors(rhythm_f=0.80, notes_f1=0.00),   # notes n/a on percussion
    "bass":    ScoreFloors(rhythm_f=0.85, notes_f1=0.70),
    "comping": ScoreFloors(rhythm_f=0.80, notes_f1=0.70),
    "lead":    ScoreFloors(rhythm_f=0.85, notes_f1=0.75),
    "pads":    ScoreFloors(rhythm_f=0.60, notes_f1=0.65),
}

# Weight given to the tuning axis on microtonal stems; the role's other
# weights are scaled by (1 - this) so the row still sums to 1.
MICROTONAL_TUNING_WEIGHT = 0.10


def weights_for(role: str, microtonal: bool) -> ScoreWeights:
    w = ROLE_WEIGHTS[role]
    if not microtonal:
        return w
    k = 1.0 - MICROTONAL_TUNING_WEIGHT
    return ScoreWeights(rhythm=w.rhythm * k, notes=w.notes * k, clap=w.clap * k,
                        fad=w.fad * k, tuning=MICROTONAL_TUNING_WEIGHT)


def _with_scores(candidate: Candidate, raw: dict, **changes) -> Candidate:
    scores = tuple((k, float(v)) for k, v in raw.items() if v is not None)
    return dataclasses.replace(candidate, scores=scores, **changes)


def combine(candidate: Candidate, raw: dict, role: str, microtonal: bool = False,
            tau: float | None = None, floors: ScoreFloors | None = None,
            weights: ScoreWeights | None = None) -> Candidate:
    """Pure. `raw` maps axis -> raw value or None (n/a / unavailable); it may also
    carry `shift_sec`. Returns the candidate with scores, accepted/reject_reason
    and total_score set."""
    floors = floors or ROLE_FLOORS[role]
    weights = weights or weights_for(role, microtonal)
    reject = lambda reason: _with_scores(candidate, raw, accepted=False,  # noqa: E731
                                         reject_reason=reason, total_score=0.0)

    # Hard floors on RAW values.
    if microtonal and raw.get("tuning") is not None and raw["tuning"] > floors.tuning_cents_p95:
        return reject("tuning_floor")
    if raw.get("shift_sec") is not None and abs(raw["shift_sec"]) > floors.max_shift_sec:
        return reject("shift_floor")
    if raw.get("rhythm") is None or raw["rhythm"] < floors.rhythm_f:
        return reject("rhythm_floor")
    if raw.get("notes") is not None and raw["notes"] < floors.notes_f1:
        return reject("notes_floor")

    normed = {
        "rhythm": norm_rhythm(raw["rhythm"]),
        "notes": None if raw.get("notes") is None else norm_notes(raw["notes"]),
        "clap": None if raw.get("clap") is None else norm_clap(raw["clap"]),
        "fad": None if raw.get("fad") is None or tau is None else norm_fad(raw["fad"], tau),
        "tuning": (norm_tuning(raw["tuning"], floors.tuning_cents_p95)
                   if microtonal and raw.get("tuning") is not None else None),
    }
    live = {k: v for k, v in normed.items() if v is not None}
    wsum = sum(getattr(weights, k) for k in live)
    if wsum <= 0:
        total = float(np.mean(list(live.values())))
    else:
        total = sum(getattr(weights, k) / wsum * v for k, v in live.items())
    return _with_scores(candidate, raw, accepted=True, reject_reason=None,
                        total_score=float(np.clip(total, 0.0, 1.0)))


class StemScorer:
    """Computes raw axes for candidates of the stems in one run, then `combine()`s.

    Deterministic given its inputs; holds only read-only context (paths, the
    pre-embedded reference set, optional CLAP encoder).
    """

    def __init__(self, expressive_midi: str, structure: StructureJSON, stems: list[Stem],
                 tuning: TuningSpec | None = None, refs: ReferenceSet | None = None,
                 clap: ClapEncoder | None = None, embedder: Embedder | None = None,
                 transcriber=None):
        self.expressive_midi = expressive_midi
        self.structure = structure
        self.tuning = tuning or TuningSpec(edo=structure.edo, base_tuning=structure.base_tuning)
        self.refs = refs
        self.clap = clap
        self.embedder = embedder or MelStatsEmbedder()
        self.transcriber = transcriber
        self._stems = {s.track: s for s in stems}
        self._track_idx = {t.label: t.idx for t in structure.tracks}

    def raw_scores(self, audio_path: str, stem: Stem, prompt: str = "",
                   negative_prompt: str = "") -> dict:
        idx = self._track_idx[stem.track]
        rhythm = rhythm_report(audio_path, self.expressive_midi, stem.role, idx)
        raw: dict = {
            "rhythm": rhythm.f_measure,
            "shift_sec": rhythm.shift_sec,
            "notes": note_fidelity(audio_path, stem.path, self.expressive_midi, stem.role,
                                   stem.microtonal, idx, self.tuning, self.transcriber),
            "clap": (clap_similarity(audio_path, prompt, negative_prompt, self.clap)
                     if self.clap is not None and prompt else None),
            "fad": fad_score(audio_path, self.refs, self.embedder) if self.refs is not None else None,
            "tuning": None,
        }
        if stem.microtonal and stem.role != "drums":
            f0 = track_f0(audio_path)
            intended = intended_f0_contour(self.expressive_midi, idx, self.tuning, frame_times(len(f0)))
            raw["tuning"] = tuning_check(audio_path, self.tuning.edo, self.tuning.base_tuning,
                                         intended_f0=intended, f0=f0).p95_cents
        return raw

    def score(self, candidate: Candidate) -> Candidate:
        stem = self._stems[candidate.stem]
        if not candidate.audio_path:
            return candidate
        raw = self.raw_scores(candidate.audio_path, stem, candidate.prompt, candidate.negative_prompt)
        return combine(candidate, raw, stem.role, stem.microtonal,
                       tau=self.refs.tau if self.refs is not None else None)
