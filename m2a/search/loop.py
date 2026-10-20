# m2a/search/loop.py
"""Budget-guarded candidate search for one stem (Step 3.4).

Grid over (divergence × prompt variant × seed), then successive halving:
the top quarter of accepted candidates is re-sampled at finer divergence
steps around their own settings. Every request is keyed by its request
hash and looked up in the on-disk cache **before** the budget is consulted —
a cache hit is free (P5) and a resumed run never re-buys a candidate.

P4 — the loop never repairs a candidate that failed a floor; it only tries
other settings. P3 — endpoint failures come back as rejected candidates and
are logged, never raised. D012 — when nothing clears the floors the stem
falls back to its deterministic render (`on_no_candidate="fallback_dry"`),
or the run aborts (`"strict"`, for CI).
"""
from __future__ import annotations

import dataclasses
import json
import logging
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from m2a.artifacts import file_sha256
from m2a.config import SearchConfig
from m2a.contracts import Candidate, Stem, StructureJSON
from m2a.manifest import CallRecord, RunManifest
from m2a.restyle.base import RestyleEndpoint, request_hash
from m2a.restyle.chunking import chunk_and_restyle, n_chunks_for
from m2a.restyle.policy import divergence_grid, mean_section_energy, refine_grid

log = logging.getLogger(__name__)


class NoAcceptableCandidateError(RuntimeError):
    def __init__(self, stem: str, candidates: list[Candidate]):
        self.candidates = candidates
        super().__init__(f"no candidate for {stem} cleared the floors "
                         f"({len(candidates)} tried, reasons={dict(reason_histogram(candidates))})")


def reason_histogram(candidates: list[Candidate]) -> Counter:
    return Counter(c.reject_reason or "accepted" for c in candidates)


class BudgetGuard:
    """Global and per-stem limits on paid calls and estimated spend (P5)."""

    def __init__(self, max_calls: int | None = None, max_usd: float | None = None,
                 max_calls_per_stem: int | None = None):
        self.max_calls = max_calls
        self.max_usd = max_usd
        self.max_calls_per_stem = max_calls_per_stem
        self.calls = 0
        self.usd = 0.0
        self.per_stem: Counter = Counter()

    def can_spend(self, cost_usd: float = 0.0, stem: str | None = None, n_calls: int = 1) -> bool:
        if self.max_calls is not None and self.calls + n_calls > self.max_calls:
            return False
        if self.max_usd is not None and self.usd + cost_usd > self.max_usd + 1e-12:
            return False
        if (stem is not None and self.max_calls_per_stem is not None
                and self.per_stem[stem] + n_calls > self.max_calls_per_stem):
            return False
        return True

    def record_call(self, cost_usd: float = 0.0, stem: str | None = None, n_calls: int = 1) -> None:
        self.calls += n_calls
        self.usd += cost_usd
        if stem is not None:
            self.per_stem[stem] += n_calls


class CandidateCache:
    """request_hash -> unscored endpoint result (audio + metadata) on disk."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _meta(self, req: str) -> Path:
        return self.root / f"{req}.json"

    def get(self, req: str) -> Candidate | None:
        meta = self._meta(req)
        if not meta.exists():
            return None
        data = json.loads(meta.read_text())
        data["scores"] = tuple(tuple(x) for x in data["scores"])
        return Candidate(**data)

    def put(self, req: str, c: Candidate) -> Candidate:
        audio_path = ""
        if c.audio_path:
            audio_path = str(self.root / f"{req}.wav")
            if Path(c.audio_path).resolve() != Path(audio_path).resolve():
                shutil.copyfile(c.audio_path, audio_path)
        cached = dataclasses.replace(c, audio_path=audio_path, request_hash=req, scores=(),
                                     total_score=0.0)
        self._meta(req).write_text(json.dumps(dataclasses.asdict(cached)))
        return cached


def dry_candidate(stem: Stem) -> Candidate:
    """The deterministic render as a candidate: always note-faithful."""
    return Candidate(stem=stem.track, audio_path=stem.path, prompt="", negative_prompt="",
                     strength=0.0, seed=0, endpoint="dry", endpoint_version="", scores=(),
                     accepted=True, reject_reason=None)


@dataclass
class SearchResult:
    best: Candidate
    candidates: list[Candidate] = field(default_factory=list)
    fallback_reason: str | None = None
    stopped_by_budget: bool = False


class SearchLoop:
    def __init__(self, endpoint: RestyleEndpoint, scorer, budget: BudgetGuard,
                 cache: CandidateCache, work_dir: str, manifest: RunManifest | None = None):
        self.endpoint = endpoint
        self.scorer = scorer
        self.budget = budget
        self.cache = cache
        self.work_dir = Path(work_dir)
        self.manifest = manifest

    # ── one request ──────────────────────────────────────────────────────────

    def _record(self, stem: Stem, c: Candidate, req: str, n_calls: int, latency: float,
                outcome: str) -> None:
        if self.manifest is None:
            return
        self.manifest.endpoint_calls.append(CallRecord(
            stem=stem.track, endpoint=self.endpoint.name, endpoint_version=self.endpoint.version,
            request_hash=req, prompt=c.prompt, negative_prompt=c.negative_prompt,
            divergence=c.strength,
            raw_params=self.endpoint.raw_params(c.prompt, c.negative_prompt, c.strength, c.seed),
            seed=c.seed, input_audio_hash=stem.content_hash,
            output_audio_hash=file_sha256(c.audio_path) if c.audio_path else "",
            usd_estimate=0.0 if outcome == "cache_hit" else self.endpoint.usd_per_call * n_calls,
            latency_s=round(latency, 4), n_calls=0 if outcome == "cache_hit" else n_calls,
            outcome=outcome,
        ))

    def _try(self, stem: Stem, structure: StructureJSON, prompt: str, neg: str,
             divergence: float, seed: int) -> Candidate | None:
        """Score one request; None means the budget refused it."""
        req = request_hash(stem.content_hash, prompt, neg, divergence, seed,
                           self.endpoint.name, self.endpoint.version)
        hit = self.cache.get(req)
        if hit is not None:
            hit = dataclasses.replace(hit, stem=stem.track)
            self._record(stem, hit, req, 0, 0.0, "cache_hit")
            return self.scorer.score(hit) if hit.audio_path else hit

        n_calls = n_chunks_for(stem, structure, self.endpoint)
        cost = n_calls * self.endpoint.usd_per_call
        if not self.budget.can_spend(cost, stem.track, n_calls):
            return None
        work = self.work_dir / req[:16]
        work.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        raw, used = chunk_and_restyle(stem, structure, self.endpoint, prompt, neg, divergence,
                                      seed, str(work / "out.wav"), str(work))
        latency = time.perf_counter() - t0
        self.budget.record_call(used * self.endpoint.usd_per_call, stem.track, used)
        raw = dataclasses.replace(raw, request_hash=req, usd_estimate=used * self.endpoint.usd_per_call)
        if raw.accepted or raw.reject_reason == "content_policy":
            raw = dataclasses.replace(self.cache.put(req, raw), stem=stem.track,
                                      usd_estimate=raw.usd_estimate)
        self._record(stem, raw, req, used, latency, "ok" if raw.accepted else raw.reject_reason or "fatal")
        shutil.rmtree(work, ignore_errors=True)
        if not raw.accepted:
            log.info("%s: endpoint %s for d=%.2f seed=%d", stem.track, raw.reject_reason, divergence, seed)
            return raw
        return self.scorer.score(raw)

    # ── the search ───────────────────────────────────────────────────────────

    def search(self, stem: Stem, prompts: list[tuple[str, str]], structure: StructureJSON,
               config: SearchConfig) -> SearchResult:
        grid = list(config.strength_grid) if config.strength_grid else divergence_grid(
            stem.role, stem.microtonal and not self.endpoint.follows_f0, mean_section_energy(structure))
        if not grid:
            # P6: routed, not retried — the deterministic render is the output.
            return SearchResult(best=dry_candidate(stem), fallback_reason="microtonal_routed")

        candidates: list[Candidate] = []
        tried: set[tuple[float, str, int]] = set()
        stopped = False

        def attempt(d: float, prompt: str, neg: str, seed: int) -> bool:
            key = (round(d, 4), prompt, seed)
            if key in tried:
                return True
            tried.add(key)
            c = self._try(stem, structure, prompt, neg, d, seed)
            if c is None:
                return False
            candidates.append(c)
            return True

        for d in grid:
            for prompt, neg in prompts[:config.n_prompt_variants]:
                for seed in range(config.seeds):
                    if not attempt(d, prompt, neg, seed):
                        stopped = True
                        break
                if stopped:
                    break
            if stopped:
                break

        if config.refine and not stopped:
            accepted = sorted((c for c in candidates if c.accepted),
                              key=lambda c: c.total_score, reverse=True)
            top = accepted[:max(1, len(accepted) // 4)] if accepted else []
            tried_d = {d for d, _, _ in tried}
            for c in top:
                for d in refine_grid([c.strength], tried_d):
                    if not attempt(d, c.prompt, c.negative_prompt, c.seed):
                        stopped = True
                        break
                if stopped:
                    break

        if stopped:
            log.warning("%s: budget exhausted after %d candidates; using best so far",
                        stem.track, len(candidates))
        accepted = [c for c in candidates if c.accepted]
        if not accepted:
            if config.on_no_candidate == "strict":
                raise NoAcceptableCandidateError(stem.track, candidates)
            reason = "budget_exhausted" if stopped and not candidates else "all_candidates_below_floor"
            log.warning("no candidate cleared the floors for %s (%d tried, reasons=%s); "
                        "falling back to the dry stem", stem.track, len(candidates),
                        dict(reason_histogram(candidates)))
            return SearchResult(best=dry_candidate(stem), candidates=candidates,
                                fallback_reason=reason, stopped_by_budget=stopped)
        best = max(accepted, key=lambda c: (c.total_score, -c.strength, -c.seed))
        return SearchResult(best=best, candidates=candidates, stopped_by_budget=stopped)
