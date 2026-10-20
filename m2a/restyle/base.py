# m2a/restyle/base.py
"""RestyleEndpoint protocol and the shared endpoint vocabulary (Step 3.1).

D011 — the pipeline searches a normalised `divergence` in [0, 1] (higher =
further from the input). Each endpoint declares `strength_semantics` and
converts with `provider_strength`, so a provider whose knob means
"preservation" cannot silently walk the search grid backwards.

P3 — endpoint-side failures never raise out of `transform`: they come back as
a rejected `Candidate` whose `reject_reason` is one of `OUTCOMES`, so the
search loop can log and move on. Only programming errors (bad arguments,
missing input file) raise.
"""
from __future__ import annotations

import hashlib
import json
from typing import Protocol, runtime_checkable

import numpy as np
import soundfile as sf
import soxr

from m2a.contracts import Candidate

PIPELINE_SR = 48_000

OUTCOMES = ("retryable", "fatal", "content_policy")


class EndpointError(Exception):
    kind = "fatal"


class RetryableEndpointError(EndpointError):
    kind = "retryable"


class FatalEndpointError(EndpointError):
    kind = "fatal"


class ContentPolicyError(EndpointError):
    kind = "content_policy"


@runtime_checkable
class RestyleEndpoint(Protocol):
    name: str
    version: str
    is_diffusion: bool
    follows_f0: bool          # True for DDSP/RAVE — eligible for microtonal stems
    max_input_seconds: float
    strength_semantics: str   # "divergence" | "preservation"
    usd_per_call: float

    def raw_params(self, prompt: str, negative_prompt: str, strength: float, seed: int) -> dict:
        """The provider-level parameters a call with these pipeline values would send."""

    def transform(self, audio_path: str, prompt: str, negative_prompt: str,
                  strength: float, seed: int, extra: dict) -> Candidate:
        """`strength` is normalised divergence. `extra["out_path"]` is where the
        result WAV (48 kHz) must be written."""


def write_pipeline_wav(audio: np.ndarray, sr: int, out_path: str) -> None:
    """Write endpoint output as 48 kHz WAV (endpoints commonly return 44.1 kHz)."""
    if sr != PIPELINE_SR:
        audio = soxr.resample(audio, sr, PIPELINE_SR)
    sf.write(out_path, audio, PIPELINE_SR)


def provider_strength(divergence: float, semantics: str) -> float:
    if not 0.0 <= divergence <= 1.0:
        raise ValueError(f"divergence must be in [0, 1], got {divergence}")
    if semantics == "divergence":
        return divergence
    if semantics == "preservation":
        return 1.0 - divergence
    raise ValueError(f"unknown strength_semantics {semantics!r}")


def request_hash(audio_hash: str, prompt: str, negative_prompt: str, divergence: float,
                 seed: int, endpoint: str, endpoint_version: str) -> str:
    """Identical request hash => reuse (cache hits are free, P5)."""
    payload = json.dumps([audio_hash, prompt, negative_prompt, round(divergence, 6), seed,
                          endpoint, endpoint_version])
    return hashlib.sha256(payload.encode()).hexdigest()


def candidate(endpoint: RestyleEndpoint, stem: str, audio_path: str, prompt: str,
              negative_prompt: str, strength: float, seed: int,
              reject_reason: str | None = None) -> Candidate:
    """An unscored candidate as returned by `transform` (accepted means the
    endpoint produced audio; the scorer decides acceptance on merit)."""
    return Candidate(
        stem=stem, audio_path=audio_path if reject_reason is None else "",
        prompt=prompt, negative_prompt=negative_prompt, strength=strength, seed=seed,
        endpoint=endpoint.name, endpoint_version=endpoint.version, scores=(),
        accepted=reject_reason is None, reject_reason=reject_reason,
        usd_estimate=endpoint.usd_per_call,
    )
