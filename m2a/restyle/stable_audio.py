# m2a/restyle/stable_audio.py
"""Stable Audio 2.5 audio-to-audio endpoint (Step 3.1).

Synchronous multipart POST to Stability's v2beta audio-to-audio route; the
response body is the generated audio. Provider `strength` is how far the
output may move from the input (0 = identical, 1 = input ignored), i.e.
`strength_semantics = "divergence"` (D011, verified per endpoint by the
opt-in smoke test).

Errors are normalised: 429 / 5xx / transport errors are retryable (tenacity
exponential backoff), 403 is the provider's content-moderation refusal,
everything else is fatal. After retries are exhausted the failure comes back
as a rejected Candidate, never an exception (P3).

Field names and the URL are constructor arguments because provider APIs
move; the defaults match the v2beta API at the time of writing and are
exercised only by the `smoke` test (MIDI2AUDIO_SMOKE=1). `negative_prompt`
is sent only when `send_negative_prompt=True`, since not every model
version accepts it.
"""
from __future__ import annotations

import io
import logging
import math
import os

import httpx
import soundfile as sf
from tenacity import (
    retry, retry_if_exception_type, stop_after_attempt, wait_exponential,
)

from m2a.contracts import Candidate
from m2a.restyle.base import (
    ContentPolicyError, EndpointError, FatalEndpointError, RetryableEndpointError, candidate,
    provider_strength, write_pipeline_wav,
)

log = logging.getLogger(__name__)

DEFAULT_URL = "https://api.stability.ai/v2beta/audio/stable-audio-2/audio-to-audio"


class StableAudioEndpoint:
    name = "stable_audio_a2a"
    version = "2.5"
    is_diffusion = True
    follows_f0 = False
    strength_semantics = "divergence"      # higher value => further from the input
    # Length caps change between model versions; the chunker asserts its chunk
    # length against this value rather than assuming one.
    max_input_seconds = 190.0

    def __init__(self, api_key: str | None = None, *, url: str = DEFAULT_URL,
                 model: str = "stable-audio-2.5", usd_per_call: float = 0.06,
                 max_retries: int = 3, backoff_sec: float = 2.0, timeout_sec: float = 180.0,
                 send_negative_prompt: bool = False, transport: httpx.BaseTransport | None = None):
        self.api_key = api_key or os.environ.get("STABILITY_API_KEY", "")
        if not self.api_key:
            raise FatalEndpointError("no API key: pass api_key or set STABILITY_API_KEY")
        self.url = url
        self.model = model
        self.usd_per_call = usd_per_call
        self.max_retries = max_retries
        self.backoff_sec = backoff_sec
        self.send_negative_prompt = send_negative_prompt
        self._client = httpx.Client(timeout=timeout_sec, transport=transport)

    def raw_params(self, prompt: str, negative_prompt: str, strength: float, seed: int) -> dict:
        params = {
            "prompt": prompt,
            "model": self.model,
            "strength": provider_strength(strength, self.strength_semantics),
            "seed": int(seed) % 4_294_967_295,
            "output_format": "wav",
        }
        if self.send_negative_prompt and negative_prompt:
            params["negative_prompt"] = negative_prompt
        return params

    def _post(self, audio_bytes: bytes, params: dict) -> bytes:
        try:
            resp = self._client.post(
                self.url,
                headers={"Authorization": f"Bearer {self.api_key}", "Accept": "audio/*"},
                files={"audio": ("input.wav", audio_bytes, "audio/wav")},
                data={k: str(v) for k, v in params.items()},
            )
        except httpx.TransportError as e:
            raise RetryableEndpointError(f"transport error: {e}") from e
        if resp.status_code == 200:
            return resp.content
        detail = resp.text[:300]
        if resp.status_code == 429 or resp.status_code >= 500:
            raise RetryableEndpointError(f"HTTP {resp.status_code}: {detail}")
        if resp.status_code == 403:
            raise ContentPolicyError(f"HTTP 403: {detail}")
        raise FatalEndpointError(f"HTTP {resp.status_code}: {detail}")

    def transform(self, audio_path: str, prompt: str, negative_prompt: str,
                  strength: float, seed: int, extra: dict) -> Candidate:
        out_path = extra["out_path"]
        stem = extra.get("stem", "")
        info = sf.info(audio_path)
        if info.duration > self.max_input_seconds:
            raise ValueError(f"{audio_path} is {info.duration:.1f}s > {self.max_input_seconds}s; "
                             "chunk it first (restyle.chunking)")
        params = self.raw_params(prompt, negative_prompt, strength, seed)
        params["duration"] = max(1, min(int(self.max_input_seconds), math.ceil(info.duration)))
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        post = retry(
            retry=retry_if_exception_type(RetryableEndpointError),
            stop=stop_after_attempt(self.max_retries),
            wait=wait_exponential(multiplier=self.backoff_sec, max=60),
            reraise=True,
        )(self._post)
        try:
            body = post(audio_bytes, params)
            audio, sr = sf.read(io.BytesIO(body))
        except EndpointError as e:
            log.warning("%s failed (%s): %s", self.name, e.kind, e)
            return candidate(self, stem, "", prompt, negative_prompt, strength, seed, reject_reason=e.kind)
        except (sf.LibsndfileError, RuntimeError) as e:
            log.warning("%s returned undecodable audio: %s", self.name, e)
            return candidate(self, stem, "", prompt, negative_prompt, strength, seed, reject_reason="fatal")
        write_pipeline_wav(audio, sr, out_path)
        return candidate(self, stem, out_path, prompt, negative_prompt, strength, seed)
