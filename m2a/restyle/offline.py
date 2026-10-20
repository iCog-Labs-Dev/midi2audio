# m2a/restyle/offline.py
"""Deterministic offline endpoint: the mocked endpoint for CI and dry runs.

No network, no cost. It imitates the properties of a generative a2a
endpoint that the rest of the pipeline must cope with, as a function of
divergence and seed:

* timbre moves further from the input as divergence rises (a seeded
  resonant filter mixed in proportionally);
* a seed-dependent latency shift of a few ms (endpoints shift audio — the
  scorer must align before scoring);
* above `drift_threshold` divergence the material is "re-performed" (a
  doubled attack 80-120 ms late), so a high-divergence candidate fails the
  rhythm floor, as real endpoints do when they drift from the input.
"""
from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy import signal

from m2a.contracts import Candidate
from m2a.restyle.base import candidate, provider_strength, write_pipeline_wav


class OfflineEndpoint:
    name = "offline_mock"
    version = "1"
    is_diffusion = True
    follows_f0 = False
    strength_semantics = "divergence"

    def __init__(self, usd_per_call: float = 0.0, max_input_seconds: float = 190.0,
                 max_latency_ms: float = 12.0, drift_threshold: float = 0.85):
        self.usd_per_call = usd_per_call
        self.max_input_seconds = max_input_seconds
        self.max_latency_ms = max_latency_ms
        self.drift_threshold = drift_threshold
        self.calls = 0

    def raw_params(self, prompt: str, negative_prompt: str, strength: float, seed: int) -> dict:
        return {"prompt": prompt, "negative_prompt": negative_prompt,
                "strength": provider_strength(strength, self.strength_semantics), "seed": seed}

    def transform(self, audio_path: str, prompt: str, negative_prompt: str,
                  strength: float, seed: int, extra: dict) -> Candidate:
        self.calls += 1
        d = provider_strength(strength, self.strength_semantics)
        audio, sr = sf.read(audio_path, always_2d=True)
        if audio.shape[0] / sr > self.max_input_seconds:
            raise ValueError(f"{audio_path} exceeds {self.max_input_seconds}s")
        rng = np.random.default_rng([seed, int(round(d * 1000)), len(prompt)])

        centre = rng.uniform(300.0, 3000.0)
        b, a = signal.iirpeak(centre, Q=2.0, fs=sr)
        coloured = signal.lfilter(b, a, audio, axis=0)
        out = (1.0 - d) * audio + d * coloured * 2.0

        if d > self.drift_threshold:
            # re-performed material: a doubled attack 80-120 ms late, outside
            # every rhythm window, so precision collapses
            delay = int(rng.uniform(0.08, 0.12) * sr)
            smear = np.zeros_like(out)
            smear[delay:] = out[:-delay]
            out = out + smear

        shift = int(rng.uniform(0.0, self.max_latency_ms) / 1000.0 * sr)
        out = np.concatenate([np.zeros((shift, out.shape[1])), out])[:audio.shape[0]]
        peak = float(np.max(np.abs(out))) or 1.0
        out = out / peak * 10 ** (-12.0 / 20.0)

        write_pipeline_wav(out, sr, extra["out_path"])
        return candidate(self, extra.get("stem", ""), extra["out_path"], prompt, negative_prompt,
                         strength, seed)
