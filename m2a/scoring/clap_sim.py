# m2a/scoring/clap_sim.py
"""CLAP audio-text similarity (Step 2.3): cos(audio, prompt) - cos(audio, negative).

The encoder is pluggable (`ClapEncoder`). `LaionClapEncoder` wraps
`laion_clap` from the [audio] extra; when it is not installed
`default_clap_encoder()` returns None and the combined scorer drops the
`clap` axis and renormalises the remaining weights rather than inventing a
number (D010: heavy scorers may live behind a service boundary).
"""
from __future__ import annotations

import logging
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class ClapEncoder(Protocol):
    def encode_audio(self, audio_path: str) -> np.ndarray: ...
    def encode_text(self, text: str) -> np.ndarray: ...


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a), np.ravel(b)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 0 else 0.0


def clap_similarity(audio_path: str, prompt: str, negative_prompt: str,
                    encoder: ClapEncoder) -> float:
    audio_emb = encoder.encode_audio(audio_path)
    score = cosine(audio_emb, encoder.encode_text(prompt))
    if negative_prompt:
        score -= cosine(audio_emb, encoder.encode_text(negative_prompt))
    return score


class LaionClapEncoder:
    """laion_clap CLAP_Module; loads the default checkpoint once, lazily."""

    def __init__(self, checkpoint: str | None = None):
        import laion_clap  # optional heavy dependency ([audio] extra)
        self._model = laion_clap.CLAP_Module(enable_fusion=False)
        if checkpoint:
            self._model.load_ckpt(checkpoint)
        else:
            self._model.load_ckpt()
        self._text_cache: dict[str, np.ndarray] = {}

    def encode_audio(self, audio_path: str) -> np.ndarray:
        return np.asarray(self._model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=False)[0])

    def encode_text(self, text: str) -> np.ndarray:
        if text not in self._text_cache:
            emb = self._model.get_text_embedding([text, ""], use_tensor=False)[0]
            self._text_cache[text] = np.asarray(emb)
        return self._text_cache[text]


def default_clap_encoder(backend: str = "auto") -> ClapEncoder | None:
    """`auto`: laion_clap if importable, else None. `off`: None. `laion`: required."""
    if backend == "off":
        return None
    try:
        return LaionClapEncoder()
    except ImportError:
        if backend == "laion":
            raise
        log.warning("laion_clap not installed; CLAP axis disabled (weights renormalised)")
        return None
