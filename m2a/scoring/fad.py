# m2a/scoring/fad.py
"""Fréchet Audio Distance against a reference corpus (Step 2.4).

FAD is a *distance* — lower is better and unbounded. `combine.norm_fad` maps
it to (0, 1] with `exp(-fad / tau)` before it enters any weighted sum; never
add raw FAD to a score (v1.0 bug A).

Embeddings are pluggable (`Embedder`). The default `MelStatsEmbedder` is a
deterministic log-mel window embedding that needs no model download; a CLAP
window embedder can be swapped in once D013 (estimator and embedding for
small-N corpora) is resolved in M4. Covariances are diagonally loaded so the
distance is defined when a corpus has fewer windows than dimensions.

`tau` is calibrated per corpus as the median leave-one-out FAD of each
reference against the rest (D013's suggestion); with a single reference
file there is nothing to leave out and `DEFAULT_TAU` is used.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import librosa
import numpy as np
from scipy import linalg

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif")
DEFAULT_TAU = 10.0
COV_EPS = 1e-4


class Embedder(Protocol):
    name: str

    def embed_windows(self, audio_path: str) -> np.ndarray:
        """(n_windows, dim) embeddings."""


class MelStatsEmbedder:
    """Per-window mean and std of log-mel bands."""

    name = "mel_stats_v1"

    def __init__(self, sr: int = 22_050, n_mels: int = 32, window_sec: float = 1.0,
                 hop_sec: float = 0.5):
        self.sr, self.n_mels = sr, n_mels
        self.window_sec, self.hop_sec = window_sec, hop_sec

    def embed_windows(self, audio_path: str) -> np.ndarray:
        y, _ = librosa.load(audio_path, sr=self.sr, mono=True)
        mel = librosa.power_to_db(
            librosa.feature.melspectrogram(y=y, sr=self.sr, n_mels=self.n_mels, hop_length=512),
            ref=1.0, top_db=None, amin=1e-10)
        frames_per_sec = self.sr / 512
        win = max(1, int(self.window_sec * frames_per_sec))
        hop = max(1, int(self.hop_sec * frames_per_sec))
        rows = []
        for start in range(0, max(1, mel.shape[1] - win + 1), hop):
            seg = mel[:, start:start + win]
            rows.append(np.concatenate([seg.mean(axis=1), seg.std(axis=1)]))
        return np.array(rows)


def _stats(emb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = emb.mean(axis=0)
    sigma = np.cov(emb, rowvar=False) if len(emb) > 1 else np.zeros((emb.shape[1], emb.shape[1]))
    return mu, np.atleast_2d(sigma) + COV_EPS * np.eye(emb.shape[1])


def frechet_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Fréchet distance between Gaussians fitted to two embedding sets."""
    mu1, s1 = _stats(a)
    mu2, s2 = _stats(b)
    covmean = np.real(linalg.sqrtm(s1 @ s2))
    diff = mu1 - mu2
    fd = float(diff @ diff + np.trace(s1) + np.trace(s2) - 2.0 * np.trace(covmean))
    return max(fd, 0.0)


def list_audio(reference_dir: str) -> list[str]:
    return sorted(str(p) for p in Path(reference_dir).iterdir()
                  if p.suffix.lower() in AUDIO_EXTENSIONS)


@dataclass(frozen=True)
class ReferenceSet:
    """A reference corpus embedded once, with its calibrated `tau`."""
    paths: tuple[str, ...]
    embeddings: np.ndarray
    tau: float
    embedder_name: str

    @classmethod
    def from_dir(cls, reference_dir: str, embedder: Embedder | None = None) -> "ReferenceSet":
        embedder = embedder or MelStatsEmbedder()
        paths = list_audio(reference_dir)
        if not paths:
            raise ValueError(f"no reference audio found in {reference_dir}")
        per_file = [embedder.embed_windows(p) for p in paths]
        return cls(tuple(paths), np.vstack(per_file), calibrate_tau(per_file), embedder.name)


def calibrate_tau(per_file: list[np.ndarray]) -> float:
    if len(per_file) < 2:
        return DEFAULT_TAU
    loo = [frechet_distance(e, np.vstack(per_file[:i] + per_file[i + 1:]))
           for i, e in enumerate(per_file)]
    tau = float(np.median(loo))
    return tau if tau > 0 else DEFAULT_TAU


def fad_score(audio_path: str, refs: ReferenceSet, embedder: Embedder | None = None) -> float:
    embedder = embedder or MelStatsEmbedder()
    if embedder.name != refs.embedder_name:
        raise ValueError(f"embedder {embedder.name!r} != reference set's {refs.embedder_name!r}")
    return frechet_distance(embedder.embed_windows(audio_path), refs.embeddings)
