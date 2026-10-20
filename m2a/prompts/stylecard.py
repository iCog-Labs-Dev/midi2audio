# m2a/prompts/stylecard.py
"""Reference corpus → house style card (Step 3.3, §8.7).

Audio captioning may not be available, so the card is built from a concrete
feature list measured on the references: tempo, spectral centroid / rolloff
percentiles, crest factor, integrated loudness, stereo width and band
energy ratios. Descriptors and the negative list are derived from those
features by fixed rules (deterministic, P1); an optional `llm_fn` may turn
the features into richer descriptors, and its output is validated as a
JSON object with `descriptors` / `negatives` string lists.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import librosa
import numpy as np
import pyloudnorm
import soundfile as sf

from m2a.scoring.fad import list_audio

BANDS_HZ = {"low": (20, 250), "mid": (250, 4000), "high": (4000, 16000)}


@dataclass(frozen=True)
class StyleCard:
    descriptors: tuple[str, ...] = ()
    negatives: tuple[str, ...] = ()
    features: dict = field(default_factory=dict)
    user_notes: str = ""


def measure_features(audio_path: str) -> dict:
    audio, sr = sf.read(audio_path, always_2d=True)
    mono = audio.mean(axis=1)
    y = librosa.resample(mono, orig_sr=sr, target_sr=22_050) if sr != 22_050 else mono
    centroid = librosa.feature.spectral_centroid(y=y, sr=22_050)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=22_050)[0]
    tempo = float(np.atleast_1d(librosa.beat.beat_track(y=y, sr=22_050)[0])[0])
    spec = np.abs(librosa.stft(y)) ** 2
    freqs = librosa.fft_frequencies(sr=22_050)
    total = float(spec.sum()) or 1.0
    bands = {name: float(spec[(freqs >= lo) & (freqs < hi)].sum() / total)
             for name, (lo, hi) in BANDS_HZ.items()}
    rms = float(np.sqrt(np.mean(mono ** 2))) or 1e-12
    crest_db = float(20 * np.log10(max(float(np.max(np.abs(mono))), 1e-12) / rms))
    try:
        lufs = float(pyloudnorm.Meter(sr).integrated_loudness(audio))
    except ValueError:      # shorter than one gating block
        lufs = float("nan")
    if audio.shape[1] >= 2:
        mid, side = audio[:, 0] + audio[:, 1], audio[:, 0] - audio[:, 1]
        width = float(np.sqrt(np.mean(side ** 2)) / max(np.sqrt(np.mean(mid ** 2)), 1e-12))
    else:
        width = 0.0
    return {
        "tempo_bpm": tempo,
        "centroid_hz_p50": float(np.percentile(centroid, 50)),
        "centroid_hz_p90": float(np.percentile(centroid, 90)),
        "rolloff_hz_p50": float(np.percentile(rolloff, 50)),
        "crest_db": crest_db,
        "integrated_lufs": lufs,
        "stereo_width": width,
        "band_ratio": bands,
    }


def _aggregate(per_file: list[dict]) -> dict:
    keys = [k for k in per_file[0] if k != "band_ratio"]
    agg = {k: float(np.nanmedian([f[k] for f in per_file])) for k in keys}
    agg["band_ratio"] = {b: float(np.median([f["band_ratio"][b] for f in per_file])) for b in BANDS_HZ}
    return agg


def descriptors_from_features(f: dict) -> tuple[list[str], list[str]]:
    desc, neg = [], []
    if f["centroid_hz_p50"] > 2500:
        desc.append("bright")
        neg.append("muffled")
    elif f["centroid_hz_p50"] < 1200:
        desc.append("warm")
        neg.append("harsh")
    if f["crest_db"] > 14:
        desc.append("dynamic")
        neg.append("over-compressed")
    elif f["crest_db"] < 9:
        desc.append("dense, compressed")
    if f["band_ratio"]["low"] > 0.5:
        desc.append("bass-heavy")
    if f["stereo_width"] > 0.4:
        desc.append("wide stereo image")
    neg += ["lo-fi", "distorted"]
    return desc, neg


def make_style_card(reference_dir: str | None, user_notes: str = "",
                    llm_fn: Callable[[str, str], str] | None = None) -> StyleCard:
    """Extract recurring descriptors from reference audio (or just carry user notes)."""
    if not reference_dir or not list_audio(reference_dir):
        return StyleCard(user_notes=user_notes)
    features = _aggregate([measure_features(p) for p in list_audio(reference_dir)])
    desc, neg = descriptors_from_features(features)
    if llm_fn is not None:
        raw = llm_fn("Turn these measured audio features of a reference corpus into style "
                     "descriptors. Reply with JSON {\"descriptors\": [...], \"negatives\": [...]}.",
                     json.dumps({"features": features, "user_notes": user_notes}))
        try:
            data = json.loads(raw)
            desc += [str(d) for d in data.get("descriptors", [])]
            neg += [str(n) for n in data.get("negatives", [])]
        except (json.JSONDecodeError, AttributeError):
            pass    # the rule-based card stands on its own
    dedup = lambda xs: tuple(dict.fromkeys(xs))  # noqa: E731
    return StyleCard(descriptors=dedup(desc), negatives=dedup(neg), features=features,
                     user_notes=user_notes)
