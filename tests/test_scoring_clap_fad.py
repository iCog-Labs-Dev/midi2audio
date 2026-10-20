"""tests/test_scoring_clap_fad.py — CLAP similarity (Step 2.3) and FAD (Step 2.4)."""
from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from m2a.scoring.clap_sim import clap_similarity, default_clap_encoder
from m2a.scoring.fad import (
    DEFAULT_TAU, MelStatsEmbedder, ReferenceSet, calibrate_tau, fad_score, frechet_distance,
)
from tests.synth import render_events

# ── CLAP ─────────────────────────────────────────────────────────────────────


class FakeClap:
    """Deterministic stand-in: audio and text embed onto fixed unit vectors."""
    vectors = {"funk drum kit": np.array([1.0, 0.0, 0.0]),
               "string quartet": np.array([0.0, 1.0, 0.0]),
               "drums": np.array([0.9, 0.1, 0.0])}

    def encode_audio(self, audio_path):
        return self.vectors["drums"]

    def encode_text(self, text):
        return self.vectors[text]


def test_matching_prompt_scores_higher():
    enc = FakeClap()
    assert clap_similarity("x.wav", "funk drum kit", "", enc) > clap_similarity("x.wav", "string quartet", "", enc)


def test_negative_prompt_subtraction_rewards_contrast():
    enc = FakeClap()
    with_neg = clap_similarity("x.wav", "funk drum kit", "string quartet", enc)
    assert with_neg < clap_similarity("x.wav", "funk drum kit", "", enc)
    assert with_neg > clap_similarity("x.wav", "string quartet", "funk drum kit", enc)


def test_clap_off_returns_none():
    assert default_clap_encoder("off") is None


@pytest.mark.slow
@pytest.mark.skipif(importlib.util.find_spec("laion_clap") is None, reason="laion_clap not installed")
def test_real_clap_drums_vs_strings(tmp_path):
    wav = render_events(str(tmp_path / "d.wav"), [(i * 0.25, 0.1, 0) for i in range(20)], 5.5,
                        percussive=True)
    enc = default_clap_encoder("laion")
    assert clap_similarity(wav, "funk drum kit", "", enc) > clap_similarity(wav, "string quartet", "", enc)


# ── FAD ──────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def corpora(tmp_path_factory):
    d = tmp_path_factory.mktemp("fad")
    tones, noise = d / "tones", d / "noise"
    tones.mkdir()
    noise.mkdir()
    for k, base in enumerate([48, 55, 60]):
        render_events(str(tones / f"t{k}.wav"), [(i * 0.5, 0.45, base + (i % 5)) for i in range(16)], 8.5)
        render_events(str(noise / f"n{k}.wav"), [(i * 0.25, 0.1, 0) for i in range(32)], 8.5, percussive=True)
    cand = render_events(str(d / "cand.wav"), [(i * 0.5, 0.45, 52 + (i % 4)) for i in range(16)], 8.5)
    return str(tones), str(noise), cand


def test_frechet_of_identical_sets_is_zero():
    x = np.random.default_rng(0).standard_normal((50, 8))
    assert frechet_distance(x, x) < 1e-6


def test_fad_against_itself_is_near_zero(corpora, tmp_path):
    _, _, cand = corpora
    import shutil
    shutil.copy(cand, tmp_path / "self.wav")
    refs = ReferenceSet.from_dir(str(tmp_path))
    assert fad_score(cand, refs) < 1e-3


def test_similar_corpus_scores_closer_than_different(corpora):
    tones, noise, cand = corpora
    assert fad_score(cand, ReferenceSet.from_dir(tones)) < fad_score(cand, ReferenceSet.from_dir(noise))


def test_tau_calibrated_from_leave_one_out(corpora):
    tones, _, _ = corpora
    refs = ReferenceSet.from_dir(tones)
    assert refs.tau > 0 and refs.tau != DEFAULT_TAU
    assert calibrate_tau([np.zeros((3, 4))]) == DEFAULT_TAU


def test_embedder_mismatch_rejected(corpora):
    tones, _, cand = corpora
    refs = ReferenceSet.from_dir(tones)

    class Other(MelStatsEmbedder):
        name = "other"

    with pytest.raises(ValueError):
        fad_score(cand, refs, Other())
