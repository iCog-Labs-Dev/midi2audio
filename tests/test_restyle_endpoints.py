"""tests/test_restyle_endpoints.py — endpoint protocol, Stable Audio wrapper, offline mock.

Tests never hit paid endpoints: the Stable Audio wrapper runs against an
`httpx.MockTransport` (no cassettes — recorded audio bodies would bloat the
repo, §7 point 3). The live check of D011 is opt-in (`MIDI2AUDIO_SMOKE=1`).
"""
from __future__ import annotations

import io
import os

import httpx
import numpy as np
import pytest
import soundfile as sf

from m2a.restyle.base import (
    RestyleEndpoint, provider_strength, request_hash, write_pipeline_wav,
)
from m2a.restyle.offline import OfflineEndpoint
from m2a.restyle.stable_audio import StableAudioEndpoint
from m2a.restyle.base import FatalEndpointError

STEM = "tests/fixtures/five_sec_stem.wav"


def _wav_bytes(sr=44_100, seconds=1.0):
    buf = io.BytesIO()
    t = np.arange(int(sr * seconds)) / sr
    sf.write(buf, 0.1 * np.sin(2 * np.pi * 220 * t), sr, format="WAV")
    return buf.getvalue()


def _endpoint(handler, **kw):
    return StableAudioEndpoint(api_key="test", transport=httpx.MockTransport(handler),
                               backoff_sec=0.0, **kw)


# ── protocol helpers ─────────────────────────────────────────────────────────

def test_provider_strength_semantics():
    assert provider_strength(0.8, "divergence") == 0.8
    assert provider_strength(0.8, "preservation") == pytest.approx(0.2)
    with pytest.raises(ValueError):
        provider_strength(1.2, "divergence")
    with pytest.raises(ValueError):
        provider_strength(0.5, "sideways")


def test_request_hash_is_deterministic_and_sensitive():
    base = ("h", "funk drums", "EDM", 0.7, 42, "sa", "2.5")
    assert request_hash(*base) == request_hash(*base)
    for i, changed in enumerate(["h2", "jazz drums", "", 0.71, 43, "sb", "2.6"]):
        args = list(base)
        args[i] = changed
        assert request_hash(*args) != request_hash(*base)


def test_endpoints_satisfy_protocol():
    assert isinstance(OfflineEndpoint(), RestyleEndpoint)
    assert isinstance(_endpoint(lambda r: httpx.Response(200)), RestyleEndpoint)


def test_write_pipeline_wav_resamples_to_48k(tmp_path):
    out = str(tmp_path / "o.wav")
    write_pipeline_wav(np.zeros(44_100), 44_100, out)
    info = sf.info(out)
    assert info.samplerate == 48_000 and info.frames == 48_000


# ── Stable Audio wrapper ─────────────────────────────────────────────────────

def test_successful_transform(tmp_path):
    seen = {}

    def handler(request: httpx.Request):
        seen["auth"] = request.headers["authorization"]
        seen["body"] = request.read()
        return httpx.Response(200, content=_wav_bytes())

    ep = _endpoint(handler)
    c = ep.transform(STEM, "funk drums", "EDM", 0.7, 42, {"out_path": str(tmp_path / "o.wav"), "stem": "drums"})
    assert c.accepted and c.reject_reason is None
    assert c.audio_path and sf.info(c.audio_path).samplerate == 48_000
    assert c.endpoint == "stable_audio_a2a" and c.endpoint_version == "2.5" and c.stem == "drums"
    assert seen["auth"] == "Bearer test"
    assert b'name="strength"' in seen["body"] and b"0.7" in seen["body"]
    assert b"negative_prompt" not in seen["body"]          # opt-in only


def test_rate_limit_retries_then_succeeds(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(429, text="slow down") if len(calls) < 3 else httpx.Response(200, content=_wav_bytes())

    c = _endpoint(handler, max_retries=3).transform(STEM, "p", "n", 0.7, 1, {"out_path": str(tmp_path / "o.wav")})
    assert len(calls) == 3
    assert c.accepted


def test_retries_exhausted_returns_rejected_candidate(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(503)

    c = _endpoint(handler, max_retries=2).transform(STEM, "p", "n", 0.7, 1, {"out_path": str(tmp_path / "o.wav")})
    assert len(calls) == 2
    assert not c.accepted and c.reject_reason == "retryable" and c.audio_path == ""


def test_content_policy_is_not_retried(tmp_path):
    calls = []

    def handler(request):
        calls.append(1)
        return httpx.Response(403, text="flagged by content moderation")

    c = _endpoint(handler, max_retries=3).transform(STEM, "p", "n", 0.7, 1, {"out_path": str(tmp_path / "o.wav")})
    assert len(calls) == 1
    assert c.reject_reason == "content_policy"


def test_bad_request_is_fatal(tmp_path):
    c = _endpoint(lambda r: httpx.Response(400, text="bad")).transform(
        STEM, "p", "n", 0.7, 1, {"out_path": str(tmp_path / "o.wav")})
    assert c.reject_reason == "fatal"


def test_transport_error_is_retryable(tmp_path):
    def handler(request):
        raise httpx.ConnectError("down", request=request)

    c = _endpoint(handler, max_retries=2).transform(STEM, "p", "n", 0.7, 1, {"out_path": str(tmp_path / "o.wav")})
    assert c.reject_reason == "retryable"


def test_missing_api_key_is_fatal(monkeypatch):
    monkeypatch.delenv("STABILITY_API_KEY", raising=False)
    with pytest.raises(FatalEndpointError):
        StableAudioEndpoint()


def test_overlong_input_must_be_chunked(tmp_path):
    ep = _endpoint(lambda r: httpx.Response(200, content=_wav_bytes()))
    ep.max_input_seconds = 2.0
    with pytest.raises(ValueError, match="chunk"):
        ep.transform(STEM, "p", "n", 0.7, 1, {"out_path": str(tmp_path / "o.wav")})


# ── offline mock ─────────────────────────────────────────────────────────────

def test_offline_endpoint_is_deterministic(tmp_path):
    ep = OfflineEndpoint()
    a = ep.transform(STEM, "p", "n", 0.7, 3, {"out_path": str(tmp_path / "a.wav")})
    b = ep.transform(STEM, "p", "n", 0.7, 3, {"out_path": str(tmp_path / "b.wav")})
    np.testing.assert_array_equal(sf.read(a.audio_path)[0], sf.read(b.audio_path)[0])
    assert ep.calls == 2


def _spectral_distance(a_path, b_path):
    a, _ = sf.read(a_path, always_2d=True)
    b, _ = sf.read(b_path, always_2d=True)
    n = min(len(a), len(b))
    A = np.abs(np.fft.rfft(a[:n, 0]))
    B = np.abs(np.fft.rfft(b[:n, 0]))
    return float(np.linalg.norm(np.log1p(A) - np.log1p(B)))


@pytest.fixture
def tonal_stem(tmp_path):
    t = np.arange(48_000 * 3) / 48_000
    path = str(tmp_path / "tone.wav")
    sf.write(path, 0.2 * np.sin(2 * np.pi * 220 * t) * (1 + 0.5 * np.sin(2 * np.pi * 2 * t)), 48_000)
    return path


def test_offline_divergence_moves_output_further(tmp_path, tonal_stem):
    ep = OfflineEndpoint()
    lo = ep.transform(tonal_stem, "p", "n", 0.1, 0, {"out_path": str(tmp_path / "lo.wav")})
    hi = ep.transform(tonal_stem, "p", "n", 0.8, 0, {"out_path": str(tmp_path / "hi.wav")})
    assert _spectral_distance(hi.audio_path, tonal_stem) > _spectral_distance(lo.audio_path, tonal_stem)


@pytest.mark.smoke
@pytest.mark.skipif(os.environ.get("MIDI2AUDIO_SMOKE") != "1", reason="live endpoint; set MIDI2AUDIO_SMOKE=1")
def test_stable_audio_strength_semantics_live(tmp_path, tonal_stem):
    """D011: divergence 0.9 must land spectrally further from the input than 0.1."""
    ep = StableAudioEndpoint()
    lo = ep.transform(tonal_stem, "warm bass", "", 0.1, 7, {"out_path": str(tmp_path / "lo.wav")})
    hi = ep.transform(tonal_stem, "warm bass", "", 0.9, 7, {"out_path": str(tmp_path / "hi.wav")})
    assert lo.accepted and hi.accepted
    assert _spectral_distance(hi.audio_path, tonal_stem) > _spectral_distance(lo.audio_path, tonal_stem)
