"""tests/test_restyle_chunking.py — boundary-aware chunking and the D002 crossfade law."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from m2a.contracts import Stem, StructureJSON
from m2a.restyle.chunking import (
    _bar_time, chunk_and_restyle, chunk_audio, chunk_spans, crossfade_rejoin,
    crossfade_rejoin_passthrough, n_chunks_for, select_seam_bars, select_seams,
)
from m2a.restyle.offline import OfflineEndpoint

SR = 48_000
BAR_SEC = 2.0
N_BARS = 32


def _structure(sections=None) -> StructureJSON:
    sections = sections or [
        {"label": "A", "bars": [0, 8], "energy": 0.5, "boundary_lvl": 0},
        {"label": "B", "bars": [8, 16], "energy": 0.6, "boundary_lvl": 3},
        {"label": "A2", "bars": [16, 24], "energy": 0.5, "boundary_lvl": 1},
        {"label": "C", "bars": [24, 32], "energy": 0.7, "boundary_lvl": 2},
    ]
    return StructureJSON.model_validate({
        "schema": "midi2audio.structure/1", "provenance": "planner", "source_hash": "x",
        "edo": 12, "base_tuning": 60.0, "tempo_map": [[0.0, 120.0]], "meter": [[0, "4/4"]],
        "key": [], "chords": [], "sections": sections,
        "bar_table": [[i, i * BAR_SEC] for i in range(N_BARS)],
        "tracks": [{"idx": 0, "name": "lead", "role": "lead"}],
    })


@pytest.fixture(scope="module")
def long_stem(tmp_path_factory):
    path = tmp_path_factory.mktemp("ch") / "long.wav"
    rng = np.random.default_rng(0)
    t = np.arange(int(N_BARS * BAR_SEC * SR)) / SR
    y = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.05 * rng.standard_normal(len(t))
    sf.write(str(path), np.stack([y, 0.8 * y], axis=1), SR, subtype="FLOAT")
    return str(path)


def test_seams_on_downbeats():
    s = _structure()
    seams = select_seams(s.sections, s.bar_table, N_BARS * BAR_SEC, max_chunk_sec=30.0)
    assert seams
    bar_times = [r[1] for r in s.bar_table]
    for seam_sec in seams:
        assert min(abs(seam_sec - t) for t in bar_times) < 0.01


def test_seams_prefer_strongest_boundary():
    s = _structure()
    # boundary levels: bar 8 -> 3, bar 24 -> 2, bar 16 -> 1.
    # One seam fits 52 s chunks: the lvl-3 boundary, not the bar nearest the middle.
    assert select_seam_bars(s.sections, s.bar_table, 64.0, max_chunk_sec=52.0) == [8]
    # 40 s chunks need a second seam: lvl 2 (bar 24) beats lvl 1 (bar 16).
    assert select_seam_bars(s.sections, s.bar_table, 64.0, max_chunk_sec=40.0) == [8, 24]


def test_every_chunk_fits_the_cap():
    s = _structure()
    seams = select_seam_bars(s.sections, s.bar_table, 64.0, max_chunk_sec=12.0)
    for a, b in chunk_spans(seams, s.bar_table, int(64.0 * SR), SR):
        assert (b - a) / SR <= 12.0 + 1e-6


def test_no_seam_when_stem_fits():
    s = _structure()
    assert select_seam_bars(s.sections, s.bar_table, 64.0, max_chunk_sec=100.0) == []


def test_unsplittable_raises():
    s = _structure()
    with pytest.raises(ValueError):
        select_seam_bars(s.sections, s.bar_table, 64.0, max_chunk_sec=3.0)


def test_overlap_is_two_bars_centred_on_seam(long_stem, tmp_path):
    s = _structure()
    chunks = chunk_audio(long_stem, [16], s.bar_table, str(tmp_path))
    (_, a0, b0), (_, a1, b1) = chunks
    assert a1 == int(_bar_time(15, s.bar_table) * SR) and b0 == int(_bar_time(17, s.bar_table) * SR)
    assert (b0 - a1) / SR == pytest.approx(2 * BAR_SEC)


def test_crossfade_is_lossless_without_restyle(long_stem, tmp_path):
    """Only satisfiable with an EQUAL-GAIN (linear) crossfade — equal-power on two
    copies of the same signal would put +3 dB on every seam (D002)."""
    s = _structure()
    seams = select_seam_bars(s.sections, s.bar_table, N_BARS * BAR_SEC, max_chunk_sec=20.0)
    assert len(seams) >= 2
    out = crossfade_rejoin_passthrough(long_stem, seams, s.bar_table, str(tmp_path))
    np.testing.assert_allclose(sf.read(out)[0], sf.read(long_stem)[0], atol=1e-5)


def test_rejoin_aligns_a_late_chunk(long_stem, tmp_path):
    s = _structure()
    chunks = chunk_audio(long_stem, [16], s.bar_table, str(tmp_path / "c"))
    path, a, b = chunks[1]
    late = 480                                   # 10 ms endpoint latency on chunk 2
    y, _ = sf.read(path, always_2d=True)
    sf.write(path, np.concatenate([np.zeros((late, 2)), y[:-late]]), SR, subtype="FLOAT")
    out = crossfade_rejoin(chunks, sf.info(long_stem).frames, str(tmp_path / "r.wav"))
    original, _ = sf.read(long_stem)
    joined, _ = sf.read(out)
    seam = int(16 * BAR_SEC * SR)
    region = slice(seam - SR // 2, seam + SR // 2)
    assert np.max(np.abs(joined[region] - original[region])) < 1e-4


def test_chunk_and_restyle_calls_endpoint_once_per_chunk(long_stem, tmp_path):
    s = _structure()
    ep = OfflineEndpoint(max_input_seconds=25.0)
    stem = Stem(track="lead", role="lead", path=long_stem, microtonal=False, peak_dbfs=-12.0,
                content_hash="h")
    c, n = chunk_and_restyle(stem, s, ep, "p", "n", 0.6, 0, str(tmp_path / "out.wav"), str(tmp_path / "w"))
    assert c.accepted and n == ep.calls == n_chunks_for(stem, s, ep) >= 3
    assert sf.info(c.audio_path).frames == sf.info(long_stem).frames
    assert c.stem == "lead" and c.usd_estimate == pytest.approx(n * ep.usd_per_call)
