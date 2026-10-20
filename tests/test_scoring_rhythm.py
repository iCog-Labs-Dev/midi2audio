"""tests/test_scoring_rhythm.py — onset F-measure scorer (Step 2.1).

Corruptions are applied to the *audio* and scored against the clean MIDI.
A uniform shift of every onset is deliberately NOT a monotonicity corruption:
FIX 2 aligns the candidate first, so a global shift must be recovered and
reported instead (`test_global_shift_is_recovered_and_reported`). Timing
damage is modelled as a growing fraction of displaced notes.
"""
from __future__ import annotations

import numpy as np
import pytest

from m2a.scoring.rhythm import collapse_simultaneous, rhythm_report
from tests.synth import render_events, write_midi

N = 32
BEAT = 0.5
TOTAL = N * BEAT + 1.0


@pytest.fixture(scope="module")
def ref_midi(tmp_path_factory):
    d = tmp_path_factory.mktemp("rh")
    return write_midi(str(d / "ref.mid"), [[(i, 0.9, 36) for i in range(N)]], channels=[9])


def _score(tmp_path, ref_midi, onsets, name):
    wav = render_events(str(tmp_path / f"{name}.wav"), [(t, 0.3, 0) for t in onsets],
                        TOTAL, percussive=True)
    return rhythm_report(wav, ref_midi, "drums", 0)


def _clean():
    return [i * BEAT for i in range(N)]


def _displace(fraction, ms, seed=0):
    rng = np.random.default_rng(seed)
    onsets = _clean()
    for i in rng.choice(N, int(N * fraction), replace=False):
        onsets[i] += ms / 1000.0
    return onsets


def test_clean_render_scores_near_one(tmp_path, ref_midi):
    r = _score(tmp_path, ref_midi, _clean(), "clean")
    assert r.f_measure > 0.95
    assert abs(r.shift_sec) < 0.03


def test_timing_damage_is_monotonic(tmp_path, ref_midi):
    clean = _score(tmp_path, ref_midi, _clean(), "c").f_measure
    q = _score(tmp_path, ref_midi, _displace(0.25, 60), "q").f_measure
    h = _score(tmp_path, ref_midi, _displace(0.5, 60), "h").f_measure
    assert clean > q > h


def test_dropped_notes_lower_recall(tmp_path, ref_midi):
    keep = [t for i, t in enumerate(_clean()) if i % 5 != 0]       # drop 20 %
    clean = _score(tmp_path, ref_midi, _clean(), "c")
    dropped = _score(tmp_path, ref_midi, keep, "d")
    assert dropped.recall < clean.recall
    assert dropped.f_measure < clean.f_measure


def test_extra_onsets_lower_precision(tmp_path, ref_midi):
    extra = sorted(_clean() + [i * BEAT + 0.25 for i in range(0, N, 5)])   # +20 % off-beat
    clean = _score(tmp_path, ref_midi, _clean(), "c")
    noisy = _score(tmp_path, ref_midi, extra, "x")
    assert noisy.precision < clean.precision
    assert noisy.f_measure < clean.f_measure


def test_global_shift_is_recovered_and_reported(tmp_path, ref_midi):
    clean = _score(tmp_path, ref_midi, _clean(), "c")
    late = _score(tmp_path, ref_midi, [t + 0.08 for t in _clean()], "late")
    assert late.f_measure > 0.95
    assert abs((late.shift_sec - clean.shift_sec) - 0.08) < 0.015


def test_collapse_simultaneous_counts_chord_once():
    times = [0.0, 0.003, 0.009, 0.5, 0.5, 1.0]
    assert list(collapse_simultaneous(times)) == [0.0, 0.5, 1.0]
