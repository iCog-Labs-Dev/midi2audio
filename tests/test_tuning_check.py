"""tests/test_tuning_check.py — f0 deviation from the EDO lattice (Step 1.9)."""
from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from m2a.scoring.tuning_check import frame_times, tuning_check

SR = 22_050
NOTE_SEC = 0.5
MAJOR_12 = [0, 2, 4, 5, 7, 9, 11, 12]       # semitones
MAJOR_19 = [0, 3, 6, 8, 11, 14, 17, 19]     # nearest 19-EDO steps


def _sine_sequence(path, cents_list, base_midi=60.0):
    chunks = []
    for c in cents_list:
        f = 440.0 * 2 ** ((base_midi + c / 100.0 - 69.0) / 12.0)
        # 0.4 s tone + 0.1 s silence, with 10 ms fades: frames straddling two
        # pitches would otherwise dominate the nearest-step percentile.
        t = np.arange(int((NOTE_SEC - 0.1) * SR)) / SR
        fade = np.minimum(1.0, np.minimum(t, t[-1] - t) / 0.01)
        chunks.append(0.3 * np.sin(2 * np.pi * f * t) * fade)
        chunks.append(np.zeros(int(0.1 * SR)))
    sf.write(path, np.concatenate(chunks), SR)
    return str(path)


def _contour(cents_list, n_frames, base_midi=60.0):
    times = frame_times(n_frames)
    f0 = np.full(n_frames, np.nan)
    for i, c in enumerate(cents_list):
        lo, hi = i * NOTE_SEC + 0.05, (i + 1) * NOTE_SEC - 0.05
        f0[(times >= lo) & (times < hi)] = 440.0 * 2 ** ((base_midi + c / 100.0 - 69.0) / 12.0)
    return f0


@pytest.fixture(scope="module")
def tones_12(tmp_path_factory):
    return _sine_sequence(tmp_path_factory.mktemp("t") / "12.wav", [s * 100.0 for s in MAJOR_12])


def test_exact_edo_tones_pass(tmp_path):
    cents = [s * 1200.0 / 19 for s in MAJOR_19]
    wav = _sine_sequence(tmp_path / "19.wav", cents)
    report = tuning_check(wav, edo=19, base_tuning_midi=60.0)
    assert report.p95_cents < 2.0
    assert report.passes


def test_12_edo_tones_fail_19_edo_check(tones_12):
    n_frames = int(len(sf.read(tones_12)[0]) / 512) + 1
    intended = _contour([s * 1200.0 / 19 for s in MAJOR_19], n_frames)
    report = tuning_check(tones_12, edo=19, base_tuning_midi=60.0, intended_f0=intended)
    assert report.p95_cents > 15.0
    assert not report.passes
    assert report.gate_eligible


def test_nearest_step_fallback_is_documented_as_weak(tones_12):
    report = tuning_check(tones_12, edo=19, base_tuning_midi=60.0, intended_f0=None)
    assert report.gate_eligible is False


def test_monotonicity(tmp_path):
    scores = []
    for detune in [0, 5, 10, 20]:
        wav = _sine_sequence(tmp_path / f"d{detune}.wav", [s * 100.0 + detune for s in MAJOR_12])
        scores.append(tuning_check(wav, edo=12, base_tuning_midi=60.0).p95_cents)
    assert scores == sorted(scores)
    assert scores[-1] > scores[0] + 10.0
