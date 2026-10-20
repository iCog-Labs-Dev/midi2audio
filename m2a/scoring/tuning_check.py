# m2a/scoring/tuning_check.py
"""f0-track audio and measure cents deviation from the intended EDO pitch (Step 1.9).

[v1.1] Deviation is measured against the **intended** f0 contour (built from
the expressive MIDI and tuning.json) whenever one is supplied. The
nearest-EDO-step fallback is weak for large EDOs — in 19-EDO arbitrary
12-EDO audio already lands within 31.6 cents of *some* step — so a report
produced without an intended contour is marked `gate_eligible=False` and
must not be used as a hard rejection gate.

f0 comes from YIN with parabolic interpolation rather than pYIN: pYIN
reports f0 on a quantised 10-cent bin grid, which cannot verify a 15-cent
floor with any margin, and is ~10x slower. Voicing is an RMS gate plus a
local-stability check (frames that disagree with their neighbours' median
by more than `STABILITY_CENTS` are transients and are dropped). Octave
errors are folded out (deviation is taken modulo 1200 cents): this check is
about tuning, and octave slips on bright timbres would otherwise dominate
the percentile.
"""
from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

from m2a.midi_io import track_notes
from m2a.render.tuning import TuningSpec, note_pitch, pitch_to_hz

DEFAULT_FLOOR_P95_CENTS = 15.0   # D004
F0_SR = 22_050
F0_FRAME = 2048
F0_HOP = 512
F0_MIN_MIDI, F0_MAX_MIDI = 28, 96
VOICING_GATE_DB = -35.0          # frame RMS relative to the loudest frame
STABILITY_CENTS = 30.0
REFINE_NFFT = 8192
REFINE_HARMONICS = 4
REFINE_MAX_CENTS = 30.0
TRANSITION_GUARD_SEC = 0.04      # ignore frames this close to a note boundary


@dataclass(frozen=True)
class TuningReport:
    mean_cents: float
    p95_cents: float
    voiced_frames: int
    passes: bool
    gate_eligible: bool = True


def frame_times(n_frames: int, sr: int = F0_SR, hop: int = F0_HOP) -> np.ndarray:
    return librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=hop)


def intended_f0_contour(midi_path: str, track_idx: int, tuning: TuningSpec,
                        times: np.ndarray) -> np.ndarray:
    """Intended f0 (Hz) at each time in `times`; NaN where no note sounds or
    within `TRANSITION_GUARD_SEC` of a note boundary. For overlapping notes
    the most recently started one wins (monophonic-lead semantics)."""
    f0 = np.full(len(times), np.nan)
    for n in track_notes(midi_path, track_idx):
        hz = pitch_to_hz(note_pitch(n.pitch, n.bend, tuning))
        lo, hi = n.start_sec + TRANSITION_GUARD_SEC, n.end_sec - TRANSITION_GUARD_SEC
        if hi > lo:
            f0[(times >= lo) & (times < hi)] = hz
    return f0


def track_f0(audio_path: str) -> np.ndarray:
    """f0 in Hz per frame (NaN where unvoiced or unstable), at F0_SR / F0_HOP framing."""
    y, sr = librosa.load(audio_path, sr=F0_SR, mono=True)
    if y.size < F0_FRAME or not np.any(y):
        return np.full(max(1, 1 + len(y) // F0_HOP), np.nan)
    f0 = librosa.yin(y, fmin=librosa.midi_to_hz(F0_MIN_MIDI), fmax=librosa.midi_to_hz(F0_MAX_MIDI),
                     sr=sr, frame_length=F0_FRAME, hop_length=F0_HOP)
    rms = librosa.feature.rms(y=y, frame_length=F0_FRAME, hop_length=F0_HOP)[0][:len(f0)]
    loud = 20.0 * np.log10(np.maximum(rms, 1e-12) / max(float(rms.max()), 1e-12)) > VOICING_GATE_DB
    cents = 1200.0 * np.log2(f0)
    padded = np.pad(cents, 2, mode="edge")
    local_median = np.median(np.lib.stride_tricks.sliding_window_view(padded, 5), axis=1)
    stable = np.abs(cents - local_median) < STABILITY_CENTS
    f0 = np.where(loud & stable, f0, np.nan)
    return _refine_spectral(y, sr, f0)


def _refine_spectral(y: np.ndarray, sr: int, f0: np.ndarray) -> np.ndarray:
    """Refine each voiced YIN estimate by quadratic interpolation of the
    strongest of the first `REFINE_HARMONICS` spectral peaks near k*f0, in a
    zero-padded spectrum. YIN's lag interpolation alone wobbles by a few
    cents; the interpolated harmonic peak is sub-cent on steady tones.
    Estimates that would move by more than `REFINE_MAX_CENTS` are kept as-is."""
    spec = np.abs(librosa.stft(y, n_fft=REFINE_NFFT, hop_length=F0_HOP, win_length=F0_FRAME))
    logmag = np.log(spec + 1e-12)
    bin_hz = sr / REFINE_NFFT
    out = f0.copy()
    for i in np.flatnonzero(np.isfinite(f0)):
        if i >= spec.shape[1]:
            break
        best = None
        for k in range(1, REFINE_HARMONICS + 1):
            target = k * f0[i] / bin_hz
            lo = int(np.floor(target * 2 ** (-REFINE_MAX_CENTS / 1200)))
            hi = int(np.ceil(target * 2 ** (REFINE_MAX_CENTS / 1200))) + 1
            if lo < 1 or hi >= spec.shape[0] - 1:
                continue
            b = lo + int(np.argmax(spec[lo:hi, i]))
            if best is None or spec[b, i] > best[0]:
                best = (spec[b, i], b, k)
        if best is None:
            continue
        _, b, k = best
        a, m, c = logmag[b - 1, i], logmag[b, i], logmag[b + 1, i]
        denom = a - 2 * m + c
        offset = 0.5 * (a - c) / denom if denom < 0 else 0.0
        refined = (b + offset) * bin_hz / k
        if abs(1200.0 * np.log2(refined / f0[i])) <= REFINE_MAX_CENTS:
            out[i] = refined
    return out


def _fold(cents: np.ndarray) -> np.ndarray:
    return np.abs((cents + 600.0) % 1200.0 - 600.0)


def tuning_check(audio_path: str, edo: int, base_tuning_midi: float,
                 intended_f0: np.ndarray | None = None,
                 floor_p95_cents: float = DEFAULT_FLOOR_P95_CENTS,
                 f0: np.ndarray | None = None) -> TuningReport:
    f0 = track_f0(audio_path) if f0 is None else f0
    if intended_f0 is not None:
        n = min(len(f0), len(intended_f0))
        f0, ref = f0[:n], np.asarray(intended_f0[:n], dtype=float)
        mask = np.isfinite(f0) & np.isfinite(ref) & (f0 > 0) & (ref > 0)
        deviations = _fold(1200.0 * np.log2(f0[mask] / ref[mask]))
    else:
        mask = np.isfinite(f0) & (f0 > 0)
        base_hz = librosa.midi_to_hz(base_tuning_midi)
        cents = 1200.0 * np.log2(f0[mask] / base_hz)
        step = 1200.0 / edo
        deviations = np.abs(cents - np.round(cents / step) * step)

    gate_eligible = intended_f0 is not None
    if deviations.size == 0:
        return TuningReport(0.0, 0.0, 0, passes=True, gate_eligible=gate_eligible)
    p95 = float(np.percentile(deviations, 95))
    return TuningReport(
        mean_cents=float(np.mean(deviations)),
        p95_cents=p95,
        voiced_frames=int(deviations.size),
        passes=p95 < floor_p95_cents,
        gate_eligible=gate_eligible,
    )
