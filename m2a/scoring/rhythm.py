# m2a/scoring/rhythm.py
"""Onset F-measure between candidate audio and the expressive MIDI (Step 2.1).

Three [v1.1] fixes over a naive implementation:

1. Simultaneous note-ons are collapsed (`collapse_simultaneous`): a four-note
   chord is one audio onset, and counting it as four capped polyphonic
   recall at 25 %.
2. The candidate is aligned before scoring (`estimate_shift_sec`):
   generative endpoints shift audio by tens of ms, and scoring before
   alignment measures endpoint latency rather than musical timing. The shift
   is reported; above `MAX_ACCEPTABLE_SHIFT_SEC` it is itself a rejection
   reason (the endpoint re-barred the material).
3. Matching uses `mir_eval.onset.f_measure` (optimal bipartite matching), not
   a hand-rolled greedy pairing.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass

import librosa
import mir_eval
import numpy as np

from m2a.midi_io import track_notes

ANALYSIS_SR = 22_050
HOP = 256
PAD_FRAMES = 8
DRUM_TOLERANCE_MS = 25.0
DEFAULT_TOLERANCE_MS = 50.0
MAX_SHIFT_SEC = 0.25
MAX_ACCEPTABLE_SHIFT_SEC = 0.20


@dataclass(frozen=True)
class RhythmResult:
    f_measure: float
    precision: float
    recall: float
    shift_sec: float          # positive: candidate audio is late relative to the MIDI


def file_key(path: str) -> tuple[str, int, int]:
    """Cache key for derived analyses: a rewritten file gets a new key."""
    st = os.stat(path)
    return os.path.abspath(path), st.st_mtime_ns, st.st_size


# Scorers analyse the same files repeatedly (every candidate is compared with
# its dry stem; rhythm and notes both need onsets), so loads and envelopes are
# memoised per file version. Returned arrays are read-only.
@functools.lru_cache(maxsize=16)
def _load(key: tuple[str, int, int], sr: int) -> np.ndarray:
    y, _ = librosa.load(key[0], sr=sr, mono=True)
    y.flags.writeable = False
    return y


@functools.lru_cache(maxsize=16)
def _envelope_and_onsets(key: tuple[str, int, int]) -> tuple[np.ndarray, np.ndarray]:
    y = _load(key, ANALYSIS_SR)
    if y.size == 0 or not np.any(y):
        return np.zeros(0), np.zeros(0)
    env = onset_envelope(y)
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=ANALYSIS_SR, hop_length=HOP)
    onsets = librosa.frames_to_time(frames, sr=ANALYSIS_SR, hop_length=HOP)
    env.flags.writeable = False
    onsets.flags.writeable = False
    return env, onsets


def load_mono(audio_path: str, sr: int = ANALYSIS_SR) -> np.ndarray:
    return _load(file_key(audio_path), sr)


def audio_envelope_and_onsets(audio_path: str) -> tuple[np.ndarray, np.ndarray]:
    return _envelope_and_onsets(file_key(audio_path))


def onset_envelope(y: np.ndarray, sr: int = ANALYSIS_SR) -> np.ndarray:
    """Onset-strength envelope framed like `y` itself. The signal is analysed
    with `PAD_FRAMES` of leading silence (then trimmed) so a note at t=0 has
    a preceding frame to rise from and is detectable."""
    padded = np.concatenate([np.zeros(PAD_FRAMES * HOP, dtype=y.dtype), y])
    return librosa.onset.onset_strength(y=padded, sr=sr, hop_length=HOP)[PAD_FRAMES:]


def onset_detect_audio(audio_path: str) -> np.ndarray:
    """Onset times in seconds (envelope peaks). No backtracking: on sustained
    tones the preceding energy minimum lands at arbitrary points inside the
    previous note, whereas the peak lag is a near-constant bias that
    `estimate_shift_sec` absorbs."""
    return audio_envelope_and_onsets(audio_path)[1]


def midi_note_on_times(midi_path: str, track_idx: int) -> np.ndarray:
    return np.array(sorted(n.start_sec for n in track_notes(midi_path, track_idx)))


def collapse_simultaneous(times, eps_ms: float = 15.0) -> np.ndarray:
    """Merge onsets closer than `eps_ms` to the first onset of their cluster."""
    out: list[float] = []
    for t in sorted(times):
        if not out or (t - out[-1]) * 1000.0 > eps_ms:
            out.append(float(t))
    return np.array(out)


def estimate_shift_sec(env: np.ndarray, ref_times: np.ndarray, sr: int = ANALYSIS_SR,
                       max_shift_sec: float = MAX_SHIFT_SEC) -> float:
    """Lag (s) maximising the cross-correlation between an onset-strength
    envelope and an impulse train at the reference onsets, within ±max_shift."""
    if len(ref_times) == 0 or env.size == 0:
        return 0.0
    env = env - env.mean()
    frame_sec = HOP / sr
    impulses = np.zeros(len(env))
    idx = np.round(np.asarray(ref_times) / frame_sec).astype(int)
    impulses[idx[(idx >= 0) & (idx < len(env))]] = 1.0
    max_lag = int(round(max_shift_sec / frame_sec))
    best_lag, best = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            c = float(np.dot(env[lag:], impulses[:len(env) - lag]))
        else:
            c = float(np.dot(env[:lag], impulses[-lag:]))
        # prefer the smallest |lag| on ties so a perfect render reports 0
        if c > best + 1e-12 or (abs(c - best) <= 1e-12 and abs(lag) < abs(best_lag)):
            best, best_lag = c, lag
    return best_lag * frame_sec


def rhythm_report(audio_path: str, expressive_midi_path: str, role: str, track_idx: int,
                  tolerance_ms: float = DEFAULT_TOLERANCE_MS) -> RhythmResult:
    tol = DRUM_TOLERANCE_MS if role == "drums" else tolerance_ms
    env, onsets = audio_envelope_and_onsets(audio_path)
    ref = collapse_simultaneous(midi_note_on_times(expressive_midi_path, track_idx))
    shift = estimate_shift_sec(env, ref)
    cand = onsets - shift
    if len(ref) == 0 and len(cand) == 0:
        return RhythmResult(1.0, 1.0, 1.0, shift)
    if len(ref) == 0 or len(cand) == 0:
        return RhythmResult(0.0, 0.0, 0.0, shift)
    f, p, r = mir_eval.onset.f_measure(ref, np.sort(cand), window=tol / 1000.0)
    return RhythmResult(float(f), float(p), float(r), float(shift))


def rhythm_score(audio_path: str, expressive_midi_path: str, role: str, track_idx: int,
                 tolerance_ms: float = DEFAULT_TOLERANCE_MS) -> float:
    return rhythm_report(audio_path, expressive_midi_path, role, track_idx, tolerance_ms).f_measure
