# m2a/scoring/notes.py
"""Note-fidelity scorer (Step 2.2): transcription F1 + CQT-chroma cosine.

[v1.1] fixes carried from the plan:

1. Not applicable to percussion — returns None and `combine` renormalises.
2. Microtonal stems are compared by f0 contour against the intended EDO
   pitch, since semitone-binned transcription cannot validate them.
3. Chroma cosine is candidate vs the **dry stem** (like-for-like audio), not
   candidate vs a MIDI-synthesised chroma.

Transcription is pluggable (`Transcriber`). basic-pitch is used when
installed and selected; the built-in default is a CQT harmonic-salience
transcriber with iterative harmonic cancellation (Klapuri-style), which is
deterministic, fast and dependency-free so the M2 gate runs in a light venv
(D010). Transcribed onsets are shift-aligned the same way the rhythm scorer
aligns, so endpoint latency is not mistaken for wrong notes.
"""
from __future__ import annotations

import functools
from typing import Callable, Protocol

import librosa
import mir_eval
import numpy as np

from m2a.midi_io import track_notes
from m2a.render.tuning import TuningSpec, note_pitch, pitch_to_hz
from m2a.scoring.rhythm import (
    ANALYSIS_SR, HOP, audio_envelope_and_onsets, collapse_simultaneous, estimate_shift_sec, file_key,
    load_mono,
)
from m2a.scoring.tuning_check import frame_times, intended_f0_contour, track_f0

F1_WEIGHT = 0.7
CHROMA_WEIGHT = 0.3
ONSET_TOLERANCE_SEC = 0.05

CQT_FMIN_MIDI = 24            # C1
CQT_BINS_PER_SEMITONE = 3
CQT_OCTAVES = 7
HARMONICS = 6
HARMONIC_DECAY = 0.8
SALIENCE_STOP_RATIO = 0.35    # stop when the next pitch's salience < this × the strongest
FUNDAMENTAL_MIN_RATIO = 0.1   # a pitch needs its own fundamental >= this × the frame's peak bin
MAX_POLYPHONY = 6
SEGMENT_SKIP_SEC = 0.02       # skip the attack transient
SEGMENT_ANALYSIS_SEC = 0.15


class Transcriber(Protocol):
    def __call__(self, audio_path: str) -> list[tuple[float, float, float]]:
        """Return [(onset_sec, offset_sec, midi_pitch), ...]."""


def _harmonic_offsets() -> list[tuple[int, float]]:
    return [(int(round(12 * np.log2(h) * CQT_BINS_PER_SEMITONE)), HARMONIC_DECAY ** (h - 1))
            for h in range(1, HARMONICS + 1)]


def _segment_pitches(spectrum: np.ndarray) -> list[int]:
    """Iterative estimate-and-cancel multi-pitch estimation on one CQT frame."""
    spec = spectrum.copy()
    n_bins = len(spec)
    offsets = _harmonic_offsets()
    n_semis = n_bins // CQT_BINS_PER_SEMITONE
    pitches: list[int] = []
    first = None
    fundamental_floor = FUNDAMENTAL_MIN_RATIO * float(spec.max())
    for _ in range(MAX_POLYPHONY):
        salience = np.zeros(n_semis)
        for s in range(n_semis):
            b0 = s * CQT_BINS_PER_SEMITONE + CQT_BINS_PER_SEMITONE // 2
            # Missing-fundamental guard: without it C3 "explains" a C-major
            # triad (C4 G4 C5 E5 G5 are all its harmonics) better than C4 does.
            if spectrum[b0] < fundamental_floor:
                continue
            salience[s] = sum(w * spec[b0 + o] for o, w in offsets if b0 + o < n_bins)
        best = int(np.argmax(salience))
        if first is None:
            first = salience[best]
        if first <= 0 or salience[best] < SALIENCE_STOP_RATIO * first:
            break
        pitches.append(CQT_FMIN_MIDI + best)
        b0 = best * CQT_BINS_PER_SEMITONE + CQT_BINS_PER_SEMITONE // 2
        for o, _ in offsets:
            lo, hi = max(0, b0 + o - 1), min(n_bins, b0 + o + 2)
            spec[lo:hi] = 0.0
    return pitches


CQT_FMIN_HZ = float(librosa.midi_to_hz(CQT_FMIN_MIDI - 1.0 / CQT_BINS_PER_SEMITONE))


@functools.lru_cache(maxsize=8)
def _cqt(key: tuple[str, int, int]) -> np.ndarray:
    """|CQT| at 3 bins/semitone, bin 3s+1 centred on MIDI CQT_FMIN_MIDI+s. Shared by
    transcription and chroma so each file is transformed once."""
    y = load_mono(key[0])
    C = np.abs(librosa.cqt(y, sr=ANALYSIS_SR, hop_length=HOP, fmin=CQT_FMIN_HZ,
                           n_bins=12 * CQT_OCTAVES * CQT_BINS_PER_SEMITONE,
                           bins_per_octave=12 * CQT_BINS_PER_SEMITONE))
    C.flags.writeable = False
    return C


@functools.lru_cache(maxsize=8)
def _chroma(key: tuple[str, int, int]) -> np.ndarray:
    chroma = librosa.feature.chroma_cqt(C=_cqt(key), sr=ANALYSIS_SR, hop_length=HOP,
                                        fmin=CQT_FMIN_HZ, bins_per_octave=12 * CQT_BINS_PER_SEMITONE)
    chroma.flags.writeable = False
    return chroma


def cqt_transcribe(audio_path: str) -> list[tuple[float, float, float]]:
    y = load_mono(audio_path)
    if y.size == 0 or not np.any(y):
        return []
    C = _cqt(file_key(audio_path))
    _, onsets = audio_envelope_and_onsets(audio_path)
    duration = len(y) / ANALYSIS_SR
    frame_sec = HOP / ANALYSIS_SR
    notes = []
    for i, on in enumerate(onsets):
        off = onsets[i + 1] if i + 1 < len(onsets) else duration
        a = int((on + SEGMENT_SKIP_SEC) / frame_sec)
        b = max(a + 1, int(min(off, on + SEGMENT_SKIP_SEC + SEGMENT_ANALYSIS_SEC) / frame_sec))
        if a >= C.shape[1]:
            continue
        spectrum = C[:, a:min(b, C.shape[1])].mean(axis=1)
        for p in _segment_pitches(spectrum):
            notes.append((float(on), float(off), float(p)))
    return notes


def basic_pitch_transcribe(audio_path: str) -> list[tuple[float, float, float]]:
    from basic_pitch.inference import predict   # optional heavy dependency ([audio] extra)
    _, _, note_events = predict(audio_path)
    return [(float(s), float(e), float(p)) for s, e, p, *_ in note_events]


def default_transcriber(name: str = "cqt") -> Callable[[str], list[tuple[float, float, float]]]:
    if name == "basic_pitch":
        return basic_pitch_transcribe
    if name == "cqt":
        return cqt_transcribe
    raise ValueError(f"unknown transcriber {name!r}")


def reference_notes(midi_path: str, track_idx: int,
                    tuning: TuningSpec | None = None) -> list[tuple[float, float, float]]:
    tuning = tuning or TuningSpec()
    return [(n.start_sec, n.end_sec, note_pitch(n.pitch, n.bend, tuning))
            for n in track_notes(midi_path, track_idx)]


def note_f1(est: list[tuple[float, float, float]], ref: list[tuple[float, float, float]],
            onset_tolerance: float = ONSET_TOLERANCE_SEC) -> float:
    if not ref and not est:
        return 1.0
    if not ref or not est:
        return 0.0

    def arrays(notes):
        iv = np.array([[max(s, 0.0), max(e, s, 0.0) + 1e-3] for s, e, _ in notes])
        hz = np.array([pitch_to_hz(p) for _, _, p in notes])
        return iv, hz

    ref_iv, ref_hz = arrays(ref)
    est_iv, est_hz = arrays(est)
    _, _, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ref_iv, ref_hz, est_iv, est_hz, onset_tolerance=onset_tolerance, offset_ratio=None)
    return float(f)


def cqt_chroma_cosine(audio_path: str, dry_stem_path: str) -> float:
    """Mean framewise cosine between CQT chromagrams, over frames where either has energy."""
    a = _chroma(file_key(audio_path))
    b = _chroma(file_key(dry_stem_path))
    n = min(a.shape[1], b.shape[1])
    if n == 0:
        return 0.0
    a, b = a[:, :n], b[:, :n]
    na, nb = np.linalg.norm(a, axis=0), np.linalg.norm(b, axis=0)
    active = (na > 1e-6) | (nb > 1e-6)
    if not np.any(active):
        return 1.0
    cos = np.sum(a * b, axis=0)[active] / np.maximum(na[active] * nb[active], 1e-12)
    return float(np.clip(np.mean(cos), 0.0, 1.0))


def f0_contour_similarity(audio_path: str, intended_f0: np.ndarray,
                          tolerance_cents: float = 50.0, f0: np.ndarray | None = None) -> float:
    """Fraction-like similarity of the audio's f0 to the intended contour:
    mean over intended-voiced frames of max(0, 1 - |dev| / tolerance), with
    unvoiced audio frames counting as 0."""
    f0 = track_f0(audio_path) if f0 is None else f0
    n = min(len(f0), len(intended_f0))
    f0, ref = f0[:n], intended_f0[:n]
    want = np.isfinite(ref) & (ref > 0)
    if not np.any(want):
        return 1.0
    got = np.where(np.isfinite(f0) & (f0 > 0), f0, np.nan)[want]
    dev = np.abs(1200.0 * np.log2(got / ref[want]))
    dev = np.abs((dev + 600.0) % 1200.0 - 600.0)
    sim = np.where(np.isfinite(dev), np.clip(1.0 - dev / tolerance_cents, 0.0, 1.0), 0.0)
    return float(np.mean(sim))


def note_fidelity(audio_path: str, dry_stem_path: str, expressive_midi_path: str,
                  role: str, microtonal: bool, track_idx: int,
                  tuning: TuningSpec | None = None,
                  transcriber: Callable[[str], list] | None = None) -> float | None:
    if role == "drums":
        return None
    if microtonal:
        f0 = track_f0(audio_path)
        intended = intended_f0_contour(expressive_midi_path, track_idx, tuning or TuningSpec(),
                                       frame_times(len(f0)))
        return f0_contour_similarity(audio_path, intended, f0=f0)

    transcriber = transcriber or cqt_transcribe
    ref = reference_notes(expressive_midi_path, track_idx, tuning)
    est = transcriber(audio_path)
    if est:
        env, _ = audio_envelope_and_onsets(audio_path)
        shift = estimate_shift_sec(env, collapse_simultaneous([s for s, _, _ in ref]))
        est = [(s - shift, e - shift, p) for s, e, p in est]
    f1 = note_f1(est, ref)
    chroma_sim = cqt_chroma_cosine(audio_path, dry_stem_path)
    return F1_WEIGHT * f1 + CHROMA_WEIGHT * chroma_sim
