# m2a/restyle/chunking.py
"""Boundary-aware chunk / restyle / rejoin (Step 3.2).

Stems longer than an endpoint's input cap are split at **downbeats**,
preferring section boundaries with the highest `boundary_lvl` (a seam inside
a phrase is where restyled chunks audibly disagree). Neighbouring chunks
share an `overlap_bars`-bar region centred on the seam — bars, not seconds,
so the fade stays centred on a downbeat in any meter.

D002 — chunks restyled with the same prompt and seed over a shared overlap
are strongly correlated, so they are joined with an **equal-gain (linear)**
crossfade; equal-power would put a +3 dB bump on every seam. Before the fade
the incoming chunk is aligned to the outgoing one by cross-correlation over
the overlap (the counter to tempo drift across seams).
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy import signal

from m2a.contracts import Candidate, SectionInfo, Stem, StructureJSON
from m2a.restyle.base import RestyleEndpoint

OVERLAP_BARS = 2
MAX_ALIGN_MS = 50.0
CAP_MARGIN_SEC = 1.0


def _bar_time(i: float, bar_table: list[list[float]]) -> float:
    """Time of (possibly fractional or past-the-end) bar index `i`."""
    times = [t for _, t in bar_table]
    if len(times) == 1:
        times = times + [times[0] + 2.0]
    last_dur = times[-1] - times[-2]
    lo = int(np.floor(i))
    frac = i - lo
    def at(k: int) -> float:
        if k < 0:
            return times[0] + k * (times[1] - times[0])
        if k < len(times):
            return times[k]
        return times[-1] + (k - len(times) + 1) * last_dur
    return at(lo) + frac * (at(lo + 1) - at(lo))


def select_seam_bars(sections: list[SectionInfo], bar_table: list[list[float]], total_sec: float,
                     max_chunk_sec: float, overlap_bars: float = OVERLAP_BARS) -> list[int]:
    """Bar indices to cut at, so that every chunk (with its overlap) fits
    `max_chunk_sec`. Splits the longest offending chunk at the candidate bar
    with the highest boundary level, ties broken toward the chunk's middle."""
    half = overlap_bars / 2.0
    n_bars = len(bar_table)
    level = {b: -1 for b in range(1, n_bars)}
    for s in sections:
        if 0 < s.bars[0] < n_bars:
            level[s.bars[0]] = max(level[s.bars[0]], s.boundary_lvl)

    def span(a: int | None, b: int | None) -> float:
        start = 0.0 if a is None else max(0.0, _bar_time(a - half, bar_table))
        end = total_sec if b is None else min(total_sec, _bar_time(b + half, bar_table))
        return end - start

    seams: list[int] = []
    while True:
        bounds = [None] + seams + [None]
        chunks = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
        too_long = [(span(a, b), a, b) for a, b in chunks if span(a, b) > max_chunk_sec]
        if not too_long:
            return seams
        _, a, b = max(too_long, key=lambda c: c[0])
        lo = (a if a is not None else 0) + 1
        hi = (b if b is not None else n_bars) - 1
        options = [k for k in range(lo, hi + 1) if k in level]
        if not options:
            raise ValueError(f"cannot split bars {a}..{b} to fit {max_chunk_sec:.1f}s chunks")
        mid = (lo + hi) / 2.0
        seams = sorted(seams + [max(options, key=lambda k: (level[k], -abs(k - mid)))])


def select_seams(sections: list[SectionInfo], bar_table: list[list[float]], total_sec: float,
                 max_chunk_sec: float, overlap_bars: float = OVERLAP_BARS) -> list[float]:
    """Seam times in seconds (each one a downbeat)."""
    return [_bar_time(k, bar_table)
            for k in select_seam_bars(sections, bar_table, total_sec, max_chunk_sec, overlap_bars)]


def chunk_spans(seam_bars: list[int], bar_table: list[list[float]], n_samples: int, sr: int,
                overlap_bars: float = OVERLAP_BARS) -> list[tuple[int, int]]:
    half = overlap_bars / 2.0
    starts = [0] + [max(0, int(round(_bar_time(k - half, bar_table) * sr))) for k in seam_bars]
    ends = [min(n_samples, int(round(_bar_time(k + half, bar_table) * sr))) for k in seam_bars] + [n_samples]
    return list(zip(starts, ends))


def chunk_audio(audio_path: str, seam_bars: list[int], bar_table: list[list[float]], out_dir: str,
                overlap_bars: float = OVERLAP_BARS) -> list[tuple[str, int, int]]:
    audio, sr = sf.read(audio_path, always_2d=True)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out = []
    for k, (a, b) in enumerate(chunk_spans(seam_bars, bar_table, len(audio), sr, overlap_bars)):
        path = str(Path(out_dir) / f"chunk_{k:03d}.wav")
        sf.write(path, audio[a:b], sr, subtype="FLOAT")
        out.append((path, a, b))
    return out


def _fit(x: np.ndarray, n: int, channels: int) -> np.ndarray:
    if x.shape[1] != channels:
        x = np.repeat(x.mean(axis=1, keepdims=True), channels, axis=1)
    if len(x) >= n:
        return x[:n]
    return np.concatenate([x, np.zeros((n - len(x), channels))])


def _alignment_lag(reference: np.ndarray, incoming: np.ndarray, max_lag: int) -> int:
    """Samples by which `incoming` is late relative to `reference` (mono sums)."""
    a, b = reference.mean(axis=1), incoming.mean(axis=1)
    if np.dot(a, a) < 1e-12 or np.dot(b, b) < 1e-12:
        return 0
    corr = signal.correlate(b, a, mode="full", method="fft")
    lags = signal.correlation_lags(len(b), len(a), mode="full")
    window = np.abs(lags) <= max_lag
    corr, lags = corr[window], lags[window]
    best = np.flatnonzero(corr >= corr.max() - 1e-9 * abs(corr.max()))
    return int(lags[best[np.argmin(np.abs(lags[best]))]])


def crossfade_rejoin(chunks: list[tuple[str, int, int]], n_samples: int, out_path: str,
                     align: bool = True, max_align_ms: float = MAX_ALIGN_MS) -> str:
    """Rejoin chunk files at their spans with equal-gain fades over each overlap."""
    first, sr = sf.read(chunks[0][0], always_2d=True)
    channels = first.shape[1]
    out = np.zeros((n_samples, channels))
    prev_end = 0
    max_lag = int(max_align_ms / 1000.0 * sr)
    for k, (path, a, b) in enumerate(chunks):
        cur, _ = sf.read(path, always_2d=True)
        cur = _fit(cur, b - a, channels)
        ov = max(0, prev_end - a)
        if k > 0 and ov > 0:
            if align:
                lag = _alignment_lag(out[a:prev_end], cur[:ov], max_lag)
                if lag > 0:
                    cur = np.concatenate([cur[lag:], np.zeros((lag, channels))])
                elif lag < 0:
                    cur = np.concatenate([np.zeros((-lag, channels)), cur[:lag]])
            w = np.linspace(0.0, 1.0, ov, endpoint=False)[:, None] if ov > 1 else np.ones((ov, 1))
            out[a:prev_end] = out[a:prev_end] * (1.0 - w) + cur[:ov] * w
            out[prev_end:b] = cur[ov:]
        else:
            out[a:b] = cur
        prev_end = b
    sf.write(out_path, out, sr, subtype="FLOAT")
    return out_path


def crossfade_rejoin_passthrough(audio_path: str, seam_bars: list[int], bar_table: list[list[float]],
                                 work_dir: str) -> str:
    """Chunk and rejoin without restyling — must be lossless (the D002 law test)."""
    chunks = chunk_audio(audio_path, seam_bars, bar_table, work_dir)
    n = sf.info(audio_path).frames
    return crossfade_rejoin(chunks, n, str(Path(work_dir) / "rejoined.wav"))


def chunk_and_restyle(stem: Stem, structure: StructureJSON, endpoint: RestyleEndpoint,
                      prompt: str, negative_prompt: str, strength: float, seed: int,
                      out_path: str, work_dir: str) -> tuple[Candidate, int]:
    """Restyle a whole stem, chunking when it exceeds the endpoint's cap.

    Returns (candidate, n_endpoint_calls). If any chunk fails, the whole
    candidate is rejected with that chunk's reason (a stem with a hole in it
    is not a candidate).
    """
    info = sf.info(stem.path)
    extra = {"stem": stem.track}
    cap = endpoint.max_input_seconds - CAP_MARGIN_SEC
    if info.duration <= endpoint.max_input_seconds:
        c = endpoint.transform(stem.path, prompt, negative_prompt, strength, seed,
                               {**extra, "out_path": out_path})
        return c, 1

    seam_bars = select_seam_bars(structure.sections, structure.bar_table, info.duration, cap)
    chunks = chunk_audio(stem.path, seam_bars, structure.bar_table, str(Path(work_dir) / "in"))
    restyled: list[tuple[str, int, int]] = []
    last: Candidate | None = None
    for k, (path, a, b) in enumerate(chunks):
        if (b - a) / info.samplerate > endpoint.max_input_seconds:
            raise AssertionError(f"chunk {k} is {(b - a) / info.samplerate:.1f}s > endpoint cap")
        chunk_out = str(Path(work_dir) / f"out_{k:03d}.wav")
        last = endpoint.transform(path, prompt, negative_prompt, strength, seed,
                                  {**extra, "out_path": chunk_out})
        if not last.accepted:
            return dataclasses.replace(last, stem=stem.track), k + 1
        restyled.append((chunk_out, a, b))
    crossfade_rejoin(restyled, info.frames, out_path)
    assert last is not None
    return dataclasses.replace(last, stem=stem.track, audio_path=out_path,
                               usd_estimate=last.usd_estimate * len(chunks)), len(chunks)


def n_chunks_for(stem: Stem, structure: StructureJSON, endpoint: RestyleEndpoint) -> int:
    """How many paid calls restyling this stem will take (for the budget guard)."""
    info = sf.info(stem.path)
    if info.duration <= endpoint.max_input_seconds:
        return 1
    cap = endpoint.max_input_seconds - CAP_MARGIN_SEC
    return 1 + len(select_seam_bars(structure.sections, structure.bar_table, info.duration, cap))
