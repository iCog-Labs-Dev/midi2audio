# m2a/restyle/policy.py
"""Divergence schedule: role × section energy × EDO (§8.3).

Decides where the search starts. Percussive and textural stems tolerate
more divergence than pitched harmonic ones; high-energy sections mask
artefacts while sparse ones expose them. The published ranges are the search
interval, not settings — the scorer decides.

P6 — a microtonal stem gets an empty grid: it is routed away from diffusion
entirely and keeps its deterministic render. `tension` is accepted but not
required (D016: it is null for inferred structure).
"""
from __future__ import annotations

from m2a.contracts import StructureJSON

BASE_GRID = {
    "drums":   [0.70, 0.80, 0.90],  # percussive: tolerates high divergence
    "bass":    [0.60, 0.70, 0.80],
    "comping": [0.55, 0.65, 0.75],  # pitched + harmonic: least tolerant
    "lead":    [0.55, 0.65, 0.75],
    "pads":    [0.70, 0.80, 0.90],
}
MIN_DIVERGENCE, MAX_DIVERGENCE = 0.5, 0.95
ENERGY_SWING = 0.05
REFINE_STEP = 0.05


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def divergence_grid(role: str, microtonal: bool, section_energy: float,
                    tension: float | None = None) -> list[float]:
    if microtonal:
        return []
    base = BASE_GRID[role]
    # energy 0.5 is neutral; 1.0 adds ENERGY_SWING, 0.0 subtracts it
    bump = ENERGY_SWING * (section_energy - 0.5) * 2.0
    return [round(clamp(d + bump, MIN_DIVERGENCE, MAX_DIVERGENCE), 4) for d in base]


def mean_section_energy(structure: StructureJSON) -> float:
    """Bar-weighted mean energy across sections (0.5 when there are none)."""
    total = weight = 0.0
    for s in structure.sections:
        bars = max(1, s.bars[1] - s.bars[0])
        total += s.energy * bars
        weight += bars
    return total / weight if weight else 0.5


def refine_grid(centres: list[float], tried: set[float], step: float = REFINE_STEP / 2) -> list[float]:
    """Finer divergences around the best-scoring ones (successive halving)."""
    out = []
    for c in centres:
        for d in (c - step, c + step):
            d = round(clamp(d, MIN_DIVERGENCE, MAX_DIVERGENCE), 4)
            if d not in tried and d not in out:
                out.append(d)
    return out
