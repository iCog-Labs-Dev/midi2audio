# m2a/render/registry.py
"""Stage 2: renderer registry and the per-package stem rendering entry point.

Picks a renderer for each track by (role, microtonal) so that microtonal
routing (added in M1.5) can slot in without changing call sites here.
"""
from __future__ import annotations

from pathlib import Path

import pretty_midi

from m2a.analysis.from_midi import load_structure
from m2a.contracts import RenderPackage, Stem
from m2a.render.fluidsynth_r import FluidSynthRenderer

DEFAULT_SOUNDFONT = str(Path(pretty_midi.__file__).parent / "TimGM6mb.sf2")


class RendererRegistry:
    def __init__(self, renderers: list):
        self.renderers = renderers

    def get(self, role: str, microtonal: bool):
        for renderer in self.renderers:
            if renderer.supports(role, microtonal):
                return renderer
        raise LookupError(f"no renderer supports role={role!r} microtonal={microtonal!r}")


def default_registry() -> RendererRegistry:
    return RendererRegistry([FluidSynthRenderer()])


def stem_filename(track) -> str:
    return f"{track.name or f'track{track.idx:02d}_{track.role}'}.wav"


def render_stems(
    midi_path: str,
    structure,
    work_dir: str,
    soundfont: str = DEFAULT_SOUNDFONT,
    registry: RendererRegistry | None = None,
) -> list[Stem]:
    """Render every track in `structure` from `midi_path` to a WAV stem in `work_dir`."""
    registry = registry or default_registry()
    out_dir = Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems: list[Stem] = []
    for track in structure.tracks:
        renderer = registry.get(track.role, track.microtonal)
        track_label = track.name or f"track{track.idx:02d}_{track.role}"
        out_path = str(out_dir / stem_filename(track))
        stem = renderer.render(
            midi_path, track.idx, track_label, track.role, soundfont, out_path,
        )
        stems.append(stem)
    return stems


def render_all_stems(
    package: RenderPackage,
    work_dir: str,
    soundfont: str = DEFAULT_SOUNDFONT,
    registry: RendererRegistry | None = None,
) -> list[Stem]:
    """Render every track in `package`'s structure.json to a WAV stem."""
    structure = load_structure(package.structure_path)
    return render_stems(package.midi_path, structure, work_dir, soundfont, registry)
