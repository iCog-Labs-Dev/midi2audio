# m2a/render/registry.py
"""Stage 2: renderer registry and the per-package stem rendering entry point.

Routing is a table keyed by (role, microtonal) → an ordered preference list;
the first renderer that is available and supports the pair wins (P6: a
microtonal track is *routed*, never retried through a 12-EDO path).

    (role, 12-EDO)       sfizz (if an .sfz is configured) → FluidSynth
    (drums, microtonal)  FluidSynth — unpitched, tuning is irrelevant
    (bass|comping|pads, microtonal)  FluidSynth + MPE pitch-bend rotation
    (lead, microtonal)   MidiDdspRenderer(mode="f0_contour")
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import pretty_midi

from m2a.analysis.from_midi import load_structure
from m2a.contracts import RenderPackage, Stem
from m2a.render.ddsp_r import MidiDdspRenderer
from m2a.render.fluidsynth_r import FluidSynthMpeRenderer, FluidSynthRenderer
from m2a.render.sfizz_r import SfizzRenderer
from m2a.render.tuning import TuningSpec, load_tuning

DEFAULT_SOUNDFONT = str(Path(pretty_midi.__file__).parent / "TimGM6mb.sf2")
ROLES = ("drums", "bass", "comping", "lead", "pads")


class RendererRegistry:
    def __init__(self, config=None):
        self.config = config
        fluid = FluidSynthRenderer()
        mpe = FluidSynthMpeRenderer()
        sfizz = SfizzRenderer(getattr(config, "sfz_instruments", None))
        self._routes: dict[tuple[str, bool], list] = {}
        for role in ROLES:
            self._routes[(role, False)] = [sfizz, fluid]
            self._routes[(role, True)] = [mpe]
        self._routes[("drums", True)] = [fluid]
        self._routes[("lead", True)] = [MidiDdspRenderer(config, mode="f0_contour")]

    def candidates(self, role: str, microtonal: bool) -> list:
        key = (role, microtonal)
        if key not in self._routes:
            raise ValueError(f"No renderer for ({role}, microtonal={microtonal})")
        return self._routes[key]

    def get(self, role: str, microtonal: bool):
        """The routed renderer for (role, microtonal), regardless of availability."""
        return next((r for r in self.candidates(role, microtonal) if r.supports(role, microtonal)),
                    self.candidates(role, microtonal)[-1])

    def resolve(self, role: str, microtonal: bool):
        """The first renderer that supports the pair *and* is usable on this machine."""
        for r in self.candidates(role, microtonal):
            if r.supports(role, microtonal) and r.available():
                return r
        raise LookupError(f"no available renderer for role={role!r} microtonal={microtonal!r}")


def default_registry(config=None) -> RendererRegistry:
    return RendererRegistry(config)


def stem_filename(track) -> str:
    return f"{track.label}.wav"


def render_stems(
    midi_path: str,
    structure,
    work_dir: str,
    soundfont: str = DEFAULT_SOUNDFONT,
    registry: RendererRegistry | None = None,
    tuning: TuningSpec | None = None,
) -> list[Stem]:
    """Render every track in `structure` from `midi_path` to a WAV stem in `work_dir`."""
    registry = registry or default_registry()
    tuning = tuning or TuningSpec(edo=structure.edo, base_tuning=structure.base_tuning)
    out_dir = Path(work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stems: list[Stem] = []
    for track in structure.tracks:
        renderer = registry.resolve(track.role, track.microtonal)
        out_path = str(out_dir / stem_filename(track))
        stem = renderer.render(
            midi_path, track.idx, track.label, track.role, soundfont, out_path, tuning=tuning,
        )
        stems.append(dataclasses.replace(stem, microtonal=track.microtonal))
    return stems


def render_all_stems(
    package: RenderPackage,
    work_dir: str,
    soundfont: str = DEFAULT_SOUNDFONT,
    registry: RendererRegistry | None = None,
) -> list[Stem]:
    """Render every track in `package`'s structure.json to a WAV stem."""
    structure = load_structure(package.structure_path)
    tuning = load_tuning(package.tuning_path) if Path(package.tuning_path).exists() else None
    return render_stems(package.midi_path, structure, work_dir, soundfont, registry, tuning)
