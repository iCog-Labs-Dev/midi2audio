# m2a/render/sfizz_r.py
"""sfizz renderer for 12-EDO pitched roles (opt-in).

Used only when the `sfizz_render` CLI is on PATH **and** the config maps the
role to an .sfz instrument (`AudioConfig.sfz_instruments`); otherwise the
registry falls through to FluidSynth. No SFZ library ships with the repo.

The .scl path (`SfizzSclRenderer` in the plan) is deferred: microtonal
pitched tracks render tuning-exact through MPE pitch bends instead (see
DECISIONS.md D017).
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import mido

from m2a.contracts import Stem
from m2a.render.fluidsynth_r import (
    SAMPLE_RATE, TARGET_PEAK_DBFS, _extract_single_track_midi, _normalise_peak, _stem,
)
from m2a.render.tuning import TuningSpec


class SfizzRenderer:
    name = "sfizz"

    def __init__(self, instruments: dict[str, str] | None = None):
        self.instruments = dict(instruments or {})

    def supports(self, role: str, microtonal: bool) -> bool:
        return not microtonal and role in self.instruments

    def available(self) -> bool:
        return shutil.which("sfizz_render") is not None and bool(self.instruments)

    def render(self, midi_path: str, track_idx: int, track_name: str, role: str,
               soundfont: str, out_path: str, tuning: TuningSpec | None = None) -> Stem:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        tmp_midi = str(Path(out_path).with_suffix(".track.mid"))
        _extract_single_track_midi(mido.MidiFile(midi_path), track_idx).save(tmp_midi)
        subprocess.run([
            "sfizz_render", "--sfz", self.instruments[role], "--midi", tmp_midi,
            "--wav", out_path, "--samplerate", str(SAMPLE_RATE),
        ], check=True, capture_output=True)
        _normalise_peak(out_path, target_dbfs=TARGET_PEAK_DBFS)
        return _stem(out_path, track_name, role, microtonal=False)
