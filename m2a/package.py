# m2a/package.py
"""RenderPackage construction and the boundary checks of the input contract.

`build_render_package` assembles a `run_<hash>/` directory (score.mid,
structure.json, tuning.json, manifest.json) from loose inputs; per D005 a
caller-supplied structure.json is always accepted and inference runs only
when none is given. `validate_package` enforces the five contract invariants
and aborts with `ContractViolation` (P3: fail loudly at boundaries).
"""
from __future__ import annotations

import io
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mido

from m2a import __version__
from m2a.analysis.from_midi import analyse_midi, load_structure
from m2a.artifacts import content_hash, file_sha256
from m2a.contracts import RenderPackage, StructureJSON
from m2a.render.tuning import SAFE_MICROTONAL_METHODS, TuningSpec, load_tuning

DEFAULT_TUNING = {"edo": 12, "base_tuning": 60.0, "method": "direct_12"}


class ContractViolation(ValueError):
    """A RenderPackage broke one of the five input-contract invariants."""


def _onsets_by_track(mid: "mido.MidiFile") -> list[list[int]]:
    out = []
    for track in mid.tracks:
        clock, onsets = 0, []
        for msg in track:
            clock += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                onsets.append(clock)
        out.append(onsets)
    return out


def validate_package(structure: StructureJSON, tuning: TuningSpec, midi_path: str) -> None:
    mid = mido.MidiFile(midi_path)
    onsets = _onsets_by_track(mid)

    # 1. every track with notes is identified by its index exactly once
    #    (uniqueness itself is enforced by StructureJSON's validator).
    idxs = {t.idx for t in structure.tracks}
    out_of_range = sorted(i for i in idxs if i >= len(mid.tracks))
    if out_of_range:
        raise ContractViolation(f"structure.tracks idx {out_of_range} not in score.mid "
                                f"({len(mid.tracks)} tracks)")
    unlisted = [i for i, o in enumerate(onsets) if o and i not in idxs]
    if unlisted:
        raise ContractViolation(f"score.mid tracks {unlisted} have notes but no structure.tracks entry")

    # 2. bar_table monotonic: enforced by StructureJSON's validator.

    # 3. tuning.edo == structure.edo
    if tuning.edo != structure.edo:
        raise ContractViolation(f"tuning.json edo={tuning.edo} != structure.json edo={structure.edo}")

    # 4. score.mid round-trips with onsets within 1 tick
    buf = io.BytesIO()
    mid.save(file=buf)
    buf.seek(0)
    reparsed = _onsets_by_track(mido.MidiFile(file=buf))
    for i, (a, b) in enumerate(zip(onsets, reparsed)):
        if len(a) != len(b) or any(abs(x - y) > 1 for x, y in zip(a, b)):
            raise ContractViolation(f"score.mid track {i} does not round-trip")

    # 5. microtonal tracks need a tuning method that can carry them
    micro = [t.label for t in structure.tracks if t.microtonal]
    if micro and tuning.method not in SAFE_MICROTONAL_METHODS:
        raise ContractViolation(f"tracks {micro} are microtonal but tuning.json method is "
                                f"{tuning.method!r}; expected one of {SAFE_MICROTONAL_METHODS}")


def load_package(package: RenderPackage) -> tuple[StructureJSON, TuningSpec]:
    """Load and validate a package's structure + tuning."""
    structure = load_structure(package.structure_path)
    tuning = load_tuning(package.tuning_path)
    validate_package(structure, tuning, package.midi_path)
    return structure, tuning


def build_render_package(midi_path: str, out_root: str, structure_path: str | None = None,
                         tuning_path: str | None = None, producer: str = "midi2audio.cli") -> RenderPackage:
    """Assemble and validate a content-hashed `run_<hash>/` package under `out_root`."""
    staging = Path(out_root) / "_staging"
    staging.mkdir(parents=True, exist_ok=True)
    midi = staging / "score.mid"
    shutil.copyfile(midi_path, midi)

    tuning_file = staging / "tuning.json"
    if tuning_path:
        shutil.copyfile(tuning_path, tuning_file)
    else:
        tuning_file.write_text(json.dumps(DEFAULT_TUNING, indent=2))
    tuning = load_tuning(str(tuning_file))

    structure_file = staging / "structure.json"
    if structure_path:
        shutil.copyfile(structure_path, structure_file)
        provenance = load_structure(str(structure_file)).provenance
    else:
        analyse_midi(str(midi), str(structure_file), edo=tuning.edo, base_tuning=tuning.base_tuning)
        provenance = "inferred"

    h = content_hash(str(midi), str(structure_file), str(tuning_file), config={},
                     code_version=__version__)
    root = Path(out_root) / f"run_{h[:16]}"
    if root.exists():
        shutil.rmtree(staging)
    else:
        staging.rename(root)
    manifest_file = root / "manifest.json"
    if not manifest_file.exists():
        manifest_file.write_text(json.dumps({
            "schema": "midi2audio.manifest/1",
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "pipeline_version": __version__,
            "provenance": provenance,
            "source": producer,
            "midi_sha256": file_sha256(str(root / "score.mid")),
        }, indent=2))

    package = RenderPackage(
        root=str(root), midi_path=str(root / "score.mid"),
        structure_path=str(root / "structure.json"), tuning_path=str(root / "tuning.json"),
        manifest_path=str(manifest_file), content_hash=h,
    )
    load_package(package)
    return package
