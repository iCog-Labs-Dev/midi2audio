"""tests/conftest.py — shared helpers for the midi2audio test suite."""
from __future__ import annotations

import json
import pathlib

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load and parse a JSON fixture file by filename."""
    return json.loads((FIXTURES / name).read_text())


def fixture_render_package():
    """Build a `RenderPackage` pointing at the eight_bar fixture set."""
    from m2a.artifacts import content_hash
    from m2a.contracts import RenderPackage

    midi_path = str(FIXTURES / "eight_bar.mid")
    structure_path = str(FIXTURES / "eight_bar_structure.json")
    tuning_path = str(FIXTURES / "eight_bar_tuning.json")
    manifest_path = str(FIXTURES / "eight_bar_manifest.json")
    h = content_hash(midi_path, structure_path, tuning_path, config={}, code_version="0.1.0")
    return RenderPackage(
        root=str(FIXTURES), midi_path=midi_path, structure_path=structure_path,
        tuning_path=tuning_path, manifest_path=manifest_path, content_hash=h,
    )
