"""tests/conftest.py — shared helpers for the midi2audio test suite."""
from __future__ import annotations

import json
import pathlib

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def load_fixture(name: str) -> dict:
    """Load and parse a JSON fixture file by filename."""
    return json.loads((FIXTURES / name).read_text())
