"""tests/test_bootstrap.py — Step 0.1 exit criteria."""
from __future__ import annotations

import pathlib


def test_package_importable():
    import m2a
    assert m2a.__version__ == "0.1.0"


def test_decisions_file_exists():
    assert pathlib.Path("DECISIONS.md").exists()


def test_decisions_has_all_entries():
    text = pathlib.Path("DECISIONS.md").read_text()
    for d in [f"D{i:03d}" for i in range(1, 17)]:   # [v1.1] D001-D016
        assert d in text, f"{d} missing from DECISIONS.md"
