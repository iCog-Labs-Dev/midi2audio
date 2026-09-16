"""tests/test_cli.py — scripts/run_pipeline.py subcommands."""
from __future__ import annotations

import json
import subprocess
import sys

FIXTURE_MIDI = "tests/fixtures/eight_bar.mid"


def test_analyse_command(tmp_path):
    result = subprocess.run([
        sys.executable, "scripts/run_pipeline.py", "analyse",
        FIXTURE_MIDI, "--out", str(tmp_path),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "structure.json").exists()


def test_expressivize_command_with_neutral_spec(tmp_path):
    spec_path = tmp_path / "spec.json"
    spec_path.write_text(json.dumps({}))
    result = subprocess.run([
        sys.executable, "scripts/run_pipeline.py", "expressivize",
        FIXTURE_MIDI, "--groove-spec", str(spec_path), "--out", str(tmp_path),
    ], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "expressive.mid").exists()


def test_render_audio_not_yet_wired(tmp_path):
    result = subprocess.run([
        sys.executable, "scripts/run_pipeline.py", "render-audio", FIXTURE_MIDI,
    ], capture_output=True, text=True)
    assert result.returncode != 0
