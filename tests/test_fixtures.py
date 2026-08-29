"""tests/test_fixtures.py — Step 0.4 exit criteria.

These tests run against the committed fixture files and the generate_fixtures
script output. They guard against regressions in the fixture data itself.
"""
from __future__ import annotations

import json
import pathlib

import mido
import pytest

from m2a.contracts import StructureJSON
from tests.conftest import load_fixture

FIXTURES = pathlib.Path(__file__).parent / "fixtures"


# ── MIDI fixture ─────────────────────────────────────────────────────────────

def test_fixture_midi_parses():
    mid = mido.MidiFile(str(FIXTURES / "eight_bar.mid"))
    names = [t.name for t in mid.tracks]
    assert "drums" in names and "bass" in names


def test_fixture_midi_has_four_tracks():
    mid = mido.MidiFile(str(FIXTURES / "eight_bar.mid"))
    # mido counts the tempo track separately; we expect 4 named tracks
    named = [t for t in mid.tracks if t.name]
    assert len(named) == 4


@pytest.mark.parametrize("track_name", ["drums", "bass", "comping", "lead"])
def test_fixture_has_32_notes_per_track(track_name):
    """8 bars × 4 beats = 32 note_on events per track."""
    mid = mido.MidiFile(str(FIXTURES / "eight_bar.mid"))
    for track in mid.tracks:
        if track.name == track_name:
            onsets = [m for m in track if m.type == "note_on" and m.velocity > 0]
            assert len(onsets) == 32, f"{track_name}: expected 32 note_on, got {len(onsets)}"
            return
    pytest.fail(f"Track '{track_name}' not found in fixture MIDI")


def test_fixture_onsets_are_exactly_one_beat_apart():
    """[v1.1] regression for the 470-vs-480 tick bug — every inter-onset delta must == tpb."""
    mid = mido.MidiFile(str(FIXTURES / "eight_bar.mid"))
    for track in mid.tracks:
        onsets, clock = [], 0
        for msg in track:
            clock += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                onsets.append(clock)
        if len(onsets) > 1:
            deltas = {b - a for a, b in zip(onsets, onsets[1:])}
            assert deltas == {mid.ticks_per_beat}, \
                f"Track '{track.name}': unexpected onset deltas {deltas}"


# ── structure.json fixture ────────────────────────────────────────────────────

def test_fixture_structure_validates():
    data = load_fixture("eight_bar_structure.json")
    s = StructureJSON.model_validate(data)
    assert len(s.tracks) == 4


def test_fixture_structure_has_sequential_idxs():
    data = load_fixture("eight_bar_structure.json")
    s = StructureJSON.model_validate(data)
    assert [t.idx for t in s.tracks] == [0, 1, 2, 3]


def test_fixture_structure_12_edo():
    data = load_fixture("eight_bar_structure.json")
    s = StructureJSON.model_validate(data)
    assert s.edo == 12


# ── tuning.json fixture ───────────────────────────────────────────────────────

def test_fixture_tuning_has_required_fields():
    data = json.loads((FIXTURES / "eight_bar_tuning.json").read_text())
    assert "edo" in data and "method" in data


def test_fixture_tuning_edo_matches_structure():
    tuning = json.loads((FIXTURES / "eight_bar_tuning.json").read_text())
    structure = load_fixture("eight_bar_structure.json")
    assert tuning["edo"] == structure["edo"]


# ── manifest.json fixture ─────────────────────────────────────────────────────

def test_fixture_manifest_has_required_fields():
    data = json.loads((FIXTURES / "eight_bar_manifest.json").read_text())
    assert "schema" in data
    assert "pipeline_version" in data


# ── WAV fixtures ──────────────────────────────────────────────────────────────

def test_fixture_wav_files_exist():
    assert (FIXTURES / "five_sec_stem.wav").exists()
    assert (FIXTURES / "synthetic_host.wav").exists()


def test_fixture_wav_readable():
    """WAV files must be parseable (correct header)."""
    import soundfile as sf
    for name in ("five_sec_stem.wav", "synthetic_host.wav"):
        info = sf.info(str(FIXTURES / name))
        assert info.samplerate == 48000
        assert info.duration > 0
