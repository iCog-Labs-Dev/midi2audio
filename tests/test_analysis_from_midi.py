"""tests/test_analysis_from_midi.py — Stage 0 MIDI analysis."""
from __future__ import annotations

from m2a.analysis.from_midi import analyse_midi

FIXTURE_MIDI = "tests/fixtures/eight_bar.mid"


def test_analyses_fixture_midi(tmp_path):
    structure = analyse_midi(FIXTURE_MIDI, str(tmp_path / "s.json"))
    assert structure.edo == 12
    assert len(structure.tracks) == 4
    assert all(t.quantized for t in structure.tracks)
    assert all(t.flat_velocity for t in structure.tracks)


def test_bar_table_monotonic(tmp_path):
    structure = analyse_midi(FIXTURE_MIDI, str(tmp_path / "s.json"))
    times = [r[1] for r in structure.bar_table]
    assert times == sorted(times)


def test_tempo_close_to_120(tmp_path):
    structure = analyse_midi(FIXTURE_MIDI, str(tmp_path / "s.json"))
    assert abs(structure.tempo_map[0][1] - 120.0) < 2.0


def test_four_tracks_with_correct_roles(tmp_path):
    structure = analyse_midi(FIXTURE_MIDI, str(tmp_path / "s.json"))
    roles = {t.role for t in structure.tracks}
    assert roles == {"drums", "bass", "comping", "lead"}


def test_output_file_written_and_valid(tmp_path):
    out_path = tmp_path / "s.json"
    analyse_midi(FIXTURE_MIDI, str(out_path))
    assert out_path.exists()
    import json
    data = json.loads(out_path.read_text())
    assert data["schema"] == "midi2audio.structure/1"


def test_sections_tension_is_null(tmp_path):
    """D016: inferred provenance never populates tension."""
    structure = analyse_midi(FIXTURE_MIDI, str(tmp_path / "s.json"))
    assert all(s.tension is None for s in structure.sections)
