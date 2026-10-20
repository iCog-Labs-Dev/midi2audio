"""tests/test_package.py — RenderPackage assembly and the five boundary invariants."""
from __future__ import annotations

import json

import pytest

from m2a.analysis.from_midi import analyse_midi, load_structure
from m2a.package import ContractViolation, build_render_package, load_package, validate_package
from m2a.render.tuning import TuningSpec
from tests.conftest import FIXTURES
from tests.synth import write_midi

FIXTURE_MIDI = str(FIXTURES / "eight_bar.mid")


def _structure():
    return load_structure(str(FIXTURES / "eight_bar_structure.json"))


def test_fixture_package_is_valid():
    validate_package(_structure(), TuningSpec(), FIXTURE_MIDI)


def test_edo_mismatch_rejected():                                      # invariant 3
    with pytest.raises(ContractViolation, match="edo"):
        validate_package(_structure(), TuningSpec(edo=19, method="scala"), FIXTURE_MIDI)


def test_microtonal_track_needs_safe_method():                         # invariant 5
    s = _structure()
    s.tracks[3].microtonal = True
    with pytest.raises(ContractViolation, match="microtonal"):
        validate_package(s, TuningSpec(), FIXTURE_MIDI)
    validate_package(s, TuningSpec(method="mpe"), FIXTURE_MIDI)


def test_track_with_notes_must_be_listed():                            # invariant 1
    s = _structure()
    s.tracks.pop()
    with pytest.raises(ContractViolation, match="no structure.tracks entry"):
        validate_package(s, TuningSpec(), FIXTURE_MIDI)


def test_out_of_range_idx_rejected():                                  # invariant 1
    s = _structure()
    s.tracks[3].idx = 9
    with pytest.raises(ContractViolation, match="not in score.mid"):
        validate_package(s, TuningSpec(), FIXTURE_MIDI)


def test_conductor_track_indices_match_mido_tracks(tmp_path):
    """Regression: from_midi numbered tracks by pretty_midi instrument order, so a
    note-less conductor track 0 shifted every idx by one and the renderer
    extracted the wrong track."""
    midi = write_midi(str(tmp_path / "c.mid"), [[], [(0, 1, 40)], [(0, 1, 72)]],
                      programs=[0, 33, 81])
    s = analyse_midi(midi, str(tmp_path / "s.json"))
    assert [(t.idx, t.role) for t in s.tracks] == [(1, "bass"), (2, "lead")]
    validate_package(s, TuningSpec(), midi)


def test_build_package_infers_structure_and_default_tuning(tmp_path):
    pkg = build_render_package(FIXTURE_MIDI, str(tmp_path))
    structure, tuning = load_package(pkg)
    assert structure.provenance == "inferred"
    assert tuning.is_12_edo
    assert pkg.root.endswith(pkg.content_hash[:16])
    assert json.loads(open(pkg.manifest_path).read())["schema"] == "midi2audio.manifest/1"


def test_build_package_accepts_supplied_structure(tmp_path):            # D005
    pkg = build_render_package(FIXTURE_MIDI, str(tmp_path),
                               structure_path=str(FIXTURES / "eight_bar_structure.json"),
                               tuning_path=str(FIXTURES / "eight_bar_tuning.json"))
    assert load_structure(pkg.structure_path).source_hash.endswith("0" * 64)


def test_build_package_is_content_addressed(tmp_path):
    a = build_render_package(FIXTURE_MIDI, str(tmp_path))
    b = build_render_package(FIXTURE_MIDI, str(tmp_path))
    assert a == b
