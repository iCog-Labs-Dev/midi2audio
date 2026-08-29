"""tests/test_contracts.py — Step 0.2 exit criteria."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from m2a.contracts import Candidate, RenderPackage, StructureJSON
from tests.conftest import load_fixture


def test_structure_json_valid():
    data = load_fixture("eight_bar_structure.json")
    s = StructureJSON.model_validate(data)
    assert s.edo == 12
    assert len(s.tracks) == 4


def test_structure_json_rejects_non_monotonic_bar_table():
    data = load_fixture("eight_bar_structure.json")
    data["bar_table"] = [[0, 2.0], [1, 0.0]]  # reversed
    with pytest.raises(ValidationError):
        StructureJSON.model_validate(data)


def test_structure_json_rejects_duplicate_track_idx():
    data = load_fixture("eight_bar_structure.json")
    # duplicate the first track's idx on the second track
    data["tracks"][1]["idx"] = data["tracks"][0]["idx"]
    with pytest.raises(ValidationError):
        StructureJSON.model_validate(data)


def test_render_package_is_frozen():
    pkg = RenderPackage(
        root="r",
        midi_path="m",
        structure_path="s",
        tuning_path="t",
        manifest_path="mf",
        content_hash="h",
    )
    with pytest.raises(Exception):
        pkg.root = "other"  # type: ignore[misc]


def test_candidate_rejected():
    c = Candidate(
        stem="drums",
        audio_path="/tmp/x.wav",
        prompt="",
        negative_prompt="",
        strength=0.7,
        seed=1,
        endpoint="sa",
        endpoint_version="2.5",
        scores=(("rhythm", 0.4),),
        accepted=False,
        reject_reason="rhythm_floor",
    )
    assert not c.accepted
    assert c.reject_reason == "rhythm_floor"


def test_section_tension_optional():
    """[v1.1 / D016] tension may be null for inferred-provenance structure.json."""
    data = load_fixture("eight_bar_structure.json")
    s = StructureJSON.model_validate(data)
    # The fixture has tension=null; model must accept it
    assert s.sections[0].tension is None


def test_track_idx_present():
    """[v1.1] tracks must carry an idx field."""
    data = load_fixture("eight_bar_structure.json")
    s = StructureJSON.model_validate(data)
    for i, track in enumerate(s.tracks):
        assert track.idx == i
