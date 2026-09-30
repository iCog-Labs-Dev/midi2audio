"""tests/test_groove_spec.py — GrooveSpec schema validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from m2a.groove.spec import (
    GrooveSpec,
    MicrotimingSpec,
    SwingSpec,
    positions_per_bar,
)


def test_minimal_spec():
    s = GrooveSpec()
    assert s.swing is None


def test_swing_ratio_bounds():
    with pytest.raises(ValidationError):
        SwingSpec(ratio=0.3, applies_to=["drums"])


def test_jitter_sd_bounds():
    with pytest.raises(ValidationError):
        GrooveSpec(microtiming_ms={"drums": MicrotimingSpec(jitter_sd=20.0)})


def test_round_trip():
    spec = GrooveSpec(
        swing=SwingSpec(ratio=0.58, subdivision=8, applies_to=["drums", "bass"]),
        microtiming_ms={"drums": MicrotimingSpec(by_position_16=[0.0] * 16, jitter_sd=2.0)},
    )
    data = spec.model_dump()
    spec2 = GrooveSpec.model_validate(data)
    assert spec == spec2


def test_track_override_key_allowed():
    spec = GrooveSpec(microtiming_ms={"lead": MicrotimingSpec(), "track:3": MicrotimingSpec(global_offset=5.0)})
    assert spec.microtiming_ms["track:3"].global_offset == 5.0


def test_positions_per_bar_four_four_sixteen():
    assert positions_per_bar("4/4", 16) == 16


def test_positions_per_bar_seven_eight_sixteen():
    assert positions_per_bar("7/8", 16) == 14


def test_positions_per_bar_four_four_triplet_grid():
    assert positions_per_bar("4/4", 24) == 24
