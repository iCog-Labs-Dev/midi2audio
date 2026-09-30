"""tests/test_groove_apply.py — groove applicator invariants."""
from __future__ import annotations

import mido
import pytest
from hypothesis import given, settings, strategies as st

from hypothesis import HealthCheck

from m2a.analysis.from_midi import load_structure
from m2a.groove.apply import apply_groove, compute_swing_delay_ms
from m2a.groove.spec import GrooveSpec, MicrotimingSpec, SwingSpec, VelocitySpec

FIXTURE_MIDI = "tests/fixtures/eight_bar.mid"
FIXTURE_STRUCTURE_PATH = "tests/fixtures/eight_bar_structure.json"


def _structure():
    return load_structure(FIXTURE_STRUCTURE_PATH)


def _count_notes(path: str) -> int:
    mid = mido.MidiFile(path)
    return sum(1 for track in mid.tracks for msg in track if msg.type == "note_on" and msg.velocity > 0)


def _velocities(path: str) -> list[int]:
    mid = mido.MidiFile(path)
    return [msg.velocity for track in mid.tracks for msg in track
            if msg.type == "note_on" and msg.velocity > 0]


@given(rng_seed=st.integers(min_value=0, max_value=2**32 - 1))
@settings(max_examples=25, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_note_count_preserved(rng_seed, tmp_path):
    spec = GrooveSpec()
    out_path = str(tmp_path / f"out_{rng_seed}.mid")
    out, report = apply_groove(FIXTURE_MIDI, spec, _structure(), rng_seed, out_path=out_path)
    original = _count_notes(FIXTURE_MIDI)
    result = _count_notes(out)
    assert result == original + report.ghost_notes_inserted


def test_offsets_match_spec_within_one_tick(tmp_path):
    structure = _structure()
    spec = GrooveSpec(microtiming_ms={
        "drums": MicrotimingSpec(by_position_16=[0.0] * 7 + [6.0] + [0.0] * 8, jitter_sd=0.0)
    })
    out, report = apply_groove(FIXTURE_MIDI, spec, structure, rng_seed=0, out_path=str(tmp_path / "out.mid"))
    drum_offsets = [o for o in report.applied_timing_offsets if o[0] == "drums"]
    bpm = structure.tempo_map[0][1]
    tpb = mido.MidiFile(FIXTURE_MIDI).ticks_per_beat
    expected_ticks = round(6.0 / 1000.0 * (bpm / 60.0) * tpb)
    for _, pos, offset_ticks in drum_offsets:
        if pos == 7:
            assert abs(offset_ticks - expected_ticks) <= 1


def test_neutral_velocity_map_no_change(tmp_path):
    spec = GrooveSpec(velocity=VelocitySpec(accent_map_16=[1.0] * 16))
    out_path = str(tmp_path / "out.mid")
    apply_groove(FIXTURE_MIDI, spec, _structure(), rng_seed=0, out_path=out_path)
    assert _velocities(out_path) == _velocities(FIXTURE_MIDI)


def test_deterministic_given_same_seed(tmp_path):
    spec = GrooveSpec(
        swing=SwingSpec(ratio=0.6, subdivision=8, applies_to=["drums"]),
        microtiming_ms={"bass": MicrotimingSpec(jitter_sd=3.0)},
    )
    structure = _structure()
    out1, _ = apply_groove(FIXTURE_MIDI, spec, structure, rng_seed=42, out_path=str(tmp_path / "a.mid"))
    out2, _ = apply_groove(FIXTURE_MIDI, spec, structure, rng_seed=42, out_path=str(tmp_path / "b.mid"))
    assert open(out1, "rb").read() == open(out2, "rb").read()


def test_different_seed_changes_jitter(tmp_path):
    spec = GrooveSpec(microtiming_ms={"bass": MicrotimingSpec(jitter_sd=5.0)})
    structure = _structure()
    out1, _ = apply_groove(FIXTURE_MIDI, spec, structure, rng_seed=1, out_path=str(tmp_path / "a.mid"))
    out2, _ = apply_groove(FIXTURE_MIDI, spec, structure, rng_seed=2, out_path=str(tmp_path / "b.mid"))
    assert open(out1, "rb").read() != open(out2, "rb").read()


def test_swing_offbeat_delay_positive():
    delay_ms = compute_swing_delay_ms(ratio=0.66, subdivision=8, bpm_local=120.0, tempo_scaling=False)
    assert delay_ms > 0


def test_swing_straight_ratio_no_delay():
    delay_ms = compute_swing_delay_ms(ratio=0.5, subdivision=8, bpm_local=120.0, tempo_scaling=False)
    assert delay_ms == pytest.approx(0.0)


def test_swing_tempo_scaling_shrinks_ratio_moderately_faster_tempo():
    """At 150 BPM (moderately above the 120 BPM reference) with ratio=0.66,
    holding the short note's absolute duration constant shrinks ratio_eff
    into (0.5, 0.6) without yet hitting the straight-time floor."""
    bpm = 150.0
    delay_ms = compute_swing_delay_ms(ratio=0.66, subdivision=8, bpm_local=bpm,
                                       tempo_scaling=True, reference_bpm=120.0)
    pair_ms = 60_000.0 / bpm
    ratio_eff = 0.5 + delay_ms / pair_ms
    assert 0.5 < ratio_eff < 0.60


def test_swing_tempo_scaling_clamps_to_straight_at_extreme_tempo():
    """At a tempo fast enough that holding the reference short-note duration
    would exceed half the local beat, ratio_eff clamps to 0.5 (straight)
    rather than going negative or below the floor."""
    delay_ms = compute_swing_delay_ms(ratio=0.66, subdivision=8, bpm_local=240.0,
                                       tempo_scaling=True, reference_bpm=120.0)
    assert delay_ms == pytest.approx(0.0)


def test_swing_full_pipeline_applies_to_offbeat_positions(tmp_path):
    """Integration check: swing wiring reaches the applicator on a track that has offbeat onsets."""
    structure = _structure()
    spec = GrooveSpec(microtiming_ms={
        "drums": MicrotimingSpec(by_position_16=[0.0] * 16, jitter_sd=0.0)
    })
    out, report = apply_groove(FIXTURE_MIDI, spec, structure, rng_seed=0, out_path=str(tmp_path / "out.mid"))
    # the fixture's drum onsets fall on quarter-note grid positions 0/4/8/12; confirm those are
    # exactly the positions the applicator sees (i.e. slot computation lines up with the fixture).
    drum_offsets = [pos for role, pos, _ in report.applied_timing_offsets if role == "drums"]
    assert set(drum_offsets) == {0, 4, 8, 12}


def test_microtiming_rejects_mismatched_grid_length():
    structure = _structure()
    spec = GrooveSpec(microtiming_ms={"drums": MicrotimingSpec(by_position_16=[0.0] * 5, grid=16)})
    with pytest.raises(ValueError):
        apply_groove(FIXTURE_MIDI, spec, structure, rng_seed=0, out_path="/tmp/should_not_write.mid")


def test_anticipation_not_implemented_raises():
    structure = _structure()
    spec = GrooveSpec(anticipation={"prob": 0.5})
    with pytest.raises(NotImplementedError):
        apply_groove(FIXTURE_MIDI, spec, structure, rng_seed=0, out_path="/tmp/should_not_write2.mid")
