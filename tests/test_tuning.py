"""tests/test_tuning.py — EDO tuning utilities (Step 1.8)."""
from __future__ import annotations

import pytest

from m2a.render.tuning import (
    TuningSpec, channel_rotation_plan, mts_tuning_table, note_pitch, split_pitch, step_to_cents,
    to_scala,
)


def test_scala_12_edo_last_line():
    s = to_scala(12)
    assert "1200.000000" in s.splitlines()[-1]


def test_scala_19_edo_step_size():
    lines = to_scala(19).splitlines()
    first_step = float(lines[3])
    assert abs(first_step - 63.157895) < 0.0001
    assert int(lines[2]) == 19 and len(lines) == 3 + 19


def test_step_to_cents_perfect_fifth_12():
    assert abs(step_to_cents(7, 12) - 700.0) < 0.001


def test_step_to_cents_fifth_19():
    assert abs(step_to_cents(11, 19) - 694.737) < 0.01


def test_note_pitch_scala_key_mapping():
    t = TuningSpec(edo=19, base_tuning=60.0, method="scala")
    assert note_pitch(60, 0, t) == 60.0
    assert abs(note_pitch(71, 0, t) - (60.0 + 6.94737)) < 1e-4   # 11 steps = 19-EDO fifth
    assert abs(note_pitch(79, 0, t) - 72.0) < 1e-9               # 19 steps = octave


def test_note_pitch_mpe_uses_bend():
    t = TuningSpec(edo=19, method="mpe", bend_range_semitones=2.0)
    assert note_pitch(60, 4096, t) == 61.0
    assert note_pitch(60, 4096, TuningSpec()) == 60.0              # direct_12 ignores bend


def test_split_pitch_round_trips():
    for pitch in [60.0, 60.37, 61.49, 59.51, 72.947]:
        key, bend = split_pitch(pitch, 2.0)
        assert abs(key + bend / 8192 * 2.0 - pitch) < 0.001


def test_unknown_method_rejected():
    with pytest.raises(ValueError):
        TuningSpec(method="cents_table")


def test_mts_bulk_dump_shape_and_checksum():
    sysex = mts_tuning_table(19, 60)
    assert sysex[0] == 0xF0 and sysex[-1] == 0xF7
    assert sysex[1:4] == bytes([0x7E, 0x7F, 0x08]) and sysex[4] == 0x01
    assert len(sysex) == 1 + 5 + 16 + 128 * 3 + 2      # F0, header, name, data, checksum+F7
    body = sysex[1:-2]
    chk = 0
    for b in body:
        chk ^= b
    assert sysex[-2] == chk & 0x7F
    assert all(b < 0x80 for b in sysex[1:-1])


def test_mts_bulk_dump_base_key_and_step():
    sysex = mts_tuning_table(19, 60)
    data = sysex[1 + 5 + 16:-2]
    key60 = data[60 * 3:60 * 3 + 3]
    assert key60 == bytes([60, 0, 0])
    semitone, msb, lsb = data[61 * 3:61 * 3 + 3]
    pitch = semitone + ((msb << 7) | lsb) / (1 << 14)
    assert abs(pitch - (60 + 1200 / 19 / 100)) < 1e-4


def test_channel_rotation_separates_simultaneous_different_bends():
    notes = [(0, 10, 60.0), (0, 10, 60.3), (0, 10, 60.6)]
    plan = channel_rotation_plan(notes, 2.0)
    chans = [plan[i][0] for i in range(3)]
    assert len(set(chans)) == 3
    assert 9 not in chans


def test_channel_rotation_shares_channel_for_equal_bend():
    notes = [(0, 10, 60.0), (0, 10, 64.0)]       # both bend 0
    plan = channel_rotation_plan(notes, 2.0)
    assert plan[0][2] == plan[1][2] == 0


def test_channel_rotation_overflow_raises():
    notes = [(0, 10, 60.0 + i * 0.05) for i in range(16)]
    with pytest.raises(ValueError):
        channel_rotation_plan(notes, 2.0)
