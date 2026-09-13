from pathlib import Path

import mido
import pytest

from m2a.analysis import analyze_midi


def make_basic_midi(
    path: Path,
    velocities=(80, 80, 80, 80),
    starts=(0, 480, 960, 1440),
):
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(
        mido.MetaMessage(
            "set_tempo",
            tempo=mido.bpm2tempo(120),
            time=0,
        )
    )
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=4,
            denominator=4,
            time=0,
        )
    )

    pitches = [60, 64, 67, 72]
    previous = 0

    for pitch, velocity, start in zip(pitches, velocities, starts):
        track.append(
            mido.Message(
                "note_on",
                note=pitch,
                velocity=velocity,
                time=start - previous,
            )
        )
        track.append(
            mido.Message(
                "note_off",
                note=pitch,
                velocity=0,
                time=120,
            )
        )
        previous = start + 120

    mid.save(path)


def make_polyphonic_midi(path: Path):
    mid = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    mid.tracks.append(track)

    track.append(
        mido.MetaMessage(
            "set_tempo",
            tempo=mido.bpm2tempo(120),
            time=0,
        )
    )
    track.append(
        mido.MetaMessage(
            "time_signature",
            numerator=4,
            denominator=4,
            time=0,
        )
    )

    for pitch in (60, 64, 67):
        track.append(
            mido.Message(
                "note_on",
                note=pitch,
                velocity=80,
                time=0,
            )
        )

    track.append(
        mido.Message("note_off", note=60, velocity=0, time=480)
    )
    track.append(
        mido.Message("note_off", note=64, velocity=0, time=0)
    )
    track.append(
        mido.Message("note_off", note=67, velocity=0, time=0)
    )

    mid.save(path)


def test_global_tempo_meter_and_ppq(tmp_path):
    path = tmp_path / "basic.mid"
    make_basic_midi(path)

    result = analyze_midi(path)

    assert result.source.ppq == 480
    assert len(result.tempo_map) == 1
    assert result.tempo_map[0].position_ticks == 0
    assert result.tempo_map[0].bpm == pytest.approx(120)

    assert result.meter[0].numerator == 4
    assert result.meter[0].denominator == 4


def test_pitch_range_and_note_density(tmp_path):
    path = tmp_path / "basic.mid"
    make_basic_midi(path)

    result = analyze_midi(path)
    track = result.tracks[0]

    assert track.range == (60, 72)
    assert track.note_density[0] == 4


def test_onset_profile(tmp_path):
    path = tmp_path / "basic.mid"
    make_basic_midi(path)

    result = analyze_midi(path)
    profile = result.tracks[0].onset_profile_16

    assert profile is not None

    for index in (0, 4, 8, 12):
        assert profile[index] == pytest.approx(0.25)

    assert sum(profile) == pytest.approx(1.0)


def test_flat_velocity(tmp_path):
    path = tmp_path / "basic.mid"
    make_basic_midi(path, velocities=(80, 80, 80, 80))

    result = analyze_midi(path)
    velocity = result.tracks[0].velocity

    assert velocity.mean == pytest.approx(80)
    assert velocity.std == pytest.approx(0)
    assert velocity.unique_values == 1
    assert velocity.flat is True


def test_polyphony(tmp_path):
    path = tmp_path / "poly.mid"
    make_polyphonic_midi(path)

    result = analyze_midi(path)
    poly = result.tracks[0].polyphony

    assert poly.max == 3
    assert poly.mean == pytest.approx(3.0)


def test_perfect_quantization(tmp_path):
    path = tmp_path / "basic.mid"
    make_basic_midi(path)

    result = analyze_midi(path)
    quant = result.tracks[0].quantization

    assert quant.fraction_within_tolerance == pytest.approx(1.0)
    assert quant.quantized is True


def test_shifted_quantization(tmp_path):
    path = tmp_path / "shifted.mid"
    make_basic_midi(
        path,
        starts=(0, 485, 965, 1450),
    )

    result = analyze_midi(path)
    quant = result.tracks[0].quantization

    assert quant.fraction_within_tolerance == pytest.approx(0.75)


def test_analysis_is_deterministic(tmp_path):
    path = tmp_path / "basic.mid"
    make_basic_midi(path)

    a = analyze_midi(path)
    b = analyze_midi(path)

    assert a.model_dump() == b.model_dump()

from m2a.analysis import (
    NoteEvent,
    group_notes_by_track,
    pitch_range,
)

def test_notes_are_grouped_by_track():
    notes = [
        NoteEvent(
            track_idx=0,
            pitch=60,
            velocity=80,
            start_tick=0,
            end_tick=120,
        ),
        NoteEvent(
            track_idx=1,
            pitch=36,
            velocity=100,
            start_tick=0,
            end_tick=120,
        ),
        NoteEvent(
            track_idx=0,
            pitch=64,
            velocity=80,
            start_tick=480,
            end_tick=600,
        ),
    ]

    grouped = group_notes_by_track(notes)

    assert set(grouped) == {0, 1}
    assert len(grouped[0]) == 2
    assert len(grouped[1]) == 1

    assert grouped[0][0].pitch == 60
    assert grouped[0][1].pitch == 64
    assert grouped[1][0].pitch == 36

def test_pitch_range():
    notes = [
        NoteEvent(
            track_idx=0,
            pitch=60,
            velocity=80,
            start_tick=0,
            end_tick=120,
        ),
        NoteEvent(
            track_idx=0,
            pitch=64,
            velocity=80,
            start_tick=480,
            end_tick=600,
        ),
        NoteEvent(
            track_idx=0,
            pitch=72,
            velocity=80,
            start_tick=960,
            end_tick=1080,
        ),
    ]

    assert pitch_range(notes) == (60, 72)

def test_pitch_range_empty():
    assert pitch_range([]) is None