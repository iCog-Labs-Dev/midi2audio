"""tests/test_render_microtonal.py — microtonal routing (Step 1.10) and the M1.5 exit:
a 19-EDO fixture renders within the tuning-check floor."""
from __future__ import annotations

import shutil

import mido
import pytest

from m2a.render.ddsp_r import MidiDdspRenderer
from m2a.render.fluidsynth_r import FluidSynthMpeRenderer, FluidSynthRenderer, mpe_track_midi
from m2a.render.registry import DEFAULT_SOUNDFONT, RendererRegistry
from m2a.render.tuning import TuningSpec, note_pitch
from m2a.scoring.tuning_check import frame_times, intended_f0_contour, track_f0, tuning_check
from tests.synth import write_midi

TUNING_19 = TuningSpec(edo=19, base_tuning=60.0, method="scala")
SCALE_19 = [0, 3, 6, 8, 11, 14, 17, 19]


@pytest.fixture(scope="module")
def edo19_midi(tmp_path_factory):
    path = tmp_path_factory.mktemp("m") / "edo19.mid"
    bass = [(i, 1, 41 + s) for i, s in enumerate(SCALE_19)]
    lead = [(i, 1, 60 + s) for i, s in enumerate(SCALE_19)]
    return write_midi(str(path), [[], bass, lead], programs=[0, 33, 81])


def test_routing_table_complete():
    registry = RendererRegistry()
    for role in ["drums", "bass", "comping", "lead", "pads"]:
        for micro in [False, True]:
            assert registry.get(role, micro) is not None


def test_19_edo_lead_uses_ddsp():
    assert isinstance(RendererRegistry().get("lead", microtonal=True), MidiDdspRenderer)


def test_microtonal_pitched_tracks_never_route_to_12_edo_path():
    registry = RendererRegistry()
    for role in ["bass", "comping", "pads"]:
        assert isinstance(registry.get(role, True), FluidSynthMpeRenderer)
    assert isinstance(registry.get("drums", True), FluidSynthRenderer)


def test_unknown_role_rejected():
    with pytest.raises(ValueError):
        RendererRegistry().get("kazoo", False)


def test_mpe_track_midi_carries_exact_pitch(edo19_midi):
    out = mpe_track_midi(mido.MidiFile(edo19_midi), 1, TUNING_19)
    bends: dict[int, int] = {}
    sounded = []
    for msg in out.tracks[1]:
        if msg.type == "pitchwheel":
            bends[msg.channel] = msg.pitch
        if msg.type == "note_on" and msg.velocity > 0:
            sounded.append(msg.note + bends.get(msg.channel, 0) / 8192 * 2.0)
    expected = [note_pitch(41 + s, 0, TUNING_19) for s in SCALE_19]
    assert len(sounded) == len(expected)
    for got, want in zip(sounded, expected):
        assert abs(got - want) < 0.001


def _intended(path, midi, idx):
    f0 = track_f0(path)
    return f0, intended_f0_contour(midi, idx, TUNING_19, frame_times(len(f0)))


@pytest.mark.slow
def test_19_edo_lead_renders_within_tuning_floor(edo19_midi, tmp_path):
    stem = MidiDdspRenderer().render(edo19_midi, 2, "lead", "lead", DEFAULT_SOUNDFONT,
                                     str(tmp_path / "lead.wav"), tuning=TUNING_19)
    f0, intended = _intended(stem.path, edo19_midi, 2)
    report = tuning_check(stem.path, 19, 60.0, intended_f0=intended, f0=f0)
    assert report.passes and report.gate_eligible, report
    assert stem.microtonal


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth CLI not installed")
def test_19_edo_bass_renders_within_tuning_floor(edo19_midi, tmp_path):
    stem = FluidSynthMpeRenderer().render(edo19_midi, 1, "bass", "bass", DEFAULT_SOUNDFONT,
                                          str(tmp_path / "bass.wav"), tuning=TUNING_19)
    f0, intended = _intended(stem.path, edo19_midi, 1)
    report = tuning_check(stem.path, 19, 60.0, intended_f0=intended, f0=f0)
    assert report.passes, report


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("fluidsynth") is None, reason="fluidsynth CLI not installed")
def test_12_edo_render_of_19_edo_part_fails_the_gate(edo19_midi, tmp_path):
    """The gate must catch a 19-EDO part rendered through the plain 12-EDO path."""
    stem = FluidSynthRenderer().render(edo19_midi, 1, "bass", "bass", DEFAULT_SOUNDFONT,
                                       str(tmp_path / "bass12.wav"))
    f0, intended = _intended(stem.path, edo19_midi, 1)
    report = tuning_check(stem.path, 19, 60.0, intended_f0=intended, f0=f0)
    assert not report.passes, report
