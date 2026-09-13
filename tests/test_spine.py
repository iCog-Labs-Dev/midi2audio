"""Property-based verification and deterministic tests for the M1 spine."""

from __future__ import annotations

import tempfile
from pathlib import Path
import numpy as np
import pretty_midi
import pytest
from m2a.analysis import analyze_midi
from m2a.groove.apply import apply_groove_to_midi
from m2a.groove.spec import GrooveSpec, SwingSpec, TrackMicrotiming
from m2a.orchestrator import PipelineOrchestrator


@pytest.fixture
def quantized_midi_file(tmp_path: Path) -> Path:
    """Generate an exact, quantized 4-bar 120 BPM MIDI fixture."""
    pm = pretty_midi.PrettyMIDI(initial_tempo=120.0)

    # Track 0: Drums (Kick on 1, Snare on 2 & 4, Hi-Hat on all 16ths)
    drums = pretty_midi.Instrument(program=0, is_drum=True, name="Standard Kit")
    sec_per_16th = 0.125  # 120 BPM -> 0.5s/beat -> 0.125s/16th

    for bar in range(4):
        bar_offset = bar * 2.0
        # Kick on 1 and 3
        drums.notes.append(pretty_midi.Note(100, 36, bar_offset + 0.0, bar_offset + 0.1))
        drums.notes.append(pretty_midi.Note(100, 36, bar_offset + 1.0, bar_offset + 1.1))
        # Snare on 2 and 4
        drums.notes.append(pretty_midi.Note(95, 38, bar_offset + 0.5, bar_offset + 0.6))
        drums.notes.append(pretty_midi.Note(95, 38, bar_offset + 1.5, bar_offset + 1.6))
        # Hats on all 16ths with flat velocity
        for step in range(16):
            t = bar_offset + (step * sec_per_16th)
            drums.notes.append(pretty_midi.Note(80, 42, t, t + 0.05))

    pm.instruments.append(drums)

    # Track 1: Bass
    bass = pretty_midi.Instrument(program=33, is_drum=False, name="Fingered Bass")
    for bar in range(4):
        bar_offset = bar * 2.0
        bass.notes.append(pretty_midi.Note(90, 36, bar_offset + 0.0, bar_offset + 0.4))
        bass.notes.append(pretty_midi.Note(90, 39, bar_offset + 0.5, bar_offset + 0.9))
        bass.notes.append(pretty_midi.Note(90, 41, bar_offset + 1.0, bar_offset + 1.4))
        bass.notes.append(pretty_midi.Note(90, 36, bar_offset + 1.5, bar_offset + 1.9))
    pm.instruments.append(bass)

    midi_path = tmp_path / "test_quantized.mid"
    pm.write(str(midi_path))
    return midi_path


def test_groove_spec_validation():
    """Verify pydantic raises strict validation errors on malformed specs."""
    with pytest.raises(ValueError):
        # Invalid swing ratio > 0.75
        SwingSpec(ratio=0.85)

    with pytest.raises(ValueError):
        # Invalid position array length (!= 16)
        TrackMicrotiming(by_position_16=[0.0] * 8)


def test_deterministic_microtiming_offset_within_one_tick(quantized_midi_file: Path, tmp_path: Path):
    """Prove that applied microtiming matches the spec within 1 tick (strict non-negotiable)."""
    analysis = analyze_midi(quantized_midi_file, {})

    spec = GrooveSpec(
        swing=SwingSpec(ratio=0.5),  # Straight, test only microtiming offsets
        microtiming_ms={
            "drums": TrackMicrotiming(
                by_position_16=[0.0, 5.0, 10.0, -5.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                global_offset=0.0,
                jitter_sd=0.0,  # 0 jitter for deterministic verification
            )
        },
    )

    out_midi_path = tmp_path / "expressive.mid"
    apply_groove_to_midi(quantized_midi_file, spec, analysis, out_midi_path, seed=42)

    # Re-read and check offsets
    pm_orig = pretty_midi.PrettyMIDI(str(quantized_midi_file))
    pm_expr = pretty_midi.PrettyMIDI(str(out_midi_path))

    drums_orig = pm_orig.instruments[0]
    drums_expr = pm_expr.instruments[0]

    # Calculate tick duration at 120 BPM: 0.5s / 480 ppq ≈ 1.0416 ms
    tick_dur_sec = (60.0 / 120.0) / pm_orig.resolution

    for n_orig, n_expr in zip(drums_orig.notes, drums_expr.notes):
        if n_orig.pitch == 42:  # Hi-hat check
            pos_16 = int(round((n_orig.start % 2.0) / 0.125)) % 16
            expected_offset_sec = spec.microtiming_ms["drums"].by_position_16[pos_16] / 1000.0
            actual_offset_sec = n_expr.start - n_orig.start
            assert abs(actual_offset_sec - expected_offset_sec) <= (tick_dur_sec + 1e-4)


def test_end_to_end_spine_pipeline(quantized_midi_file: Path, tmp_path: Path):
    """Verify execution of Orchestrator across Stages 0, 1, and 2."""
    config = {
        "pipeline": {
            "sample_rate": 48000,
            "headroom_dbfs": -12.0,
            "artifacts_dir": str(tmp_path / "artifacts"),
            "soundfont_path": "non_existent.sf2",  # Triggers pure DSP fallback
        },
        "stage0_analysis": {},
        "stage1_groove": {
            "swing": {"ratio": 0.60, "subdivision": 8},
        },
    }

    orchestrator = PipelineOrchestrator(config)
    results = orchestrator.run_deterministic_spine(quantized_midi_file)

    assert Path(results["groove_midi_path"]).is_file()
    assert len(results["stems"]) == 2
    for stem in results["stems"]:
        assert Path(stem["audio_path"]).is_file()
        assert stem["sample_rate"] == 48000