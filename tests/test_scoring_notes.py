"""tests/test_scoring_notes.py — note-fidelity scorer (Step 2.2)."""
from __future__ import annotations

import pytest

from m2a.scoring.notes import (
    cqt_chroma_cosine, cqt_transcribe, note_f1, note_fidelity, reference_notes,
)
from tests.synth import render_events, write_midi

N = 16
BEAT = 0.5
TOTAL = N * BEAT + 1.0
CHORD = (60, 64, 67)                  # C major, root position
REVOICED = (64, 67, 72)               # same chord, first inversion


def _chords(revoice_every: int = 0):
    out = []
    for i in range(N):
        chord = REVOICED if revoice_every and i % revoice_every == 0 else CHORD
        out.append([(i, 0.9, p) for p in chord])
    return out


@pytest.fixture(scope="module")
def ref(tmp_path_factory):
    d = tmp_path_factory.mktemp("notes")
    midi = write_midi(str(d / "ref.mid"), [[n for c in _chords() for n in c]], programs=[4])
    dry = _wav(d / "dry.wav", _chords())
    return midi, dry, d


def _wav(path, chords, transpose=0, drop_every=0):
    events = []
    for i, chord in enumerate(chords):
        if drop_every and i % drop_every == 0:
            continue
        events += [(start * BEAT, dur * BEAT, p + transpose) for start, dur, p in chord]
    return render_events(str(path), events, TOTAL)


def test_transcriber_recovers_triads(ref):
    midi, dry, _ = ref
    assert note_f1(cqt_transcribe(dry), reference_notes(midi, 0)) > 0.9


def test_dry_render_scores_high(ref):
    midi, dry, _ = ref
    assert note_fidelity(dry, dry, midi, "comping", False, 0) > 0.9


def test_drums_not_applicable(ref):
    midi, dry, _ = ref
    assert note_fidelity(dry, dry, midi, "drums", False, 0) is None


def test_monotonic_under_transpose_and_drop(ref):
    midi, dry, d = ref
    clean = note_fidelity(dry, dry, midi, "comping", False, 0)
    dropped = note_fidelity(_wav(d / "drop.wav", _chords(), drop_every=10), dry, midi, "comping", False, 0)
    transposed = note_fidelity(_wav(d / "tr.wav", _chords(), transpose=2), dry, midi, "comping", False, 0)
    assert clean > dropped > transposed


def test_revoicing_separates_f1_from_chroma(ref):
    """[v1.1] Re-voicing keeps pitch classes (chroma barely moves) but changes the
    notes (F1 drops): proof that the two terms measure different things."""
    midi, dry, d = ref
    revoiced = _wav(d / "rv.wav", _chords(revoice_every=4))
    ref_notes = reference_notes(midi, 0)
    f1_drop = note_f1(cqt_transcribe(dry), ref_notes) - note_f1(cqt_transcribe(revoiced), ref_notes)
    chroma_drop = cqt_chroma_cosine(dry, dry) - cqt_chroma_cosine(revoiced, dry)
    assert f1_drop > 0.05
    assert chroma_drop < f1_drop / 2
