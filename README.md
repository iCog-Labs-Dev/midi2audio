# midi2audio

**Standalone MIDI-to-production-audio pipeline.**

Takes a multitrack MIDI file as input and produces a finished, mastered audio
file as output. No dependency on any symbolic composer — any tool that can
produce a MIDI file can feed it.

> Implementation reference: *From Multitrack MIDI to Production-Quality Audio* (Aug 2026)  
> Plan: [`MIDI2AUDIO_IMPLEMENTATION_PLAN_v1.1.md`](MIDI2AUDIO_IMPLEMENTATION_PLAN_v1.1.md)

---

## Status

| Milestone | Description | Status |
|-----------|-------------|--------|
| **M0** | Contract & scaffolding | ✅ Complete |
| **M1** | Deterministic spine (MIDI → WAV stems) | ✅ Complete |
| M1.5 | Microtonal path | 🔲 Not started |
| M2 | Scoring | 🔲 Not started |
| M3 | One endpoint end-to-end | 🔲 Not started |
| M4 | Reference corpus | 🔲 Not started |
| M5 | Breadth | 🔲 Not started |
| M6 | Single-reference mode | 🔲 Not started |

---

## What's in M0

M0 establishes the contract, scaffolding, and test fixtures that all later
milestones build on. Nothing here performs audio rendering or calls any paid
endpoint.

```
m2a/
├── __init__.py        # __version__ = "0.1.0"
├── contracts.py       # RenderPackage, Stem, Candidate, StructureJSON (pydantic)
└── artifacts.py       # Content-addressed stage store + streaming hash

scripts/
└── generate_fixtures.py   # Produces tests/fixtures/*.mid and *.wav

tests/
├── conftest.py
├── fixtures/
│   ├── eight_bar.mid              # 4-track, 8-bar, 120 BPM, fully quantised
│   ├── eight_bar_structure.json   # Validated structure.json for the MIDI
│   ├── eight_bar_tuning.json      # 12-EDO tuning config
│   ├── eight_bar_manifest.json    # Minimal provenance record
│   ├── five_sec_stem.wav          # 5 s silence (replaced with real render in M1)
│   └── synthetic_host.wav         # Kick+bass click track for single-ref testing
├── test_bootstrap.py
├── test_contracts.py
├── test_artifacts.py
└── test_fixtures.py

DECISIONS.md   # 16 pre-seeded design decisions (D001–D016)
```

---

## What's in M1

M1 is the deterministic spine: MIDI in → expressivized MIDI → dry WAV stems
out. Zero network calls, zero paid endpoints. The only impure operations are
disk I/O and the (fully mocked/injectable) LLM groove-author call, which is
cached immediately.

```
m2a/
├── analysis/
│   └── from_midi.py    # Stage 0: MIDI -> structure.json (tempo/meter/key/
│                        #   chords/sections/tracks via pretty_midi + music21)
├── groove/
│   ├── spec.py          # GrooveSpec — the LLM output contract (pydantic)
│   ├── author.py        # LLM call + validate + retry (never patches invalid specs)
│   └── apply.py         # Pure applicator: swing, meter-aware microtiming,
│                         #   velocity accent, ghost notes, articulation
├── render/
│   ├── fluidsynth_r.py  # Stage 2: dry WAV stem via the FluidSynth CLI
│   └── registry.py      # Renderer selection by (role, microtonal)
├── config.py             # AudioConfig / GrooveConfig (D008: midi_gpt off by default)
└── orchestrator.py       # S0->S1->S2 DAG runner with content-addressed caching

scripts/
└── run_pipeline.py       # CLI: analyse | expressivize | render-audio (M3)
```

Notes on scope and decisions made while implementing M1:

- **D001** (tick rounding) resolved: round-half-even at serialization time;
  per-transformation offsets accumulate additively but are each computed
  from the note's *original* metrical position (see `DECISIONS.md`).
- Track identity is index-based (`structure.tracks[i].idx`), matching D016's
  convention from M0 — the applicator and renderer expand **roles** to track
  indices via a role map (D009), never by name.
- `GrooveSpec.anticipation`, `.tempo_curve`, and `VelocitySpec.phrase_arc` are
  accepted by the schema but raise `NotImplementedError` if set — deferred to
  a later milestone rather than silently no-op'd.
- Swing's tempo-scaling formula is implemented exactly as specified (Friberg
  & Sundström 2002, holding the off-beat note's absolute duration constant);
  at large tempo jumps it can clamp to straight time (ratio_eff = 0.5) sooner
  than the plan's worked example assumed — see the test suite for the exact
  behaviour at 150 vs. 240 BPM.

---

## Quick start

```bash
# 1. Create and activate a virtual environment (Python 3.11)
python3.11 -m venv .venv
source .venv/bin/activate

# 2. Install the package in editable mode with dev deps
pip install -e ".[dev]"

# 3. Install the FluidSynth CLI (M1 Stage 2 renderer)
brew install fluid-synth   # macOS; apt-get install fluidsynth on Debian/Ubuntu

# 4. Generate binary fixture files (MIDI + WAV)
python scripts/generate_fixtures.py

# 5. Run the full test suite
pytest -v
```

Expected output: **76 passed**.

> **D010 note:** on some machines, `pip install -e ".[dev]"` fails building
> `librosa`'s `numba → llvmlite` dependency from source (no prebuilt wheel
> for that Python build). None of the M1 code actually imports `librosa` —
> if you hit this, either use a Python interpreter/distribution that has a
> prebuilt `llvmlite` wheel available, or drop `librosa` from the `core`
> extra in `pyproject.toml` until a later milestone genuinely needs it.

---

## The input/output contract

Every pipeline run is anchored to a **RenderPackage** — a content-hashed directory:

```
run_<hash>/
├── score.mid          # multitrack MIDI (any source)
├── structure.json     # MIDI analysis artifact (schema v1)
├── tuning.json        # EDO, base tuning, per-track pitch method
└── manifest.json      # provenance record
```

`structure.json` is the key artifact. When the MIDI comes from an external
source, `midi2audio` infers it (M1). When it comes from `aimusic`, that system
produces it directly from its planner state. Either way, downstream stages see
the same schema — validated at the boundary by `StructureJSON` in
[`m2a/contracts.py`](m2a/contracts.py).

---

## Design principles

| # | Principle |
|---|-----------|
| P1 | Pure core, impure edges — every module except `restyle/*` is a pure function `(artifact_path, config) → artifact_path` |
| P2 | Dataclasses hold paths, not buffers — audio arrays are never kept in memory |
| P3 | Fail loudly at boundaries, tolerantly inside search |
| P4 | Hard floors before weighted scores — bad candidates are dropped, not repaired |
| P5 | Money guard consulted before every paid call |
| P6 | Microtonal tracks are routed, not retried |

---

## Open design decisions

See [`DECISIONS.md`](DECISIONS.md) for all 16 decisions (D001–D016). Key
unresolved items before M1.5 begins:

- **D004** — Microtonal cents threshold
- **D010** — Dependency environment topology (`madmom`, `demucs`, `basic-pitch`,
  `laion-clap` co-resolution on Python 3.11) — see the D010 note under
  Quick start; M1 itself needs none of these.

---

## Running tests

```bash
# M0 tests only (no audio deps required)
pytest tests/test_bootstrap.py tests/test_contracts.py \
       tests/test_artifacts.py tests/test_fixtures.py -v

# All tests (M0 + M1; requires the fluidsynth CLI on PATH)
pytest -v

# With coverage
pytest --cov=m2a --cov-report=term-missing
```

Markers:

| Marker | Meaning |
|--------|---------|
| `smoke` | Live endpoint calls — opt-in via `MIDI2AUDIO_SMOKE=1` |
| `slow` | Tests that take > 10 seconds |

---

## Repository conventions

- **Python ≥ 3.11** required.
- All source lives under `m2a/`. Sub-packages mirror the pipeline stages
  (`analysis/`, `groove/`, `render/`, `restyle/`, `scoring/`, `mixmaster/`).
- Stages are added milestone-by-milestone; stub `__init__.py` files are added
  as each milestone begins.
- Config lives in `config/default.yaml` (added in M1). Per-genre profiles
  go in `config/profiles/`.
- Every design decision is logged in `DECISIONS.md` before the relevant
  milestone starts. Never start a milestone with a TBD decision it depends on.
