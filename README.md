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
| M1 | Deterministic spine (MIDI → WAV stems) | 🔲 Not started |
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

## Quick start

```bash
# 1. Create and activate a virtual environment (Python 3.11)
uv venv .venv --python 3.11
source .venv/bin/activate

# 2. Install the package in editable mode with core + test deps
#    (The full [dev] extra requires llvmlite to build — see D010 note below)
uv pip install -e . pydantic pyyaml mido soundfile numpy pytest pytest-cov hypothesis vcrpy

# 3. Generate binary fixture files (MIDI + WAV)
python scripts/generate_fixtures.py

# 4. Run the M0 test suite
pytest tests/test_bootstrap.py tests/test_contracts.py \
       tests/test_artifacts.py tests/test_fixtures.py -v
```

Expected output: **32 passed**.

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
unresolved items before M1 begins:

- **D001** — Tick rounding policy (groove applicator)
- **D004** — Microtonal cents threshold
- **D010** — Dependency environment topology (`madmom`, `demucs`, `basic-pitch`,
  `laion-clap` co-resolution on Python 3.11) — **verify in week 1**

> **D010 note:** `librosa → numba → llvmlite` fails to build from source on
> some machines (x86 macOS). The `dev` extra in `pyproject.toml` intentionally
> separates heavy audio deps into `[audio]`/`[restyle]` extras. Until D010 is
> resolved, install only the packages listed in the Quick start above.

---

## Running tests

```bash
# M0 tests only (no audio deps required)
pytest tests/test_bootstrap.py tests/test_contracts.py \
       tests/test_artifacts.py tests/test_fixtures.py -v

# All tests (once M1+ deps are installed)
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
