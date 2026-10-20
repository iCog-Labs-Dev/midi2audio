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
| **M1.5** | Microtonal path | ✅ Complete |
| **M2** | Scoring | ✅ Complete |
| **M3** | One endpoint end-to-end | ✅ Complete |
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

## What's in M1.5 — microtonal path

```
m2a/
├── midi_io.py            # per-track notes by mido track index + merged tempo map
├── package.py            # RenderPackage assembly + the five contract invariants
├── render/
│   ├── tuning.py         # EDO -> .scl, MTS bulk dump, pitch decoding, MPE channel rotation
│   ├── fluidsynth_r.py   # + FluidSynthMpeRenderer: tuning-exact via per-note pitch bends
│   ├── ddsp_r.py         # MidiDdspRenderer(mode="f0_contour"): harmonic+noise synth on the EDO f0
│   ├── sfizz_r.py        # opt-in 12-EDO sfizz renderer (needs sfz_instruments config)
│   └── registry.py       # (role, microtonal) -> ordered renderer preference table
└── scoring/
    └── tuning_check.py   # cents deviation from the *intended* EDO pitch (D004)
```

- **Pitch encoding** is declared by `tuning.json.method`: `direct_12`, key-mapped
  `scala` / `mts_esp` (key `k` = EDO step `k - base_tuning`), or `mpe` (key + bend).
- Microtonal pitched tracks never touch a 12-EDO path (P6); a 19-EDO part
  rendered on the 12-EDO path is caught by the tuning gate (tested).
- Deviations from the plan's renderer table are recorded as **D017**.
- Fixed while wiring the boundary: `from_midi` numbered tracks by pretty_midi
  instrument order, so a note-less conductor track shifted every `idx` and the
  renderer extracted the wrong track. Tracks are now indexed by mido track.

---

## What's in M2 — scoring

```
m2a/scoring/
├── rhythm.py      # onset F-measure: chords collapsed, shift-aligned first, mir_eval matching
├── notes.py       # 0.7 * transcription F1 + 0.3 * CQT-chroma cosine vs the dry stem
├── clap_sim.py    # cos(audio, prompt) - cos(audio, negative); pluggable encoder
├── fad.py         # Fréchet Audio Distance + per-corpus tau (median leave-one-out)
├── tuning_check.py
└── combine.py     # per-role weights and floors; floors on raw values, then a
                   #   weighted sum over axes normalised to [0, 1] (FAD as exp(-fad/tau))
```

Every scorer has monotonicity tests. Axes that are not applicable (notes on
drums) or not available (no CLAP model, no reference corpus) are dropped and
their weight redistributed, so totals stay in [0, 1] and comparable. Heavy
models are optional (D010): the built-in CQT transcriber and log-mel FAD
embedder need no downloads; basic-pitch and laion-clap plug in when installed.

---

## What's in M3 — one endpoint end-to-end

```
m2a/
├── restyle/
│   ├── base.py           # RestyleEndpoint protocol, error taxonomy, divergence <-> provider strength (D011)
│   ├── stable_audio.py   # Stable Audio 2.5 a2a: httpx + tenacity, 403 = content policy
│   ├── offline.py        # deterministic offline endpoint (CI / dry runs, zero cost)
│   ├── chunking.py       # seams on downbeats by boundary_lvl, 2-bar overlap, aligned linear fade (D002)
│   └── policy.py         # divergence grid: role x section energy; microtonal -> [] (P6)
├── prompts/
│   ├── stylecard.py      # measured reference features -> descriptors + negatives
│   └── generate.py       # per-stem variants; "instrumental"/"vocals" and odd-meter counters baked in
├── search/loop.py        # BudgetGuard (calls, USD, per stem), request-hash cache, successive halving
├── manifest.py           # run provenance: stages, every call, per-stem outcome, decisions (§8.1)
└── orchestrator.py       # + run(): spine -> S3 search per stem -> stems, rough mix, click, manifest
```

- Cache hits never count against the budget; a re-run with `max_calls: 0`
  reproduces the same output hashes (tested).
- When nothing clears the floors a stem falls back to its dry render and is
  flagged in the manifest (D012); `--strict` exits 2 instead.
- `render-audio` on the 8-bar fixture with the offline endpoint completes in
  well under 60 s (tested).

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

# 6. Render the fixture end to end (offline endpoint: no network, no cost)
python scripts/run_pipeline.py render-audio tests/fixtures/eight_bar.mid --out ./output
```

Expected output: **214 passed, 2 skipped** (live CLAP and the live-endpoint smoke test).

> **D010 note:** on Intel macOS, `llvmlite >= 0.46` (pulled in by
> `librosa -> numba`) has no prebuilt wheel and fails to build. Pin a wheel-backed
> pair when installing: `pip install -e ".[dev]" "numba==0.62.1" "llvmlite==0.45.1" "numpy<2.4"`.
> librosa is required from M1.5 on (f0 tracking, onsets, CQT).

`render-audio` options: `--endpoint offline|stable_audio` (the latter reads
`STABILITY_API_KEY`), `--refs DIR` (reference corpus for FAD + the style card),
`--structure` / `--tuning` (supply instead of infer; required for microtonal
material), `--budget-calls`, `--budget-usd`, `--strict`, `--profile` (YAML
overlay of `config/default.yaml`). Output: `stems/`, `rough_mix.wav`,
`click.wav`, `bar_table.json`, `scores.json`, `manifest.json`.

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

See [`DECISIONS.md`](DECISIONS.md) (D001–D018). Resolved through M3: D001,
D002, D004, D005, D008, D009, D010, D011, D012, D016, plus D017 (microtonal
renderer deviations) and D018 (endpoint tests without cassettes). D013 is
resolved for the interim; still open before M4: **D003** (LUFS targets),
**D013** (final FAD embedding/estimator), **D015** (full-mix scoring).

---

## Running tests

```bash
# M0 tests only (no audio deps required)
pytest tests/test_bootstrap.py tests/test_contracts.py \
       tests/test_artifacts.py tests/test_fixtures.py -v

# Fast suite (skips the >10 s end-to-end runs)
pytest -m "not slow"

# All tests (requires the fluidsynth CLI on PATH)
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
