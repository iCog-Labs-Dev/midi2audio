# midi2audio — Implementation Plan

**Standalone MIDI-to-production-audio pipeline**
Reference paper: *From Multitrack MIDI to Production-Quality Audio* (Aug 2026)
Status: **v1.1** · standalone repo

> **v1.1 changelog.** This revision fixes correctness bugs found by reviewing v1.0 against the
> design document (swing formula, FAD sign in the weighted sum, polyphonic onset counting,
> fixture note spacing, crossfade law, hardcoded 16-position grid), closes the gaps where the
> plan was silent on things the paper specifies (per-role score weights, provenance manifest,
> final audit gate, cover-blend scoring rule, strength semantics, dry-stem fallback), and adds
> the missing infrastructure the milestones assumed but never built. All new material is marked
> **[v1.1]**. Section 8 collects the additions that did not fit inside an existing step.

---

## 0. What This Is and What It Is Not

`midi2audio` is a **standalone pipeline**. It takes a multitrack MIDI file as input and produces a
finished, well-produced audio master as output. It has no dependency on `aimusic` or any other
symbolic composer. Any tool that can produce a MIDI file can feed it.

`aimusic` (the `musicGeneration` repo) remains a separate, independent symbolic composer. When the
team wants those two systems to work together, `aimusic` ships a small optional adapter
(`aimusic/render/package.py`) that converts its internal `Score` + `BeatState` output into the
`midi2audio` input format (`RenderPackage`). That adapter is `aimusic`'s responsibility, not this
repo's.

**The interface between the two systems is a directory on disk.** No shared Python types, no import
coupling. This is the contract, and it is the only thing that needs to stay stable across both repos.

---

## 1. The Input/Output Contract

Every run of `midi2audio` is anchored to a **RenderPackage** — a content-hashed directory:

```
run_<hash>/
├── score.mid          # multitrack MIDI file (any source)
├── structure.json     # Stage 0 analysis artifact (schema v1)
├── tuning.json        # EDO, base tuning, per-track pitch method
└── manifest.json      # provenance: who produced this and how
```

`structure.json` is the key artifact. When the MIDI comes from an external source (DAW, hand-authored,
third-party tool), `midi2audio` infers it using `pretty_midi`, `miditoolkit`, and `music21`.
When the MIDI comes from `aimusic`, `aimusic` produces `structure.json` directly from its internal
planner state — more accurately and cheaply than inference. Either way, downstream stages see the
same schema.

### structure.json schema (v1)

```json
{
  "schema": "midi2audio.structure/1",
  "provenance": "inferred",
  "source_hash": "sha256:...",
  "edo": 12,
  "base_tuning": 60,
  "tempo_map": [[0.0, 120.0], [32.0, 118.5]],
  "meter": [[0, "4/4"]],
  "key": [{"start_bar": 0, "key": "F# minor"}],
  "chords": [{"bar": 0, "beat": 0, "chord": "Cm7"}],
  "sections": [
    {"label": "A", "bars": [0, 16], "energy": 0.42,
     "tension": 0.31, "boundary_lvl": 3}
  ],
  "bar_table": [[0, 0.0], [1, 2.0]],
  "tracks": [
    {"name": "drums", "role": "drums", "program": null,
     "percussion": true, "quantized": true, "flat_velocity": true,
     "onset_profile_16": [0.9, 0.0, 0.1, 0.2],
     "pitch_range": [35, 51], "microtonal": false},
    {"name": "lead", "role": "lead", "program": 81,
     "percussion": false, "quantized": true, "flat_velocity": true,
     "pitch_range": [40, 76], "microtonal": true}
  ]
}
```

Five invariants are checked at the boundary and abort the run if violated:

1. **[v1.1]** Every track in `score.mid` is identified by its **zero-based track index**, which
   appears exactly once in `structure.tracks[].idx`. `name` is advisory and may be empty or
   duplicated (most real-world MIDI has unnamed or same-named tracks); when absent, the
   canonical name is `track{idx:02d}_{role}`. Aborting a run because a DAW export had two
   tracks called "Audio 1" was a v1.0 bug, not a safety property.
2. `bar_table` is monotonically increasing.
3. `tuning.json.edo == structure.edo`.
4. `score.mid` round-trips: re-parsing reproduces note onsets within 1 tick.
5. If any track has `microtonal: true`, `tuning.json.method` is `mpe`, `mts_esp`, or `scala` — never `direct_12`.

---

## 2. Repository Layout

```
midi2audio/
├── pyproject.toml          # pinned deps, python >= 3.11
├── README.md
├── DECISIONS.md            # decision log — filled during development
│
├── m2a/
│   ├── __init__.py
│   ├── contracts.py        # RenderPackage, Stem, Candidate, StructureJSON (pydantic)
│   ├── artifacts.py        # content-addressed store + cache
│   ├── orchestrator.py     # stage DAG + budget guard
│   ├── manifest.py         # [v1.1] provenance record — required by the design document,
│   │                       # absent from v1.0's layout entirely (see §8.1)
│   ├── config.py           # AudioConfig tree (frozen dataclasses, YAML-loaded)
│   │
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── from_midi.py    # primary: pretty_midi / music21 → structure.json
│   │   ├── host.py         # single-ref: beat grid, Demucs separation, onset lattice
│   │   └── reconcile.py    # cross-check two structure.jsons, emit disagreement report
│   │                        # [v1.1] no milestone builds this — see §8.7 (spec it or cut it)
│   │
│   ├── groove/
│   │   ├── __init__.py
│   │   ├── spec.py         # GrooveSpec pydantic schema
│   │   ├── author.py       # LLM call → GrooveSpec (validate + reject/retry loop)
│   │   ├── apply.py        # deterministic applicator — pure function
│   │   ├── extract.py      # groove template from reference audio (madmom)
│   │   ├── lattice.py      # bounded re-quantization to host onset lattice
│   │   └── models.py       # GrooVAE / MIDI-GPT wrappers (off by default)
│   │
│   ├── render/
│   │   ├── __init__.py
│   │   ├── registry.py     # (role, microtonal, policy) → renderer
│   │   ├── fluidsynth_r.py
│   │   ├── sfizz_r.py
│   │   ├── ddsp_r.py
│   │   ├── pianoteq_r.py
│   │   ├── hostsampler.py  # auto-SFZ from separated host stem
│   │   └── tuning.py       # EDO → .scl / MTS-ESP / MPE channel-rotation
│   │
│   ├── restyle/
│   │   ├── __init__.py
│   │   ├── base.py         # RestyleEndpoint protocol
│   │   ├── stable_audio.py # Stable Audio 2.5 a2a
│   │   ├── suno.py         # Suno cover endpoint
│   │   ├── musicgen_local.py
│   │   ├── chunking.py     # boundary_lvl-aware chunk/overlap/crossfade/inpaint
│   │   └── policy.py       # strength schedule: role × tension × EDO
│   │
│   ├── prompts/
│   │   ├── __init__.py
│   │   ├── stylecard.py    # reference corpus → house style card
│   │   └── generate.py     # per-stem prompt variants
│   │
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── rhythm.py       # onset F-measure vs expressive MIDI
│   │   ├── notes.py        # transcription F1 / CQT-chroma cosine
│   │   ├── clap_sim.py     # CLAP audio-text similarity
│   │   ├── fad.py          # Fréchet Audio Distance
│   │   ├── tuning_check.py # f0 deviation from EDO lattice
│   │   ├── single_ref.py   # per-window CLAP cosine (single-ref mode)
│   │   ├── syncopation.py  # [v1.1] LHL syncopation index — makes "same rhythmic
│   │   │                   # conversation" measurable rather than hand-tuned (see §8.4)
│   │   ├── mixfit.py       # mix-compatibility scorer
│   │   └── combine.py      # weighted sum + hard floors
│   │
│   ├── search/
│   │   └── loop.py         # grid → successive halving; caching; budget guard
│   │
│   └── mixmaster/
│       ├── mix.py          # LUFS norm, static mix, cover blend
│       ├── master.py       # Matchering against reference
│       ├── align.py        # cross-correlation stem alignment
│       └── audit.py        # [v1.1] final audit gate: loudness, true-peak, seam scan,
│                           # rhythm re-verification of the finished master (see §8.2)
│
├── config/
│   ├── default.yaml
│   └── profiles/
│       ├── funk.yaml
│       ├── prog.yaml
│       ├── jazz_fusion.yaml
│       ├── single_ref.yaml
│       └── microtonal.yaml
│
├── scripts/
│   └── run_pipeline.py     # CLI entry point
│
└── tests/
    ├── fixtures/
    │   ├── eight_bar.mid
    │   ├── eight_bar_structure.json
    │   ├── eight_bar_tuning.json
    │   ├── five_sec_stem.wav
    │   └── synthetic_host.wav
    └── test_*.py
```

---

## 3. Design Principles

These are not suggestions — they are constraints that every module must satisfy.

**P1 — Pure core, impure edges.** Every module except `restyle/*` and `orchestrator.py` is a pure
function: `(artifact_path, config) → artifact_path`. No hidden state, no network calls, no random
draws except through an explicit seed parameter. The LLM call in `groove/author.py` is the only
exception inside the pipeline proper, and its output is cached to disk immediately so reruns are
deterministic.

**P2 — Dataclasses hold paths, not buffers.** `RenderPackage`, `Stem`, `Candidate` hold `str` paths
and content hashes. Never audio arrays. A stage that produces audio writes it to disk and returns
the path.

**P3 — Fail loudly at boundaries, tolerantly inside search.** Schema violations, missing stems,
budget breaches, and tuning failures abort with a named error and a clear message. A single failed
endpoint call inside the search loop is logged, retried per policy, and otherwise skipped.

**P4 — Hard floors before weighted scores.** The search loop never repairs a candidate that failed a
hard floor (rhythm adherence, note fidelity, tuning preservation). It logs the rejection reason and
moves on. Repair happens only by re-generating with different strength or seed.

**P5 — Money guard consulted before every paid call.** `max_endpoint_calls` and `max_usd_estimate`
— **[v1.1]** note that `BudgetGuard` in M3 exposes only `max_calls`; `max_usd_estimate` needs a
per-endpoint `usd_per_call` and a `can_spend(cost)` signature, or half of P5 is decorative. Budget
is enforced both globally and **per stem** (the design document's figure is 20-40 endpoint calls
per stem, and one pathological stem should not eat the run). Cache hits never count against it.
are config parameters. The search loop checks them before every paid call and stops gracefully,
reporting best-so-far. Tests never hit paid endpoints.

**P6 — Microtonal tracks are routed, not retried.** A 19-EDO pitched track goes to an f0-following
renderer (DDSP/RAVE) or sfizz + `.scl`, not to a diffusion-style restyle endpoint. If no safe
restyle path exists, the deterministic stem is kept. This is a routing decision, not a fallback.

---

## 4. Milestones

Six milestones, sequenced strictly. Each is independently testable and independently useful as
a stopping point. The critical sequencing rule: **M3 must not begin before M2 is merged.** A restyle
layer without a scorer produces unaccountable output and wastes paid calls.

```
M0  Contract & scaffolding     Week 1
M1  Deterministic spine        Weeks 2–3
M1.5  Microtonal path          Week 4
M2  Scoring                    Weeks 5–6
M3  One endpoint end-to-end    Weeks 7–8
M4  Reference corpus           Weeks 9–10
M5  Breadth                    Week 11
M6  Single-reference mode      Week 12
    Hardening & handoff        Week 13
```

---

## 5. Step-by-Step Implementation

---

### M0 — Contract and Scaffolding (Week 1)

Goal: the repo exists, the contract is defined, everything downstream can be developed against
fixture data without waiting for anything else.

---

#### Step 0.1 — Repository bootstrap

**Deliverable:** repo skeleton, `pyproject.toml`, empty package structure.

Create the repo and `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "midi2audio"
version = "0.1.0"
description = "Automated MIDI to production-quality audio pipeline."
requires-python = ">=3.11"
dependencies = []   # intentionally empty — everything is in extras

[project.optional-dependencies]
core = [
  "pretty-midi",
  "miditoolkit",
  "music21",
  "mido",
  "numpy",
  "scipy",
  "librosa",
  "soundfile",
  "soxr",            # [v1.1] endpoints return 44.1 kHz; pipeline is 48 kHz
  "mir_eval",        # [v1.1] matched onset F-measure
  "pyloudnorm",      # [v1.1] LUFS — the plan sets LUFS targets in five places
  "pedalboard",      # [v1.1] high-pass / EQ / bus compression
  "pydantic>=2.0",
  "pyyaml",
  "tenacity",
]
audio = [
  "midi2audio[core]",
  "madmom",
  "demucs",
  "basic-pitch",
  "laion-clap",
  "matchering",
]
restyle = [
  "midi2audio[audio]",
  "httpx",
]
dev = [
  "midi2audio[restyle]",
  "pytest",
  "pytest-cov",
  "hypothesis",
  "vcrpy",
]

[tool.setuptools.packages.find]
include = ["m2a*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = [
  "smoke: opt-in live endpoint tests (set MIDI2AUDIO_SMOKE=1)",
  "slow: tests that take > 10 seconds",
]
```

Create `m2a/__init__.py` with version:

```python
__version__ = "0.1.0"
```

Create `DECISIONS.md` at repo root with these eight decisions pre-seeded (to be resolved before
the relevant milestone):

```markdown
# DECISIONS.md

## D001 — Tick rounding policy
Before: M1 Step 1.2 (groove applicator)
Question: When a groove timing offset lands between ticks, do we round-half-even or truncate?
Do offsets accumulate across multiple transformations or are they absolute per note?
Resolution: [TBD]

## D002 — Crossfade law
Before: M3 Step 3.2 (chunking)
Question: Equal-power or linear crossfade at chunk seams?
Overlap length in bars for odd meters (5/4, 7/8)?
Resolution: **[v1.1] Equal-gain (linear), preceded by chunk-local alignment.** Equal-power sums
correlated signals to +3 dB; chunks generated with identical prompt and seed over a shared 2-bar
overlap are correlated, so equal-power would put an audible level bump on every seam. Measure
the residual offset between overlapping regions by cross-correlation and correct it *before*
the fade (this is also the counter for the "tempo drift across chunk seams" failure mode named
in the design document). Overlap length: 2 bars in all meters, i.e. a time-varying number of
samples — bars, not seconds, is what keeps the fade centred on a downbeat.

## D003 — LUFS targets
Before: M4 Step 4.3 (mix and master)
Question: Per-role LUFS targets and final master target?
Resolution: drums -12, bass -13, comping -15, lead -12, master -10 [confirm]

## D004 — Microtonal cents threshold
Before: M1.5 Step 1.7 (tuning check)
Question: Above how many cents from 12-EDO is a track classified microtonal?
What is the p95 tuning floor for the hard rejection gate?
Resolution: [TBD — suggest: threshold 8 cents, floor p95 < 15 cents]

## D005 — structure.json always emitted
Before: M0 Step 0.4
Question: Should structure.json be emitted for runs where the caller already
supplies it (e.g. aimusic adapter)? Or only produced by from_midi?
Resolution: Always accepted as input; from_midi produces it when absent.

## D006 — E_host normalization (single-ref mode)
Before: M6 Step 6.2 (lattice re-quantization)
Question: Does the host energy term participate in the scoring normalization
or is it applied as a post-hoc gate?
Resolution: [TBD]

## D007 — Complement-mode δ
Before: M6 Step 6.2
Question: Minimum distance (ms) from a host onset for interlocking parts?
Resolution: [TBD — suggest: 30 ms = one 16th at 120 BPM]

## D009 — Groove-spec keys: roles or track names? **[v1.1]**
Before: M1 Step 1.4
Question: v1.0 documents `applies_to` and `microtiming_ms` keys as track names but the design
document specifies per-role semantics. Which wins, and how are two drum tracks handled?
Resolution: Roles are canonical; `track:<idx>` prefix for explicit per-track overrides; the
applicator expands via a role map built from `structure.tracks`.

## D010 — Dependency environment topology **[v1.1]**
Before: M0 Step 0.1 (day one)
Question: Do madmom, demucs, basic-pitch and laion-clap co-resolve on Python 3.11 in one venv?
Resolution: [TBD — verify in week 1; fallback is a scorer service boundary, see §7]

## D011 — Strength semantics per endpoint **[v1.1]**
Before: M3 Step 3.1
Question: Does a higher `strength` mean more divergence or more preservation, per endpoint?
Resolution: Pipeline searches normalised `divergence in [0,1]`; each endpoint declares
`strength_semantics` and converts. Verified per endpoint by a smoke test that renders at
divergence 0.1 and 0.9 and asserts the 0.9 output is spectrally further from the input.

## D012 — Behaviour when no candidate clears the floors **[v1.1]**
Before: M3 Step 3.4
Question: Abort, or fall back to the deterministic stem?
Resolution: `on_no_candidate: fallback_dry` by default (the dry stem is always note-faithful);
`strict` in CI. Logged in the manifest and flagged in the audit report.

## D013 — FAD estimator and corpus size **[v1.1]**
Before: M4 Step 4.x
Question: FAD is strongly biased at small N, and the design document's corpus is 5-30 tracks.
Which embedding, which estimator, and what is `tau`?
Resolution: [TBD — suggest CLAP or EnCodec embeddings rather than VGGish, and the FAD-infinity
extrapolation of Gui et al. (2024) rather than a raw small-N FAD; `tau` = median leave-one-out
FAD of the corpus against itself]

## D014 — Suno cover endpoint: is there a licensed programmatic path? **[v1.1]**
Before: M5 Step 5.1
Question: Does an official, ToS-compliant API exist for upload-and-cover at the account tier we
hold, and do the upload terms permit our material?
Resolution: [TBD — this is a legal/procurement question, not an engineering one. Until it is
answered "yes" in writing, the full-mix cohesion pass runs through the Stable Audio a2a endpoint
on the rough mix instead. Third-party unofficial Suno wrappers are out of scope: they breach ToS
and would put the provenance manifest in an indefensible state.]

## D015 — Full-mix scoring **[v1.1]**
Before: M4
Question: The design document scores "each stem (and the full mix)"; v1.0 scores only stems.
Which axes apply to the mix, and does the mix have hard floors?
Resolution: [TBD — suggest rhythm + FAD + CLAP on the mix, floors on rhythm only]

## D016 — Sections for inferred provenance **[v1.1]**
Before: M1 Step 1.1
Question: `SectionInfo` carries `tension` and `boundary_lvl`, which only the aimusic planner can
produce. What does `from_midi.py` emit?
Resolution: `boundary_lvl` from novelty-curve peak height, quantised to 0-3; `tension` null.
Consumers must treat `tension` as optional and `policy.py` must not require it.

## D008 — MIDI-GPT license gate
Before: M5 (breadth)
Question: MIDI-GPT weights are CC-BY-NC. Behind what config flag does that
branch sit, and how does CI assert it defaults to off?
Resolution: config flag midi_gpt: false; CI test asserts default config loads
with midi_gpt=False.
```

**Test:** `tests/test_bootstrap.py`:

```python
def test_package_importable():
    import m2a
    assert m2a.__version__ == "0.1.0"

def test_decisions_file_exists():
    import pathlib
    assert pathlib.Path("DECISIONS.md").exists()

def test_decisions_has_all_entries():
    text = pathlib.Path("DECISIONS.md").read_text()
    for d in [f"D{i:03d}" for i in range(1, 17)]:   # [v1.1] D001-D016
        assert d in text, f"{d} missing from DECISIONS.md"
```

**Exit:** `pip install -e ".[dev]"` succeeds. Bootstrap tests pass.

---

#### Step 0.2 — Contracts and schemas

**Deliverable:** `m2a/contracts.py` — every inter-stage data type.

These are frozen dataclasses for Python-side objects and a pydantic model for the `structure.json`
schema (because it validates external input).

```python
# m2a/contracts.py
from __future__ import annotations
from dataclasses import dataclass
from pydantic import BaseModel, Field, model_validator

# ── Python-side contracts (frozen dataclasses) ──────────────────────────────

@dataclass(frozen=True)
class RenderPackage:
    """The input contract. Everything downstream reads from this."""
    root: str           # run_<hash>/ directory
    midi_path: str      # score.mid inside root
    structure_path: str # structure.json inside root
    tuning_path: str    # tuning.json inside root
    manifest_path: str  # manifest.json inside root
    content_hash: str   # sha256 of midi + structure + tuning

@dataclass(frozen=True)
class Stem:
    """One rendered audio stem."""
    track: str          # matches structure.tracks[].name
    role: str           # drums | bass | comping | lead | pads
    path: str           # absolute path to .wav
    microtonal: bool
    peak_dbfs: float
    content_hash: str

@dataclass(frozen=True)
class Candidate:
    """One restyled audio candidate for a single stem."""
    stem: str           # track name
    audio_path: str
    prompt: str
    negative_prompt: str
    strength: float
    seed: int
    endpoint: str
    endpoint_version: str
    # [v1.1] a dict inside a frozen dataclass is mutable and unhashable — the freeze is
    # cosmetic. Store an ordered tuple of pairs; `dict(c.scores)` at the read site.
    scores: tuple[tuple[str, float], ...]
    accepted: bool
    reject_reason: str | None
    total_score: float = 0.0
    # [v1.1] every field the manifest needs to reproduce this candidate:
    request_hash: str = ""       # sha256 of (audio hash, prompt, neg, divergence, seed, endpoint_version)
    usd_estimate: float = 0.0

# ── External-input schema (pydantic, validates structure.json) ───────────────

class TrackInfo(BaseModel):
    name: str
    role: str           # drums | bass | comping | lead | pads
    program: int | None
    percussion: bool
    quantized: bool
    flat_velocity: bool
    onset_profile_16: list[float] = Field(default_factory=list)
    pitch_range: list[int] = Field(default_factory=list)
    microtonal: bool

class SectionInfo(BaseModel):
    label: str
    bars: list[int]     # [start_bar, end_bar]
    energy: float = Field(ge=0.0, le=1.0)
    tension: float = Field(ge=0.0, le=1.0)
    boundary_lvl: int = Field(ge=0)

class StructureJSON(BaseModel):
    schema_version: str = Field(alias="schema", default="midi2audio.structure/1")
    provenance: str     # inferred | planner | host
    source_hash: str
    edo: int = Field(ge=1)
    base_tuning: float
    tempo_map: list[list[float]]
    meter: list[list]
    key: list[dict]
    chords: list[dict]
    sections: list[SectionInfo]
    bar_table: list[list[float]]
    tracks: list[TrackInfo]

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def check_invariants(self) -> "StructureJSON":
        # [v1.1] raise ValueError, never assert: bare asserts are stripped under `python -O`,
        # and pydantic only converts ValueError/AssertionError into ValidationError reliably
        # when the message is preserved. A validator that silently vanishes in an optimised
        # build is worse than no validator.
        times = [row[1] for row in self.bar_table]
        if times != sorted(times):
            raise ValueError("bar_table must be monotonically increasing")
        idxs = [t.idx for t in self.tracks]
        if len(set(idxs)) != len(idxs):
            raise ValueError(f"duplicate track idx in structure.tracks: {idxs}")
        # Invariant 5: microtonal tracks must have safe method in tuning.json
        # (cross-file check performed by orchestrator at run time)
        return self
```

**Test:** `tests/test_contracts.py`:

```python
def test_structure_json_valid():
    data = load_fixture("eight_bar_structure.json")
    s = StructureJSON.model_validate(data)
    assert s.edo == 12
    assert len(s.tracks) == 4

def test_structure_json_rejects_non_monotonic_bar_table():
    data = load_fixture("eight_bar_structure.json")
    data["bar_table"] = [[0, 2.0], [1, 0.0]]  # reversed
    with pytest.raises(ValidationError):
        StructureJSON.model_validate(data)

def test_render_package_is_frozen():
    pkg = RenderPackage(root="r", midi_path="m", structure_path="s",
                        tuning_path="t", manifest_path="mf", content_hash="h")
    with pytest.raises(Exception):
        pkg.root = "other"

def test_candidate_rejected():
    c = Candidate(stem="drums", audio_path="/tmp/x.wav", prompt="", negative_prompt="",
                  strength=0.7, seed=1, endpoint="sa", endpoint_version="2.5",
                  scores={"rhythm": 0.4}, accepted=False, reject_reason="rhythm_floor")
    assert not c.accepted
    assert c.reject_reason == "rhythm_floor"
```

**Exit:** All contract tests pass.

---

#### Step 0.3 — Content-addressed artifact store

**Deliverable:** `m2a/artifacts.py` — cache key derivation and stage-output management.

```python
# m2a/artifacts.py
import hashlib, json, pathlib, shutil
from dataclasses import dataclass

def content_hash(*file_paths: str, config: dict, code_version: str) -> str:
    # [v1.1] stream instead of read_bytes(): reference corpora and stems are hundreds of MB,
    # and v1.0 loaded every one of them into RAM to hash it.
    h = hashlib.sha256()
    for p in sorted(file_paths):
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    h.update(json.dumps(config, sort_keys=True).encode())
    h.update(code_version.encode())
    return h.hexdigest()          # [v1.1] full digest; truncate only for directory names

class ArtifactStore:
    def __init__(self, work_dir: str):
        self.root = pathlib.Path(work_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: str, hash_: str) -> pathlib.Path:
        d = self.root / f"{stage}_{hash_}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def is_complete(self, stage: str, hash_: str) -> bool:
        return (self.root / f"{stage}_{hash_}" / "_complete").exists()

    def mark_complete(self, stage: str, hash_: str) -> None:
        (self.root / f"{stage}_{hash_}" / "_complete").touch()
```

**Test:** `tests/test_artifacts.py`:

```python
def test_same_inputs_same_hash(tmp_path):
    f = tmp_path / "f.mid"
    f.write_bytes(b"midi")
    h1 = content_hash(str(f), config={"a": 1}, code_version="0.1")
    h2 = content_hash(str(f), config={"a": 1}, code_version="0.1")
    assert h1 == h2

def test_different_config_different_hash(tmp_path):
    f = tmp_path / "f.mid"
    f.write_bytes(b"midi")
    h1 = content_hash(str(f), config={"a": 1}, code_version="0.1")
    h2 = content_hash(str(f), config={"a": 2}, code_version="0.1")
    assert h1 != h2

def test_cache_hit_detection(tmp_path):
    store = ArtifactStore(str(tmp_path))
    assert not store.is_complete("s1", "abc")
    store.mark_complete("s1", "abc")
    assert store.is_complete("s1", "abc")
```

**Exit:** Determinism and idempotency tests pass.

---

#### Step 0.4 — Fixture corpus

**Deliverable:** `tests/fixtures/` with six files used throughout all milestones.

| File | What it is | How to produce |
|---|---|---|
| `eight_bar.mid` | 4-track MIDI (drums/bass/comping/lead), 8 bars, 120 BPM, 4/4, 12-EDO, fully quantized, flat velocity | `scripts/generate_fixtures.py` using `mido` |
| `eight_bar_structure.json` | Valid `structure.json` for the fixture MIDI | Hand-authored to match; validated by `StructureJSON` |
| `eight_bar_tuning.json` | `{"edo": 12, "base_tuning": 60.0, "method": "direct_12"}` | Hand-authored |
| `eight_bar_manifest.json` | Minimal provenance record | Hand-authored |
| `five_sec_stem.wav` | 5 seconds of silence initially; replaced with real render in M1 | `soundfile.write` |
| `synthetic_host.wav` | 5 seconds of kick + bass at 120 BPM for single-ref testing in M6 | `scripts/generate_fixtures.py` |

`scripts/generate_fixtures.py`:

```python
import mido, soundfile as sf, numpy as np, json, pathlib

def make_eight_bar_midi(out: str, bpm: float = 120.0, tpb: int = 480) -> None:
    mid = mido.MidiFile(ticks_per_beat=tpb)
    us_per_beat = int(60_000_000 / bpm)
    tracks = {"drums": 9, "bass": 0, "comping": 1, "lead": 2}
    programs = {"bass": 33, "comping": 4, "lead": 81}
    for i, (name, ch) in enumerate(tracks.items()):
        t = mido.MidiTrack()
        t.name = name
        mid.tracks.append(t)
        # [v1.1] tempo and meter are global: emit them once, on track 0 only. v1.0 wrote
        # set_tempo into all four tracks, which is legal but makes pretty_midi's tempo-change
        # list report four coincident changes and complicates the tempo-map assertion.
        if i == 0:
            t.append(mido.MetaMessage("set_tempo", tempo=us_per_beat, time=0))
            t.append(mido.MetaMessage("time_signature", numerator=4, denominator=4, time=0))
        if name in programs:
            t.append(mido.Message("program_change", channel=ch,
                                  program=programs[name], time=0))
        # 8 bars x 4 beats, one note per beat
        # [v1.1] BUG FIX. v1.0 emitted note_on(time=0) + note_off(time=tpb-10) per beat, so the
        # delta between consecutive onsets was 470 ticks, not 480: the "120 BPM quantised"
        # fixture was actually 2 % sharp and drifted ~150 ms over 8 bars. Every quantisation
        # diagnosis, groove-offset and rhythm-F-measure test built on it would have been
        # measuring the fixture's drift. The 10-tick gap must be charged to the next note_on.
        for beat in range(32):
            pitch = {"drums": 36, "bass": 40, "comping": 60, "lead": 72}[name]
            t.append(mido.Message("note_on",  channel=ch, note=pitch,
                                  velocity=64, time=(0 if beat == 0 else 10)))
            t.append(mido.Message("note_off", channel=ch, note=pitch,
                                  velocity=0, time=tpb - 10))
    mid.save(out)

def make_silent_wav(out: str, seconds: float = 5.0, sr: int = 48000) -> None:
    sf.write(out, np.zeros(int(seconds * sr)), sr)

if __name__ == "__main__":
    fx = pathlib.Path("tests/fixtures")
    fx.mkdir(parents=True, exist_ok=True)
    make_eight_bar_midi(str(fx / "eight_bar.mid"))
    make_silent_wav(str(fx / "five_sec_stem.wav"))
    make_silent_wav(str(fx / "synthetic_host.wav"))
    print("Fixtures written.")
```

**Test:** `tests/test_fixtures.py`:

```python
def test_fixture_midi_parses():
    mid = mido.MidiFile("tests/fixtures/eight_bar.mid")
    names = [t.name for t in mid.tracks]
    assert "drums" in names and "bass" in names

def test_fixture_structure_validates():
    data = json.loads(Path("tests/fixtures/eight_bar_structure.json").read_text())
    s = StructureJSON.model_validate(data)
    assert len(s.tracks) == 4

def test_fixture_onsets_are_exactly_one_beat_apart():
    """[v1.1] regression test for the 470-vs-480 tick bug."""
    mid = mido.MidiFile("tests/fixtures/eight_bar.mid")
    for track in mid.tracks:
        onsets, clock = [], 0
        for msg in track:
            clock += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                onsets.append(clock)
        if len(onsets) > 1:
            deltas = {b - a for a, b in zip(onsets, onsets[1:])}
            assert deltas == {mid.ticks_per_beat}, deltas

def test_fixture_tuning_has_required_fields():
    data = json.loads(Path("tests/fixtures/eight_bar_tuning.json").read_text())
    assert "edo" in data and "method" in data
```

**Exit:** All fixture tests pass. Fixture files committed.

---

### M1 — Deterministic Spine (Weeks 2–3)

Goal: MIDI file in → expressivized MIDI → WAV stems out. Zero network calls, zero paid endpoints.
The only impure operations are disk I/O and the LLM groove-spec call (which is cached immediately).

---

#### Step 1.1 — Stage 0: MIDI analysis (`analysis/from_midi.py`)

**Deliverable:** `m2a/analysis/from_midi.py` — produce `structure.json` from a MIDI file.

Use `pretty_midi` for tempo/meter/note extraction and `music21` for key and chord inference:

```python
import pretty_midi, music21, json
from m2a.contracts import StructureJSON

def analyse_midi(midi_path: str, out_path: str) -> StructureJSON:
    pm = pretty_midi.PrettyMIDI(midi_path)

    tempo_map = _extract_tempo_map(pm)
    meter_map = _extract_meter_map(pm)
    key_map   = _infer_key(pm)          # music21 chroma template matching
    chords    = _infer_chords(pm)       # music21 chord analyser, per beat
    sections  = _detect_sections(pm)    # novelty detection on bar-level piano-roll
    bar_table = _build_bar_table(tempo_map, meter_map, pm.get_end_time())
    tracks    = [_analyse_track(inst, bar_table) for inst in pm.instruments]

    data = {
        "schema": "midi2audio.structure/1",
        "provenance": "inferred",
        "source_hash": _file_hash(midi_path),
        "edo": 12,           # default; caller overrides for microtonal MIDI
        "base_tuning": 60.0,
        "tempo_map": tempo_map,
        "meter": meter_map,
        "key": key_map,
        "chords": chords,
        "sections": sections,
        "bar_table": bar_table,
        "tracks": tracks,
    }
    structure = StructureJSON.model_validate(data)
    with open(out_path, "w") as f:
        json.dump(structure.model_dump(by_alias=True), f, indent=2)
    return structure
```

Key analysis functions to implement:

- `_extract_tempo_map(pm)` — `pm.get_tempo_changes()` → list of `[time_sec, bpm]`.
- `_extract_meter_map(pm)` — `pm.time_signature_changes` → list of `[bar_idx, "N/D"]`.
- `_infer_key(pm)` — build chroma, run `music21.analysis.discrete.KrumhanslSchmuckler`.
- `_infer_chords(pm)` — beat-aligned chroma, template-match against 24 major/minor triads + 7ths.
- `_detect_sections(pm)` — bar-level piano-roll embedding, self-similarity matrix, novelty curve peaks.
- `_analyse_track(inst, bar_table)` — role from GM program number (GM drum = percussion, bass 32–39, etc.), onset histogram over 16 positions, quantization fraction.
- `_build_bar_table(tempo_map, meter_map, duration)` — list of `[bar_idx, time_sec]`.

**Quantization diagnosis:** fraction of onsets within 10 ms of nearest 16th-note grid position. Above 0.85 → `quantized: true`. RMS velocity variance below threshold → `flat_velocity: true`.

**Test:** `tests/test_analysis_from_midi.py`:

```python
def test_analyses_fixture_midi():
    structure = analyse_midi("tests/fixtures/eight_bar.mid", "/tmp/s.json")
    assert structure.edo == 12
    assert len(structure.tracks) == 4
    assert all(t.quantized for t in structure.tracks)
    assert all(t.flat_velocity for t in structure.tracks)

def test_bar_table_monotonic():
    structure = analyse_midi("tests/fixtures/eight_bar.mid", "/tmp/s.json")
    times = [r[1] for r in structure.bar_table]
    assert times == sorted(times)

def test_tempo_close_to_120():
    structure = analyse_midi("tests/fixtures/eight_bar.mid", "/tmp/s.json")
    assert abs(structure.tempo_map[0][1] - 120.0) < 2.0

def test_four_tracks_with_correct_roles():
    structure = analyse_midi("tests/fixtures/eight_bar.mid", "/tmp/s.json")
    roles = {t.role for t in structure.tracks}
    assert roles == {"drums", "bass", "comping", "lead"}
```

**Exit:** `from_midi` produces a valid `StructureJSON` from the fixture MIDI. All four tests pass.

---

#### Step 1.2 — Stage 1: GrooveSpec schema (`groove/spec.py`)

**Deliverable:** `m2a/groove/spec.py` — pydantic schema for the groove specification.

This schema is the LLM output contract. Every groove spec produced by `groove/author.py` is
validated against it before being applied. Invalid specs are rejected and regenerated.

```python
from pydantic import BaseModel, Field

class SwingSpec(BaseModel):
    ratio: float = Field(ge=0.5, le=0.75,
        description="0.5=straight, 0.66=triplet swing")
    subdivision: int = Field(default=8,
        description="8 or 16 — which subdivision swings")
    applies_to: list[str] = Field(default_factory=list,
        description="track names: drums, bass, comping, lead")

class MicrotimingSpec(BaseModel):
    by_position_16: list[float] | None = Field(default=None,
        description="16 ms offsets indexed by 16th-note position in bar")
    global_offset: float = Field(default=0.0,
        description="ms — positive = late, negative = ahead")
    jitter_sd: float = Field(default=2.0, ge=0.0, le=10.0,
        description="standard deviation of random jitter in ms")

class VelocitySpec(BaseModel):
    accent_map_16: list[float] | None = Field(default=None,
        description="16 multipliers 0.0–1.0 for metrical accent")
    ghost_notes: dict | None = None
    phrase_arc: dict | None = None

class ArticulationSpec(BaseModel):
    duration_scale: float = Field(default=1.0, ge=0.1, le=2.0)
    strum_ms: float = Field(default=0.0, ge=0.0)

class GrooveSpec(BaseModel):
    swing: SwingSpec | None = None
    microtiming_ms: dict[str, MicrotimingSpec] = Field(default_factory=dict,
        description="keys are track names")
    velocity: VelocitySpec | None = None
    anticipation: dict | None = None
    tempo_curve: dict | None = None
    articulation: dict[str, ArticulationSpec] = Field(default_factory=dict)
    # Repo-specific extensions
    section_overrides: dict[str, "GrooveSpec"] | None = None
    tension_coupling: dict | None = None
    edo_safe: bool = Field(default=False,
        description="If true, applicator asserts no 12-EDO pitch operations are applied")
    groove_id_hint: int | None = None

GrooveSpec.model_rebuild()  # resolves forward ref
```

**Test:** `tests/test_groove_spec.py`:

```python
def test_minimal_spec():
    s = GrooveSpec()
    assert s.swing is None

def test_swing_ratio_bounds():
    with pytest.raises(ValidationError):
        GrooveSpec(swing=SwingSpec(ratio=0.3, applies_to=["drums"]))

def test_jitter_sd_bounds():
    with pytest.raises(ValidationError):
        GrooveSpec(microtiming_ms={"drums": MicrotimingSpec(jitter_sd=20.0)})

def test_round_trip():
    spec = GrooveSpec(
        swing=SwingSpec(ratio=0.58, subdivision=8, applies_to=["drums", "bass"]),
        microtiming_ms={"drums": MicrotimingSpec(by_position_16=[0]*16, jitter_sd=2.0)},
    )
    data = spec.model_dump()
    spec2 = GrooveSpec.model_validate(data)
    assert spec == spec2
```

**Exit:** Schema validates known-good specs; rejects known-invalid specs. Round-trip passes.

---

#### Step 1.3 — Stage 1: LLM groove author (`groove/author.py`)

**Deliverable:** `m2a/groove/author.py` — call an LLM to produce a `GrooveSpec`, validate it, retry on failure.

```python
import json
from m2a.groove.spec import GrooveSpec
from pydantic import ValidationError

SYSTEM_PROMPT = """
You are a music production expert. Given a MIDI analysis and style intent,
emit a JSON groove specification. Output ONLY valid JSON matching the
GrooveSpec schema. No prose, no markdown fences.
"""

def author_groove_spec(
    structure: dict,
    style_intent: str,
    llm_fn,           # callable(system, user) -> str
    max_retries: int = 3,
) -> GrooveSpec:
    user_prompt = f"""
MIDI analysis:
{json.dumps(structure, indent=2)}

Style intent: {style_intent}

Emit a GrooveSpec JSON. Track names are: {[t['name'] for t in structure['tracks']]}.
"""
    last_error = None
    for attempt in range(max_retries):
        raw = llm_fn(SYSTEM_PROMPT, user_prompt)
        try:
            data = json.loads(raw)
            spec = GrooveSpec.model_validate(data)
            return spec
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            user_prompt += f"\n\nPrevious attempt was invalid: {e}\nTry again."
    raise RuntimeError(f"Failed to author valid GrooveSpec after {max_retries} attempts: {last_error}")
```

The LLM call is the **only impure operation in groove/**. Its output is written to
`{stage_dir}/groove_spec.json` immediately so reruns use the cached version.

**Test:** `tests/test_groove_author.py`:

```python
def test_valid_llm_response_accepted():
    def mock_llm(sys, usr):
        return '{"swing": {"ratio": 0.58, "subdivision": 8, "applies_to": ["drums"]}}'
    spec = author_groove_spec(fixture_structure, "funk", mock_llm)
    assert spec.swing.ratio == 0.58

def test_invalid_json_retried():
    calls = []
    def mock_llm(sys, usr):
        calls.append(usr)
        if len(calls) < 2:
            return "not json"
        return '{"swing": null}'
    spec = author_groove_spec(fixture_structure, "funk", mock_llm, max_retries=3)
    assert len(calls) == 2

def test_max_retries_raises():
    def bad_llm(sys, usr):
        return "always bad"
    with pytest.raises(RuntimeError):
        author_groove_spec(fixture_structure, "funk", bad_llm, max_retries=2)
```

**Exit:** Author retries on validation failure; raises after `max_retries`; never patches invalid specs.

---

#### Step 1.4 — Stage 1: Groove applicator (`groove/apply.py`)

**Deliverable:** `m2a/groove/apply.py` — the pure function
`apply_groove(midi_path, spec, structure, rng_seed) → (out_midi_path, DiffReport)`.

This is the most algorithmically dense module. Implement transformations in this order:

1. **Swing** — **[v1.1] corrected formula.** Delay the off-beat member of each swung pair.
   The pair spans one beat when `subdivision == 8` and half a beat when `subdivision == 16`:

   ```
   delay_ms = (ratio - 0.5) * pair_duration_ms
   ```

   v1.0's `* 2` factor was wrong: at `ratio = 0.66` it placed the off-eighth at 82 % of the
   beat instead of 66 %, i.e. roughly a dotted-eighth feel labelled as triplet swing. There is
   no factor of two — `ratio` already *is* the position of the second note as a fraction of the
   pair.

   `pair_duration_ms` must come from the **local** tempo at that note (interpolated from
   `tempo_map`), not `tempo_map[0][1]`.

   **[v1.1] Tempo scaling.** The paper requires swing to shrink as tempo rises. Implement the
   empirical rule from the jazz-timing literature (Friberg & Sundström 2002): performers hold
   the *short* (off-beat) note roughly constant in absolute duration — around 100 ms — rather
   than holding the ratio constant, so the ratio collapses toward 0.5 at fast tempi:

   ```
   short_ms   = (1.0 - ratio) * beat_ms_at(reference_bpm)      # reference_bpm default 120
   ratio_eff  = clamp(1.0 - short_ms / beat_ms_local, 0.5, ratio)
   ```

   Expose `tempo_scaling: bool = True` and `reference_bpm: float = 120.0` on `SwingSpec`.
   Test: at 240 BPM with `ratio = 0.66`, `ratio_eff` must be strictly between 0.5 and 0.60.

2. **Microtiming by metrical position** — **[v1.1] meter-aware grid.** Positions are indexed
   within the bar on a grid derived from the meter and the spec's new `grid` field, **not** a
   hardcoded 16. `4/4` at `grid=16` has 16 positions; `7/8` at `grid=16` has 14; a shuffle or
   triplet feel uses `grid=24` (the design document says "16th- **or 24th-**grid profile", and
   v1.0 dropped the 24 case entirely, which also makes swing and microtiming fight each other
   on triplet material). The applicator computes
   `positions_per_bar = grid * numerator / denominator * (4 / grid_reference)` via a single
   helper `positions_per_bar(meter, grid)` and **rejects** any spec whose `by_position` length
   disagrees. Without this, every odd-meter profile in `config/profiles/prog.yaml` silently
   applies the wrong offsets from bar 1.

   For each note: look up its position index, convert the ms offset to ticks **at the local
   tempo**, apply it, then add Gaussian jitter with `jitter_sd` (the paper caps human noise at
   sigma <= 4 ms; `SpecValidator` warns above that).
3. **Velocity accent map:** multiply velocity by `accent_map_16[pos]`, clamp to [1, 127].
4. **Ghost notes:** insert notes with `prob` probability at flagged positions, at `vel_scale × normal_velocity`.
5. **Phrase arcs:** apply a time-varying velocity envelope over phrases.
6. **Anticipations:** at `chord_id` transition beats (from `structure.chords`), push comping onsets early with given probability.
7. **Tempo curve:** modify the MIDI tempo map, applied to all tracks jointly.
8. **Articulation:** scale note duration, apply strum delay for chords.

**[v1.1] Role vs. track-name resolution — fix the ambiguity before writing the applicator.**
v1.0 is inconsistent: `SwingSpec.applies_to` is documented as "track names: drums, bass,
comping, lead", `microtiming_ms` keys are documented as "track names", and the design document
specifies everything **per role**. These collide the moment a piece has two percussion tracks or
a doubled lead. Resolution (record as D009):

* Groove-spec keys are always **roles**. The LLM only ever sees and emits roles.
* The applicator expands roles to track indices through
  `role_map = {role: [idx, ...]}` built from `structure.tracks`.
* A per-track override is possible but must be explicit:
  `microtiming_ms: {"lead": ..., "track:3": ...}`, where a `track:` prefix bypasses the role map.
* `DiffReport` records both role and track idx for every applied offset.

Use `mido` for MIDI I/O. Seed all random draws with `rng_seed` via Python's `random.Random`,
one generator **per (track, transformation)** so that adding a track does not change the jitter
drawn for existing tracks — otherwise a one-track edit invalidates every cached stem.

```python
@dataclass(frozen=True)
class DiffReport:
    applied_timing_offsets: tuple    # (track, position, offset_ticks) per note
    applied_velocity_changes: tuple  # (track, original, final) per note
    ghost_notes_inserted: int
    bar_crossing_flags: tuple        # note ids that crossed bar lines
    rule_violations: tuple           # e.g. collision from anticipation
```

**Invariants the applicator must maintain:**

- Note count preserved (except flagged ghost insertions counted in `DiffReport`).
- No note crosses a bar line without being flagged.
- `edo_safe=True` ⇒ no operation indexes into a 12-EDO pitch table.
- Applied timing offset for each `(role, metrical_position)` equals the spec value within 1 tick.

**Test:** `tests/test_groove_apply.py` — use `hypothesis` for property-based tests:

```python
from hypothesis import given, strategies as st, settings

@given(seed=st.integers(min_value=0, max_value=2**32-1))
@settings(max_examples=50)
def test_note_count_preserved(seed):
    spec = GrooveSpec()  # neutral spec
    _, report = apply_groove(FIXTURE_MIDI, spec, FIXTURE_STRUCTURE, seed)
    original = count_notes(FIXTURE_MIDI)
    result = count_notes(RESULT_MIDI)
    assert result == original + report.ghost_notes_inserted

def test_offsets_match_spec_within_one_tick():
    spec = GrooveSpec(microtiming_ms={
        "drums": MicrotimingSpec(by_position_16=[0]*7 + [+6.0] + [0]*8, jitter_sd=0.0)
    })
    out, report = apply_groove(FIXTURE_MIDI, spec, FIXTURE_STRUCTURE, seed=0)
    drum_offsets = [o for o in report.applied_timing_offsets if o[0] == "drums"]
    for track, pos, offset_ticks in drum_offsets:
        if pos == 7:
            # [v1.1] local tempo, not tempo_map[0][1] — the global-tempo version silently
            # passes on the constant-tempo fixture and silently fails on anything real.
            bpm = tempo_at(FIXTURE_STRUCTURE, position_time_sec(track, pos))
            expected_ticks = ms_to_ticks(6.0, bpm, 480)
            assert abs(offset_ticks - expected_ticks) <= 1

def test_neutral_velocity_map_no_change():
    spec = GrooveSpec(velocity=VelocitySpec(accent_map_16=[1.0] * 16))
    out, _ = apply_groove(FIXTURE_MIDI, spec, FIXTURE_STRUCTURE, seed=0)
    assert velocities(out) == velocities(FIXTURE_MIDI)

def test_deterministic_given_same_seed():
    spec = make_expressive_spec()
    out1, _ = apply_groove(FIXTURE_MIDI, spec, FIXTURE_STRUCTURE, seed=42)
    out2, _ = apply_groove(FIXTURE_MIDI, spec, FIXTURE_STRUCTURE, seed=42)
    assert read_bytes(out1) == read_bytes(out2)
```

**Exit:** All property-based tests pass. Applicator is provably deterministic.

---

#### Step 1.5 — Stage 2: FluidSynth renderer (`render/fluidsynth_r.py` + `registry.py`)

**Deliverable:** Render expressive MIDI to per-track WAV stems using FluidSynth.

```python
# m2a/render/fluidsynth_r.py
import subprocess, pathlib

class FluidSynthRenderer:
    def supports(self, role: str, microtonal: bool) -> bool:
        return not microtonal  # microtonal routing added in M1.5

    def render(self, midi_path: str, track_name: str,
               soundfont: str, out_path: str) -> "Stem":
        # 1. Extract single-track MIDI (mido: keep only the target track)
        single_track_mid = _extract_track(midi_path, track_name)
        # 2. Shell out to fluidsynth CLI.
        # [v1.1] --reverb=0 --chorus=0: the deterministic layer must be dry. Effects here get
        # baked in before the restyle pass and re-reverbed in STAGE 6, which is where the
        # "low-end mud / smeared room" failure mode starts.
        subprocess.run([
            "fluidsynth", "-ni", soundfont, single_track_mid,
            "-F", out_path, "-r", "48000", "-g", "0.5",
            "--reverb=0", "--chorus=0",
        ], check=True)
        # [v1.1] -g is an input gain, not a peak guarantee. Normalise explicitly to the
        # -12 dBFS peak headroom the design document asks for.
        _normalise_peak(out_path, target_dbfs=-12.0)
        peak = _measure_peak_dbfs(out_path)
        return Stem(track=track_name, role=..., path=out_path,
                    microtonal=False, peak_dbfs=peak,
                    content_hash=_file_hash(out_path))
```

Also emit per run: click track, stereo rough mix, bar-boundary timestamp table
(`bar_table.json` — copied from `structure.json`, no re-derivation).

**Test:** `tests/test_render_fluidsynth.py`:

```python
@pytest.mark.slow
def test_renders_four_stems(tmp_path):
    stems = render_all_stems(FIXTURE_PACKAGE, str(tmp_path))
    assert len(stems) == 4
    for stem in stems:
        assert pathlib.Path(stem.path).stat().st_size > 0

def test_stems_are_48khz(tmp_path):
    stems = render_all_stems(FIXTURE_PACKAGE, str(tmp_path))
    for stem in stems:
        _, sr = sf.read(stem.path)
        assert sr == 48000

def test_peak_within_headroom(tmp_path):
    # [v1.1] the paper specifies -12 dBFS peaks; v1.0 asserted only < -6 dBFS, which any
    # near-clipping render passes.
    stems = render_all_stems(FIXTURE_PACKAGE, str(tmp_path))
    for stem in stems:
        assert -12.5 < stem.peak_dbfs < -11.5

def test_render_is_deterministic(tmp_path):
    stems1 = render_all_stems(FIXTURE_PACKAGE, str(tmp_path / "r1"))
    stems2 = render_all_stems(FIXTURE_PACKAGE, str(tmp_path / "r2"))
    for s1, s2 in zip(stems1, stems2):
        # [v1.1] compare PCM, not file bytes: WAV metadata chunks can carry timestamps.
        a, _ = sf.read(s1.path); b, _ = sf.read(s2.path)
        np.testing.assert_array_equal(a, b)
```

**Exit:** 4 WAV stems produced from fixture MIDI. Each passes 48 kHz and headroom tests.

---

#### Step 1.6 — Orchestrator with caching (`orchestrator.py`)

**Deliverable:** `m2a/orchestrator.py` — the stage DAG runner for the deterministic spine (S0→S1→S2).

```python
class Orchestrator:
    def __init__(self, config: "AudioConfig", work_dir: str):
        self.config = config
        self.store = ArtifactStore(work_dir)

    def run_spine(self, package: RenderPackage) -> "SpineResult":
        # Stage 0: analyse or load structure
        s0_hash = content_hash(package.midi_path,
                               config={"stage": "s0"}, code_version=VERSION)
        if not self.store.is_complete("s0", s0_hash):
            structure = analyse_midi(package.midi_path,
                                     str(self.store.stage_dir("s0", s0_hash) / "structure.json"))
            self.store.mark_complete("s0", s0_hash)
        else:
            structure = load_structure(self.store.stage_dir("s0", s0_hash) / "structure.json")

        # Stage 1: expressivize
        s1_hash = content_hash(package.midi_path, str(self.config.groove),
                               config={"stage": "s1"}, code_version=VERSION)
        if not self.store.is_complete("s1", s1_hash):
            spec = author_groove_spec(structure, self.config.groove.style_intent, llm_fn)
            expressive_midi = apply_groove(package.midi_path, spec, structure,
                                           self.config.groove.seed)
            self.store.mark_complete("s1", s1_hash)
        ...

        # Stage 2: render
        ...
        return SpineResult(structure=structure, expressive_midi=expressive_midi, stems=stems)
```

**Test:** `tests/test_orchestrator.py`:

```python
def test_spine_produces_stems(tmp_path):
    result = Orchestrator(default_config(), str(tmp_path)).run_spine(FIXTURE_PACKAGE)
    assert len(result.stems) == 4

def test_cache_hit_skips_render(tmp_path, mocker):
    orch = Orchestrator(default_config(), str(tmp_path))
    orch.run_spine(FIXTURE_PACKAGE)
    spy = mocker.spy(FluidSynthRenderer, "render")
    orch.run_spine(FIXTURE_PACKAGE)  # second run
    spy.assert_not_called()

def test_config_change_invalidates_s1_but_not_s0(tmp_path, mocker):
    orch1 = Orchestrator(default_config(), str(tmp_path))
    orch1.run_spine(FIXTURE_PACKAGE)
    spy_s0 = mocker.spy(analyse_midi, "__call__")
    orch2 = Orchestrator(config_with_different_groove(), str(tmp_path))
    orch2.run_spine(FIXTURE_PACKAGE)
    spy_s0.assert_not_called()   # S0 was cached
```

**Exit:** Orchestrator chains correctly. Cache hit/miss behavior verified. End-to-end: MIDI in → stems out.

---

#### Step 1.7 — CLI (`scripts/run_pipeline.py`)

**Deliverable:** command-line entry point.

```python
# scripts/run_pipeline.py
import argparse, pathlib
from m2a.orchestrator import Orchestrator
from m2a.config import AudioConfig

parser = argparse.ArgumentParser(description="midi2audio pipeline")
sub = parser.add_subparsers(dest="command")

# analyse: MIDI → structure.json
p_analyse = sub.add_parser("analyse")
p_analyse.add_argument("midi", help="Input MIDI file")
p_analyse.add_argument("--out", default="./output")

# expressivize: MIDI → expressive MIDI
p_expr = sub.add_parser("expressivize")
p_expr.add_argument("midi", help="Input MIDI file")
p_expr.add_argument("--structure", help="Existing structure.json (skip inference)")
p_expr.add_argument("--groove-spec", help="Existing groove spec JSON (skip LLM)")
p_expr.add_argument("--profile", help="Style profile YAML")
p_expr.add_argument("--out", default="./output")

# render-audio: full pipeline
p_render = sub.add_parser("render-audio")
p_render.add_argument("midi", help="Input MIDI file")
p_render.add_argument("--structure", help="Existing structure.json")
p_render.add_argument("--profile", default="config/default.yaml")
p_render.add_argument("--refs", nargs="*", help="Reference audio files")
p_render.add_argument("--budget-calls", type=int, default=200)
p_render.add_argument("--out", default="./output")
```

**Test:** `tests/test_cli.py`:

```python
def test_analyse_command(tmp_path):
    result = subprocess.run([
        "python", "scripts/run_pipeline.py", "analyse",
        "tests/fixtures/eight_bar.mid", "--out", str(tmp_path)
    ], capture_output=True)
    assert result.returncode == 0
    assert (tmp_path / "structure.json").exists()
```

**Exit:** `analyse` and `expressivize` subcommands work on fixture data. Full `render-audio` not yet wired (M3).

---

### M1.5 — Microtonal Path (Week 4)

---

#### Step 1.8 — EDO tuning utilities (`render/tuning.py`)

**Deliverable:** `m2a/render/tuning.py` — EDO-to-tuning-format converters. These are pure functions.

```python
import math

def to_scala(edo: int, base_hz: float = 440.0) -> str:
    """Generate a Scala .scl file string for the given EDO."""
    lines = [f"! {edo}-edo.scl",
             f"{edo}-equal temperament",
             f" {edo}"]
    for step in range(1, edo + 1):
        lines.append(f" {step * 1200.0 / edo:.6f}")
    return "\n".join(lines)

def step_to_cents(step: int, edo: int) -> float:
    return step * (1200.0 / edo)

def mts_tuning_table(edo: int, base_tuning_midi: int = 60) -> bytes:
    """Generate an MTS bulk tuning dump SysEx for the given EDO."""
    ...

def channel_rotation_plan(notes: list, bend_range_semitones: float) -> dict:
    """Assign MPE channels to notes to avoid simultaneous same-channel pitch bends."""
    ...
```

**Test:** `tests/test_tuning.py`:

```python
def test_scala_12_edo_last_line():
    s = to_scala(12)
    assert "1200.000000" in s.splitlines()[-1]

def test_scala_19_edo_step_size():
    s = to_scala(19)
    lines = s.splitlines()
    first_step = float(lines[4])
    assert abs(first_step - 63.157895) < 0.0001

def test_step_to_cents_perfect_fifth_12():
    assert abs(step_to_cents(7, 12) - 700.0) < 0.001

def test_step_to_cents_fifth_19():
    assert abs(step_to_cents(11, 19) - 694.737) < 0.01
```

**Exit:** All EDO utility tests pass.

---

#### Step 1.9 — Tuning check scorer (`scoring/tuning_check.py`)

**Deliverable:** `m2a/scoring/tuning_check.py` — f0-track audio and measure cents deviation from the intended EDO lattice.

```python
import librosa, numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class TuningReport:
    mean_cents: float
    p95_cents: float
    voiced_frames: int
    passes: bool

# [v1.1] IMPORTANT DESIGN CHANGE. v1.0 measured deviation from the *nearest* EDO step, which
# is a weak test: in 19-EDO the step is 63.2 cents, so arbitrary 12-EDO audio lands within
# 31.6 cents of some step and averages ~15.8 cents error — right on the proposed floor. The
# check barely separates the two cases it exists to separate. Measure deviation from the
# **intended** pitch instead: the expressive MIDI already knows which scale degree each note
# is, so pass the intended f0 contour and compare against it. Fall back to nearest-step only
# when no note reference is available (e.g. a full-mix pass).
def tuning_check(audio_path: str, edo: int, base_tuning_midi: float,
                 intended_f0: "np.ndarray | None" = None,
                 floor_p95_cents: float = 15.0) -> TuningReport:
    y, sr = librosa.load(audio_path, sr=None, mono=True)
    f0, voiced, _ = librosa.pyin(y, fmin=librosa.midi_to_hz(40),
                                 fmax=librosa.midi_to_hz(90), sr=sr)
    base_hz = librosa.midi_to_hz(base_tuning_midi)
    deviations = []
    for freq in f0[voiced]:
        if freq <= 0:
            continue
        cents_from_base = 1200 * np.log2(freq / base_hz)
        step = round(cents_from_base / (1200.0 / edo))
        expected = step * (1200.0 / edo)
        deviations.append(abs(cents_from_base - expected))
    if not deviations:
        return TuningReport(0.0, 0.0, 0, passes=True)
    return TuningReport(
        mean_cents=float(np.mean(deviations)),
        p95_cents=float(np.percentile(deviations, 95)),
        voiced_frames=len(deviations),
        passes=float(np.percentile(deviations, 95)) < floor_p95_cents,
    )
```

**Test:** `tests/test_tuning_check.py`:

```python
def test_exact_edo_tones_pass(tmp_path):
    # Synthesize pure tones at exact 19-EDO steps
    wav = synthesize_19edo_tones(tmp_path)
    report = tuning_check(wav, edo=19, base_tuning_midi=60.0)
    assert report.p95_cents < 2.0
    assert report.passes

def test_12_edo_tones_fail_19_edo_check(tmp_path):
    # [v1.1] scored against the *intended* 19-EDO contour, where 12-EDO content is unambiguously
    # wrong (errors up to 31.6 cents, mean ~16), not against the nearest 19-EDO step.
    wav = synthesize_12edo_scale(tmp_path)
    report = tuning_check(wav, edo=19, base_tuning_midi=60.0,
                          intended_f0=intended_contour_19edo())
    assert report.p95_cents > 15.0
    assert not report.passes

def test_nearest_step_fallback_is_documented_as_weak(tmp_path):
    """[v1.1] Guards the claim above: without an intended contour the check is near-chance
    for large EDOs, so the fallback path must not be used as a hard gate."""
    wav = synthesize_12edo_scale(tmp_path)
    report = tuning_check(wav, edo=19, base_tuning_midi=60.0, intended_f0=None)
    assert report.gate_eligible is False

def test_monotonicity(tmp_path):
    """Progressively detuned tones should score progressively worse."""
    scores = []
    for detune_cents in [0, 5, 10, 20]:
        wav = synthesize_detuned_tone(tmp_path, detune_cents)
        r = tuning_check(wav, edo=12, base_tuning_midi=60.0)
        scores.append(r.p95_cents)
    assert scores == sorted(scores)
```

**Exit:** Tuning check distinguishes 12-EDO from 19-EDO content. Monotonicity holds.

---

#### Step 1.10 — Microtonal renderer routing

**Deliverable:** Extend `render/registry.py` to route microtonal tracks.

```python
# m2a/render/registry.py
class RendererRegistry:
    def __init__(self, config):
        self.config = config
        self._renderers = {
            ("drums",   False): FluidSynthRenderer(config),
            ("bass",    False): SfizzRenderer(config),
            ("comping", False): SfizzRenderer(config),
            ("lead",    False): SfizzRenderer(config),
            ("drums",   True):  FluidSynthRenderer(config),  # unpitched — no concern
            ("bass",    True):  SfizzSclRenderer(config),
            ("comping", True):  SfizzSclRenderer(config),
            # [v1.1] Caveat on this row. MIDI-DDSP's note encoder takes *integer* MIDI pitch;
            # feeding it a 19-EDO part loses the microtonality at the front door. Tuning-exact
            # rendering requires bypassing the note encoder and driving the DDSP **synthesis**
            # stage with an explicit f0 contour computed from the EDO lattice, using the
            # expression module only for envelope/vibrato. Implement `MidiDdspRenderer` in that
            # mode from the start, or the M1.5 exit criterion ("19-EDO fixture renders within
            # the tuning-check floor") cannot be met.
            ("lead",    True):  MidiDdspRenderer(config, mode="f0_contour"),
        }

    def get(self, role: str, microtonal: bool) -> Renderer:
        key = (role, microtonal)
        if key not in self._renderers:
            raise ValueError(f"No renderer for ({role}, microtonal={microtonal})")
        return self._renderers[key]
```

**Test:**

```python
def test_routing_table_complete():
    registry = RendererRegistry(default_config())
    for role in ["drums", "bass", "comping", "lead"]:
        for micro in [False, True]:
            r = registry.get(role, micro)
            assert r is not None

def test_19_edo_lead_uses_ddsp():
    registry = RendererRegistry(default_config())
    r = registry.get("lead", microtonal=True)
    assert isinstance(r, MidiDdspRenderer)
```

**Exit:** Registry routes correctly. 19-EDO fixture renders within tuning check floor.

---

### M2 — Scoring (Weeks 5–6)

**Sequencing rule: all five scorers must be tested with monotonicity assertions before M3 begins.**

---

#### Step 2.1 — Rhythm scorer (`scoring/rhythm.py`)

Onset F-measure between candidate audio and expressive MIDI ground truth.

```python
import mir_eval, numpy as np

def rhythm_score(audio_path: str, expressive_midi_path: str, role: str,
                 tolerance_ms: float = 50.0) -> float:
    tol = 25.0 if role == "drums" else tolerance_ms
    cand = onset_detect_audio(audio_path)                       # librosa onset_detect
    ref  = midi_note_on_times(expressive_midi_path)

    # [v1.1] FIX 1 — collapse simultaneous note-ons. A four-note comping chord is ONE audio
    # onset but FOUR note-ons; v1.0 capped recall at 25 % on any polyphonic stem, so every
    # comping and pad candidate would have failed the 0.85 rhythm floor forever and the search
    # would have found "no acceptable candidate" on a perfect render.
    ref = collapse_simultaneous(ref, eps_ms=15.0)

    # [v1.1] FIX 2 — align first. The design document notes generative endpoints shift audio by
    # tens of ms (STAGE 6 step 1 exists precisely to undo this). Scoring before alignment
    # measures endpoint latency, not musical timing. Estimate the shift by cross-correlating
    # onset-strength envelopes, apply it, then score. Record the shift in the manifest; a shift
    # above ~200 ms is itself a rejection reason (the endpoint re-barred the material).
    shift = estimate_shift_sec(audio_path, expressive_midi_path, max_shift_sec=0.25)
    cand = np.asarray(cand) - shift

    # [v1.1] FIX 3 — use mir_eval's matched F-measure (optimal bipartite matching in the
    # window) rather than a hand-rolled greedy pairing, which double-counts near ties.
    f, _p, _r = mir_eval.onset.f_measure(np.asarray(ref), cand, window=tol / 1000.0)
    return float(f)
```

**Monotonicity test corruptions:**

| Corruption | How | Expected |
|---|---|---|
| Shift all onsets +30 ms | `mido` time offset | F-measure drops |
| Shift +60 ms | same | Drops further |
| Drop 20% of notes | remove random note-ons | Recall drops |
| Add 20% random extra onsets | inject random note-ons | Precision drops |

Assert each level strictly worse than the previous.

---

#### Step 2.2 — Note fidelity scorer (`scoring/notes.py`)

```python
def note_fidelity(audio_path: str, dry_stem_path: str,
                  expressive_midi_path: str, role: str, microtonal: bool) -> float:
    # [v1.1] FIX 1 — not applicable to percussion. basic-pitch on a drum stem returns noise;
    # v1.0 would have scored every drum candidate on a meaningless number. Return None and let
    # combine.py renormalise the remaining weights (drums carry ~0.05 notes weight anyway).
    if role == "drums":
        return None

    # [v1.1] FIX 2 — basic-pitch quantises to 12-TET semitone bins, so it cannot validate a
    # 19-EDO stem: a correct microtonal render scores as a wrong 12-EDO one. Route microtonal
    # material to an f0-contour comparison instead and let tuning_check carry the pitch gate.
    if microtonal:
        return f0_contour_similarity(audio_path, intended_contour(expressive_midi_path))

    f1 = note_f1(basic_pitch_transcribe(audio_path), load_midi_notes(expressive_midi_path))

    # [v1.1] FIX 3 — the design document specifies chroma cosine "between candidate and the
    # DRY STEM", not candidate vs MIDI. The dry stem is the like-for-like signal; comparing
    # audio chroma against a synthesised-from-MIDI chroma adds a rendering artefact to every
    # measurement.
    chroma_sim = cqt_chroma_cosine(audio_path, dry_stem_path)
    return 0.7 * f1 + 0.3 * chroma_sim
```

**Monotonicity test corruptions:** transpose +2 semitones, drop 10% of notes, quantize groove away.

**[v1.1] Additional corruption:** re-voice one chord (same root, different inversion). Note
fidelity must drop; chroma cosine must *not* drop much. This is the test that proves the two
terms are measuring different things and justifies keeping both.

---

#### Step 2.3 — CLAP similarity scorer (`scoring/clap_sim.py`)

```python
def clap_similarity(audio_path: str, prompt: str, negative_prompt: str) -> float:
    audio_emb = clap_encode_audio(audio_path)
    pos_emb = clap_encode_text(prompt)
    neg_emb = clap_encode_text(negative_prompt)
    return cosine(audio_emb, pos_emb) - cosine(audio_emb, neg_emb)
```

**Test:** Drum stem scores higher against "funk drum kit" than "string quartet". Negative prompt subtraction increases score on matching stems.

---

#### Step 2.4 — FAD scorer (`scoring/fad.py`)

```python
def fad_score(audio_path: str, reference_dir: str) -> float:
    candidate_embs = embed_windows(audio_path)     # CLAP or VGGish
    reference_embs = embed_directory(reference_dir)
    return frechet_distance(candidate_embs, reference_embs)
```

**Test:** FAD of stem against itself ≈ 0. FAD against sonically different reference > FAD against similar.

---

#### Step 2.5 — Combined scorer with hard floors (`scoring/combine.py`)

**[v1.1] Two blocking bugs in v1.0 are fixed here.**

**Bug A — the FAD sign.** Frechet Audio Distance is a *distance*: lower is better. v1.0 added it
into the weighted sum with a positive weight of 0.20, so the objective actively preferred
candidates that sounded *less* like the reference corpus. Since FAD is also unbounded and
typically an order of magnitude larger than the other axes (which live in [0, 1]), it would have
dominated the sum outright — the search would have been an anti-optimiser wearing a scorer's
coat. Every axis must be mapped to "higher is better, in [0, 1]" **before** the sum.

**Bug B — weights that do not sum to 1 and a `tuning` weight that is never used.** v1.0's
weights sum to 0.90 over the four axes actually summed, and `tuning` (0.10) is declared but
excluded from the sum. Totals are then not comparable across roles or runs.

**[v1.1] Per-role weights.** The design document is explicit that weights are per role — "drums
weight rhythm heavily; pads weight timbre and reference proximity; leads weight note fidelity" —
and v1.0 has a single global table. Restored:

```python
import numpy as np

# ── Axis normalisation: every axis -> [0, 1], higher is better ───────────────
def norm_rhythm(f: float) -> float:              # already 0..1
    return float(np.clip(f, 0.0, 1.0))

def norm_notes(f1: float) -> float:              # already 0..1
    return float(np.clip(f1, 0.0, 1.0))

def norm_clap(delta: float) -> float:
    # cosine(pos) - cosine(neg) lives in [-2, 2]; squash to [0, 1]
    return float(np.clip((delta + 2.0) / 4.0, 0.0, 1.0))

def norm_fad(fad: float, tau: float) -> float:
    # tau is calibrated once per corpus: the median leave-one-out FAD of the reference
    # corpus against itself. That makes "as close as the references are to each other"
    # score ~0.37 and gives the axis a stable scale across corpora.
    return float(np.exp(-fad / tau))

def norm_tuning(p95_cents: float, floor: float) -> float:
    return float(np.clip(1.0 - p95_cents / floor, 0.0, 1.0))


@dataclass(frozen=True)
class ScoreWeights:
    rhythm: float
    notes: float
    clap: float
    fad: float
    tuning: float = 0.0        # non-zero only for microtonal stems

    def __post_init__(self):
        total = self.rhythm + self.notes + self.clap + self.fad + self.tuning
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"weights must sum to 1.0, got {total}")

# Per-role, per the design document's guidance. Tested: every row sums to 1.0.
ROLE_WEIGHTS = {
    "drums":   ScoreWeights(rhythm=0.45, notes=0.05, clap=0.20, fad=0.30),
    "bass":    ScoreWeights(rhythm=0.30, notes=0.30, clap=0.15, fad=0.25),
    "comping": ScoreWeights(rhythm=0.25, notes=0.30, clap=0.20, fad=0.25),
    "lead":    ScoreWeights(rhythm=0.20, notes=0.45, clap=0.15, fad=0.20),
    "pads":    ScoreWeights(rhythm=0.10, notes=0.20, clap=0.30, fad=0.40),
}

# Per-role floors too: a 25 ms drum window is a harder test than a 50 ms lead window,
# so a single 0.85 floor is not the same standard across roles.
ROLE_FLOORS = {
    "drums":   ScoreFloors(rhythm_f=0.80, notes_f1=0.00),   # notes n/a on percussion
    "bass":    ScoreFloors(rhythm_f=0.85, notes_f1=0.70),
    "comping": ScoreFloors(rhythm_f=0.80, notes_f1=0.70),
    "lead":    ScoreFloors(rhythm_f=0.85, notes_f1=0.75),
    "pads":    ScoreFloors(rhythm_f=0.60, notes_f1=0.65),
}

@dataclass(frozen=True)
class ScoreFloors:
    rhythm_f: float = 0.85
    notes_f1: float = 0.70
    tuning_cents_p95: float = 15.0   # used only when microtonal=True

def score(candidate: Candidate, stems: list[Stem], refs: str,
          weights: ScoreWeights, floors: ScoreFloors) -> Candidate:
    """Pure. Returns candidate with scores filled; accepted/reject_reason set."""
    stem = next(s for s in stems if s.track == candidate.stem)
    scores = {
        "rhythm": rhythm_score(candidate.audio_path,
                                expressive_midi_for(candidate.stem)),
        "notes":  note_fidelity(candidate.audio_path,
                                 expressive_midi_for(candidate.stem)),
        "clap":   clap_similarity(candidate.audio_path,
                                   candidate.prompt, candidate.negative_prompt),
        "fad":    fad_score(candidate.audio_path, refs),
    }
    if stem.microtonal:
        report = tuning_check(candidate.audio_path, edo=..., base_tuning_midi=...)
        scores["tuning"] = report.p95_cents
        if report.p95_cents > floors.tuning_cents_p95:
            return _reject(candidate, scores, "tuning_floor")
    # Hard floors, evaluated on the RAW axis values
    if scores["rhythm"] < floors.rhythm_f:
        return _reject(candidate, scores, "rhythm_floor")
    if scores["notes"] is not None and scores["notes"] < floors.notes_f1:
        return _reject(candidate, scores, "notes_floor")

    # [v1.1] Weighted sum over NORMALISED axes. Any axis that is None for this role
    # (e.g. notes on drums) is dropped and its weight redistributed proportionally, so
    # totals stay comparable across roles.
    normed = {
        "rhythm": norm_rhythm(scores["rhythm"]),
        "notes":  None if scores["notes"] is None else norm_notes(scores["notes"]),
        "clap":   norm_clap(scores["clap"]),
        "fad":    norm_fad(scores["fad"], tau=refs_tau),          # <-- lower FAD now scores HIGHER
        "tuning": norm_tuning(scores["tuning"], floors.tuning_cents_p95) if stem.microtonal else None,
    }
    weights = ROLE_WEIGHTS[stem.role]
    live = {k: v for k, v in normed.items() if v is not None}
    wsum = sum(getattr(weights, k) for k in live) or 1.0
    total = sum(getattr(weights, k) / wsum * v for k, v in live.items())
    return dataclasses.replace(candidate, scores=tuple(scores.items()),
                               accepted=True, total_score=total)
```

**Test:**

- Candidate below rhythm floor -> `accepted=False`, `reject_reason="rhythm_floor"`.
- Candidate above all floors -> `accepted=True`, `total_score` computed.
- Pure: calling `score()` twice with same input returns identical result.
- **[v1.1]** `test_lower_fad_scores_higher`: two candidates identical on every other axis, one
  with FAD 5.0 and one with FAD 50.0 — the FAD-5 candidate must win. This is the regression
  test for Bug A, and it is the single most important test in `scoring/`.
- **[v1.1]** `test_role_weights_sum_to_one`: every row of `ROLE_WEIGHTS` sums to 1.0.
- **[v1.1]** `test_total_score_bounded`: `0.0 <= total_score <= 1.0` for all fixtures.
- **[v1.1]** `test_drums_drop_notes_axis`: a drum candidate with `notes=None` still produces a
  total in [0, 1] and the remaining weights are renormalised.

**Exit:** All five scorers pass monotonicity. Combined scorer correctly applies hard floors.

---

### M3 — One Endpoint End-to-End (Weeks 7–8)

---

#### Step 3.1 — RestyleEndpoint protocol and Stable Audio wrapper

`m2a/restyle/base.py`:

```python
from typing import Protocol
from m2a.contracts import Candidate

class RestyleEndpoint(Protocol):
    name: str
    is_diffusion: bool
    follows_f0: bool          # True for DDSP/RAVE — skips diffusion path for microtonal
    max_seconds: float

    def transform(self, audio_path: str, prompt: str, negative_prompt: str,
                  strength: float, seed: int, extra: dict) -> Candidate: ...
```

`m2a/restyle/stable_audio.py` — wrap Stable Audio 2.5 a2a with `httpx` + `tenacity` retry:

```python
class StableAudioEndpoint:
    name = "stable_audio_a2a"
    version = "2.5"
    is_diffusion = True
    follows_f0 = False
    # [v1.1] The single most dangerous unstated assumption in v1.0. Providers disagree about
    # what "strength" means: for some it is how far the output may DIVERGE from the input, for
    # others how much of the input is PRESERVED. Get it backwards and the search walks the grid
    # in the wrong direction — and still returns a "best" candidate, so nothing fails loudly.
    # Internally the pipeline searches over normalised `divergence` in [0, 1]; each endpoint
    # declares its convention and converts.
    strength_semantics = "divergence"      # higher value => further from the input
    usd_per_call = 0.06                    # config-overridable; feeds the budget guard
    # Length caps change between model versions. Treat this as a default, query the endpoint's
    # capability response at startup, and assert the chunker's chunk length against the live
    # value rather than this constant.
    max_input_seconds = 190.0

    def transform(self, audio_path, prompt, negative_prompt,
                  strength, seed, extra) -> Candidate:
        # upload audio, poll for result, download, return Candidate
        # normalize errors to retryable | fatal | content_policy
        ...
```

**Test:** Use `vcrpy` for record/replay. Tests never hit paid endpoints.

```python
@pytest.mark.vcr  # replays cassette in tests/cassettes/
def test_successful_transform():
    ep = StableAudioEndpoint(api_key="test")
    c = ep.transform("tests/fixtures/five_sec_stem.wav",
                     "funk drums", "EDM", 0.7, 42, {})
    assert c.audio_path is not None

@pytest.mark.vcr
def test_rate_limit_retries():
    ep = StableAudioEndpoint(api_key="test", max_retries=3)
    c = ep.transform(...)
    assert c.accepted or c.reject_reason in ("content_policy", "fatal")
```

---

#### Step 3.2 — Boundary-aware chunking (`restyle/chunking.py`)

```python
def chunk_and_restyle(stem: Stem, structure: StructureJSON,
                      endpoint: RestyleEndpoint, prompt: str,
                      negative_prompt: str, strength: float,
                      seed: int) -> str:
    # 1. Select seams by descending boundary_lvl
    seams = select_seams(structure.sections, structure.bar_table)
    # 2. Chunk audio at seams with 2-bar overlap
    chunks = chunk_audio(stem.path, seams, overlap_bars=2, bar_table=structure.bar_table)
    # 3. Restyle each chunk
    restyled = [endpoint.transform(c, prompt, negative_prompt, strength, seed, {})
                for c in chunks]
    # 4. Crossfade rejoin at downbeats
    return crossfade_rejoin(restyled, seams, bar_table=structure.bar_table)
```

**Test:**

```python
def test_seams_on_downbeats():
    seams = select_seams(FIXTURE_STRUCTURE.sections, FIXTURE_STRUCTURE.bar_table)
    for seam_sec in seams:
        # seam must be within 10 ms of a bar boundary
        bar_times = [r[1] for r in FIXTURE_STRUCTURE.bar_table]
        assert min(abs(seam_sec - t) for t in bar_times) < 0.01

def test_crossfade_is_lossless_without_restyle():
    # [v1.1] This test is only satisfiable with an EQUAL-GAIN (linear) crossfade. An
    # equal-power (sin/cos) fade applied to two copies of the SAME signal sums to sqrt(2),
    # a +3 dB bump at the seam centre — equal-power is correct for *uncorrelated* material.
    # Restyled chunks share prompt, seed and a 2-bar overlap of identical input context, so
    # they are strongly correlated: linear is the right law here. See D002.
    out = crossfade_rejoin_passthrough(FIXTURE_STEM.path, SEAMS, BAR_TABLE)
    original = np.array(sf.read(FIXTURE_STEM.path)[0])
    result = np.array(sf.read(out)[0])
    np.testing.assert_allclose(original, result, atol=1e-5)
```

---

#### Step 3.3 — Prompt generation

`m2a/prompts/stylecard.py` + `generate.py`:

```python
def make_style_card(reference_dir: str, user_notes: str = "") -> StyleCard:
    """Extract recurring descriptors from reference audio."""
    ...

def generate_prompts(structure: StructureJSON, track: TrackInfo,
                     style_card: StyleCard, n_variants: int = 4) -> list[tuple[str, str]]:
    """Return list of (prompt, negative_prompt) pairs."""
    meter = structure.meter[0][1] if structure.meter else "4/4"
    is_odd_meter = not meter.startswith("4/")
    ...
```

**Test:** Odd-meter section gets explicit meter in prompt. Non-4/4 gets "four-on-the-floor" in negative.

---

#### Step 3.4 — Search loop (`search/loop.py`)

```python
class SearchLoop:
    def __init__(self, endpoint, scorer, budget: BudgetGuard):
        self.endpoint = endpoint
        self.scorer = scorer
        self.budget = budget

    def search(self, stem: Stem, prompts: list, structure: StructureJSON,
               refs: str, config: SearchConfig) -> Candidate:
        candidates = []
        for strength in config.strength_grid:
            for prompt, neg in prompts:
                for seed in range(config.seeds):
                    # [v1.1] request-hash cache before the budget check — a cache hit is free
                    # and must not count against the budget. The design document requires
                    # "identical request hash => reuse"; v1.0 never implemented it, so every
                    # re-run of a resumed job re-bought the same candidates.
                    req = request_hash(stem, prompt, neg, strength, seed,
                                       self.endpoint.name, self.endpoint.version)
                    if (hit := self.cache.get(req)) is not None:
                        candidates.append(self.scorer.score(hit, ...))
                        continue
                    if not self.budget.can_spend(self.endpoint.usd_per_call):
                        return _best_or_fallback(candidates, stem, config)
                    raw = self.endpoint.transform(stem.path, prompt, neg,
                                                   strength, seed, {})
                    self.cache.put(req, raw)
                    scored = self.scorer.score(raw, ...)
                    candidates.append(scored)
                    self.budget.record_call()
        # Successive halving on accepted candidates
        accepted = [c for c in candidates if c.accepted]
        if not accepted:
            # [v1.1] Do NOT abort the run. The pipeline's contract is "produces a good result
            # without a human in the loop"; the deterministic stem from STAGE 2 is always a
            # valid, note-faithful output, and the whole point of the layering is that the
            # restyle layer is optional per stem. Fall back, log loudly, mark it in the
            # manifest and the audit report, and keep going. `strict` remains available for CI.
            if config.on_no_candidate == "fallback_dry":
                log.warning("no candidate cleared the floors for %s (%d tried, reasons=%s); "
                            "falling back to the dry stem",
                            stem.track, len(candidates), _reason_histogram(candidates))
                return dry_candidate(stem, reason="all_candidates_below_floor")
            raise NoAcceptableCandidateError(candidates)
        top = sorted(accepted, key=lambda c: c.total_score, reverse=True)
        top_quarter = top[:max(1, len(top) // 4)]
        # Re-sample around top_quarter with finer strength steps
        ...
        return max(all_accepted, key=lambda c: c.total_score)
```

**Test:**

```python
def test_budget_guard_stops_search(mock_endpoint, mock_scorer):
    loop = SearchLoop(mock_endpoint, mock_scorer, BudgetGuard(max_calls=5))
    loop.search(FIXTURE_STEM, PROMPTS, FIXTURE_STRUCTURE, REFS, config)
    assert mock_endpoint.call_count <= 5

def test_selects_highest_scoring_candidate(mock_endpoint, mock_scorer):
    # Mock scorer assigns deterministic scores by seed
    best = loop.search(...)
    assert best.total_score == max(c.total_score for c in all_scored)
```

**Exit:** `render-audio` CLI runs end-to-end on fixture with mocked endpoint. Completes unattended in < 60 s.

---

### M4 — Reference Corpus (Weeks 9–10)

---

#### Step 4.1 — Groove extraction from reference audio (`groove/extract.py`)

```python
def extract_groove_template(reference_path: str) -> GrooveSpec:
    """Measure groove from reference audio using madmom beat tracking."""
    # 1. madmom DBN beat tracker → beat times
    beats = madmom_beat_track(reference_path)
    # 2. Demucs source separation → drum/bass stems
    stems = demucs_separate(reference_path)
    # 3. Onset detect per stem
    # 4. For each onset: deviation from nearest 16th-note grid
    # 5. Aggregate → by_position_16 (median deviation per position)
    return GrooveSpec(microtiming_ms={"drums": MicrotimingSpec(by_position_16=...)})
```

**Test:** Extract from a fixture with known swing → measured ratio within 0.02 of ground truth.

---

#### Step 4.2 — Mix and master (`mixmaster/`)

```python
# mix.py
def mix_stems(stems: list[Stem], structure: StructureJSON,
              targets: LufsTargets) -> str:
    # 1. Align restyled stems to dry counterparts (cross-correlation)
    # 2. Normalize each to role LUFS target
    # 3. Static pan: drums/bass center, comping panned, lead forward
    # 4. Bus compression
    # 5. Optional cover blend (high-passed)
    ...

# master.py
def master(mix_path: str, reference_path: str, target_lufs: float) -> str:
    # Matchering against reference master
    import matchering as mg
    mg.process(target=mix_path, reference=reference_path, ...)
    ...
```

**Test:** Alignment corrects a 50 ms stem offset to within 5 ms. Master LUFS within 1 LU of target.

---

#### Step 4.3 — HTML audit report

One HTML file per run with stem audio players, spectrograms, metric table, prompts/params per candidate.

**Test:** Report renders without error. Contains one section per stem.

---

### M5 — Breadth (Week 11)

- **Step 5.1** `restyle/suno.py` + cover blend in `mix.py`. Cover blend improves weighted objective on fixture without breaching floors.
- **Step 5.2** `restyle/musicgen_local.py`. Local MusicGen-melody for cheap search. No paid calls.
- **Step 5.3** Inpainting seam repair in `chunking.py`. Spectral discontinuity at a seam triggers inpainting call.

---

### M6 — Single-Reference Mode (Week 12)

---

#### Step 6.1 — Host analysis (`analysis/host.py`)

```python
def analyse_host(host_path: str, out_dir: str) -> HostAnalysis:
    # madmom DBN beat grid → bar boundaries
    beats = madmom_beat_track(host_path)
    # Demucs separation → per-role stems
    stems = demucs_separate(host_path)
    # Onset lattice: per-role onset probability over 16th-note grid
    lattice = extract_onset_lattice(stems)
    # Accent profile
    accent = extract_accent_profile(stems)
    return HostAnalysis(beats=beats, lattice=lattice, accent=accent)
```

**Test:** Self-consistency — render a known MIDI at known groove into a synthetic host, then recover beat grid within 20 ms tolerance.

---

#### Step 6.2 — Lattice re-quantization (`groove/lattice.py`)

```python
def quantize_to_lattice(midi_path: str, lattice: OnsetLattice,
                        max_displacement_ticks: int,
                        mode: str = "lock") -> str:
    # mode="lock": snap to lattice positions
    # mode="complement": snap to positions lattice does NOT occupy
    # max_displacement_ticks: never move a note more than this
    ...
```

**Test:**

```python
def test_noop_when_already_on_lattice():
    # Part that is already on the lattice → no notes move
    out = quantize_to_lattice(on_lattice_midi, lattice, max_displacement_ticks=96)
    assert onset_times(out) == onset_times(on_lattice_midi)

def test_bound_always_respected():
    out = quantize_to_lattice(off_lattice_midi, lattice, max_displacement_ticks=48)
    for orig, moved in zip(onset_times(off_lattice_midi), onset_times(out)):
        assert abs(orig - moved) <= 48

def test_complement_mode_avoids_host_onsets():
    out = quantize_to_lattice(midi, lattice, 96, mode="complement")
    host_onset_times = lattice.onset_times_sec()
    for onset in onset_times_sec(out):
        assert all(abs(onset - h) > 0.030 for h in host_onset_times)  # δ = 30 ms
```

---

#### Step 6.3 — Mix-fit scorer (`scoring/mixfit.py`)

```python
def mixfit_score(candidate: Candidate, host_path: str,
                 host_analysis: HostAnalysis) -> float:
    # 1. Beat alignment: onset F-measure against host measured grid
    # 2. Spectral clash: band-wise energy overlap with host mix
    # 3. Level fit: candidate LUFS vs comparable host stems
    # 4. Embedding coherence: composite similarity to adjacent host sections
    ...
```

**Test:** Monotonicity — shifting candidate off grid, boosting into occupied band, or over-leveling each degrade their sub-score.

---

### Hardening and Handoff (Week 13)

- **CI end-to-end:** 8-bar fixture through M1+M2, fluidsynth only, mocked endpoints, < 60 seconds.
- **Coverage target:** 80% line coverage on `m2a/` excluding `restyle/` wrappers.
- **Consolidate `DECISIONS.md`:** all eight decisions resolved and recorded.
- **Update README:** installation, quick-start, full CLI reference.
- **Next-quarter scope:** document deferred items — closed-loop groove optimization, GTTM-weight optimization via the aimusic adapter, MIDI-DDSP production quality, hostsampler auto-SFZ.

---

## 6. The aimusic Adapter (Separate Work, aimusic Repo)

When the `aimusic` team wants to pipe their composer output into `midi2audio`, they add one file
to their repo — **not to this one**:

```python
# In aimusic repo: aimusic/render/package.py
from aimusic.core.core_types import Score, BeatState
from aimusic.core.diagnostics import StructuralDiagnostics

def emit_render_package(
    score: Score,
    beatstates: list[BeatState],
    diagnostics: StructuralDiagnostics,
    edo_config,
    out_dir: str,
) -> str:
    """
    Produce a midi2audio-compatible RenderPackage directory from aimusic outputs.
    Writes: score.mid, structure.json (provenance=planner), tuning.json, manifest.json
    """
    # structure.json produced from BeatState fields directly (B1 structural bypass)
    # tempo_map    ← score.tempo_bpm
    # meter        ← meter_id via vocabularies
    # key          ← key_id via vocabularies
    # chords       ← chord_id transitions in beatstate path
    # sections     ← boundary_lvl changes
    # track roles  ← role_id + decoder track names (exact, no inference needed)
    # onset_profile← groove_id + decoded onsets
    # quantized    ← always True (decoder emits grid-aligned material)
    ...
```

`midi2audio` never imports this. It only reads the files on disk.

---

## 7. Dependency Reference

| Package | Used in | Why |
|---|---|---|
| `pretty-midi` | `analysis/from_midi.py` | MIDI parsing, tempo/meter extraction |
| `music21` | `analysis/from_midi.py` | Key inference, chord analysis |
| `mido` | `groove/apply.py`, `render/*` | Low-level MIDI I/O and editing |
| `numpy`, `scipy` | `scoring/*`, `mixmaster/*` | DSP, statistics |
| `librosa` | `scoring/rhythm.py`, `scoring/tuning_check.py` | Onset detection, f0 tracking, CQT |
| `soundfile` | all render/audio modules | WAV I/O |
| `madmom` | `groove/extract.py`, `analysis/host.py` | Beat tracking (DBN) — **[v1.1] see installation warning below** |
| `mir_eval` | `scoring/rhythm.py` | Standard matched onset F-measure (do not hand-roll) |
| `pyloudnorm` | `mixmaster/mix.py`, `master.py`, `audit.py` | ITU-R BS.1770 / EBU R128 LUFS — **[v1.1] the plan specifies LUFS targets everywhere and shipped no library that can measure LUFS** |
| `pedalboard` | `mixmaster/mix.py` | High-pass, EQ, bus compression (v1.0 said "bus compression" with nothing to do it) |
| `soxr` | resampling | Endpoints return 44.1 kHz; the pipeline is 48 kHz |
| `demucs` | `groove/extract.py`, `analysis/host.py` | Source separation |
| `basic-pitch` | `scoring/notes.py` | Polyphonic MIDI transcription |
| `laion-clap` | `scoring/clap_sim.py`, `scoring/fad.py` | Audio-text embedding |
| `matchering` | `mixmaster/master.py` | Reference mastering |
| `pydantic>=2.0` | `contracts.py`, `groove/spec.py` | Schema validation, LLM output contracts |
| `pyyaml` | `config.py` | Profile and config loading |
| `httpx` | `restyle/stable_audio.py`, `restyle/suno.py` | Async HTTP for endpoint calls |
| `tenacity` | `restyle/*.py` | Retry with backoff |
| `pytest`, `hypothesis`, `vcrpy` | `tests/` | Testing, property tests, HTTP replay |

Install groups:

```bash
pip install -e ".[core]"     # analysis + groove + render (no endpoints, no ML)
pip install -e ".[audio]"    # + madmom + demucs + basic-pitch + clap + matchering
pip install -e ".[restyle]"  # + httpx (endpoint wrappers)
pip install -e ".[dev]"      # everything + pytest + hypothesis + vcrpy
```

### [v1.1] Dependency reality check — read before Week 1

The `[audio]` group as written is unlikely to resolve in a single environment, and this will
cost days if it is discovered in Week 9 instead of Week 1. Verify all of this on day one with a
throwaway venv; record the outcome as **D010**.

1. **`madmom` and Python 3.11.** The last PyPI release (0.16.1, 2018) predates the removal of
   `collections` ABC aliases and several deprecated NumPy aliases, and does not build cleanly on
   modern Pythons. Options, in order of preference:
   (a) define a `BeatTracker` Protocol in `analysis/beats.py` and keep madmom behind it;
   (b) install from git rather than PyPI and pin the commit;
   (c) substitute a modern tracker — *Beat This!* (Foscarin et al., ISMIR 2024) or BeatNet — both
   pip-installable and torch-native, which also removes a Cython/NumPy pin from the tree.
   The Protocol is cheap and makes (c) a config change instead of a refactor.
2. **Torch pin collisions.** `demucs`, `basic-pitch` (TensorFlow, or ONNX/CoreML runtimes in
   recent versions) and `laion-clap` (transformers + torch) each pin conflicting versions. The
   robust arrangement is a **scorer service boundary**: heavy scorers run in their own venv
   behind a thin subprocess/HTTP interface returning JSON. This also keeps `[core]` genuinely
   light, which the deterministic spine needs for CI to stay under a minute.
3. **`vcrpy` and audio payloads.** Cassettes will embed multi-MB WAV bodies. Use a custom
   matcher that hashes the request body and store a 0.5 s stub response, or the repo grows by
   hundreds of MB in M3.
4. **System binaries** are undeclared in v1.0 and must be in the README and CI image:
   `fluidsynth`, `sfizz`, `ffmpeg`, plus at least one SoundFont. `test_renders_four_stems` fails
   confusingly without them.

---

## 8. [v1.1] Additions — What the Design Document Requires and v1.0 Did Not Build

Section 5's milestones are amended by the steps below. Each one exists because the reference
paper specifies it and no step in v1.0 delivers it. They are placed in the milestone where they
belong; none of them extends the critical path by more than a day or two, and two of them
(§8.1, §8.2) are prerequisites for being able to claim the pipeline works at all.

---

### 8.1 — `m2a/manifest.py` (M0, one day)

The design document's contract section is unambiguous: *"All requests and responses (minus audio
payloads) are logged to the manifest"*, and *"Every render carries a manifest of all parameters,
prompts, seeds, and endpoint versions."* v1.0 has a `manifest.json` in the input package that
records who produced the MIDI, and nothing that records what the pipeline then did. Without it,
provenance, cost accounting, reproducibility and the rights position are all unevidenced.

```python
@dataclass
class RunManifest:
    run_id: str
    code_version: str
    config_hash: str
    started_at: str
    inputs: dict                 # midi hash, structure hash, tuning hash, reference corpus hashes
    stages: list[StageRecord]    # per stage: inputs, config subset, outputs, wall-clock, cache hit
    endpoint_calls: list[CallRecord]
    decisions: dict              # resolved values of D001-D016 for this run
    totals: dict                 # calls, usd_estimate, wall_clock, cache_hit_rate

@dataclass
class CallRecord:
    endpoint: str
    endpoint_version: str
    request_hash: str
    prompt: str
    negative_prompt: str
    divergence: float            # normalised; also record the raw provider parameter sent
    raw_params: dict
    seed: int
    input_audio_hash: str
    output_audio_hash: str
    usd_estimate: float
    latency_s: float
    outcome: str                 # ok | retryable | fatal | content_policy | cache_hit
    watermark_declared: bool     # e.g. SynthID on Google models — record what the provider says
    terms_version: str           # upload-terms version acknowledged at call time
```

**Test:** a completed run's manifest, replayed through the orchestrator with all endpoints
mocked to return the recorded output hashes, reproduces the same final master hash.

---

### 8.2 — `m2a/mixmaster/audit.py` (M4, one to two days)

STAGE 6 step 4 of the design document specifies a *final audit render*: loudness stats, true-peak
check, seam-discontinuity scan, and **rhythm re-verification of the finished file against the
expressive MIDI**. v1.0 has an HTML report (a display surface) but no gate. The rhythm
re-verification matters most: per-stem scoring can pass while the mixed and mastered file has
drifted, because alignment, blending and mastering all happen after the last time anything was
scored.

```python
@dataclass(frozen=True)
class AuditReport:
    integrated_lufs: float
    short_term_max_lufs: float
    true_peak_dbtp: float          # 4x-oversampled — pyloudnorm measures sample peak only
    seam_discontinuities: tuple    # (time_sec, spectral_flux_z) above threshold
    rhythm_f_final: float          # finished master vs expressive MIDI
    stems_fallen_back: tuple       # stems that used the dry fallback (D012)
    passes: bool
```

Gate thresholds in config: `true_peak_dbtp <= -1.0`, `|integrated_lufs - target| <= 1.0`,
`rhythm_f_final >= 0.80`, `len(seam_discontinuities) == 0`. A failing audit does not silently
ship; it exits non-zero and names the failing check.

**Test:** monotonicity, as with every other metric — splice a 2 ms gap into the master and the
seam scan must find it; boost the master 3 dB and the true-peak check must fail.

---

### 8.3 — `m2a/restyle/policy.py` (M3, one day)

Listed in v1.0's repository layout as *"strength schedule: role x tension x EDO"* and then never
specified in any step. It is the module that decides where the search even starts, so leaving it
undefined means every stem searches the same grid regardless of how much divergence it can
survive.

```python
def divergence_grid(role: str, microtonal: bool, section_energy: float,
                    tension: float | None) -> list[float]:
    if microtonal:
        return []                       # P6: routed away from diffusion entirely
    base = {
        "drums":   [0.70, 0.80, 0.90],  # percussive: tolerates high divergence
        "bass":    [0.60, 0.70, 0.80],
        "comping": [0.55, 0.65, 0.75],  # pitched + harmonic: least tolerant
        "lead":    [0.55, 0.65, 0.75],
        "pads":    [0.70, 0.80, 0.90],
    }[role]
    # High-energy sections mask artefacts; sparse ones expose them.
    return [clamp(d + 0.05 * (section_energy - 0.5) * 2, 0.5, 0.95) for d in base]
```

Anchored on the design document's guidance (raise toward 0.85-0.9 when output hugs the input,
lower toward 0.6-0.75 when it deviates) and on its note that percussive and textural stems
tolerate more divergence than pitched harmonic ones. Treat the published ranges as the search
interval, not as settings; the scorer decides.

---

### 8.4 — `m2a/scoring/syncopation.py` (M6, two days)

Single-reference mode distinguishes *feel* (where events land relative to the grid) from
*syncopation vocabulary* (the part locking into the host's onset lattice). v1.0 implements the
second as an unmeasured transformation: `lattice.py` snaps notes, and nothing scores whether the
result actually reads as the same rhythmic conversation. That violates the plan's own rule that
quality claims must cite a metric.

Add a syncopation index — the Longuet-Higgins & Lee (1984) metrical-weight measure is the
standard one and is about forty lines of code — and score the candidate's syncopation profile
against the host's:

```python
def syncopation_index(onsets_by_position, meter, grid) -> float:
    """LHL: sum over rests-preceded-by-onsets of (weight(rest) - weight(onset))."""

def syncopation_match(candidate_midi, host_lattice, meter, grid) -> float:
    """1 - |LHL(candidate) - LHL(host)| / max_possible, per bar, averaged."""
```

This also gives the outer groove-optimisation loop (§8.6) a real objective in single-reference
mode, and it is testable by construction: pushing an on-beat note to the preceding sixteenth
must raise the index.

---

### 8.5 — Cover blend: score the blend, never the cover (M5, half a day)

v1.0's Step 5.1 says only that the cover blend must "improve the weighted objective without
breaching floors". The design document is more specific and the specifics are the safety
property: the covered mix is **layered under** the literal stem mix at 20-40 % by loudness,
aligned by cross-correlation, high-passed (the stated counter to the low-end mud failure mode),
and **the scorer evaluates the blend, not the raw cover**. Scoring the raw cover would let a
re-performed, rhythmically drifted mix pass on timbre alone.

```python
def cover_blend(stem_mix: str, cover: str, blend_pct: float,
                highpass_hz: float = 180.0) -> str:
    cover_al = align_by_xcorr(cover, stem_mix)          # endpoints shift by tens of ms
    cover_hp = highpass(cover_al, highpass_hz)          # counter: low-end mud
    return sum_at_loudness_ratio(stem_mix, cover_hp, blend_pct)   # 0.20-0.40
```

Search `blend_pct` in {0.0, 0.2, 0.3, 0.4}; `0.0` (no cover at all) must be in the grid so the
pass can lose. The blend is scored with the full-mix axes (D015) including the rhythm floor.

---

### 8.6 — Deferred, with the objective named (post-M6)

The design document's closed-loop groove optimisation — using the same scorer to tune STAGE 1
parameters (swing ratio, anticipation probability) in an outer loop — is correctly deferred in
v1.0's handoff section, but deferred items need their objective written down or they get
re-litigated. The objective is: **rhythm-profile similarity between the rendered stem and the
groove template extracted from the references**, i.e. distance between by-position microtiming
and accent profiles, plus (in single-reference mode) the syncopation match of §8.4. Deterministic
and cheap: it needs no endpoint calls at all, because it can be evaluated on the STAGE 2 dry
render. That makes it the highest-value deferred item, not the lowest.

---

### 8.7 — Loose ends to close before M1

* **`analysis/reconcile.py`** appears in the layout and in no milestone. It is presumably meant
  to cross-check an aimusic-supplied `structure.json` against an inferred one. Either give it a
  step (a disagreement report is genuinely useful the first time the adapter is wrong about a
  section boundary) or delete it from the layout. Dead entries in a repo skeleton get built by
  someone eventually.
* **`render/pianoteq_r.py`** likewise: the design document names headless Pianoteq for piano and
  microtonal `.scl` support, but it is licensed software with no CI story. Keep it, gate it on a
  config flag and a licence check, and say so.
* **`render/hostsampler.py`** is deferred to "next quarter" in v1.0's handoff, but the design
  document makes auto-sampling the host the **default** timbre policy for any role that exists in
  the host — it is policy (1) of three, ahead of both DDSP and generative restyle. Deferring it
  means M6 ships single-reference mode with its primary timbre path missing and its exit
  criterion met only through the fallback path. Either build it in M6 (it is the smallest of the
  three: onset-segment, pitch-detect, bucket by energy into velocity layers, emit SFZ) or record
  the deviation explicitly in DECISIONS.md rather than in a handoff bullet.
* **Full-mix scoring** (D015): the design document scores stems *and* the full mix. Wire the mix
  through `combine.py` in M4 with rhythm and FAD axes.
* **Prompt failure-mode defaults**: the design document's four named failure modes each have a
  stated counter, and three of them live in the prompt layer. Bake them into
  `prompts/generate.py` as non-optional defaults rather than leaving them to the style card:
  always include "instrumental" in the positive prompt and "vocals, singing" in the negative for
  instrumental material; always carry the style card's negative list; and add the meter to the
  prompt with "four-on-the-floor" negated for odd meters (v1.0's test already assumes this but no
  code produces it).
* **`prompts/stylecard.py` is a stub.** `make_style_card` has a docstring and `...`. It needs a
  concrete feature list to hand the LLM, since audio captioning may not be available: tempo,
  spectral centroid/rolloff percentiles, crest factor, integrated and short-term LUFS, stereo
  width, band energy ratios, estimated instrumentation from a tagger, and any user notes. Then:
  recurring adjectives across references, era markers, and an explicit negative list. Without the
  feature list this is a prompt written by a model that has heard nothing.

---

### 8.8 — Schedule note

Thirteen weeks for M0-M6 plus hardening assumes one experienced engineer full-time with no
dependency surprises, and §7's reality check suggests the surprises are likely. The order is
right and the sequencing rule (no M3 before M2) is the correct one to protect. If time
compresses, cut M5 (breadth: Suno, MusicGen, inpainting) before cutting M2 or the §8.1-8.2
infrastructure: breadth adds options, while the scorer and the manifest are what make any of the
output defensible.
