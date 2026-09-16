# DECISIONS.md

## D001 — Tick rounding policy
Before: M1 Step 1.2 (groove applicator)
Question: When a groove timing offset lands between ticks, do we round-half-even or truncate?
Do offsets accumulate across multiple transformations or are they absolute per note?
Resolution: Round-half-even to the nearest tick (Python's built-in `round`), applied once at
serialization time. Offsets accumulate additively across the transformation pipeline (swing +
microtiming + strum all contribute to one running delta per note), but each transformation's own
contribution is computed from the note's *original* quantised metrical position, not from the
position after previously-applied transformations — otherwise a note's grid-slot lookup would
drift off its intended position as stages compose.

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

## D008 — MIDI-GPT license gate
Before: M5 (breadth)
Question: MIDI-GPT weights are CC-BY-NC. Behind what config flag does that
branch sit, and how does CI assert it defaults to off?
Resolution: config flag midi_gpt: false; CI test asserts default config loads
with midi_gpt=False.

## D009 — Groove-spec keys: roles or track names? [v1.1]
Before: M1 Step 1.4
Question: v1.0 documents `applies_to` and `microtiming_ms` keys as track names but the design
document specifies per-role semantics. Which wins, and how are two drum tracks handled?
Resolution: Roles are canonical; `track:<idx>` prefix for explicit per-track overrides; the
applicator expands via a role map built from `structure.tracks`.

## D010 — Dependency environment topology [v1.1]
Before: M0 Step 0.1 (day one)
Question: Do madmom, demucs, basic-pitch and laion-clap co-resolve on Python 3.11 in one venv?
Resolution: [TBD — verify in week 1; fallback is a scorer service boundary, see §7]

## D011 — Strength semantics per endpoint [v1.1]
Before: M3 Step 3.1
Question: Does a higher `strength` mean more divergence or more preservation, per endpoint?
Resolution: Pipeline searches normalised `divergence in [0,1]`; each endpoint declares
`strength_semantics` and converts. Verified per endpoint by a smoke test that renders at
divergence 0.1 and 0.9 and asserts the 0.9 output is spectrally further from the input.

## D012 — Behaviour when no candidate clears the floors [v1.1]
Before: M3 Step 3.4
Question: Abort, or fall back to the deterministic stem?
Resolution: `on_no_candidate: fallback_dry` by default (the dry stem is always note-faithful);
`strict` in CI. Logged in the manifest and flagged in the audit report.

## D013 — FAD estimator and corpus size [v1.1]
Before: M4 Step 4.x
Question: FAD is strongly biased at small N, and the design document's corpus is 5-30 tracks.
Which embedding, which estimator, and what is `tau`?
Resolution: [TBD — suggest CLAP or EnCodec embeddings rather than VGGish, and the FAD-infinity
extrapolation of Gui et al. (2024) rather than a raw small-N FAD; `tau` = median leave-one-out
FAD of the corpus against itself]

## D014 — Suno cover endpoint: is there a licensed programmatic path? [v1.1]
Before: M5 Step 5.1
Question: Does an official, ToS-compliant API exist for upload-and-cover at the account tier we
hold, and do the upload terms permit our material?
Resolution: [TBD — this is a legal/procurement question, not an engineering one. Until it is
answered "yes" in writing, the full-mix cohesion pass runs through the Stable Audio a2a endpoint
on the rough mix instead. Third-party unofficial Suno wrappers are out of scope: they breach ToS
and would put the provenance manifest in an indefensible state.]

## D015 — Full-mix scoring [v1.1]
Before: M4
Question: The design document scores "each stem (and the full mix)"; v1.0 scores only stems.
Which axes apply to the mix, and does the mix have hard floors?
Resolution: [TBD — suggest rhythm + FAD + CLAP on the mix, floors on rhythm only]

## D016 — Sections for inferred provenance [v1.1]
Before: M1 Step 1.1
Question: `SectionInfo` carries `tension` and `boundary_lvl`, which only the aimusic planner can
produce. What does `from_midi.py` emit?
Resolution: `boundary_lvl` from novelty-curve peak height, quantised to 0-3; `tension` null.
Consumers must treat `tension` as optional and `policy.py` must not require it.
