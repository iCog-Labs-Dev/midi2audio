# Architectural Decision Records (ADR) - Milestone 1

## ADR 001: Separation of Concerns - Deterministic Expressive Spine vs Generative Layer
- **Status**: Accepted
- **Context**: Generative text-to-audio and audio-to-audio models drift notes, simplify complex chords, and smooth deliberate syncopation toward common statistical modes.
- **Decision**: All microtiming, swing, accentuation, dynamic phrase arcs, and anticipation are resolved symbolically on the MIDI sequence prior to invoking any audio endpoint. The deterministic output serves as the immutable ground truth for downstream scoring.
- **Consequences**: Generative models are restricted to timbre, texture, and acoustic transfer. Restyle failures are detected via rhythmic and melodic divergence, triggering prompt/seed regeneration rather than post-generation alignment hacks.

## ADR 002: Ensemble Rubato Isolation to Global Tempo Maps
- **Status**: Accepted
- **Context**: Independent, per-stem phrase-level time-stretching shifts chord hit points across stems, smearing transient coherence and breaking downbeat alignment.
- **Decision**: Phrase-level rubato, ritardandi, and tempo lifts are exclusively authored into the global MIDI tempo map (`SetTempo` events). Track-level timing adjustments are restricted to local microtiming offsets and humanizing jitter ($\sigma \le 4\text{ ms}$).
- **Consequences**: Stems rendered via SoundFonts or SFZ instruments maintain absolute rhythmic lock on downbeats across polyphonic arrangements.

## ADR 003: Deterministic Content Addressing and Artifact Isolation
- **Status**: Accepted
- **Context**: Implicit cross-DAG in-memory caching leaks transient state, complicating reproducible experiment tracking and budget audits.
- **Decision**: All stage boundaries produce deterministic on-disk files accompanied by a JSON sidecar. The sidecar contains SHA-256 digests of upstream inputs, configurations, runtime manifests, and bar-boundary mappings.