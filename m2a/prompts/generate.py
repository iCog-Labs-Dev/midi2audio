# m2a/prompts/generate.py
"""Per-stem (prompt, negative_prompt) variants (Step 3.3).

Three of the design document's named failure modes are countered in the
prompt layer, so those counters are non-optional defaults here (§8.7):

* "instrumental" in every positive prompt, "vocals, singing" in every
  negative — restyle endpoints otherwise hallucinate vocals;
* the style card's negative list is always carried;
* odd / non-4/4 meters are stated in the prompt and "four-on-the-floor" is
  negated, or endpoints re-bar the material into 4/4.
"""
from __future__ import annotations

from m2a.contracts import StructureJSON, TrackInfo
from m2a.prompts.stylecard import StyleCard

ROLE_PHRASES = {
    "drums":   ["live drum kit", "tight acoustic drums", "punchy studio drum kit", "drum groove"],
    "bass":    ["electric bass guitar", "fingerstyle bass", "warm bass line", "round studio bass"],
    "comping": ["electric piano comping", "rhythm guitar chords", "keyboard comping", "chordal accompaniment"],
    "lead":    ["lead synth melody", "expressive lead line", "solo instrument melody", "lead melody"],
    "pads":    ["lush synth pads", "string pad", "ambient pad texture", "sustained pads"],
}
ALWAYS_NEGATIVE = ("vocals", "singing")


def meter_of(structure: StructureJSON) -> str:
    return str(structure.meter[0][1]) if structure.meter else "4/4"


def generate_prompts(structure: StructureJSON, track: TrackInfo, style_card: StyleCard,
                     n_variants: int = 4) -> list[tuple[str, str]]:
    """Return `n_variants` (prompt, negative_prompt) pairs. Deterministic."""
    meter = meter_of(structure)
    is_odd_meter = meter != "4/4"
    bpm = round(structure.tempo_map[0][1]) if structure.tempo_map else 120
    key = structure.key[0]["key"] if structure.key else ""

    negatives = list(ALWAYS_NEGATIVE) + list(style_card.negatives)
    if is_odd_meter:
        negatives.append("four-on-the-floor")
    negative = ", ".join(dict.fromkeys(negatives))

    phrases = ROLE_PHRASES.get(track.role, [track.role])
    out: list[tuple[str, str]] = []
    for i in range(n_variants):
        parts = [phrases[i % len(phrases)], "instrumental", f"{bpm} BPM"]
        if is_odd_meter:
            parts.append(f"in {meter} time")
        if key and track.role != "drums":
            parts.append(f"in {key}")
        parts += list(style_card.descriptors)
        if style_card.user_notes:
            parts.append(style_card.user_notes)
        out.append((", ".join(parts), negative))
    return out
