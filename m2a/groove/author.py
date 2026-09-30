# m2a/groove/author.py
"""Stage 1: call an LLM to produce a GrooveSpec, validate it, retry on failure.

The LLM call is the only impure operation in `groove/`. Callers (the
orchestrator) write its output to `{stage_dir}/groove_spec.json` immediately
so reruns use the cached version rather than re-invoking the LLM.

The author never patches an invalid response — an invalid spec is always
regenerated from scratch with the validation error appended to the prompt.
"""
from __future__ import annotations

import json
from typing import Callable, Protocol

from pydantic import ValidationError

from m2a.groove.spec import GrooveSpec

SYSTEM_PROMPT = """
You are a music production expert. Given a MIDI analysis and style intent,
emit a JSON groove specification. Output ONLY valid JSON matching the
GrooveSpec schema. No prose, no markdown fences.
"""


class LLMFn(Protocol):
    def __call__(self, system: str, user: str) -> str: ...


def author_groove_spec(
    structure: dict,
    style_intent: str,
    llm_fn: Callable[[str, str], str],
    max_retries: int = 3,
) -> GrooveSpec:
    track_names = [t.get("name") or t.get("role", "") for t in structure.get("tracks", [])]
    user_prompt = f"""
MIDI analysis:
{json.dumps(structure, indent=2)}

Style intent: {style_intent}

Emit a GrooveSpec JSON. Track roles are: {track_names}.
"""
    last_error: Exception | None = None
    for _ in range(max_retries):
        raw = llm_fn(SYSTEM_PROMPT, user_prompt)
        try:
            data = json.loads(raw)
            return GrooveSpec.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            user_prompt += f"\n\nPrevious attempt was invalid: {e}\nTry again."
    raise RuntimeError(f"Failed to author valid GrooveSpec after {max_retries} attempts: {last_error}")
