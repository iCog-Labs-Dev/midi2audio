#!/usr/bin/env python
"""scripts/run_pipeline.py — midi2audio command-line entry point.

`analyse` and `expressivize` are wired to the M1 deterministic spine.
`render-audio` (the full restyle pipeline) is not yet wired — that lands in
M3 once the scorer (M2) exists to gate paid endpoint calls.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from m2a.analysis.from_midi import analyse_midi, load_structure
from m2a.config import GrooveConfig
from m2a.groove.apply import apply_groove
from m2a.groove.author import author_groove_spec
from m2a.groove.spec import GrooveSpec


def _cmd_analyse(args: argparse.Namespace) -> int:
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "structure.json"
    analyse_midi(args.midi, str(out_path))
    print(f"Wrote {out_path}")
    return 0


def _cmd_expressivize(args: argparse.Namespace) -> int:
    out_dir = pathlib.Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.structure:
        structure = load_structure(args.structure)
    else:
        structure = analyse_midi(args.midi, str(out_dir / "structure.json"))

    if args.groove_spec:
        spec = GrooveSpec.model_validate(json.loads(pathlib.Path(args.groove_spec).read_text()))
    else:
        cfg = GrooveConfig()
        spec = author_groove_spec(structure.model_dump(by_alias=True), cfg.style_intent, cfg.llm_fn)

    out_path = out_dir / "expressive.mid"
    _, report = apply_groove(args.midi, spec, structure, rng_seed=0, out_path=str(out_path))
    print(f"Wrote {out_path}")
    print(f"  ghost notes inserted: {report.ghost_notes_inserted}")
    if report.rule_violations:
        print(f"  rule violations: {report.rule_violations}")
    return 0


def _cmd_render_audio(args: argparse.Namespace) -> int:
    print("render-audio is not yet wired — the full restyle pipeline lands in M3.",
          file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="midi2audio pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_analyse = sub.add_parser("analyse", help="MIDI -> structure.json")
    p_analyse.add_argument("midi", help="Input MIDI file")
    p_analyse.add_argument("--out", default="./output")
    p_analyse.set_defaults(func=_cmd_analyse)

    p_expr = sub.add_parser("expressivize", help="MIDI -> expressive MIDI")
    p_expr.add_argument("midi", help="Input MIDI file")
    p_expr.add_argument("--structure", help="Existing structure.json (skip inference)")
    p_expr.add_argument("--groove-spec", help="Existing groove spec JSON (skip LLM)")
    p_expr.add_argument("--profile", help="Style profile YAML")
    p_expr.add_argument("--out", default="./output")
    p_expr.set_defaults(func=_cmd_expressivize)

    p_render = sub.add_parser("render-audio", help="Full pipeline (not yet wired; see M3)")
    p_render.add_argument("midi", help="Input MIDI file")
    p_render.add_argument("--structure", help="Existing structure.json")
    p_render.add_argument("--profile", default="config/default.yaml")
    p_render.add_argument("--refs", nargs="*", help="Reference audio files")
    p_render.add_argument("--budget-calls", type=int, default=200)
    p_render.add_argument("--out", default="./output")
    p_render.set_defaults(func=_cmd_render_audio)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
