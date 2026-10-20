#!/usr/bin/env python
"""scripts/run_pipeline.py — midi2audio command-line entry point.

`analyse` and `expressivize` run the M1 deterministic spine stages.
`render-audio` runs the whole M3 pipeline: package -> spine -> per-stem
restyle search through one endpoint, gated by the M2 scorer and the budget
guard. The default endpoint is the offline mock (no network, no cost);
`--endpoint stable_audio` uses Stable Audio 2.5 and needs STABILITY_API_KEY.

Exit codes for render-audio: 0 ok, 2 no acceptable candidate in --strict
mode, 3 input contract violation.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from m2a.analysis.from_midi import analyse_midi, load_structure
from m2a.config import GrooveConfig, load_config
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
    from m2a.orchestrator import Orchestrator
    from m2a.package import ContractViolation, build_render_package
    from m2a.search.loop import NoAcceptableCandidateError

    cfg = load_config(args.profile)
    if args.endpoint:
        cfg.restyle.endpoint = args.endpoint
    if args.budget_calls is not None:
        cfg.budget.max_calls = args.budget_calls
    if args.budget_usd is not None:
        cfg.budget.max_usd = args.budget_usd
    if args.strict:
        cfg.search.on_no_candidate = "strict"

    out_dir = pathlib.Path(args.out)
    work_dir = pathlib.Path(args.work) if args.work else out_dir / "work"
    try:
        package = build_render_package(args.midi, str(work_dir / "packages"),
                                       structure_path=args.structure, tuning_path=args.tuning)
        refs = args.refs[0] if args.refs else None
        result = Orchestrator(cfg, str(work_dir)).run(package, str(out_dir), refs_dir=refs)
    except ContractViolation as e:
        print(f"input contract violation: {e}", file=sys.stderr)
        return 3
    except NoAcceptableCandidateError as e:
        print(f"strict mode: {e}", file=sys.stderr)
        return 2

    manifest = json.loads(pathlib.Path(result.manifest_path).read_text())
    for s in manifest["stems"]:
        note = f" (fallback: {s['fallback_reason']})" if s["fallback_reason"] else ""
        print(f"  {s['stem']:<12} {s['source']:<16} score={s['total_score']:.3f} "
              f"tried={s['candidates_tried']}{note}")
    t = manifest["totals"]
    print(f"Wrote {result.out_dir}: {t['endpoint_calls']} endpoint calls, "
          f"~${t['usd_estimate']:.2f}, cache hit rate {t['cache_hit_rate']:.0%}, "
          f"{t['wall_clock_s']:.1f}s")
    return 0


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

    p_render = sub.add_parser("render-audio", help="Full pipeline: MIDI -> scored, restyled stems")
    p_render.add_argument("midi", help="Input MIDI file")
    p_render.add_argument("--structure", help="Existing structure.json (skip inference)")
    p_render.add_argument("--tuning", help="tuning.json (default: 12-EDO direct_12)")
    p_render.add_argument("--profile", default="config/default.yaml")
    p_render.add_argument("--refs", nargs="*", help="Reference corpus directory (for FAD + style card)")
    p_render.add_argument("--endpoint", choices=["offline", "stable_audio"],
                          help="Restyle endpoint (default from profile: offline)")
    p_render.add_argument("--budget-calls", type=int, default=None)
    p_render.add_argument("--budget-usd", type=float, default=None)
    p_render.add_argument("--strict", action="store_true",
                          help="Fail instead of falling back to the dry stem (D012)")
    p_render.add_argument("--work", help="Cache/work directory (default: <out>/work)")
    p_render.add_argument("--out", default="./output")
    p_render.set_defaults(func=_cmd_render_audio)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
