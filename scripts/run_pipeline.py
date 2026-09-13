"""CLI entrypoint to execute the deterministic spine."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import yaml
from m2a.orchestrator import PipelineOrchestrator


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage 0 through Stage 2 of the MIDI-to-Audio pipeline."
    )
    parser.add_argument("midi_path", type=str, help="Path to multitrack MIDI file.")
    parser.add_argument("--config", type=str, default="config/default.yaml", help="Path to config YAML.")
    parser.add_argument("--profile", type=str, default=None, help="Optional profile override YAML.")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_file():
        print(f"Error: Config file not found at {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.profile:
        profile_path = Path(args.profile)
        if profile_path.is_file():
            with open(profile_path, "r", encoding="utf-8") as f:
                profile_data = yaml.safe_load(f)
                config.update(profile_data)

    print(f"[*] Initializing Deterministic Pipeline for: {args.midi_path}")
    orchestrator = PipelineOrchestrator(config)
    results = orchestrator.run_deterministic_spine(args.midi_path)

    print("[+] Execution successful.")
    print(f"    - Expressive MIDI: {results['groove_midi_path']}")
    print(f"    - Rendered Stems:  {len(results['stems'])} files written.")
    for stem in results["stems"]:
        print(f"      * [{stem['role']}] {stem['stem_name']} -> {stem['audio_path']}")


if __name__ == "__main__":
    main()
    