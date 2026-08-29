# m2a/artifacts.py
"""
Content-addressed artifact store for the midi2audio pipeline.

Every stage output lives under a directory whose name encodes the inputs
that produced it. Stages that already have a `_complete` sentinel are
skipped entirely — giving deterministic, incremental reruns for free.
"""
from __future__ import annotations

import hashlib
import json
import pathlib


def content_hash(*file_paths: str, config: dict, code_version: str) -> str:
    """Return a hex SHA-256 over the given files, config dict, and code version string.

    [v1.1] Files are streamed in 1 MiB blocks rather than loaded whole into RAM.
    Reference corpora and stems can be hundreds of megabytes; loading them all to
    hash would OOM on typical build machines.

    The full hex digest is returned. Callers that need a shorter directory name
    should truncate after calling this function (e.g. ``hash_[:16]``).
    """
    h = hashlib.sha256()
    for p in sorted(file_paths):          # sort for determinism regardless of call order
        with open(p, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
    h.update(json.dumps(config, sort_keys=True).encode())
    h.update(code_version.encode())
    return h.hexdigest()


class ArtifactStore:
    """Manages a flat directory of stage outputs, each identified by (stage, hash).

    Directory layout::

        work_dir/
          {stage}_{hash}/          # one directory per (stage, input-hash) pair
            _complete              # sentinel written by mark_complete()
            ...                    # stage-specific files

    Usage::

        store = ArtifactStore("/tmp/run_abc")
        key   = content_hash(midi_path, config=cfg, code_version="0.1.0")
        if store.is_complete("analysis", key):
            out_dir = store.stage_dir("analysis", key)
        else:
            out_dir = store.stage_dir("analysis", key)
            # ... produce outputs into out_dir ...
            store.mark_complete("analysis", key)
    """

    def __init__(self, work_dir: str) -> None:
        self.root = pathlib.Path(work_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    def stage_dir(self, stage: str, hash_: str) -> pathlib.Path:
        """Return (and create) the output directory for this (stage, hash) pair."""
        d = self.root / f"{stage}_{hash_}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def is_complete(self, stage: str, hash_: str) -> bool:
        """True if the stage has already produced output for this hash."""
        return (self.root / f"{stage}_{hash_}" / "_complete").exists()

    def mark_complete(self, stage: str, hash_: str) -> None:
        """Write the completion sentinel. Call only after all outputs are flushed.

        Creates the stage directory if it does not already exist, so callers
        that write directly into the directory via other means (e.g. shutil)
        do not need to call stage_dir() first.
        """
        d = self.stage_dir(stage, hash_)
        (d / "_complete").touch()
