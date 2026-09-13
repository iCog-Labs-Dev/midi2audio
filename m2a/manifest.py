"""Artifact provenance, SHA-256 integrity tracking, and stage audit manifests."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


def compute_file_sha256(filepath: Path | str) -> str:
    """Compute the hex-encoded SHA-256 hash of a file on disk."""
    hasher = hashlib.sha256()
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"Cannot compute hash: file does not exist: {path}")
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def compute_dict_sha256(payload: Dict[str, Any]) -> str:
    """Compute canonical SHA-256 hash of a JSON-serializable dictionary."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ArtifactManifest(BaseModel):
    """Execution sidecar capturing lineage, parameters, and on-disk cryptographic digests."""

    stage_name: str
    code_version: str = "0.1.0"
    timestamp_utc: float = Field(default_factory=time.time)
    input_hashes: Dict[str, str] = Field(default_factory=dict)
    output_hashes: Dict[str, str] = Field(default_factory=dict)
    config_snapshot: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def write_sidecar(self, output_path: Path | str) -> Path:
        """Write the manifest sidecar JSON to the given path alongside artifacts."""
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            f.write(self.model_dump_json(indent=2))
        return target