"""Canonical audit hashes and temporary-only serialization."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def hash_k3_request(model_id: str, request: Any) -> str:
    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("K3 model_id required for request hash")
    return sha256_json({"model_id": model_id, "request": request})


def write_shadow_json(result: Any, path: Path, *, temp_root: Path | None = None) -> Path:
    """Write only inside an explicitly supplied OS temporary subtree."""
    root = Path(temp_root).resolve() if temp_root is not None else None
    if root is None or not root.is_relative_to(Path(tempfile.gettempdir()).resolve()):
        raise ValueError("shadow output requires an OS temporary root")
    target = Path(path).resolve()
    if not target.is_relative_to(root) or target.suffix not in {".json", ".jsonl"}:
        raise ValueError("shadow output path must be a JSON file inside temp_root")
    payload = result.to_dict()
    if payload.get("execution_allowed") is not False or payload.get("formal_cohort_effect") is not False:
        raise ValueError("non-shadow result")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    return target
