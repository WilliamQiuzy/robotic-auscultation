"""Per-run provenance stamp for reproducibility/traceability (a medical-device requirement).

Call ``run_stamp(seed=...)`` at the top of any training/eval run and log/save the returned dict
alongside outputs, so every artifact is traceable to a code commit + RNG seed + environment.
"""
from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from typing import Optional


def _git_sha(repo: Optional[str] = None) -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _git_dirty(repo: Optional[str] = None) -> bool:
    try:
        return bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo,
                                            stderr=subprocess.DEVNULL).decode().strip())
    except Exception:  # noqa: BLE001
        return False


def data_hash(path: str) -> str:
    """Cheap content hash of a manifest/labels file (for data traceability)."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except Exception:  # noqa: BLE001
        return "unknown"


def run_stamp(seed: Optional[int] = None, repo: Optional[str] = None, **extra) -> dict:
    """Reproducibility record: git commit (+dirty flag), seed, host/python/torch, plus any extras."""
    try:
        import torch
        tv = torch.__version__
    except Exception:  # noqa: BLE001
        tv = None
    return {"git_sha": _git_sha(repo), "git_dirty": _git_dirty(repo), "seed": seed,
            "python": platform.python_version(), "torch": tv, "host": platform.node(), **extra}
