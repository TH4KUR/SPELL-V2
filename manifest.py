"""Run-manifest writer (PROTOCOL §6 contract).

Every training run writes ``run_manifest.json`` inside its bundle: subset manifest
path, subset/train seeds, config hash, git SHA, GPU name + driver, protocol
revision, start/end timestamps. This file is what makes a run auditable months
later — the paper's protocol-deviation log points at these.
"""

from __future__ import annotations

import json
import os
import platform
import re
import socket
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config import PROTOCOL_REVISION, PROJECT_ROOT, config_hash
from hardware_guard import gpu_info

MANIFEST_SCHEMA = "spell-rq2-run-manifest-v1"


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=10,
            check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _maybe_relative(path_str: str) -> str:
    """Shrink paths under the repo to repo-relative form for readable manifests."""
    try:
        p = Path(path_str).resolve()
        return str(p.relative_to(PROJECT_ROOT))
    except (ValueError, OSError):
        return path_str


def infer_subset_seed(manifest_path: str | Path) -> int | None:
    """Subset manifests embed their identity seed in the filename (..._seed101.txt);
    fixed-split manifests (e.g. splits/train_ids.txt ceilings) yield None."""
    m = re.search(r"seed(\d+)", Path(manifest_path).stem)
    return int(m.group(1)) if m else None


def write_run_manifest(
    out_path: str | Path,
    *,
    cfg: dict[str, Any],
    config_path: str | Path,
    subset_manifest: str | Path,
    train_seed: int,
    subset_seed: int | None = None,
    started_iso: str | None = None,
    finished_iso: str | None = None,
    status: str = "running",
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write (or overwrite) the manifest atomically; returns the payload dict."""
    if started_iso is None:
        started_iso = _utc_now_iso()
    payload: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "protocol_revision": PROTOCOL_REVISION,
        "status": status,
        "started_utc": started_iso,
        "finished_utc": finished_iso,
        "subset_manifest": _maybe_relative(str(subset_manifest)),
        "subset_seed": int(subset_seed) if subset_seed is not None
        else infer_subset_seed(subset_manifest),
        "train_seed": int(train_seed),
        "config_path": _maybe_relative(str(config_path)),
        "config_hash": config_hash(cfg),
        "git_sha": _git_sha(),
        "gpu_name": gpu_info().get("gpu_name"),
        "driver_version": gpu_info().get("driver_version"),
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": _torch_version(),
        "data_root_env": os.environ.get("SPELL_DATA_ROOT") or None,
    }
    if extra:
        payload.update(extra)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, out)
    return payload


def _torch_version() -> str | None:
    try:
        import torch

        return torch.__version__
    except Exception:
        return None
