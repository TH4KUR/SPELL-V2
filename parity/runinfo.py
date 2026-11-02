"""Provenance fields shared by the parity scripts' summaries."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_state() -> dict:
    """``{"git_rev": sha|None, "git_dirty": bool|None}`` for the repo checkout."""
    def run(*args: str) -> str | None:
        try:
            out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True,
                                 text=True, timeout=10, check=True)
            return out.stdout
        except Exception:
            return None

    rev = run("rev-parse", "HEAD")
    porcelain = run("status", "--porcelain", "--untracked-files=no")
    return {
        "git_rev": rev.strip() if rev and rev.strip() else None,
        "git_dirty": None if porcelain is None else bool(porcelain.strip()),
    }


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()
