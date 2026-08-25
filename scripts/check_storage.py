#!/usr/bin/env python
"""Storage gate (PROTOCOL.md storage policy).

Checks, before any training/array launch:
  1. NAS root exists and is writable (trajectory checkpoints stream there).
  2. Free space on the LOCAL checkpoint filesystem >= min_free_gb_local.
  3. Free space on the NAS filesystem          >= min_free_gb_nas.

Exit behavior:
  default      : report only — soft-fail so single-GPU dev boxes without /share1 work
  --strict     : exit 1 on ANY failure. slurm/template.sbatch runs this mode so
                 array jobs abort at launch rather than mid-training.

    python scripts/check_storage.py [--strict] [--nas-root DIR]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_paths  # noqa: E402


def _free_gb(path: Path) -> float | None:
    try:
        return shutil.disk_usage(path).free / 1e9
    except OSError:
        return None


def run_checks(nas_root: Path, min_free_local_gb: float, min_free_nas_gb: float) -> list[str]:
    """Return a list of failure strings; empty list == all gates pass."""
    failures: list[str] = []

    local_free = _free_gb(PROJECT_ROOT)
    if local_free is None:
        failures.append(f"cannot stat local filesystem at {PROJECT_ROOT}")
    elif local_free < min_free_local_gb:
        failures.append(
            f"LOCAL disk low: {local_free:.1f} GB free < {min_free_local_gb} GB "
            "(delete outputs/ or old checkpoints before launching)"
        )

    if not nas_root.exists():
        failures.append(f"NAS root {nas_root} does not exist (mount it or set SPELL_NAS_ROOT)")
    else:
        try:
            with tempfile.NamedTemporaryFile(dir=nas_root, prefix=".gate_", delete=True):
                pass
        except OSError as e:
            failures.append(f"NAS root {nas_root} is not writable: {e}")
        nas_free = _free_gb(nas_root)
        if nas_free is not None and nas_free < min_free_nas_gb:
            failures.append(
                f"NAS low: {nas_free:.1f} GB free < {min_free_nas_gb} GB on {nas_root}"
            )
    return failures


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--strict", action="store_true",
                    help="exit nonzero on any failure (used by SLURM arrays)")
    ap.add_argument("--nas-root", type=Path, default=None,
                    help="override NAS root (else SPELL_NAS_ROOT env, else paths.yaml)")
    args = ap.parse_args(argv)

    paths = load_paths()
    nas_root = args.nas_root or paths.nas_root
    failures = run_checks(nas_root, paths.min_free_gb_local, paths.min_free_gb_nas)

    print(f"storage gate: nas_root={nas_root} "
          f"(local>={paths.min_free_gb_local}GB, nas>={paths.min_free_gb_nas}GB)")
    if not failures:
        print("storage gate: PASS")
        return 0
    for f in failures:
        print(f"storage gate: FAIL — {f}")
    if args.strict:
        return 1
    print("storage gate: soft mode (dev box) — continuing despite failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
