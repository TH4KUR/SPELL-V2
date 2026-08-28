#!/usr/bin/env python3
"""Content-provenance probe for drain_runs.sh (PROTOCOL §10 item 9).

Runs wherever a python with pandas+torch exists — on Ada that is the FROZEN
venv on a COMPUTE node, reached by ONE CPU srun job carrying the full §5.8
scheduling string. The login node never runs python itself in srun mode: the
venv's base interpreter (/usr/local/apps/python-3.12.4) is not mounted there
(2026-08-28), so only the shell-side work (discovery, structural checks,
rsync/diff/archive) happens on the node that sees /share1.

Protocol (stdout ONLY, exactly one line per input, in order, TAB-separated):
    BUNDLE<TAB><dir><TAB>content-ok
    BUNDLE<TAB><dir><TAB>FAIL<TAB><reason>
or, if this interpreter lacks pandas/torch:
    PROBE-ENV-FAIL<TAB><exception>            (exit 4)

Reads one absolute bundle dir per line from --list FILE. Checks 2-4 of the
drainability contract (COMPLETED presence and the <=20-file cap stay
shell-side in drain_runs.sh — they need no python):
    2. metrics.parquet loads AND >=1 epoch row for BOTH splits
    3. last.ckpt loads via torch.load(weights_only=True)
    4. run_manifest.json has config_hash git_sha gpu_name subset_manifest
       + a seed under 'train_seed' or 'seed'
"""

import json
import os
import sys


def _check(src: str, pd, torch) -> str:
    """Return '' iff checks 2-4 hold; otherwise the named failure reason."""
    if not os.path.isfile(os.path.join(src, "metrics.parquet")):
        return "metrics.parquet missing"
    try:
        df = pd.read_parquet(os.path.join(src, "metrics.parquet"))
        counts = {s: int((df["split"] == s).sum()) for s in ("train", "val")}
        if min(counts.values()) < 1:
            return ("metrics.parquet lacks >=1 epoch row for both splits "
                    f"(got train={counts['train']}, val={counts['val']})")
    except Exception as exc:                                   # noqa: BLE001
        return f"metrics.parquet unreadable: {exc}"
    try:
        torch.load(os.path.join(src, "last.ckpt"), map_location="cpu",
                   weights_only=True)
    except FileNotFoundError:
        return "last.ckpt missing"
    except Exception as exc:                                   # noqa: BLE001
        return f"last.ckpt unreadable: {exc}"
    try:
        with open(os.path.join(src, "run_manifest.json")) as f:
            man = json.load(f)
        for k in ("config_hash", "git_sha", "gpu_name", "subset_manifest"):
            if k not in man:
                return f"run_manifest.json missing key '{k}'"
        if ("train_seed" not in man) and ("seed" not in man):
            return "run_manifest.json missing seed ('train_seed'/'seed')"
    except FileNotFoundError:
        return "run_manifest.json missing"
    except Exception as exc:                                   # noqa: BLE001
        return f"run_manifest.json unreadable: {exc}"
    return ""


def main() -> int:
    args = sys.argv[1:]
    if len(args) != 2 or args[0] != "--list":
        print("usage: drain_probe.py --list LISTFILE", file=sys.stderr)
        return 2
    try:
        import pandas as pd
        import torch
    except Exception as exc:                                   # noqa: BLE001
        # python3 without the frozen env's libs must fail NAMED, not as a
        # misleading per-bundle "unreadable" verdict
        print(f"PROBE-ENV-FAIL\t{str(exc).replace(chr(9), ' ')}")
        return 4

    with open(args[1]) as f:
        bundle_dirs = [ln.rstrip("\n") for ln in f if ln.strip()]

    for d in bundle_dirs:
        reason = _check(d, pd, torch)
        if reason:
            print(f"BUNDLE\t{d}\tFAIL\t{reason.replace(chr(9), ' ')}")
        else:
            print(f"BUNDLE\t{d}\tcontent-ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
