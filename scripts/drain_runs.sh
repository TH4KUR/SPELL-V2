#!/bin/bash
# Relay-archive drainer (Ada storage policy — PROTOCOL.md §5).
#
# Compute nodes CANNOT see /share1 (mounts vary per node; verified). Runs
# therefore bundle under runs/<track>/<run_id>/ and THIS script — executed
# from a login/mounted node — moves validated bundles to the archive.
#
# /share1 quota: 100 GB but ~3200 INODES → bundles only (ckpt + metrics.parquet,
# <=~20 files per run). Never place loose per-utterance files on /share1.
#
# PROVENANCE (§10 item 9): COMPLETED is trainer-attested success, written
# in-process by the training entrypoint. A marker ALONE proves nothing — this
# script independently validates content before moving ANYTHING. A bundle is
# DRAINABLE iff ALL hold:
#   1. COMPLETED present                          (trainer-written)
#   2. metrics.parquet loads AND >=1 epoch row for BOTH splits
#   3. last.ckpt exists AND loads with torch.load(weights_only=True)
#   4. run_manifest.json present w/ keys config_hash git_sha gpu_name
#      subset_manifest + train_seed|seed
#   5. <=20 contract files                        (inode discipline)
# Refusals NAME the failed check and those bundles STAY in the relay; every
# selected bundle gets a verdict; valid bundles still drain; exit is non-zero
# if anything was refused.
#
# Discovery looks EXACTLY at the documented bundle depth <root>/<track>/<id>/
# (mindepth=maxdepth=3); anything else must be passed explicitly by ID.
#
# Usage:
#   scripts/drain_runs.sh [--archive "/share1/$USER/spell/runs"] [--runs-dir runs]
#                         [-n|--dry-run] [--verify-only] [RUN_ID...]
#     RUN_ID omitted -> validate+drain every marker discovered at bundle depth.
#   --verify-only : validate everything, move NOTHING (gates archiving).
#   PYTHON_BIN    : python used for probes (default "python"; tests override).

set -euo pipefail
cd "$(dirname "$0")/.."

ARCHIVE="${SPELL_ARCHIVE_ROOT:-/share1/${USER}/spell/runs}"   # FROZEN archive (§5.0)
RUNS_DIR="runs"
DRY_RUN=0
VERIFY_ONLY=0
IDS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --archive) ARCHIVE="$2"; shift 2;;
        --runs-dir) RUNS_DIR="$2"; shift 2;;
        -n|--dry-run) DRY_RUN=1; shift;;
        --verify-only) VERIFY_ONLY=1; DRY_RUN=1; shift;;
        *) IDS+=("$1"); shift;;
    esac
done

if [[ ! -d "$ARCHIVE" ]]; then
    echo "drain: archive $ARCHIVE not reachable from this node." >&2
    echo "       Run this from a node that mounts it (mounts vary per node)." >&2
    exit 1
fi

if [[ ${#IDS[@]} -eq 0 ]]; then
    # Documented layout ONLY: <runs_dir>/<track>/<run_id>/COMPLETED (depth 3).
    # Depth was silently wrong (mindepth/maxdepth 2) until 2026-08-27 — bare
    # discovery never matched a real nested bundle because no test drove this
    # path. The depth regression test below pins exact discovery behaviour.
    mapfile -t IDS < <(find "$RUNS_DIR" -mindepth 3 -maxdepth 3 \
                           -type f -name COMPLETED -printf '%h\n' 2>/dev/null \
                       | sed "s|^$RUNS_DIR/||" | LC_ALL=C sort -u)
    if [[ ${#IDS[@]} -eq 0 ]]; then
        echo "drain: no completed bundles under $RUNS_DIR — nothing to do."
        exit 0
    fi
fi

PYTHON_BIN="${PYTHON_BIN:-python}"

# check_bundle_content <bundle_dir>: stdout == "content-ok ..." iff checks
# 2..4 hold; otherwise prints the failing-check reason. Communicates validity
# through the FIRST WORD so callers never parse free-form prose.
check_bundle_content() {
    local src="$1"
    [[ -f "$src/metrics.parquet" ]] || { echo "FAIL metrics.parquet missing"; return 0; }
    BUNDLE_DIR="$src" "$PYTHON_BIN" - <<'PYEOF'
import json, os, sys

src = os.environ["BUNDLE_DIR"]
reason = ""
try:
    import pandas as pd
    df = pd.read_parquet(os.path.join(src, "metrics.parquet"))
    counts = {s: int((df["split"] == s).sum()) for s in ("train", "val")}
    if min(counts.values()) < 1:
        reason = ("metrics.parquet lacks >=1 epoch row for both splits "
                  f"(got train={counts['train']}, val={counts['val']})")
except Exception as exc:                                   # noqa: BLE001
    reason = f"metrics.parquet unreadable: {exc}"
if not reason:
    try:
        import torch
        torch.load(os.path.join(src, "last.ckpt"), map_location="cpu",
                   weights_only=True)
    except FileNotFoundError:
        reason = "last.ckpt missing"
    except Exception as exc:                               # noqa: BLE001
        reason = f"last.ckpt unreadable: {exc}"
if not reason:
    try:
        with open(os.path.join(src, "run_manifest.json")) as f:
            man = json.load(f)
        for k in ("config_hash", "git_sha", "gpu_name", "subset_manifest"):
            if k not in man:
                reason = f"run_manifest.json missing key '{k}'"
        if not reason and ("train_seed" not in man) and ("seed" not in man):
            reason = "run_manifest.json missing seed ('train_seed'/'seed')"
    except FileNotFoundError:
        reason = "run_manifest.json missing"
    except Exception as exc:                               # noqa: BLE001
        reason = f"run_manifest.json unreadable: {exc}"
print(f"FAIL {reason}" if reason else "content-ok")
PYEOF
}

# ---- phase 1: validate EVERY selected bundle loudly --------------------------
declare -a REFUSED_IDS=()
declare -a VALIDATED_IDS=()
for run_id in "${IDS[@]}"; do
    src="$RUNS_DIR/$run_id"
    dst="$ARCHIVE/$run_id"
    reason=""
    if [[ ! -d "$src" ]]; then
        reason="bundle directory does not exist"
    elif [[ ! -f "$src/COMPLETED" ]]; then
        reason="no COMPLETED marker"
    else
        n_files=$(find "$src" -type f | wc -l)
        if (( n_files > 21 )); then             # 20 contract files + the marker
            reason="$n_files files > cap (20/run inode discipline)"
        fi
    fi
    if [[ -z "$reason" ]]; then
        content=$(check_bundle_content "$src")
        [[ "$content" == "content-ok"* ]] || reason="${content#FAIL }"
    fi

    if [[ -n "$reason" ]]; then
        echo "REFUSE $run_id ($reason)" >&2
        REFUSED_IDS+=("$run_id")
    else
        echo "VALIDATE ok $run_id"
        VALIDATED_IDS+=("$run_id")
        if (( DRY_RUN == 1 )) && (( VERIFY_ONLY == 0 )); then
            echo "  [dry-run] would drain $run_id -> $dst"
        fi
    fi
done

n_refused=${#REFUSED_IDS[@]}
n_valid=${#VALIDATED_IDS[@]}
if (( n_valid == 0 )); then
    echo "drain: nothing to drain ($n_refused bundle(s) refused)." >&2
    exit 1
fi

# ---- phase 2: drain validated bundles only (skipped entirely on verify-only) -
if (( VERIFY_ONLY == 0 )); then
    for run_id in "${VALIDATED_IDS[@]}"; do
        src="$RUNS_DIR/$run_id"
        dst="$ARCHIVE/$run_id"
        mkdir -p "$dst"
        # a failed rsync must not strand a HALF-COPIED bundle behind it
        rsync -a --checksum "$src/" "$dst/" \
            || { echo "rsync FAILED draining $run_id (partial copy removed)" >&2; \
                 rm -rf "$dst"; exit 1; }
        # verify byte-identical copy BEFORE removing the source bundle
        diff -r "$src" "$dst" >/dev/null \
            || { echo "VERIFY FAILED for $run_id" >&2; exit 1; }
        rm -rf "$src"
        echo "drain: OK $run_id -> $dst"
    done
else
    echo "drain: --verify-only — moved nothing."
fi

echo "drain: done (valid=$n_valid refused=$n_refused)"
if (( n_refused > 0 )); then
    exit 1
fi
