#!/bin/bash
# Relay-archive drainer (Ada storage policy — PROTOCOL.md §5).
#
# Compute nodes CANNOT see /share1 (mounts vary per node; verified). Runs
# therefore bundle under $HOME/spell/runs/<run_id>/ and THIS script — executed
# from a login/mounted node — moves completed bundles to the archive.
#
# /share1 quota: 100 GB but ~3200 INODES → bundles only (ckpt + metrics.parquet,
# <=~20 files per run). Never place loose per-utterance files on /share1.
#
# Usage:
#   scripts/drain_runs.sh [--archive /share1/NAS/spell-rq2] [--runs-dir runs] [-n|--dry-run] [RUN_ID...]
#     RUN_ID omitted -> drain every bundle containing a COMPLETED marker.

set -euo pipefail
cd "$(dirname "$0")/.."

ARCHIVE="${SPELL_ARCHIVE_ROOT:-/share1/NAS/spell-rq2}"
RUNS_DIR="runs"
DRY_RUN=0
IDS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --archive) ARCHIVE="$2"; shift 2;;
        --runs-dir) RUNS_DIR="$2"; shift 2;;
        -n|--dry-run) DRY_RUN=1; shift;;
        *) IDS+=("$1"); shift;;
    esac
done

if [[ ! -d "$ARCHIVE" ]]; then
    echo "drain: archive $ARCHIVE not reachable from this node." >&2
    echo "       Run this from a node that mounts it (mounts vary per node)." >&2
    exit 1
fi

if [[ ${#IDS[@]} -eq 0 ]]; then
    mapfile -t IDS < <(find "$RUNS_DIR" -mindepth 2 -maxdepth 2 -name COMPLETED -printf '%h\n' 2>/dev/null | sort)
    if [[ ${#IDS[@]} -eq 0 ]]; then
        echo "drain: no bundles marked COMPLETED under $RUNS_DIR — nothing to do."
        exit 0
    fi
fi

for run_id in "${IDS[@]}"; do
    src="$RUNS_DIR/$run_id"
    dst="$ARCHIVE/$run_id"
    if [[ ! -f "$src/COMPLETED" ]]; then
        echo "drain: SKIP $run_id (no COMPLETED marker)"
        continue
    fi
    n_files=$(find "$src" -type f ! -name COMPLETED | wc -l)
    if (( n_files > 20 )); then
        echo "drain: REFUSE $run_id ($n_files files > 20/run inode discipline)" >&2
        exit 1
    fi
    echo "drain: $run_id ($n_files files) -> $dst$( [[ $DRY_RUN -eq 1 ]] && echo ' [dry-run]' )"
    (( DRY_RUN == 1 )) && continue
    mkdir -p "$dst"
    rsync -a --checksum "$src/" "$dst/"
    # verify byte-identical copy BEFORE removing the source bundle
    diff -r "$src" "$dst" >/dev/null || { echo "drain: VERIFY FAILED for $run_id" >&2; exit 1; }
    rm -rf "$src"
    echo "drain: OK $run_id"
done

echo "drain: done."
