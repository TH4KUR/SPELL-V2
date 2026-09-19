#!/bin/bash
# Relay-archive drainer (Ada storage policy — PROTOCOL.md §5).
#
# Compute nodes CANNOT see /share1 (mounts vary per node; verified). Runs
# therefore bundle under runs/<track>/<run_id>/ and THIS script — executed
# from a login/mounted node — copies validated bundles to the archive.
#
# COPY, not move, by default (2026-09-19): other scripts (e.g.
# summarize_track_b.py, run from a compute node that cannot see /share1 at
# all — §5 item 2) sometimes need the relay copy to still be there after
# draining. The relay lives on $HOME, which has a HARD 30 GB / 300k-inode
# quota (§5, gated loudly at every job start by check_storage.py --strict)
# — copy-forever means the relay never shrinks on its own, so it WILL climb
# toward that quota as more runs drain over the project's lifetime. Pass
# --prune-relay to restore the old delete-after-verify behavior for a given
# invocation once you're sure nothing local still needs those bundles.
#
# /share1 quota: 100 GB but ~3200 INODES → bundles only (ckpt + metrics.parquet,
# <=~20 files per run). Never place loose per-utterance files on /share1.
#
# PROVENANCE (§10 item 9): COMPLETED is trainer-attested success, written
# in-process by the training entrypoint. A marker ALONE proves nothing — this
# script independently validates content before moving ANYTHING. A bundle is
# DRAINABLE iff ALL hold:
#   1. COMPLETED present                          (shell-side check)
#   2. metrics.parquet loads AND >=1 epoch row for BOTH splits   (probe)
#   3. last.ckpt exists AND loads with torch.load(weights_only=True)  (probe)
#   4. run_manifest.json present w/ keys config_hash git_sha gpu_name
#      subset_manifest + train_seed|seed                          (probe)
#   5. <=20 contract files                        (shell-side check)
# Refusals NAME the failed check and those bundles STAY in the relay; every
# selected bundle gets a verdict; valid bundles still drain; exit is non-zero
# if anything was refused.
#
# NODE-CLASS SPLIT (2026-08-28): the frozen venv's base interpreter
# (/usr/local/apps/python-3.12.4) is mounted ONLY on compute nodes — module
# load succeeds on the login node but exposes no python. Content checks
# therefore run via ONE CPU srun job (full §5.8 scheduling string) executing
# scripts/drain_probe.py under the frozen venv on a compute node — the exact
# interpreter that trained the runs. The login node itself does NO python in
# srun mode: discovery, structural checks, rsync/diff/archive are shell-only.
# The srun blocks until the probe job runs (typically <1 min); a failed or
# under-reporting probe job REFUSES everything loudly — never silently passes.
#
# Usage:
#   scripts/drain_runs.sh [--archive "/share1/$USER/spell/runs"] [--runs-dir runs]
#                         [-n|--dry-run] [--verify-only] [--prune-relay] [RUN_ID...]
#     RUN_ID omitted -> validate+drain every marker discovered at bundle depth.
#   --verify-only  : validate everything, copy/move NOTHING (gates archiving).
#   --prune-relay  : after a verified byte-identical copy, DELETE the relay
#                    source too (the old default, move-not-copy, behavior) —
#                    opt in only once nothing local still needs the bundle.
#   SPELL_PROBE   : 'srun' (default when a slurm client exists) | 'local'
#                   (run scripts/drain_probe.py in THIS shell).
#   PYTHON_BIN    : local mode — probe interpreter (default $HOME/envs/spell/
#                   bin/python; NEVER bare "python": Ada login shells are
#                   CentOS 7 → /usr/bin/python is 2.7, 2026-08-27 incident).
#                   srun mode — optional override of the REMOTE probe python
#                   (default $HOME/envs/spell/bin/python on the compute node).

set -euo pipefail
cd "$(dirname "$0")/.."

ARCHIVE="${SPELL_ARCHIVE_ROOT:-/share1/${USER}/spell/runs}"   # FROZEN archive (§5.0)
RUNS_DIR="runs"
DRY_RUN=0
VERIFY_ONLY=0
PRUNE_RELAY=0
IDS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --archive) ARCHIVE="$2"; shift 2;;
        --runs-dir) RUNS_DIR="$2"; shift 2;;
        -n|--dry-run) DRY_RUN=1; shift;;
        --verify-only) VERIFY_ONLY=1; DRY_RUN=1; shift;;
        --prune-relay) PRUNE_RELAY=1; shift;;
        *) IDS+=("$1"); shift;;
    esac
done

REPO_ROOT="$PWD"
case "$RUNS_DIR" in
    /*) RUNS_ABS="$RUNS_DIR" ;;
    *)  RUNS_ABS="$REPO_ROOT/$RUNS_DIR" ;;
esac
PROBE_PY="$REPO_ROOT/scripts/drain_probe.py"

# ---- probe transport ----------------------------------------------------------
PROBE_MODE="${SPELL_PROBE:-}"
if [[ -z "$PROBE_MODE" ]]; then
    if command -v srun >/dev/null 2>&1; then
        PROBE_MODE=srun
    else
        PROBE_MODE=local
    fi
fi
case "$PROBE_MODE" in
    srun|local) ;;
    *)
        echo "drain: SPELL_PROBE must be 'srun' or 'local' (got '$PROBE_MODE')" >&2
        exit 1
        ;;
esac

PYTHON_BIN="${PYTHON_BIN:-$HOME/envs/spell/bin/python}"
if [[ "$PROBE_MODE" == "local" ]]; then
    # Local probe anchors the FROZEN env (§5.0/§5.5). The venv python symlinks
    # to the module interpreter — unusable on shells without it; refuse LOUDLY
    # with the real error + remedies (no silent fallback: a wrong python must
    # never pose as a verdict).
    _py3_ok() { "$1" -c 'import sys; assert sys.version_info[0] >= 3' 2>/dev/null; }

    if [[ ! -e "$PYTHON_BIN" ]]; then
        echo "drain: PYTHON_BIN='$PYTHON_BIN' does not exist on this node." >&2
        echo "       (Frozen env lives in \$HOME/envs/spell; if absent, setup_env" >&2
        echo "       (§5.5) has not run yet. On Ada login nodes the venv python is" >&2
        echo "       NOT usable — use srun mode, the default when slurm exists.)" >&2
        exit 1
    fi
    if ! _py3_ok "$PYTHON_BIN"; then
        echo "drain: PYTHON_BIN='$PYTHON_BIN' is not a usable python3 on this node." >&2
        echo "       resolves to: $(readlink -f "$PYTHON_BIN" 2>/dev/null || echo '?')" >&2
        "$PYTHON_BIN" -V 2>&1 | sed 's/^/       interpreter says: /' >&2 || true
        echo "       remedy A: module load u22/python/3.12.4   # in THIS shell, then rerun" >&2
        echo "       remedy B: PYTHON_BIN=/path/to/python3 bash scripts/drain_runs.sh ..." >&2
        exit 1
    fi
fi

if [[ ! -d "$ARCHIVE" ]]; then
    echo "drain: archive $ARCHIVE not reachable from this node." >&2
    echo "       Run this from a node that mounts it (mounts vary per node)." >&2
    exit 1
fi

if [[ ${#IDS[@]} -eq 0 ]]; then
    # Documented layout ONLY: <runs_dir>/<track>/<run_id>/COMPLETED (depth 3).
    # Depth was silently wrong (mindepth/maxdepth 2) until 2026-08-27 — bare
    # discovery never matched a real nested bundle because no test drove this
    # path. The depth regression test pins exact discovery behaviour.
    mapfile -t IDS < <(find "$RUNS_DIR" -mindepth 3 -maxdepth 3 \
                           -type f -name COMPLETED -printf '%h\n' 2>/dev/null \
                       | sed "s|^$RUNS_DIR/||" | LC_ALL=C sort -u)
    if [[ ${#IDS[@]} -eq 0 ]]; then
        echo "drain: no completed bundles under $RUNS_DIR — nothing to do."
        exit 0
    fi
fi

# ---- phase 1a: structural checks (shell-only, no python needed) ---------------
declare -a PROBED_IDS=()          # structurally plausible -> content probe
declare -A STRUCT_REASON=()       # run_id -> structural refusal reason
for run_id in "${IDS[@]}"; do
    src="$RUNS_ABS/$run_id"
    reason=""
    if [[ ! -d "$src" ]]; then
        reason="bundle directory does not exist"
    elif [[ ! -f "$src/COMPLETED" ]]; then
        reason="no COMPLETED marker"
    else
        n_files=$(find "$src" -type f | wc -l)
        if (( n_files > 21 )); then            # 20 contract files + the marker
            reason="$n_files files > cap (20/run inode discipline)"
        fi
    fi
    if [[ -z "$reason" ]]; then
        PROBED_IDS+=("$run_id")
    else
        STRUCT_REASON["$run_id"]="$reason"
    fi
done

# ---- phase 1b: content probe — ONE job for ALL candidates ---------------------
declare -A CONTENT_OK=()          # abs bundle dir -> present iff content-ok
declare -A CONTENT_WHY=()         # abs bundle dir -> named failure reason
PROBE_NOTE=""                     # global probe failure note (set => distrust)

# run_content_probe: fills CONTENT_OK / CONTENT_WHY / PROBE_NOTE for every
# PROBED_IDS entry. A probe that cannot run, crashes, or reports fewer
# verdicts than bundles sets PROBE_NOTE — and a missing verdict is a REFUSAL,
# never a silent pass (2026-08-27 law).
LISTFILE=""
run_content_probe() {
    LISTFILE="$HOME/.spell_drain_probe.$$.txt"
    : > "$LISTFILE"
    local d
    for d in "${PROBED_IDS[@]}"; do
        printf '%s\n' "$RUNS_ABS/$d" >> "$LISTFILE"
    done
    trap 'rm -f "$LISTFILE"' EXIT

    local out rc=0
    if [[ "$PROBE_MODE" == "srun" ]]; then
        local hq lfq ppq payload
        printf -v hq %q "$HOME"
        printf -v lfq %q "$LISTFILE"
        printf -v ppq %q "$PROBE_PY"
        payload="command -v module >/dev/null 2>&1 && module load u22/python/3.12.4; "\
"exec \"\${PYTHON_BIN:-$hq/envs/spell/bin/python}\" $ppq --list $lfq"
        out=$(srun -p u22 -A research --qos=medium --constraint=2080ti \
                  --exclude=gnode066 --gres=gpu:0 -N1 -n1 -t 10 \
                  -J spell-drain-probe --export=ALL \
                  bash -lc "$payload") || rc=$?
    else
        out=$("$PYTHON_BIN" "$PROBE_PY" --list "$LISTFILE") || rc=$?
    fi

    local tag dir status reason
    while IFS=$'\t' read -r tag dir status reason; do
        if [[ "$tag" == "PROBE-ENV-FAIL" ]]; then
            PROBE_NOTE="probe env broken: ${dir}"
            continue
        fi
        [[ "$tag" == "BUNDLE" ]] || continue
        if [[ "$status" == "content-ok" ]]; then
            CONTENT_OK["$dir"]=1
        elif [[ "$status" == "FAIL" ]]; then
            CONTENT_WHY["$dir"]="$reason"
        fi
    done <<< "$out"

    if [[ -z "$PROBE_NOTE" ]] && (( rc != 0 )); then
        PROBE_NOTE="probe job failed (rc=$rc)"
    fi
    if [[ -z "$PROBE_NOTE" ]] \
       && (( ${#CONTENT_OK[@]} + ${#CONTENT_WHY[@]} < ${#PROBED_IDS[@]} )); then
        PROBE_NOTE="probe protocol violation (fewer verdict lines than bundles; stdout noise?)"
    fi
}

if (( ${#PROBED_IDS[@]} > 0 )); then
    run_content_probe
fi

# ---- phase 1c: verdicts — every selected bundle gets one, loudly --------------
declare -a REFUSED_IDS=()
declare -a VALIDATED_IDS=()
for run_id in "${IDS[@]}"; do
    src="$RUNS_ABS/$run_id"
    dst="$ARCHIVE/$run_id"
    if [[ -n "${STRUCT_REASON[$run_id]:-}" ]]; then
        reason="${STRUCT_REASON[$run_id]}"
    elif [[ -n "${CONTENT_OK[$src]:-}" ]]; then
        reason=""
    elif [[ -n "${CONTENT_WHY[$src]:-}" ]]; then
        reason="${CONTENT_WHY[$src]}"
    else
        reason="content probe: ${PROBE_NOTE:-no verdict for this bundle}"
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
        src="$RUNS_ABS/$run_id"
        dst="$ARCHIVE/$run_id"
        mkdir -p "$dst"
        # a failed rsync must not strand a HALF-COPIED bundle behind it
        rsync -a --checksum "$src/" "$dst/" \
            || { echo "rsync FAILED draining $run_id (partial copy removed)" >&2; \
                 rm -rf "$dst"; exit 1; }
        # verify byte-identical copy BEFORE touching the source bundle
        diff -r "$src" "$dst" >/dev/null \
            || { echo "VERIFY FAILED for $run_id" >&2; exit 1; }
        if (( PRUNE_RELAY == 1 )); then
            rm -rf "$src"
            echo "drain: OK $run_id -> $dst (relay pruned)"
        else
            echo "drain: OK $run_id -> $dst (relay copy KEPT — pass --prune-relay to remove it)"
        fi
    done
else
    echo "drain: --verify-only — copied/moved nothing."
fi

echo "drain: done (valid=$n_valid refused=$n_refused)"
if (( n_refused > 0 )); then
    exit 1
fi
