#!/bin/bash
#SBATCH --job-name=spell-rq2
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

# SLURM array template — Ada (PROTOCOL.md §5; §5.0 frozen constants).
# The #SBATCH block above is FLAGS ONLY: no inline comments (some schedulers
# mangle them), no pinned --array range (always passed on the sbatch command
# line), no --ntasks-per-node (single-task jobs default to 1).
#
# HARD CONSTRAINTS (verified on the cluster):
#   * Formal runs are 2080 Ti PERMANENTLY (ihub/3080 partition inaccessible).
#     A drifted card (observed RTX 3080 on gnode077) must abort — drift guard.
#   * Compute nodes CANNOT write /share1 (mounts vary per node). Everything is
#     written under $HOME (quota 30G/300k inodes; gates warn 20G / abort 23G).
#   * Relay archiving ONLY: bundles land in runs/<track>/<run_id>/ under this
#     checkout with a COMPLETED marker written by the EXIT trap below;
#     scripts/drain_runs.sh moves them to /share1/$USER/spell/runs (§5.0).
#   * This job ALWAYS runs from ~/spell/repo, synced ONLY via git (§5.7):
#     laptop commits -> pushes to its `ada` remote; this checkout pulls.
#     rsync or any bulk copy into it is FORBIDDEN. Editing ANY tracked file
#     in-place on the cluster (sed/python patch) is FORBIDDEN — it diverges
#     silently and forces bypass scripts (the Phase-1 pilot-A lesson, §10).
#   * SLURM_JOB_GPUS is unreliable here: CUDA_VISIBLE_DEVICES /
#     torch.cuda.get_device_name() are the source of truth.
#
# RUN PLAN (PROTOCOL §5 item 10): parameters come from a TSV whose task id
# lives IN THE DATA — first column matched against $SLURM_ARRAY_TASK_ID by
# scripts/plan_reader.awk (position/offset reads are FORBIDDEN). Schema:
#
#   task | track | subset_manifest | seed | config_yaml | keep_local_traj
#
# Plans are written ONLY by `python scripts/gen_run_plan.py` (hand edits are a
# protocol violation). This file must never contain a specific plan filename.
#
# Usage (from ~/spell/repo, after `squeue --me` shows zero stale rows — §5.8):
#   sbatch --export=ALL,SPELL_RUN_PLAN=$HOME/spell/repo/slurm/run_plan_pilots.tsv \
#          slurm/template.sbatch        # array selection passed via the CLI
# SPELL_DATA_ROOT defaults to $HOME/spell/data (§5.0) and is existence-gated
# below — pass it via --export only for non-canonical staged trees.

set -euo pipefail

module purge
module load u22/python/3.12.4   # CentOS 7 / GLIBC 2.17 host: manylinux2014 wheels;
                                # cu124 OK for observed drivers 570-580.

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs runs outputs checkpoints

source "$HOME/envs/spell/bin/activate"   # FROZEN env name (§5.0); torch 2.6.0+cu124

fatal() { echo "FATAL[template]: $*" >&2; exit 2; }

# DATA ROOT (§5.0 canonical): staged data lives ONLY at $HOME/spell/data.
# Default it here so a forgotten --export can never silently fall back to the
# configured raw-audit-source default (datasets/LRS3); explicit overrides win.
: "${SPELL_DATA_ROOT:=$HOME/spell/data}"
export SPELL_DATA_ROOT
[[ -d "$SPELL_DATA_ROOT" ]] || fatal "data root missing: ${SPELL_DATA_ROOT} (docs/layout.md; §5.0)"

PLAN="${SPELL_RUN_PLAN:?set SPELL_RUN_PLAN to the run-plan TSV}"
[[ -f "$PLAN" ]] || fatal "no such plan file: ${PLAN}"

# --- run-plan read: index-free, schema-checked, LOUD --------------------------
ROWLINE="$(awk -v t="${SLURM_ARRAY_TASK_ID:?array context required}" \
               -f "${REPO_ROOT}/scripts/plan_reader.awk" "$PLAN")"
IFS=$'\t' read -r TASK_ID TRACK MANIFEST SEED CFG KEEP_LOCAL_TRAJ <<< "$ROWLINE"

[[ "$TASK_ID" == "$SLURM_ARRAY_TASK_ID" ]] \
    || fatal "reader returned task '$TASK_ID' but array context is '$SLURM_ARRAY_TASK_ID'"
[[ -f "$MANIFEST" ]] || fatal "subset manifest does not exist under repo root: ${MANIFEST}"
[[ -f "$CFG" ]]      || fatal "config yaml does not exist under repo root: ${CFG}"
[[ "$SEED" =~ ^[0-9]+$ ]] || fatal "seed '$SEED' is not a non-negative integer"
[[ "$KEEP_LOCAL_TRAJ" =~ ^[01]$ ]] || fatal "keep_local_traj '$KEEP_LOCAL_TRAJ' must be 0 or 1"
case "$TRACK" in track_a|track_b) ;; *) fatal "unknown track '$TRACK'" ;; esac

RUN_ID="${TRACK}/$(basename "$MANIFEST" .txt)_seed${SEED}_job${SLURM_JOB_ID}_t${TASK_ID}"
RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR"

cleanup() {
    # Relay contract: mark the bundle complete so drain_runs.sh will take it.
    [[ -d "$RUN_DIR" ]] && touch "$RUN_DIR/COMPLETED"
}
trap cleanup EXIT

echo "[spell] task=$TASK_ID track=$TRACK manifest=$MANIFEST seed=$SEED cfg=$CFG"
echo "[spell] run_dir=$RUN_DIR keep_local_traj=$KEEP_LOCAL_TRAJ host=$(hostname)"
echo "[spell] data_root=${SPELL_DATA_ROOT}"

# --- gates BEFORE any training ---
python scripts/check_storage.py --strict
python -c "import hardware_guard; info = hardware_guard.assert_gpu(); print('[spell] gpu:', info)"

python "scripts/train_${TRACK}.py" \
    --config "$CFG" \
    --subset-manifest "$MANIFEST" \
    --seed "$SEED" \
    --output-dir "$RUN_DIR" \
    --run-manifest-out "outputs/run_manifest_${TRACK}_${SLURM_JOB_ID}_t${TASK_ID}.json"

echo "[spell] done — bundle in $RUN_DIR; drain later via scripts/drain_runs.sh $RUN_ID"
