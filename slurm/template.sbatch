#!/bin/bash
#SBATCH --job-name=spell-rq2
#SBATCH -p u22 -A research --qos=medium          # FROZEN scheduling constants (§5.8)
#SBATCH --constraint=2080ti --exclude=gnode066   # no-drift GPU pool + documented bad node
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%5
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

# SLURM array template — Ada revision (PROTOCOL.md §5 + §5.0 frozen constants).
#
# HARD CONSTRAINTS (verified on the cluster):
#   * Formal runs are 2080 Ti PERMANENTLY (ihub/3080 partition inaccessible).
#     A drifted card (observed RTX 3080 on gnode077) must abort — the guard below.
#   * Compute nodes CANNOT write /share1 (mounts vary per node). Everything is
#     written under $HOME (quota 30G/300k inodes; gates warn 20G / abort 23G).
#   * Relay archiving ONLY: bundles land in runs/<run_id>/ under the relay dir
#     ~/spell/runs with a COMPLETED marker; scripts/drain_runs.sh moves them to
#     /share1/$USER/spell/runs from a mounted node (§5.0 canonical archive).
#   * This job ALWAYS runs from ~/spell/repo, which is synced ONLY via git
#     (§5.7 — bootstrap DONE since Phase 0b): laptop commits -> pushes to its
#     `ada` remote; this checkout pulls. rsync or any bulk copy into it is
#     FORBIDDEN (clobbers divergence).
#   * SLURM_JOB_GPUS is unreliable here: CUDA_VISIBLE_DEVICES /
#     torch.cuda.get_device_name() are the source of truth (assert_gpu reads them).
#
# Per-array-task parameters come from a run plan TSV (one row per task index):
#   columns:  track  subset_manifest  seed  config_yaml  keep_local_traj
# Usage:
#   squeue --me                                   # MUST show zero stale rows first (§5.8)
#   sbatch --export=ALL,SPELL_RUN_PLAN=$HOME/spell/repo/slurm/run_plan_pilots.tsv \
#       slurm/template.sbatch                     # from ~/spell/repo

set -euo pipefail

module purge
module load u22/python/3.12.4        # host is CentOS 7 / GLIBC 2.17 despite u22 names:
                                     # wheels must be manylinux2014-compatible.
                                     # cu124 wheels OK for observed drivers 570-580.

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs runs outputs checkpoints

source "$HOME/envs/spell/bin/activate"   # FROZEN env name (§5.0); pinned torch 2.6.0+cu124

PLAN="${SPELL_RUN_PLAN:?set SPELL_RUN_PLAN to the run-plan TSV}"
readarray -t ROW < <(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$PLAN")   # +2: header+comment lines
TRACK=${ROW[0]}
MANIFEST=${ROW[1]}
SEED=${ROW[2]}
CFG=${ROW[3]}
KEEP_LOCAL_TRAJ=${ROW[4]:-0}

RUN_ID="${TRACK}/$(basename "$MANIFEST" .txt)_seed${SEED}_job${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
RUN_DIR="runs/$RUN_ID"
mkdir -p "$RUN_DIR"

cleanup() {
    # Relay contract: mark the bundle complete so drain_runs.sh will take it.
    [[ -d "$RUN_DIR" ]] && touch "$RUN_DIR/COMPLETED"
}
trap cleanup EXIT

echo "[spell] track=$TRACK manifest=$MANIFEST seed=$SEED cfg=$CFG task=$SLURM_ARRAY_TASK_ID"
echo "[spell] run_dir=$RUN_DIR keep_local_traj=$KEEP_LOCAL_TRAJ host=$(hostname)"

# --- gates BEFORE any training ---
python scripts/check_storage.py --strict                       # $HOME quota gate (warn 20G / abort 23G)
python -c "import hardware_guard; info = hardware_guard.assert_gpu(); print('[spell] gpu:', info)"  # drift guard

python "scripts/train_${TRACK}.py" \
    --config "$CFG" \
    --subset-manifest "$MANIFEST" \
    --seed "$SEED" \
    --output-dir "$RUN_DIR" \
    --run-manifest-out "outputs/run_manifest_${TRACK}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

echo "[spell] done — bundle in $RUN_DIR; drain later via scripts/drain_runs.sh $RUN_ID"
