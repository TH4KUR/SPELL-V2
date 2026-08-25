#!/bin/bash
#SBATCH --job-name=spell-rq2
#SBATCH --account=${SPELL_SLURM_ACCOUNT:-CHANGE_ME}
#SBATCH --partition=${SPELL_SLURM_PARTITION:-gpu}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=0-9%5
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -e logs/%x-%A_%a.err

# Generic SLURM array template. NOTE: this dev machine has no SLURM binaries —
# this template targets the cluster and is exercised only there.
#
# Per-array-task parameters come from a run plan TSV (one row per task index):
#   columns:  track  subset_manifest  seed  config_yaml  keep_local_traj
#   keep_local_traj=1 marks a LESS-designated run: it alone retains FULL
#   trajectory checkpoints locally; all other runs stream ckpts to the NAS and
#   keep only last.ckpt locally (PROTOCOL.md storage policy).
# Usage on the cluster:
#   sbatch --export=ALL,SPELL_RUN_PLAN=/path/to/plan.tsv slurm/template.sbatch

set -euo pipefail

module purge
# Adjust to the cluster's modules:
module load python/3.12 cuda/12.8

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs outputs checkpoints

# pymax-style venv expected at $HOME/bin/pymax (mirrors local setup)
source "$HOME/bin/pymax/bin/activate"

PLAN="${SPELL_RUN_PLAN:?set SPELL_RUN_PLAN to the run-plan TSV}"
readarray -t ROW < <(sed -n "$((SLURM_ARRAY_TASK_ID + 2))p" "$PLAN")   # +2: header+comment lines
TRACK=${ROW[0]}
MANIFEST=${ROW[1]}
SEED=${ROW[2]}
CFG=${ROW[3]}
KEEP_LOCAL_TRAJ=${ROW[4]:-0}

# --- storage gate (PROTOCOL.md): abort BEFORE training if any fs is low ---
python scripts/check_storage.py --strict

RUN_ID="${TRACK}/$(basename "$MANIFEST" .txt)_seed${SEED}_job${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
NAS_RUN_DIR="${SPELL_NAS_ROOT:-/share1/NAS/spell-rq2}/${RUN_ID}"
LOCAL_RUN_DIR="checkpoints/${RUN_ID}"
mkdir -p "$NAS_RUN_DIR" "$LOCAL_RUN_DIR"

echo "[spell] track=$TRACK manifest=$MANIFEST seed=$SEED cfg=$CFG task=$SLURM_ARRAY_TASK_ID"
echo "[spell] nas_run_dir=$NAS_RUN_DIR local_run_dir=$LOCAL_RUN_DIR keep_local_traj=$KEEP_LOCAL_TRAJ"

# Trainer contract (implemented Phase 1+):
#   * every ckpt is written THROUGH to $NAS_RUN_DIR as it is created;
#   * locally, only last.ckpt survives (updated atomically per save);
#   * keep_local_traj=1 (LESS-designated) keeps the full local trajectory.
if [ "$KEEP_LOCAL_TRAJ" != "1" ]; then
    export SPELL_CKPT_POLICY=nas_write_through   # local keeps only last.ckpt
else
    export SPELL_CKPT_POLICY=keep_local_trajectory
fi

python "scripts/train_${TRACK}.py" \
    --config "$CFG" \
    --subset-manifest "$MANIFEST" \
    --seed "$SEED" \
    --output-dir "$LOCAL_RUN_DIR" \
    --nas-dir "$NAS_RUN_DIR" \
    --run-manifest-out "outputs/run_manifest_${TRACK}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

# Post-run cleanup: enforce "local copies removed except last.ckpt".
find "$LOCAL_RUN_DIR" -name '*.ckpt' ! -name 'last.ckpt' -delete

echo "[spell] done"
