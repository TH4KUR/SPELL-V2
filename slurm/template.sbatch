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
#   columns:  track  subset_manifest  seed  config_yaml
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

echo "[spell] track=$TRACK manifest=$MANIFEST seed=$SEED cfg=$CFG task=$SLURM_ARRAY_TASK_ID"

python "scripts/train_${TRACK}.py" \
    --config "$CFG" \
    --subset-manifest "$MANIFEST" \
    --seed "$SEED" \
    --output-dir "checkpoints/${TRACK}/$(basename "$MANIFEST" .txt)_seed${SEED}" \
    --run-manifest-out "outputs/run_manifest_${TRACK}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.json"

echo "[spell] done"
