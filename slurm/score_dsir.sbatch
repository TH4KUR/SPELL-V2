#!/bin/bash
#SBATCH --job-name=score-dsir
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err
# DSIR scorer (roster #3) — CPU-only §5.8 job, scipy sparse + L-BFGS-B (no
# sklearn). Fits hashed n-gram importance target(val) vs pool(train) —
# deterministic; §3.20 seeds live in the laptop-side resample. Reads val
# tokens in-job (never committed). GO/NO-GO: the printed weight_spread CV is
# the 72h gate. Commit-from-Ada per §5 item 12 afterwards.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs
: "${SPELL_DATA_ROOT:=$HOME/spell/data}"
export SPELL_DATA_ROOT
[[ -d "$SPELL_DATA_ROOT" ]] || { echo "FATAL[data] data root missing: $SPELL_DATA_ROOT (§5.0)"; exit 3; }
python scripts/check_storage.py --strict
source ~/envs/spell/bin/activate
python scripts/score_dsir.py --out scores/dsir_weights.parquet
echo "DONE score_dsir — read the weight_spread line, then commit scores/ (§5 item 12)."