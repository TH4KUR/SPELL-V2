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
#
# SPELL_DSIR_L2 / SPELL_DSIR_OUT (optional, default to the formal run's
# values) let a --l2 sensitivity check reuse this exact job/env setup instead
# of a one-off bypass command; point SPELL_DSIR_OUT outside scores/ for any
# such diagnostic run so it can never be mistaken for (or accidentally
# committed as) the real roster feed, e.g.:
#   sbatch --export=ALL,SPELL_DSIR_L2=1e-6,SPELL_DSIR_OUT=outputs/dsir_l2_diag.parquet \
#          slurm/score_dsir.sbatch
set -euo pipefail
module purge
module load u22/python/3.12.4

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs outputs
: "${SPELL_DATA_ROOT:=$HOME/spell/data}"
export SPELL_DATA_ROOT
[[ -d "$SPELL_DATA_ROOT" ]] || { echo "FATAL[data] data root missing: $SPELL_DATA_ROOT (§5.0)"; exit 3; }
: "${SPELL_DSIR_L2:=1e-4}"
: "${SPELL_DSIR_OUT:=scores/dsir_weights.parquet}"
source ~/envs/spell/bin/activate
python scripts/check_storage.py --strict
python scripts/score_dsir.py --out "$SPELL_DSIR_OUT" --l2 "$SPELL_DSIR_L2"
echo "DONE score_dsir (l2=${SPELL_DSIR_L2} -> ${SPELL_DSIR_OUT}) — read the"
echo "weight_spread line; commit scores/ (§5 item 12) ONLY for the formal (default) run."