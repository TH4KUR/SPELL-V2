#!/bin/bash
#SBATCH --job-name=sweep-dsir-l2
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
# DSIR --l2 held-out sweep (research decision support, NOT a roster scorer —
# writes nothing under scores/). Held-out cross-validation across several l2
# values to tell "l2 was suppressing real signal" apart from "we're now
# overfitting noise" — the ad-hoc single-point --l2 diagnostic can't
# distinguish those. Logs to W&B under job_type=dsir_l2_sweep, LIVE as each
# l2 point finishes with its own duration (§3.24 — see
# scripts/sweep_dsir_l2.py's docstring for why this doesn't touch the §3.17
# formal-run logging contract). Writes outputs/dsir_l2_sweep.parquet.
#
# Usage (from ~/spell/repo):
#   sbatch slurm/sweep_dsir_l2.sbatch
#   sbatch --export=ALL,SPELL_DSIR_SWEEP_L2S="1e-6 1e-5 1e-4" slurm/sweep_dsir_l2.sbatch
set -euo pipefail
module purge
module load u22/python/3.12.4

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs outputs
: "${SPELL_DATA_ROOT:=$HOME/spell/data}"
export SPELL_DATA_ROOT
[[ -d "$SPELL_DATA_ROOT" ]] || { echo "FATAL[data] data root missing: $SPELL_DATA_ROOT (§5.0)"; exit 3; }
: "${SPELL_DSIR_SWEEP_L2S:=1e-6 3e-6 1e-5 3e-5 1e-4 3e-4 1e-3}"
source ~/envs/spell/bin/activate
python scripts/check_storage.py --strict
python scripts/sweep_dsir_l2.py --l2s ${SPELL_DSIR_SWEEP_L2S} \
    --out outputs/dsir_l2_sweep.parquet --job-id "${SLURM_JOB_ID}"
echo "DONE sweep_dsir_l2 — check the RECOMMENDATION line and the W&B run."
