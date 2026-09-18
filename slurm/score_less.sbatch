#!/bin/bash
#SBATCH --job-name=score-less
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err
# LESS scorer (roster #5, CTC-grad variant, Track B native) — GPU job.
# Full trajectory (ALL ckpt_epoch*.ckpt), grad-ENABLED forward passes (this is
# NOT a no_grad inference job like score_proxy.sbatch — budget accordingly;
# 06:00:00 is a starting point: 4 checkpoints + a val pass vs score_proxy's 2
# checkpoints, no-grad-only). Writes scores/less_influence_seed{201,202,203}.parquet
# INTO the clone; commit-from-Ada per §5 item 12 afterwards.
#
# Usage (from ~/spell/repo):
#   sbatch --export=ALL,SPELL_LESS_BUNDLE=runs/track_b/<proxy_run_id> \
#          slurm/score_less.sbatch
set -euo pipefail
module purge
module load u22/python/3.12.4

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs

: "${SPELL_DATA_ROOT:=$HOME/spell/data}"
export SPELL_DATA_ROOT
[[ -d "$SPELL_DATA_ROOT" ]] || { echo "FATAL[data] data root missing: $SPELL_DATA_ROOT (§5.0)"; exit 3; }

BUNDLE="${SPELL_LESS_BUNDLE:?set SPELL_LESS_BUNDLE to the proxy run bundle (runs/track_b/<id>)}"
[[ -d "$BUNDLE" ]] || { echo "FATAL[bundle] no such bundle: $BUNDLE"; exit 3; }

source ~/envs/spell/bin/activate
python scripts/check_storage.py --strict
python -c "import hardware_guard; info = hardware_guard.assert_gpu(); print('[score_less] gpu:', info)"

python scripts/score_less.py --bundle "$BUNDLE" --out-dir scores --seeds 201 202 203
echo "DONE score_less — commit scores/ from this clone (§5 item 12)."
