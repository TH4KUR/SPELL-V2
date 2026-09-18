#!/bin/bash
#SBATCH --job-name=score-proxy
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err
# Proxy-trajectory scorer (roster #4 feed) — GPU job. Late-checkpoint CTC-loss
# and EL2N inference over the full train pool, feeding the deterministic
# lossrank/anti/el2n selectors (§3.20). Writes scores/proxy_scores.parquet
# INTO the clone; commit-from-Ada per §5 item 12 afterwards. Logs to W&B
# LIVE (job_type=proxy_scoring): per-checkpoint AND within-checkpoint
# progress every --report-every (default 2000) utterances, each report
# carrying its own chunk duration (§3.24), not just cumulative elapsed.
#
# Usage (from ~/spell/repo):
#   sbatch --export=ALL,SPELL_PROXY_BUNDLE=runs/track_b/<proxy_run_id> \
#          slurm/score_proxy.sbatch
set -euo pipefail
module purge
module load u22/python/3.12.4

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs

: "${SPELL_DATA_ROOT:=$HOME/spell/data}"
export SPELL_DATA_ROOT
[[ -d "$SPELL_DATA_ROOT" ]] || { echo "FATAL[data] data root missing: $SPELL_DATA_ROOT (§5.0)"; exit 3; }

BUNDLE="${SPELL_PROXY_BUNDLE:?set SPELL_PROXY_BUNDLE to the proxy run bundle (runs/track_b/<id>)}"
[[ -d "$BUNDLE" ]] || { echo "FATAL[bundle] no such proxy bundle: $BUNDLE"; exit 3; }

source ~/envs/spell/bin/activate
python scripts/check_storage.py --strict
python -c "import hardware_guard; info = hardware_guard.assert_gpu(); print('[score_proxy] gpu:', info)"

python scripts/score_proxy.py --bundle "$BUNDLE" --out scores/proxy_scores.parquet \
    --job-id "${SLURM_JOB_ID}"
echo "DONE score_proxy — check the W&B run, then commit scores/ from this clone (§5 item 12)."
