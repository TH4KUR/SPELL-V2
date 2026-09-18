#!/bin/bash
#SBATCH --job-name=score-dnsmos
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err
# DNSMOS scorer (roster #1) — CPU-only §5.8 job under the frozen env.
# Writes scores/dnsmos_scores.parquet INTO the clone; commit-from-Ada per §5
# item 12 afterwards. Weights are vendored (models/dnsmos/) — no internet.
# --time bumped 03:00:00 -> 06:00:00 (2026-09-18): job 2700964 was still
# running at 2:40:30 of a 3:00:00 wall with zero visibility into how far it
# had gotten — this same retrofit adds live W&B progress (§3.23/§3.24) so a
# future near-timeout is at least OBSERVABLE, but the wall itself needed
# headroom regardless.
# Logs to W&B LIVE (job_type=dnsmos_scoring): progress every --report-every
# (default 2000) utterances scored, each report carrying its own chunk
# duration (§3.24), not just cumulative elapsed.
set -euo pipefail
module purge
module load u22/python/3.12.4

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs
: "${SPELL_DATA_ROOT:=$HOME/spell/data}"
export SPELL_DATA_ROOT
[[ -d "$SPELL_DATA_ROOT" ]] || { echo "FATAL[data] data root missing: $SPELL_DATA_ROOT (§5.0)"; exit 3; }
source ~/envs/spell/bin/activate
python scripts/check_storage.py --strict
python scripts/score_dnsmos.py --out scores/dnsmos_scores.parquet \
    --job-id "${SLURM_JOB_ID}"
echo "DONE score_dnsmos — check the W&B run, then commit scores/ from this clone (§5 item 12)."