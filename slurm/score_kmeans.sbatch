#!/bin/bash
#SBATCH --job-name=score-kmeans
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
# k-means diversity scorer (roster #2) — CPU-only §5.8 job under the frozen
# env, scipy only (no sklearn in the lock). Builds RVQ1 histogram features
# in-job (never committed), clusters per §3.20 identity seed, emits
# kmeans_assignments_seed{S}.parquet + token_stats.parquet. Commit-from-Ada
# per §5 item 12 afterwards.
set -euo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs
: "${SPELL_DATA_ROOT:=$HOME/spell/data}"
export SPELL_DATA_ROOT
[[ -d "$SPELL_DATA_ROOT" ]] || { echo "FATAL[data] data root missing: $SPELL_DATA_ROOT (§5.0)"; exit 3; }
python scripts/check_storage.py --strict
source ~/envs/spell/bin/activate
python scripts/score_kmeans.py --seeds 201 202 203 --k 64 --out-dir scores
echo "DONE score_kmeans — commit scores/ from this clone (§5 item 12)."