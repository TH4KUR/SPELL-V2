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
#SBATCH --time=03:00:00
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err
# DNSMOS scorer (roster #1) — CPU-only §5.8 job under the frozen env.
# Writes scores/dnsmos_scores.parquet INTO the clone; commit-from-Ada per §5
# item 12 afterwards. Weights are vendored (models/dnsmos/) — no internet.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs
: "${SPELL_DATA_ROOT:=$HOME/spell/data}"
export SPELL_DATA_ROOT
[[ -d "$SPELL_DATA_ROOT" ]] || { echo "FATAL[data] data root missing: $SPELL_DATA_ROOT (§5.0)"; exit 3; }
python scripts/check_storage.py --strict
source ~/envs/spell/bin/activate
python scripts/score_dnsmos.py --out scores/dnsmos_scores.parquet
echo "DONE score_dnsmos — commit scores/ from this clone (§5 item 12)."