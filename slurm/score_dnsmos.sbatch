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
#SBATCH --time=08:00:00
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err
# DNSMOS scorer (roster #1) — CPU-only §5.8 job under the frozen env.
# Writes scores/dnsmos_scores.parquet INTO the clone; commit-from-Ada per §5
# item 12 afterwards. Weights are vendored (models/dnsmos/) — no internet.
# --time history: 03:00:00 -> 06:00:00 -> OMITTED (2026-09-19) -> 08:00:00
# (same day). Omitting --time was WRONG: job 2704082 died after only
# ~59 minutes, because SLURM applies the u22 PARTITION's own DefaultTime
# (01:00:00, confirmed via `scontrol show partition u22`) when --time is
# unset -- that is a SEPARATE setting from the medium QOS's MaxWall
# (4-00:00:00, §5 item 8a), which only caps what you're ALLOWED to
# request, not what you get by default. Real observed throughput WITH the
# thread-pinning fix (§3.27) is ~2.02 utt/s -> ~4h for the full 29,064-utt
# pool; 08:00:00 gives 2x margin. ALWAYS pass an explicit --time sized off
# real observed throughput -- never rely on cluster defaults again.
# scripts/score_dnsmos.py's checkpoint/resume feature (§3.30) means even if
# this DOES still time out, a resubmission of this exact command picks up
# from scores/dnsmos_scores.partial.parquet instead of restarting at
# utterance 0 -- job 2704082's ~59 minutes of progress was NOT lost.
# Logs to W&B LIVE (job_type=dnsmos_scoring): progress every --report-every-s
# (default 30) seconds of wall clock, each report carrying an ETA plus its
# own chunk duration (§3.24/§3.25); a crash/exception also gets logged onto
# the run's summary (error + traceback), not just an opaque exit_code=1.
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