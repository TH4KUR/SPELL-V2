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
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err
# DNSMOS scorer (roster #1) — CPU-only §5.8 job under the frozen env.
# Writes scores/dnsmos_scores.parquet INTO the clone; commit-from-Ada per §5
# item 12 afterwards. Weights are vendored (models/dnsmos/) — no internet.
# --time deliberately OMITTED (2026-09-19, was 03:00:00 -> 06:00:00 ->
# removed): job 2701385 hit the 06:00:00 wall at 82.1% done and lost ALL of
# it -- nothing was written to disk before the final write, so the timeout
# discarded ~5 hours of completed work outright. Omitting --time falls back
# to the u22/medium partition-or-QOS default, which may itself still cap the
# job (check `sacctmgr show qos medium format=Name,MaxWall` if unsure) --
# this is NOT a guarantee of unlimited time. What actually fixes the
# LOSS-of-work problem is scripts/score_dnsmos.py's checkpoint/resume
# feature (added the same day): progress is written to
# scores/dnsmos_scores.partial.parquet on the same live cadence as W&B
# reports, and a resubmission of this exact command automatically resumes
# from there instead of restarting at utterance 0 -- so if this DOES still
# time out, it now costs one more resubmit, not the lost work itself.
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