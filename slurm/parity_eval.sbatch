#!/bin/bash
#SBATCH --job-name=parity-eval
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err
# RAVEn-parity Phase-1 gate (PROTOCOL §3.33): the released RAVEn Base audio-only ASR
# checkpoint decoded over one parity split by scripts/parity_eval.py. One split per job,
# chosen by PARITY_SPLIT; no training, no tuning.
#
#   mkdir -p logs        # SLURM opens the -o/-e files before this script runs
#   sbatch --export=ALL,PARITY_SPLIT=val  slurm/parity_eval.sbatch
#   # only after the val run looked sane (PROTOCOL §3.33 unlock):
#   SPELL_FINAL_EVAL=1 sbatch --export=ALL,PARITY_SPLIT=test slurm/parity_eval.sbatch
#
# A checkpoint written by slurm/parity_train.sbatch (its final.pt) is evaluated with the same
# decoding recipe by naming it, its tokenizer's units file and switching the released-file pin off:
#   sbatch --export=ALL,PARITY_SPLIT=val,PARITY_CKPT=outputs/parity/train_R-B_<id>/final.pt,\
#          PARITY_UNITS=parity/spm/unigram1000_units.txt,PARITY_EXPECT_SHA256=none slurm/parity_eval.sbatch
# (the same with SPELL_FINAL_EVAL=1 and PARITY_SPLIT=test, only for the recipe chosen on val WER and
# only after the Phase-1 gate has passed, PROTOCOL §3.34.) Beam search on a 1.15 h val split took
# 20-45 min estimated for the released checkpoint; a from-scratch model can decode more slowly.
#
# Inputs (all outside the repo; nothing here is tracked):
#   PARITY_CKPT           released checkpoint, sha256 78212d2b... (default
#                         $HOME/spell/ckpts/raven_base_lrs3_asr.pth; verified by the script)
#   PARITY_UNITS          optional units file of the checkpoint's tokenizer (default: RAVEn's vendored list)
#   PARITY_EXPECT_SHA256  optional checkpoint sha256 to require, or "none" (default: the released file's)
#   PARITY_VALID_ID_FILE  AV-HuBERT lrs3-valid.id, ONLY needed when parity/splits has not
#                         been built in this clone yet (the id lists are untracked); the job
#                         then runs scripts/parity_build_splits.py first (seconds, no GPU).
# SPELL_FINAL_EVAL is never set by this file: the unlock is the submitter's explicit act.
#
# --time=04:00:00: no Ada measurement exists yet. The paper (App. A.4) reports ~3 s to
# decode 10 s of footage with the Base model on an A100 (batch 1), and laptop-CPU decoding
# of real speech with this checkpoint at beam 40 took ~0.25 s per audio second. The parity
# val/test splits hold 1.15 h / 0.85 h of audio: at 0.3-0.6 s per audio second (the upper
# figure allowing for a 2080 Ti) that is ~20-45 min / ~15-30 min. 04:00:00 leaves >= 5x
# margin over the slow end (§3.31: always explicit, sized from observed throughput).
# predictions.jsonl is flushed per utterance, so a killed job still leaves its partial rows.
# Logs to W&B LIVE (job_type=parity_eval): progress, per-utterance timing, ETA and the
# running corpus WER every --report-every-s (default 30) wall-clock seconds.
set -euo pipefail
module purge
module load u22/python/3.12.4

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs outputs

fatal() { echo "FATAL[parity_eval]: $*" >&2; exit 3; }

: "${SPELL_DATA_ROOT:=$HOME/spell/data}"
export SPELL_DATA_ROOT
[[ -d "$SPELL_DATA_ROOT" ]] || fatal "data root missing: ${SPELL_DATA_ROOT} (docs/layout.md; §5.0)"

SPLIT="${PARITY_SPLIT:?set PARITY_SPLIT=val or PARITY_SPLIT=test via --export}"
case "$SPLIT" in val|test) ;; *) fatal "PARITY_SPLIT must be val or test, got '$SPLIT'" ;; esac
CKPT="${PARITY_CKPT:-$HOME/spell/ckpts/raven_base_lrs3_asr.pth}"
[[ -f "$CKPT" ]] || fatal "checkpoint missing: ${CKPT}"

source ~/envs/spell/bin/activate

python scripts/check_storage.py --strict

if [[ ! -f parity/splits/provenance.json ]]; then
    VALID_ID="${PARITY_VALID_ID_FILE:?parity/splits is not built here: set PARITY_VALID_ID_FILE=/path/to/lrs3-valid.id}"
    [[ -f "$VALID_ID" ]] || fatal "valid-id file missing: ${VALID_ID}"
    python scripts/parity_build_splits.py --valid-id-file "$VALID_ID"
fi

EXTRA=()
if [[ -n "${PARITY_UNITS:-}" ]]; then
    [[ -f "$PARITY_UNITS" ]] || fatal "units file missing: ${PARITY_UNITS}"
    EXTRA+=(--units "$PARITY_UNITS")
fi
if [[ -n "${PARITY_EXPECT_SHA256:-}" ]]; then
    EXTRA+=(--expect-sha256 "$PARITY_EXPECT_SHA256")
fi

echo "[parity_eval] split=${SPLIT} ckpt=${CKPT} host=$(hostname) data_root=${SPELL_DATA_ROOT}"
python scripts/parity_eval.py \
    --checkpoint "$CKPT" \
    --split "$SPLIT" \
    --job-id "${SLURM_JOB_ID}" \
    ${EXTRA[@]+"${EXTRA[@]}"}
echo "DONE parity_eval ${SPLIT} — read outputs/parity/eval_${SPLIT}_${SLURM_JOB_ID}/summary.json and the W&B run."
