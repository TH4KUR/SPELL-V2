#!/bin/bash
#SBATCH --job-name=parity-train
#SBATCH -p u22
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH --constraint=2080ti
#SBATCH --exclude=gnode066
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err
# RAVEn-parity Phase 2 (PROTOCOL §3.34): from-scratch training of the vendored RAVEn Base
# audio-only ASR model on the parity TRAIN split under ONE pre-registered recipe, validated on the
# parity VAL split. The official test split is never read by this job; it is scored once, later, by
# slurm/parity_eval.sbatch for the recipe chosen on val WER (needs the SPELL_FINAL_EVAL unlock, which
# this file never sets).
#
#   mkdir -p logs        # SLURM opens the -o/-e files before this script runs
#   # 1) 1-epoch timing run (memory and speed on a real 2080 Ti with the recipe's own schedule; the beam
#   #    pass over 20 val utterances times the slow decoder after one epoch):
#   sbatch --time=02:00:00 --export=ALL,PARITY_RECIPE=R-B,PARITY_STOP_AFTER_EPOCHS=1,PARITY_BEAM_VAL_EVERY=1,PARITY_BEAM_VAL_LIMIT=20 \
#          slurm/parity_train.sbatch
#   # 2) the full runs, one job per recipe:
#   sbatch --export=ALL,PARITY_RECIPE=R-A slurm/parity_train.sbatch
#   sbatch --export=ALL,PARITY_RECIPE=R-B slurm/parity_train.sbatch
#   # 3) continue a run that hit the wall limit or died (outputs/parity/train_<recipe>_<RUN_ID>/last.pt is
#   #    rewritten atomically at the end of every epoch; <RUN_ID> is the FIRST job's id):
#   sbatch --export=ALL,PARITY_RECIPE=R-B,PARITY_RESUME=1,PARITY_RUN_ID=<first job id> slurm/parity_train.sbatch
#
# Inputs (all outside the frozen env; nothing here is tracked by the job itself):
#   PARITY_RECIPE          R-A or R-B (configs/parity/recipes.yaml, pinned by sha256); required
#   PARITY_RESUME          1 = continue <out>/last.pt (needs PARITY_RUN_ID); anything else = fresh run
#   PARITY_RUN_ID          directory tag; default $SLURM_JOB_ID. Resubmissions pass the first job's id.
#   PARITY_TOKENS_DIR      tokenizer artifacts from scripts/parity_train_spm.py (default parity/spm in
#                          the clone: unigram1000.model, unigram1000_units.txt, token_ids_trainval.tsv,
#                          tokenizer.json). They are produced on the LAPTOP (sentencepiece is not in the
#                          frozen env and a second venv is forbidden, §5.5) and arrive by git (§5.7).
#   PARITY_VALID_ID_FILE   AV-HuBERT lrs3-valid.id; only needed when parity/splits is not built in this
#                          clone yet (the job then runs scripts/parity_build_splits.py first)
#   PARITY_STOP_AFTER_EPOCHS, PARITY_BEAM_VAL_EVERY, PARITY_BEAM_VAL_LIMIT, PARITY_NUM_WORKERS (default 6),
#   PARITY_MICRO_BATCH_SECONDS (memory override; recorded as a deviation in summary.json): optional.
#   PARITY_NONDETERMINISTIC=1: let cuDNN pick its fastest convolution algorithms (about 1.5x faster on the
#   development GPU, recorded in summary.json). Default is PROTOCOL §10 item 1's deterministic warn_only.
# Outputs: outputs/parity/train_<recipe>_<RUN_ID>/ (last.pt, weights_epochNNNN.pt x last 10, final.pt,
#   metrics.jsonl, summary.json, val_beam_predictions.jsonl); about 3.1 GB per run (15 files). W&B run
#   parity-train-<recipe>-<job id>, job_type parity_train, live per-update series parity_train/*.
#
# --time=48:00:00: UNMEASURED ON ADA. Sizing: on the development laptop's RTX 4060 (8 GB, fp32 with TF32
# off, deterministic kernels) the real trainer measured about 110 audio-seconds per second end to end at
# 60 s micro-batches (900 synthetic LRS3-like utterances, 1 epoch, 15 updates; peak 6.1 GB allocated,
# 7.2 GB reserved; parity/REVIEW_STAGE2.md section 6), i.e. about 16 min per epoch of the 29.25 h train
# split (530 updates). R-B (75 epochs) is then about 20 h plus the final beam-search val pass, R-A (50
# epochs) about 14 h. A 2080 Ti has similar fp32 throughput but this is a guess; 48 h is 2.3x the R-B
# estimate. MaxWall is 4 days (§5 item 8a); a wall-limit kill loses at most one epoch and is resumed
# with PARITY_RESUME=1. Always explicit (§3.31). Run the 1-epoch timing job first and re-size --time
# from its summary.json (session_wall_s, history[0].epoch_wall_s) before the long runs.
set -euo pipefail
module purge
module load u22/python/3.12.4

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs outputs

fatal() { echo "FATAL[parity_train]: $*" >&2; exit 3; }

: "${SPELL_DATA_ROOT:=$HOME/spell/data}"
export SPELL_DATA_ROOT
[[ -d "$SPELL_DATA_ROOT" ]] || fatal "data root missing: ${SPELL_DATA_ROOT} (docs/layout.md; §5.0)"

RECIPE="${PARITY_RECIPE:?set PARITY_RECIPE=R-A or PARITY_RECIPE=R-B via --export}"
case "$RECIPE" in R-A|R-B) ;; *) fatal "PARITY_RECIPE must be R-A or R-B, got '$RECIPE'" ;; esac
RUN_ID="${PARITY_RUN_ID:-$SLURM_JOB_ID}"
OUT="outputs/parity/train_${RECIPE}_${RUN_ID}"
TOKENS_DIR="${PARITY_TOKENS_DIR:-parity/spm}"
[[ -f "$TOKENS_DIR/tokenizer.json" ]] \
    || fatal "tokenizer artifacts missing in ${TOKENS_DIR}: run scripts/parity_train_spm.py on the laptop and ship parity/spm"

EXTRA=()
if [[ "${PARITY_RESUME:-0}" == "1" ]]; then
    [[ -n "${PARITY_RUN_ID:-}" ]] || fatal "PARITY_RESUME=1 needs PARITY_RUN_ID=<first job id>"
    [[ -f "$OUT/last.pt" ]] || fatal "nothing to resume: ${OUT}/last.pt missing"
    EXTRA+=(--resume)
else
    [[ ! -e "$OUT" ]] || fatal "${OUT} exists: continue it with PARITY_RESUME=1 or pick another PARITY_RUN_ID"
fi
if [[ -n "${PARITY_STOP_AFTER_EPOCHS:-}" ]]; then EXTRA+=(--stop-after-epochs "$PARITY_STOP_AFTER_EPOCHS"); fi
if [[ -n "${PARITY_BEAM_VAL_EVERY:-}" ]]; then EXTRA+=(--beam-val-every "$PARITY_BEAM_VAL_EVERY"); fi
if [[ -n "${PARITY_BEAM_VAL_LIMIT:-}" ]]; then EXTRA+=(--beam-val-limit "$PARITY_BEAM_VAL_LIMIT"); fi
if [[ -n "${PARITY_MICRO_BATCH_SECONDS:-}" ]]; then EXTRA+=(--micro-batch-seconds "$PARITY_MICRO_BATCH_SECONDS"); fi
if [[ "${PARITY_NONDETERMINISTIC:-0}" == "1" ]]; then EXTRA+=(--no-deterministic-kernels); fi

source ~/envs/spell/bin/activate

# fp32 reproducibility: cuBLAS needs this workspace setting for deterministic kernels; expandable
# segments keep the caching allocator from fragmenting over variable-length batches (memory only).
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/check_storage.py --strict
python -c "import hardware_guard; print('[parity_train] gpu:', hardware_guard.enforce_gpu_policy())"

if [[ ! -f parity/splits/provenance.json ]]; then
    VALID_ID="${PARITY_VALID_ID_FILE:?parity/splits is not built here: set PARITY_VALID_ID_FILE=/path/to/lrs3-valid.id}"
    [[ -f "$VALID_ID" ]] || fatal "valid-id file missing: ${VALID_ID}"
    python scripts/parity_build_splits.py --valid-id-file "$VALID_ID"
fi

echo "[parity_train] recipe=${RECIPE} out=${OUT} resume=${PARITY_RESUME:-0} host=$(hostname) data_root=${SPELL_DATA_ROOT}"
python scripts/parity_train.py \
    --recipe "$RECIPE" \
    --tokens-dir "$TOKENS_DIR" \
    --out-dir "$OUT" \
    --job-id "${SLURM_JOB_ID}" \
    --num-workers "${PARITY_NUM_WORKERS:-6}" \
    ${EXTRA[@]+"${EXTRA[@]}"}
echo "DONE parity_train ${RECIPE} — read ${OUT}/summary.json and the W&B run; then, for the recipe chosen on val WER only, evaluate ${OUT}/final.pt with slurm/parity_eval.sbatch."
