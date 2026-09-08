#!/bin/bash
#SBATCH -p u22 -A research --qos=medium          # FROZEN scheduling constants (§5.8)
#SBATCH --constraint=2080ti --exclude=gnode066   # FULL string binds CPU jobs too (§5.8)
#SBATCH --gres=gpu:0                             # CPU-only build/reconcile job
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=spell-setup-env
#SBATCH -o logs/%x-%A.out
#SBATCH -e logs/%x-%A.err

# Build-or-reconcile the FROZEN Ada python env `~/envs/spell` from the repo's
# requirements.lock (PROTOCOL §5.0/§5.5).
#
# Contract:
#   * `requirements.lock` in the repo root IS the description of the venv.
#     A venv that isn't described by the lockfile is considered BROKEN.
#   * Missing-absent case: create ~/envs/spell fresh and install the exact lock.
#   * Existing env: RECONCILE — pip pins everything to the lock (no-op when
#     already exact). This is also how newly locked deps (e.g. pytest, lightning)
#     reach the live env without anyone hand-installing on the login node.
#   * torch comes FIRST from the cu124 extra index (the lock pins 2.6.0+cu124;
#     PyPI alone cannot resolve the +cu124 local version).
#   * NEVER deletes or recreates an existing venv outright (no second venv, ever);
#     repair-by-reconciliation only.
#
# Usage (from ~/spell/repo):
#   sbatch scripts/setup_env.sbatch

set -euo pipefail
module purge
module load u22/python/3.12.4

REPO_ROOT="${SLURM_SUBMIT_DIR}"
cd "$REPO_ROOT"
mkdir -p logs

LOCK="$REPO_ROOT/requirements.lock"
[[ -f "$LOCK" ]] || { echo "FATAL: $LOCK missing — cannot describe the env" >&2; exit 2; }

ENV_DIR="$HOME/envs/spell"
if [[ ! -d "$ENV_DIR" ]]; then
    echo "[setup_env] creating fresh venv at $ENV_DIR"
    python -m venv "$ENV_DIR"
fi
source "$ENV_DIR/bin/activate"
python -m pip install --upgrade pip --no-cache-dir

echo "[setup_env] step 1/2: pinned torch from the cu124 index"
python -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cu124 \
    "torch==2.6.0+cu124"

echo "[setup_env] step 2/2: reconcile full lockfile (installs anything missing,"
echo "            no-ops what already matches; mismatched pins are an install error)"
python -m pip install --no-cache-dir -r "$LOCK"

# onnxruntime is DELIBERATELY not in the lock (2026-09-09): it was briefly
# hand-installed for DNSMOS scoring, then confirmed to crash under this
# project's numpy>=2.0 pin (`AttributeError: _ARRAY_API not found` — a
# genuine C-ABI break, not a conservative version cap) with no version able
# to coexist with numpy in one pip resolve without --no-deps risk.
# score_dnsmos.py now runs dnsmos_model.DNSMOSTorch instead (a verified
# pure-PyTorch port of the vendored ONNX graph — see PROTOCOL §5.5 and
# models/dnsmos/PROVENANCE.md). Retire any leftover install from that
# episode so the live env matches the lock exactly.
python -m pip uninstall -y onnxruntime >/dev/null 2>&1 || true

# ---- smoke check: the suite-critical imports resolve. Nothing fancier — pip's
# ---- exact-pin reconcile above IS the version guarantee; we don't re-audit it.
python - <<'PY'
import importlib

mods = ["numpy", "pandas", "pyarrow", "yaml", "soundfile", "jiwer", "wandb",
        "pytest", "lightning", "pytorch_lightning", "torch",
        "dnsmos_model"]   # §5.5 suite-critical
for m in mods:
    importlib.import_module(m)
    print("[setup_env] import OK:", m)

import torch
if torch.cuda.is_available():
    # SLURM_JOB_GPUS is unreliable — device name is truth (§5.8)
    print("[setup_env] cuda:", torch.version.cuda, "| device:", torch.cuda.get_device_name(0))
else:
    print("[setup_env] cuda: not visible from this (gpu:0 CPU) allocation — expected here")
PY

echo "[setup_env] done — ~/envs/spell now satisfies requirements.lock"
