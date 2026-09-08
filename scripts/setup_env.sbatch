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

echo "[setup_env] step 2/3: reconcile the lockfile EXCEPT onnxruntime (installs"
echo "            anything missing, no-ops what already matches; mismatched pins"
echo "            are an install error)"
grep -v '^onnxruntime==' "$LOCK" > /tmp/spell-lock-no-onnxruntime.$$.txt
python -m pip install --no-cache-dir -r /tmp/spell-lock-no-onnxruntime.$$.txt
rm -f /tmp/spell-lock-no-onnxruntime.$$.txt

# onnxruntime==1.18.1 declares `numpy<2.0`, which HARD-CONFLICTS with
# scipy==1.18.1/lightning/torchmetrics needing numpy>=2.0 (pip's resolver
# refuses the two together: 2026-09-09 incident). Installing it with
# --no-deps skips that check entirely so numpy stays at the lock's pin, but
# CONFIRMED BROKEN AT RUNTIME (2026-09-09): its compiled pybind bindings were
# built against numpy 1.x and crash under numpy>=2.0 (`AttributeError:
# _ARRAY_API not found` — the canonical numpy-2.0 C-ABI break for an
# unrebuilt 1.x extension). This is not a conservative pin; DNSMOS scoring is
# BLOCKED on this dependency until it's replaced (newer onnxruntime build, or
# a pure-PyTorch port of score_dnsmos.py) — see PROTOCOL §5.5.
ORT_PIN="$(grep '^onnxruntime==' "$LOCK" | head -1)"
[[ -n "$ORT_PIN" ]] || { echo "FATAL: onnxruntime pin missing from $LOCK" >&2; exit 2; }
echo "[setup_env] step 3/3: ${ORT_PIN} via --no-deps (numpy<2.0 conflict, see comment above)"
python -m pip install --no-cache-dir --no-deps "$ORT_PIN"

# ---- smoke check: suite-critical imports resolve. onnxruntime is checked
# ---- SEPARATELY and NON-FATALLY — it is a KNOWN, tracked failure (above),
# ---- and must never mask a real regression in the packages that matter for
# ---- every other job (numpy/scipy/lightning etc., which DO fail this loop
# ---- hard, as they should).
python - <<'PY'
import importlib

mods = ["numpy", "pandas", "pyarrow", "yaml", "soundfile", "jiwer", "wandb",
        "pytest", "lightning", "pytorch_lightning", "torch"]   # §5.5 suite-critical
for m in mods:
    importlib.import_module(m)
    print("[setup_env] import OK:", m)

import torch
if torch.cuda.is_available():
    # SLURM_JOB_GPUS is unreliable — device name is truth (§5.8)
    print("[setup_env] cuda:", torch.version.cuda, "| device:", torch.cuda.get_device_name(0))
else:
    print("[setup_env] cuda: not visible from this (gpu:0 CPU) allocation — expected here")

try:
    importlib.import_module("onnxruntime")
    print("[setup_env] import OK: onnxruntime")
except Exception as e:
    print(f"[setup_env] WARNING: onnxruntime import FAILED (known issue, "
          f"DNSMOS scoring blocked, PROTOCOL §5.5): {e!r}")
PY

echo "[setup_env] done — ~/envs/spell satisfies requirements.lock (onnxruntime excepted, see above)"
