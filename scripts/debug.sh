#!/bin/bash
# Single-GPU debug launcher — everything Phase 0 can check without training runs.
# Usage: bash scripts/debug.sh [pytest args...]
set -euo pipefail
cd "$(dirname "$0")/.."

source "$HOME/bin/pymax/bin/activate"

echo "== python =="
python --version
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

echo; echo "== protocol sanity =="
python - <<'PY'
import config
proto = config.load_protocol()
print(proto)
print("protocol hash:", config.config_hash(proto)[:16], "(full hash in run manifests)")
assert proto.crop_samples == proto.crop_frames * proto.sample_rate // proto.token_hz
print("crop law holds:", proto.crop_frames, "frames =", proto.crop_samples, "samples")
uni = config.universe_budget
print("budget rule check: universe_budget(31071, 0.25) =", uni(31071, 0.25), "utts")
PY

echo; echo "== storage gate (soft — dev box may lack /share1) =="
python scripts/check_storage.py

echo; echo "== unit tests =="
pytest -q "${@:--q}"

echo; echo "== quick data audit (first 20 folders/clips per split) =="
python scripts/audit_data.py --sample 20 --out-dir outputs/debug_audit | tail -12

echo; echo "debug checks passed."
