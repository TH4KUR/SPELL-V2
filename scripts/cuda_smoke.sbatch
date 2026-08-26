#!/bin/bash
#SBATCH -p u22 -A research --qos=medium --gres=gpu:1 --constraint=2080ti
#SBATCH --exclude=gnode066
#SBATCH -c 4 --mem=16G -t 00:10:00
#SBATCH --job-name=cuda_smoke
#SBATCH --output=%x.%j.log

echo "NODE=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"   # the REAL gpu assignment
source "$HOME/envs/spell/bin/activate"
python - <<'PY'
import os, torch
print("torch", torch.__version__, "| cuda build", torch.version.cuda)
assert torch.cuda.is_available(), "FATAL: cuda not available"
x = torch.randn(1024, 1024, device="cuda")
name = torch.cuda.get_device_name(0)
assert "2080 Ti" in name, f"FATAL: unexpected GPU '{name}' (hardware drift!)"
print("CUDA OK |", name, "| capability:", torch.cuda.get_device_capability(0),
      "| matmul sum:", (x@x).sum().item())
PY
