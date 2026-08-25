"""HARDWARE DRIFT GUARD (locked Ada policy).

A physically swapped RTX 3080 was observed inside Ada's "2080 Ti" feature pool
(gnode077). Every train/eval entrypoint MUST call :func:`assert_gpu` before any
training or evaluation: it aborts with the hostname unless the visible GPU
matches the locked model. GPU name + driver version are captured for the run
manifest either way.

Driver range 570–580 across nodes is fine for cu124 wheels — logged, not gated.
"""

from __future__ import annotations

import socket

# Locked formal-run GPU. Permanent constraint: the ihub/3080 Ti partition is
# inaccessible (research account rejected) — do not add alternates here.
LOCKED_GPU_SUBSTRING = "RTX 2080 Ti"


def gpu_info() -> dict:
    """Best-effort GPU identity + driver for run manifests (never raises)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return {"gpu_name": None, "driver_version": None, "cuda_available": False}
        props = torch.cuda.get_device_properties(0)
        return {
            "gpu_name": torch.cuda.get_device_name(0),
            "driver_version": getattr(props, "driver_version", None)
            or _nvidia_smi_driver(),
            "cuda_available": True,
            "total_mem_gb": round(props.total_memory / 1e9, 2),
        }
    except Exception as e:  # noqa: BLE001 - manifest info must never crash a run
        return {"gpu_name": None, "driver_version": None, "error": str(e)}


def _nvidia_smi_driver() -> str | None:
    try:
        import subprocess

        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().splitlines()[0] if out.stdout.strip() else None
    except Exception:  # noqa: BLE001
        return None


def assert_gpu(expected_substring: str = LOCKED_GPU_SUBSTRING) -> dict:
    """Abort unless device 0's name contains ``expected_substring``.

    Returns the :func:`gpu_info` dict (for the run manifest) on success.
    Raises RuntimeError naming the offending host otherwise.
    """
    info = gpu_info()
    name = info.get("gpu_name")
    if not name or expected_substring not in name:
        raise RuntimeError(
            f"HARDWARE DRIFT GUARD on host {socket.gethostname()!r}: GPU is "
            f"{name!r}, expected name containing {expected_substring!r}. "
            "A non-locked card in the 2080 Ti pool invalidates hardware comparability. "
            "Refusing to train/evaluate — resubmit onto a genuine RTX 2080 Ti node."
        )
    return info
