#!/usr/bin/env python3
"""One-time OFFLINE tool: extract weights from models/dnsmos/sig_bak_ovr.onnx
into dnsmos_model.DNSMOSTorch's state dict, then verify the port is
numerically equivalent to the ONNX graph.

NOT part of the frozen Ada env / requirements.lock — this needs `onnx` and
`onnxruntime` as ground truth to verify against, and onnxruntime==1.18.1's
compiled bindings are confirmed incompatible with this project's numpy>=2.0
pin (2026-09-09; models/dnsmos/PROVENANCE.md). Run it in a THROWAWAY venv,
never inside ~/envs/spell:

    python -m venv /tmp/dnsmos_port_venv
    source /tmp/dnsmos_port_venv/bin/activate
    pip install onnx onnxruntime numpy torch
    python scripts/port_dnsmos_to_torch.py   # run from the repo root

Re-run this ONLY if models/dnsmos/sig_bak_ovr.onnx itself ever changes (it
is frozen provenance — see PROVENANCE.md sha256). It overwrites
models/dnsmos/sig_bak_ovr_torch.pt on success; a failed equivalence check
raises and writes nothing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from dnsmos_model import LEN_SAMPLES, DNSMOSTorch  # noqa: E402

ONNX_PATH = ROOT / "models" / "dnsmos" / "sig_bak_ovr.onnx"
TORCH_OUT = ROOT / "models" / "dnsmos" / "sig_bak_ovr_torch.pt"
MAX_ABS_DIFF_TOLERANCE = 1e-3


def load_weights(model: DNSMOSTorch, onnx_path: Path) -> None:
    import onnx
    from onnx import numpy_helper

    m = onnx.load(str(onnx_path))
    w = {init.name: numpy_helper.to_array(init) for init in m.graph.initializer}

    def t(x):
        return torch.from_numpy(np.ascontiguousarray(x))

    with torch.no_grad():
        model.stft_real.weight.copy_(t(w["time2freq/stft-real/kernel:0"][:, :, 0]))
        model.stft_imag.weight.copy_(t(w["time2freq/stft-imag/kernel:0"][:, :, 0]))

        conv_names = ["conv2d", "conv2d_1", "conv2d_2", "conv2d_3",
                      "conv2d_4", "conv2d_5", "conv2d_6"]
        convs = [model.conv0, model.conv1, model.conv2, model.conv3,
                 model.conv4, model.conv5, model.conv6]
        for name, layer in zip(conv_names, convs):
            layer.weight.copy_(t(w[f"{name}/kernel:0"]))
            layer.bias.copy_(t(w[f"{name}/bias:0"]))

        dense_names = ["dense", "dense_1", "dense_3"]
        denses = [model.dense0, model.dense1, model.dense3]
        for name, layer in zip(dense_names, denses):
            wt = w[f"mos_estimator_logpow/{name}/MatMul/ReadVariableOp/resource:0"]
            b = w[f"mos_estimator_logpow/{name}/BiasAdd/ReadVariableOp/resource:0"]
            layer.weight.copy_(t(wt.T))                 # TF (in,out) -> torch (out,in)
            layer.bias.copy_(t(b))


def verify(model: DNSMOSTorch, onnx_path: Path) -> float:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    max_abs_diffs = []
    trials = [np.zeros((1, LEN_SAMPLES), dtype=np.float32),
              np.full((1, LEN_SAMPLES), 0.01, dtype=np.float32)]
    trials += [rng.standard_normal((1, LEN_SAMPLES)).astype(np.float32) * 0.05
               for _ in range(4)]
    for x in trials:
        onnx_out = sess.run(None, {"input_1": x})[0][0]
        with torch.no_grad():
            torch_out = model(torch.from_numpy(x)).numpy()[0]
        diff = float(np.abs(onnx_out - torch_out).max())
        max_abs_diffs.append(diff)
        print(f"[port_dnsmos] onnx={onnx_out} torch={torch_out} max_abs_diff={diff:.3e}")
    return max(max_abs_diffs)


def main() -> int:
    if not ONNX_PATH.exists():
        print(f"FATAL: {ONNX_PATH} missing", file=sys.stderr)
        return 2

    model = DNSMOSTorch().eval()
    load_weights(model, ONNX_PATH)
    overall = verify(model, ONNX_PATH)
    print(f"[port_dnsmos] OVERALL max_abs_diff = {overall:.3e} "
          f"(tolerance {MAX_ABS_DIFF_TOLERANCE:.0e})")
    if overall >= MAX_ABS_DIFF_TOLERANCE:
        print("FATAL: torch port does not match the onnx graph closely enough — "
              "NOT writing weights", file=sys.stderr)
        return 1

    torch.save(model.state_dict(), str(TORCH_OUT))
    print(f"[port_dnsmos] PASS — wrote {TORCH_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
