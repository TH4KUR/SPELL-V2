"""dnsmos_model.DNSMOSTorch regression (2026-09-09 port, PROTOCOL §5.5):

onnxruntime==1.18.1's compiled bindings are confirmed incompatible with this
project's numpy>=2.0 pin (`AttributeError: _ARRAY_API not found`) and no
onnxruntime version can coexist with numpy in one pip resolve without
--no-deps risk, so score_dnsmos.py now runs a pure-PyTorch reimplementation
of the vendored ONNX graph instead. scripts/port_dnsmos_to_torch.py verified
the port against the ONNX graph directly (<1e-6 max abs diff, onnxruntime as
ground truth, run in a throwaway venv — see that script's docstring). The
golden values below are that verification's actual onnx-computed outputs,
pinned here so a future edit to dnsmos_model.py is checked WITHOUT needing
onnx/onnxruntime as an ongoing test dependency.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from dnsmos_model import LEN_SAMPLES, load_dnsmos_torch

GOLDEN = [
    # (input, expected [sig, bak, ovr] raw, from scripts/port_dnsmos_to_torch.py
    #  verified against the ONNX graph via onnxruntime)
    (np.zeros((1, LEN_SAMPLES), dtype=np.float32),
     [2.476402, 3.2893887, 1.8059998]),
    (np.full((1, LEN_SAMPLES), 0.01, dtype=np.float32),
     [2.5541682, 2.8879654, 1.7680275]),
]


@pytest.fixture(scope="module")
def model():
    return load_dnsmos_torch()


def test_golden_outputs_match_onnx_verified_values(model):
    for x, expected in GOLDEN:
        with torch.no_grad():
            out = model(torch.from_numpy(x)).numpy()[0]
        np.testing.assert_allclose(out, expected, atol=1e-3, rtol=1e-3)


def test_output_shape_and_dtype(model):
    x = torch.zeros(3, LEN_SAMPLES)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (3, 3)
    assert out.dtype == torch.float32


def test_wrong_length_input_raises(model):
    with pytest.raises(ValueError):
        model(torch.zeros(1, LEN_SAMPLES - 1))
