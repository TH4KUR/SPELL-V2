"""Pure-PyTorch port of the vendored DNSMOS non-personalized model
(models/dnsmos/sig_bak_ovr.onnx). Reimplements the graph exactly — see
scripts/port_dnsmos_to_torch.py for the extraction/verification tool and
models/dnsmos/PROVENANCE.md for why this port exists (onnxruntime==1.18.1's
compiled bindings crash under numpy>=2.0; no version of it can coexist with
this project's numpy pin — 2026-09-09).

Graph, reverse-engineered from the ONNX file's 48 nodes / 35 initializers:
  1. frame the 144,160-sample (9.01 s @ 16 kHz) input into 900 windows of
     320 samples (win=320, hop=160 -> 20 ms/10 ms), built from two crops
     (samples[0:144000] and samples[160:144160], each reshaped to
     [-1, 900, 160] and concatenated) rather than a native unfold op.
  2. STFT via matmul against the vendored real/imag DFT-basis matrices
     (161, 320 each) — loaded verbatim from the ONNX weights, so this step
     needs no window-function guessing.
  3. log-power spectrogram: log(max(1e-12, (sqrt(re^2+im^2))^2)) / ln(10).
  4. 7-layer 2D CNN (1->128->64->64->32->32->32->64, 3x3 pad=1, ReLU, three
     MaxPool(2,2) interleaved in the exact graph order) over the
     [N, 1, 900, 161] log-power image.
  5. global max-pool over (time, freq) -> (N, 64).
  6. 3-layer MLP (64->128->64->3, ReLU between, none after) -> [sig, bak, ovr].
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

FS = 16000
INPUT_LENGTH = 9.01
LEN_SAMPLES = int(INPUT_LENGTH * FS)   # 144,160
_WIN, _HOP, _NFRAMES = 320, 160, 900
_DEFAULT_WEIGHTS = Path(__file__).resolve().parent / "models" / "dnsmos" / "sig_bak_ovr_torch.pt"


class DNSMOSTorch(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft_real = nn.Linear(_WIN, 161, bias=False)
        self.stft_imag = nn.Linear(_WIN, 161, bias=False)
        self.conv0 = nn.Conv2d(1, 128, 3, padding=1)
        self.conv1 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv6 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dense0 = nn.Linear(64, 128)
        self.dense1 = nn.Linear(128, 64)
        self.dense3 = nn.Linear(64, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 144160) float32 -> (N, 3) raw [sig, bak, ovr]."""
        if x.shape[-1] != LEN_SAMPLES:
            raise ValueError(f"expected exactly {LEN_SAMPLES} samples, got {x.shape[-1]}")
        n = x.shape[0]
        a = x[:, :144000].reshape(n, _NFRAMES, _HOP)
        b = x[:, _HOP:].reshape(n, _NFRAMES, _HOP)
        frames = torch.cat([a, b], dim=2)               # (N, 900, 320)
        real = self.stft_real(frames)
        imag = self.stft_imag(frames)
        mag = torch.sqrt(real ** 2 + imag ** 2)
        power = torch.clamp(mag ** 2.0, min=1e-12)
        logp = torch.log(power) / 2.3025851              # ln -> log10
        spec = logp.unsqueeze(-1).permute(0, 3, 1, 2)    # (N, 1, 900, 161)
        h = F.relu(self.conv0(spec))
        h = F.relu(self.conv1(h))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = self.pool(h)
        h = F.relu(self.conv4(h))
        h = self.pool(h)
        h = F.relu(self.conv5(h))
        h = self.pool(h)
        h = F.relu(self.conv6(h))
        h = h.amax(dim=(2, 3))                            # global max pool -> (N, 64)
        h = F.relu(self.dense0(h))
        h = F.relu(self.dense1(h))
        return self.dense3(h)


def load_dnsmos_torch(weights_path: str | Path = _DEFAULT_WEIGHTS) -> DNSMOSTorch:
    model = DNSMOSTorch()
    state = torch.load(str(weights_path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval()
