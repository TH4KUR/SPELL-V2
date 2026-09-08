# DNSMOS v8 weights — provenance

- File: `sig_bak_ovr.onnx` (1,157,965 bytes; sha256 below)
- Source: microsoft/DNS-Challenge (master), `DNSMOS/DNSMOS/sig_bak_ovr.onnx` — the
  NON-personalized SIG/BAK/OVRL model. Vendored so Ada compute nodes need no
  internet. The P808 model (`model_v8.onnx`) is deliberately NOT vendored: its
  reference inference path requires librosa, which the frozen env does not carry.
- Reference inference: `DNS-Challenge/DNSMOS/dnsmos_local.py` (vendored checkout,
  untracked at repo root). Our scorer `scripts/score_dnsmos.py` reimplements the
  non-personalized path EXACTLY (16 kHz mono, 9.01 s windows = 144,160 samples,
  1 s hop, self-tiling of short clips, hop-mean raw + np.poly1d calibration with
  the non-personalized coefficients) and drops the librosa/P808 branch.
- Calibration coefficients (non-personalized, dnsmos_local.py:39-41):
  sig [-0.08397278, 1.22083953, 0.0052439];
  bak [-0.13166888, 1.60915514, -0.39604546];
  ovr [-0.06766283, 1.11546468, 0.04602535].
- Citation: Reddy et al., "DNSMOS: A Non-Intrusive Speech Quality Assessment
  metric using Deep Learning", Interspeech 2021 / DNS-Challenge ICASSP 2023.

sha256: 269fbebdb513aa23cddfbb593542ecc540284a91849ac50516870e1ac78f6edd
(verified identical to the vendored-checkout source file at vendoring time,
2026-09-08.)

## Runtime: pure-PyTorch port, not onnxruntime (2026-09-09)

`onnxruntime==1.18.1` declares `numpy<2.0,>=1.21.6`, which is irreconcilable
in one `pip install` with `scipy==1.18.1`/`lightning`/`torchmetrics` needing
`numpy>=2.0` (PROTOCOL §5.5) — and even installed standalone via `--no-deps`
(bypassing that pin check), it is CONFIRMED BROKEN AT RUNTIME under
numpy>=2.0: `import onnxruntime` raises `AttributeError: _ARRAY_API not
found`, the canonical numpy-2.0 C-ABI break for a compiled extension built
against numpy 1.x and never rebuilt. Empirically verified on Ada 2026-09-09,
not a guess.

Resolution: `scripts/score_dnsmos.py` now runs `dnsmos_model.DNSMOSTorch`, a
pure-PyTorch reimplementation of this exact ONNX graph (no onnxruntime
dependency at all — torch is already the frozen, conflict-free baseline).
The graph was reverse-engineered node-by-node from `sig_bak_ovr.onnx` (48
nodes / 35 initializers: framing via reshape+concat rather than a native
unfold op, STFT via matmul against the vendored real/imag DFT-basis kernels
reused verbatim, log-power spectrogram, a 7-layer 2D CNN, global max-pool,
a 3-layer MLP) — see `dnsmos_model.py`'s module docstring for the full
trace. `scripts/port_dnsmos_to_torch.py` extracts the weights and verifies
the port against this ONNX file via onnxruntime (run in a THROWAWAY venv,
never `~/envs/spell`), producing `sig_bak_ovr_torch.pt`:

sha256 of `sig_bak_ovr_torch.pt`:
6153a11a0162a2c7adf1320a305ec78663cf8cc6105ff581244b5790ced5f02a
(re-run `scripts/port_dnsmos_to_torch.py` and update this line if
`sig_bak_ovr.onnx` itself is ever re-vendored). Verified max abs diff vs. the ONNX graph across 6 test inputs (silence,
constant, 4 seeded noise draws): **9.537e-07** — floating-point rounding
noise, not an architectural discrepancy. `tests/test_dnsmos_model.py` pins
two of those onnx-verified outputs as a standing regression, so a future
edit to `dnsmos_model.py` is checked WITHOUT onnx/onnxruntime as an ongoing
test dependency.

`sig_bak_ovr.onnx` stays vendored as the frozen source of truth for any
future re-port (e.g. if this file is ever updated) — it is no longer a
runtime dependency of any kind.