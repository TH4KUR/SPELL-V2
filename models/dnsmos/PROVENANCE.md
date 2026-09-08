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