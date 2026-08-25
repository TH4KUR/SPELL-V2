# SPEC: Project SPELL-RQ2 — Dual-track benchmark of data-selection methods on discrete speech tokens

You are implementing this project with me. Work phase by phase — do NOT skip ahead. Each phase ends with runnable code + tests that I will execute and report back before you proceed. Prefer small, typed, well-factored modules with zero magic numbers: every constant lives in a YAML config.

## Project context (read carefully)

Research question: do data-selection methods rank the SAME when subsets are consumed by a generative vocoder (Track A) vs. an ASR model (Track B), trained on identical frozen discrete speech tokens?

- Tokenizer: frozen SpeechTokenizer, 16 kHz, 8 RVQ codebook streams @ 50 Hz (codebook 1 ≈ distilled HuBERT semantics; codebooks 2–8 ≈ acoustics/prosody). Tokenization is ALREADY DONE offline.
- Track A (generative): Vocos-style vocoder (ConvNeXt backbone + iSTFT head) + HiFi-GAN discriminators (MPD+MSD), losses = multi-resolution Mel-L1 + feature matching + adversarial. Trains from scratch per subset. Metrics: STOI + PESQ. Generator weight EMA during eval.
- Track B (discriminative): input = RVQ₁ stream ONLY (shape [1, T]); architecture = embedding(codebook_size→d_model) → 4–6-layer Conformer → linear → character-level CTC. From scratch. Greedy decoding, no external LM. Metric: WER via jiwer.
- Selection happens ONCE per method producing a subset manifest (list of utterance IDs); BOTH tracks then train on that same manifest. Budget = 25% of corpus.
- Selector roster (implement in later phases): random (5 seeds), k-means diversity, DSIR (hashed n-gram over token sequences), loss-ranking + EL2N under a small shared proxy model, DNSMOS top-25% filter, anti-selection (worst 25%), and a LESS pair (gradient influence computed with Mel-L1 grads vs. CTC grads using trajectory checkpoints saved every 5 epochs).
- Statistics contract: ≥5 seeds for random floors, ≥3 seeds per selector cell; report per-seed points; paired evaluation on IDENTICAL test crops/wavs across all models; evaluate FINAL checkpoint (never best-val) uniformly; equal-epochs protocol (every run trains the same number of epochs regardless of subset size).

## Existing data layout (verified partially, audit first!)

Dataset root contains split dirs like `trainval/` with per-video subfolders. Inside each folder, files follow the pattern:
`<video_id>.mp4` — source audio/video (audio stream = 16 kHz mono, presumed)
`<video_id>.txt` — transcript for that video
`<video_id>.tokens.pt`— torch file, PRESUMED full-utterance tensor, possibly shape [8, T] int64 (8 RVQ layers @ ~50 Hz)

These were produced by a previous extraction run. Nothing about shape/dtype/rate is guaranteed until audited.

## Hard engineering requirements (apply to all phases)

- Python 3.10+, PyTorch + PyTorch Lightning, Hydra or plain YAML configs, W&B logging, SLURM-compatible launch scripts, DDP-ready (but everything must also run single-GPU).
- Every training run: fixed seed (arg), logs batch-level grad L2 norms (∇G/∇D for Track A), saves checkpoints every 5 epochs, writes a `run_manifest.json` (subset file, seed, config hash, git SHA).
- Subset manifests are plain text files of utterance IDs stored under `subsets/`; training code NEVER does selection inline — it consumes a manifest path.
- Deterministic data ordering per seed; identical eval batches/crops across all runs of a track.
- Never read the official test split anywhere except the final-eval module. Internal validation set (~2000 utterances held out of trainval, speaker-disjoint if IDs permit) is created once in Phase 0 and frozen.
- Unit tests for every nontrivial function. An `overfit_one_batch.py` smoke script per track must reach near-zero train loss before any real run.

## PHASE 0 — Audit + foundations (deliver now)

1. `audit_data.py`: given dataset root, (a) count folders/files per split, (b) load N random `.tokens.pt` files and print shape/dtype/min/max, (c) verify every folder has all three files, (d) verify transcript non-empty after basic cleaning, (e) estimate total hours from token counts (frames ÷ 50 Hz), (f) dump `data_index.parquet` mapping utterance_id → paths → n_tokens → duration. Report any anomalies.
2. Decide (and tell me): are tokens full-utterance streams? Do we need re-extraction, or only a corrected crop sampler?
3. `text_norm.py`: ONE canonical normalization fn (lowercase, strip punctuation except apostrophes, digits→words via num2words or mapping, collapse whitespace) + char vocabulary builder (`a-z`, space, apostrophe, <pad>, <blank>). Tests included. This function is frozen forever after this phase.
4. Dataset classes: `TokenDataset` (returns full 8-stream tokens + length) and shared collate utilities; crop sampler producing 16,000 audio samples ↔ EXACTLY 50 tokens (write the unit test proving alignment).
5. Split builder: internal val split (frozen ID list committed to repo), train list, and pointers to official test (paths only, unused).
6. Config skeletons for both tracks + a `PROTOCOL.md` documenting the locked rules above.
7. SLURM template + single-GPU debug launcher.

STOP. Wait for my audit results.

## PHASE 1 — Track B (ASR) end-to-end

Conformer-CTC model, training loop (PL module), SpecAugment-style masking on token streams (fixed config), greedy decoder, jiwer-based WER evaluator emitting PER-UTTERANCE WERs to disk, W&B integration. Include: learning-rate/warmup defaults clearly flagged as PROVISIONAL pending my 100% pilot; overfit-one-batch script; unit tests (CTC blank handling, padding masks, vocab round-trip). Then two pilot scripts: `pilot_100pct.yaml` and `pilot_25pct_random_x2seeds.yaml`.

## PHASE 2 — Track A (vocoder) port + anchors

Port/rewrite the Vocos-style vocoder under the new repo conventions: Mel-L1 + feature matching + HiFi-GAN losses, MPD/MSD, generator EMA, grad-norm logging, final-checkpoint STOI/PESQ evaluator with FIXED eval crops cached once. Same anchor scripts pattern as Phase 1.

## PHASE 3 — Anchor orchestration

SLURM array driver producing ALL anchor runs: ceilings (100%, both tracks), random floors (25% × 5 seeds, both tracks), anti-selection manifests (worst 25% by proxy-model loss — requires the small proxy model trained here on ~10% data, shared across all later selectors), plus `analyze_anchors.py` computing μ±σ bands, power analysis (smallest detectable ΔSTOI/ΔWER at 95%), and go/no-go printouts.

## PHASE 4 — Selectors

Implement in risk order, each behind a common interface `select(index, budget) -> List[utterance_id]`: DNSMOS filter (decode audio from mp4s once into a cache), k-means diversity (features: duration, RVQ₁ usage entropy, prosodic variance of RVQ₂₋₈), DSIR (hashed n-grams over token id sequences, logistic classifier resampling), loss-ranking, EL2N, then LESS (trajectory checkpoint loading, random-projected per-example gradients, TWO variants: Mel-L1-val reference and CTC-val reference). Every selector writes a manifest + a characterization JSON (selected duration distribution, speaker spread, codebook entropies).

## PHASE 5 — Main grid + analysis

Driver to train both tracks on every selector manifest × 3 seeds; `analyze_rq2.py`: per-track ranking tables, Kendall τ between track rankings with bootstrap-over-seeds uncertainty, LESS-probe comparison, subset-characterization panel (matplotlib), and a results.md generator.

Begin with Phase 0 now. Ask me clarifying questions BEFORE writing code if anything above is ambiguous.
