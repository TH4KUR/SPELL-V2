# RAVEn-parity track, Stage 2: review document

For the professor and the independent code reviewer. Stage 2 adds the machinery that trains the
vendored RAVEn Base audio-only model FROM SCRATCH on the parity train split, under two
pre-registered recipes, and documents every choice that the RAVEn paper does not settle.
`PROTOCOL.md` §3.34 is the binding rule, `parity/CLAIMS.md` the evidence ledger, `parity/REVIEW.md`
the Stage-1 review (vendoring, evaluation harness). **Nothing here claims parity or any WER**: no
LRS3 audio exists on the development laptop, nothing was trained on LRS3, and every number that
depends on Ada or on LRS3 audio is `PENDING` in the ledger.

## 1. What exists

| path | purpose |
|---|---|
| `configs/parity/recipes.yaml`, `parity/recipes.py` | the two pre-registered recipes R-A / R-B, every value tagged with its provenance, pinned by sha256 (`RECIPES_SHA256`); any other bytes make every entry point refuse to start |
| `parity/spm_tokens.py`, `scripts/parity_train_spm.py` | OFFLINE tokenizer step (the only code that may import sentencepiece): trains the unigram-1000 model on the parity TRAIN text, writes the units file, a token-id file and a provenance file with the fidelity report against RAVEn's units list |
| `parity/train_data.py` | training dataset (FLAC -> float32, no normalisation), zero time-mask with the vendored class's semantics and per-(seed, epoch, utterance) RNGs, deterministic length-bucketed sampler, the collate that builds what `E2E.forward` needs |
| `parity/optim.py` | AdamW, parameter groups (uniform, or upstream's own `get_param_groups` for layer-wise decay), per-update warm-up + cosine schedule |
| `parity/trainer.py` | plain-PyTorch loop: update = micro-batches with equal utterance weights, update-level clipping, per-epoch validation (loss + CTC-greedy WER), beam-search val WER, atomic full-state checkpoint, rolling weights-only checkpoints, resume, model selection, W&B live logging |
| `parity/checkpoint_avg.py` | average of weight files with the arithmetic of the vendored `average_checkpoints` (floats mean, integer buffers floor of the mean) |
| `scripts/parity_train.py`, `slurm/parity_train.sbatch` | the entry point and the Ada job (recipe by `PARITY_RECIPE`, `PARITY_RESUME=1` to continue) |
| `parity/model.py` (+`odim` argument), `scripts/parity_eval.py` (+`--units`, `--expect-sha256 none`), `slurm/parity_eval.sbatch` (+`PARITY_UNITS`, `PARITY_EXPECT_SHA256`) | the only Stage-1 source files touched: a checkpoint written by the trainer is evaluated by the unchanged Stage-1 decoding recipe; defaults and all Stage-1 tests are unchanged |
| `tests/test_parity_{recipes,spm_tokens,train_data,batching,optim,checkpoint_avg,trainer,train_cli,train_sbatch,smoke_overfit}.py`, `tests/parity_train_helpers.py` | Stage-2 tests; `tests/test_parity_{eval,sbatch,docs}.py` gained tests, and `tests/test_parity_upstream.py` (the import-hygiene scan) now allows `sentencepiece` in exactly `parity/spm_tokens.py` and `scripts/parity_train_spm.py`, function-level only (a new test pins that, and the dynamic closure test now also imports the Stage-2 modules with `sentencepiece`, `lightning`, `torchaudio`, ... blocked) |

Frozen files untouched: `text_norm.py`, `subsets/splits/*`, `configs/protocol.yaml`,
`slurm/template.sbatch`, `scripts/gen_run_plan.py`, the run-plan TSVs, Track A/B model and trainer
code, `requirements.lock`, everything under `parity/vendor/raven/` (the provenance test still passes).

## 2. Evidence base: what was read, from where

Re-read by the Stage-2 author directly from the primary sources on 2026-10-19 (PDFs fetched from
arXiv, text extracted; the raw files of Auto-AVSR fetched from GitHub), not taken from the brief:

* **RAVEn, arXiv 2212.06246.** Table 1: from scratch, 30 h labelled, ASR WER Base 18.5, Large 9.9
  (VSR 93.4 / 95.5). Table 7 (p. 17) is headed "Fine-tuning settings": epochs 75 (HR) / 50 (LR), warm-up 20,
  AdamW, LR encoder 2e-3, LR decoder 6e-3, betas (0.9, 0.98), weight decay 0.04 (HR) / 0.1 (LR), cosine,
  layer-wise LR decay 0.75 (HR) / 0.5 (LR), minimum LR 1e-5, drop path 0.2, gradient clip 5.0, audio
  augmentation TimeMask, decoder 6 blocks, hidden 256 / MLP 1024 / 4 heads (LR). Sec. 4.1 and App. A.4: CTC
  weight alpha = 0.1 (loss = alpha CTC + (1 - alpha) attention), beam 40, lambda = 0.1 at decoding,
  SentencePiece with a vocabulary of 1,000. App. A.2: "Raw audio is used without any pre-processing nor
  normalisation". App. A.4: time masking = for each second of audio, zero-masking with a duration uniform in
  [0, 0.4] s. **The paper states no separate from-scratch recipe**: Table 1 only says the from-scratch rows
  are trained without the pre-training. Not found anywhere in the paper: label smoothing, a numerical
  precision, a fine-tuning batch size, checkpoint averaging.
* **RAVEn App. A.3** (pre-training): "The samples are sorted based on their length to minimise zero-padding,
  and then samples with the same length are randomly shuffled at the beginning of each training epoch. Each
  batch contains a maximum of 96 / 36 seconds of footage for Base / Large." The sampler in `train_data.py` is
  built the same way (upstream's `ByFrameCountSampler` does exactly this per GPU).
* **USR, arXiv 2411.02256** (same first author), Table 8 and App. C.5: supervised training from scratch, 75
  epochs, 20 warm-up, AdamW (0.9, 0.98), LR 3e-3 (LRS3) / 2e-3 (LRS3+Vox2), weight decay 0.04, cosine,
  drop rate 0.1 (Base) / 0.2 (Large), gradient clipping 3.0, **155 frames per GPU (low-resource labelled)**,
  2,400 frames per GPU (unlabelled LRS3), 32 A100 40 GB for Base, seed 42, SentencePiece 1,000. 2,400 frames
  at 25 frames/s = 96 s, i.e. USR's "frames per GPU" is the same sampler budget as RAVEn's A.3 "96 seconds of
  footage" for pre-training; that supports reading 155 frames as 6.2 s per GPU, 198 s per update on 32 GPUs.
  It does not prove RAVEn's fine-tuning batch.
* **Auto-AVSR** (github.com/mpc001/auto_avsr, Apache-2.0; `cosine.py`, `lightning.py`,
  `datamodule/transforms.py`): `WarmupCosineScheduler` (linear warm-up from 0, then
  `0.5 * base_lr * (1 + cos(pi * (step - warmup) / (total - warmup)))`, stepped per iteration, using the
  scheduler's `_step_count`, so the first update runs at `base_lr / warmup_steps`); `AdaptiveTimeMask`
  identical in algorithm to the vendored RAVEn class; the loss of every device is rescaled by
  `world_size / total_batch_size` so each utterance weighs the same; `AdamW(lr, weight_decay, betas=(0.9, 0.98))`
  over all parameters. It layer-norms the waveform, RAVEn does not, so the parity loader does not.
* **Upstream RAVEn code @ 9fd9a37** (vendored): `E2E.forward`, `Encoder`, `CTC`, `LabelSmoothingLoss`,
  `get_param_groups`, `AdaptiveLengthTimeMask`, `ByFrameCountSampler`, `collate_pad`, `finetune_learner.py`
  (test-only: it feeds one utterance at a time to the encoder).

## 3. The recipes, value by value

Generated by hand from `configs/parity/recipes.yaml`; `tests/test_parity_docs.py` fails if a value or a tag
in this table differs from the file. Tags: paper-stated (RAVEn), usr-stated (USR), checkpoint-derived,
inferred (an argument, not a statement), assumed (nothing primary supports it).

<!-- recipe-table:start -->
| key | R-A | R-A tag | R-B | R-B tag | source and caveat |
|---|---|---|---|---|---|
| seed | 42 | usr-stated | 42 | usr-stated | USR C.5; RAVEn states none |
| betas | 0.9, 0.98 | paper-stated | 0.9, 0.98 | paper-stated | RAVEn Table 7, USR Table 8 |
| ctc_weight | 0.1 | paper-stated | 0.1 | paper-stated | RAVEn Sec. 4.1, A.4 (alpha); upstream `mtlalpha` |
| label_smoothing | 0.1 | inferred | 0.1 | inferred | not in the paper; upstream model config `lsm_weight` 0.1 |
| time_mask_window_s | 0.4 | paper-stated | 0.4 | paper-stated | RAVEn A.4 |
| time_mask_stride_s | 1 | paper-stated | 1 | paper-stated | RAVEn A.4 (one mask per second) |
| waveform_normalisation | none | paper-stated | none | paper-stated | RAVEn A.2 |
| precision | fp32 | assumed | fp32 | assumed | not in the paper; upstream and Auto-AVSR run fp32 |
| effective_batch_seconds | 198 | assumed | 198 | assumed | USR 155 frames x 32 GPUs at 25 fps; RAVEn's fine-tuning batch is unstated (section 5, item 1) |
| micro_batch_seconds | 60 | assumed | 60 | assumed | memory choice for an 11 GB 2080 Ti (section 6); RAVEn's per-GPU batch is unstated (about 6 s if USR's value holds) |
| model_selection | final_epoch | assumed | final_epoch | assumed | the paper does not mention averaging |
| average_last_k | 10 | inferred | 10 | inferred | Ma et al. 2022 average the last 10; used only if model_selection is average_last_k |
| beam_size | 40 | paper-stated | 40 | paper-stated | RAVEn A.4 |
| decode_ctc_weight | 0.1 | paper-stated | 0.1 | paper-stated | RAVEn A.4 (lambda) |
| epochs | 50 | paper-stated | 75 | usr-stated | Table 7 (LR column) / USR Table 8 |
| warmup_epochs | 20 | paper-stated | 20 | usr-stated | Table 7 / Table 8 |
| lr_encoder | 0.002 | paper-stated | 0.003 | usr-stated | Table 7 / Table 8 (LRS3) |
| lr_decoder | 0.006 | paper-stated | 0.003 | usr-stated | Table 7 (decoder; also the CTC head, by upstream's `ctc_equals_other`) / one LR in USR |
| weight_decay | 0.1 | paper-stated | 0.04 | usr-stated | Table 7 (LR column) / Table 8; applied to every parameter (section 5, item 6) |
| layer_decay | 0.5 | paper-stated | none | assumed | Table 7 is FINE-TUNING; R-B has none because USR's table lists none (section 5, item 2) |
| min_lr | 1e-05 | paper-stated | 1e-05 | assumed | Table 7; USR's minimum is unstated. Used as the per-group floor AND the end of the cosine (section 4.5) |
| grad_clip | 5 | paper-stated | 3 | usr-stated | Table 7 / Table 8 |
| drop_path | 0.2 | paper-stated | 0.1 | usr-stated | Table 7 / Table 8 (Base) |
<!-- recipe-table:end -->

Other model settings come from `configs/parity/arch_base.yaml` (Stage 1): dropout 0.1 and attention dropout
0.1 are upstream's `resnet_transformer_base.yaml` defaults and are assumed to hold in training (the paper
does not list a dropout rate).

## 4. How the loop works and what was proven about it

### 4.1 The batching contract of `E2E.forward` (the most error-prone part)

Read from the vendored sources (`encoder.py`, `conv1d_extractor.py`, `ctc.py`, `attention.py`,
`e2e_asr_transformer.py`, `label_smoothing_loss.py`, `nets_utils.py`) and fixed by tests:

* `x`: `(B, T, 1)` float32, zero-padded on the right. The ResNet1D front-end cuts the batch to
  `T // 640 * 640` samples and returns `T // 640` frames (25 frames/s).
* `lengths`: `(B,)` frames per utterance = `samples // 640` (CTC input lengths). Upstream's own test code
  builds `make_non_pad_mask(lengths)` from sample counts and then feeds each utterance alone, so it never
  shows the training convention; Auto-AVSR's `lengths // 640` for audio is the same convention.
* `padding_mask`: `(B, 1, T // 640)` bool, TRUE = valid (`mask.unsqueeze(1).eq(0)` is what the attention
  masks out). A `(B, T)` mask would silently broadcast wrongly. The decoder's source attention takes the same
  tensor. `make_non_pad_mask` returns a CPU tensor even for CUDA lengths; the collate builds the mask on CPU and
  the trainer moves it.
* `label`: `(B, Lmax)` int64 padded with -1; ids are 1..n_units (0 is `<blank>`; `sos = eos = odim - 1` is added
  by the model).
* Reduction: `loss = 0.1 * CTC + 0.9 * attention`, CTC `reduction="sum"` divided by the batch size, label-smoothed
  KL summed and divided by the batch size (`normalize_length=False`): per-UTTERANCE sums averaged over the batch.
  `zero_infinity=True` zeroes a CTC alignment that is impossible; no parity-train utterance has fewer frames than
  tokens (checked on the real index with the tokenizer of section 6).

**Batch invariance** (`tests/test_parity_batching.py`, tiny transformer widths, the real front-end; measured numbers):

| situation | what differs | measured |
|---|---|---|
| eval mode, front-end BatchNorm maps 0 to 0 | nothing: padded batch of 3 vs the 3 utterances alone | loss relative difference 7.8e-8; gradient relative L2 difference 5.1e-5 (float32 noise) |
| eval mode, ordinary BatchNorm statistics | only the last valid front-end frame of an utterance shorter than its batch-mates (the front-end maps zero input to a non-zero constant after BatchNorm + Swish, and convolutions mix neighbours; alone, the next layer sees zero padding instead) | frames 0..n-2 equal to 1e-5; the last frame moves (2.4e-3 absolute on features of mean magnitude 0.04); loss relative difference 6.9e-6, gradient relative L2 difference 3.8e-4 |
| eval mode, equal lengths | nothing | loss relative difference below 1e-5 |
| TRAIN mode | BatchNorm (front-end and the encoder's pre-MLP BatchNorm) normalises with the statistics of the whole micro-batch, padded frames included: an utterance's loss depends on its batch-mates; no collate can remove it | mean of 2 single-utterance losses 31.91 vs batch loss 31.65 (0.8 %) in the test model |

Upstream's own zero-padding collate has the same last-frame property; the trainer does not try to hide it.

### 4.2 An update, and equal weight per utterance

An optimizer update is the micro-batches of one group (section 4.3). Each micro-batch loss is multiplied by
`n_b / N` (utterances in the micro-batch over utterances in the update) before `backward()`, gradients are
accumulated, clipped once with `clip_grad_norm_` (the pre-clip norm is what is logged), then one AdamW step.
`tests/test_parity_batching.py` proves that two micro-batches of 2 and 1 utterances give the gradient of the
full 3-utterance batch (relative L2 difference below 2e-4) and that the tempting mean of the micro-batch means
does not. A non-finite loss or gradient norm raises `FloatingPointError`: nothing is skipped silently.

### 4.3 Sampler (`plan_epoch`)

Utterances are ordered longest-first by frame count (`duration_s * 16000 // 640`) with a (seed, epoch)-seeded
random tie-break (frame counts have 90 distinct values on the real train split, so the tie-break reshuffles
batch membership each epoch exactly as RAVEn's sampler does), packed greedily so that `n * longest <= 60 s`
(fairseq's rule), the micro-batch ORDER is shuffled with a (seed, epoch)-seeded permutation, and consecutive
micro-batches are cut into updates at the first micro-batch whose cumulative padded seconds reach
`k * total / U`, `U = round(total / 198 s)`. `U` depends only on the multiset of frame counts, so every epoch has
the same number of updates (needed by a per-update LR schedule). On the real parity train split (opt-in test,
licensed id file): **530 updates per epoch**, 1,815 micro-batches, padding overhead 0.05 %, hence R-A = 26,500 and
R-B = 39,750 updates, warm-up 10,600 updates. The plan is a pure function of `(durations, seed, epoch, budgets)`:
after a resume it is rebuilt, not stored.

### 4.4 Time mask

`train_data.time_mask` re-implements the vendored `AdaptiveLengthTimeMask(window, stride, replace_with_zero=True)`
with explicit RNGs seeded from `(seed, epoch, utterance index)`. It is **bit-identical** to the vendored class for
the same two seeds (tested for 21 length/seed combinations). Semantics, including a quirk worth knowing: one mask
per started second; each draws two integers in [0, 6400): the first only bounds where the mask may START
(`start ~ U[0, len - t)`, skipped when `t == 0`), the second is the mask LENGTH. Mean masked fraction about 0.2
minus overlaps; measured against an independent numpy model in the tests. Applied only to the training dataset.

### 4.5 Optimizer and schedule

AdamW(0.9, 0.98) over every parameter with the recipe's weight decay (upstream's `get_param_groups` and Auto-AVSR
both give AdamW one decay for all parameters; biases, norms, LayerScale gammas and BatchNorm affine parameters
are decayed too). LR(update `s`, 1-based) = `peak * s / W` for `s < W`, then
`floor + (peak - floor) * (1 + cos(pi * (s - W) / (T - W))) / 2`, `floor = min(min_lr, peak)`: Auto-AVSR's
`WarmupCosineScheduler` with the end value generalised from 0 to `min_lr`; with `min_lr = 0` it equals Auto-AVSR's
scheduler exactly (tested against a verbatim copy of its class over the whole schedule). Stepped per UPDATE.

`min_lr` has two roles in R-A: upstream's `get_param_groups(min_lr=...)` floors each group's peak LR, and the cosine
ends at it. The brief asked for the second; the first is what the vendored function does. **R-A group table**
(`tests/test_parity_optim.py` pins it; 55,743,572 parameters with the 1,002-output head):

| group | peak LR | parameters |
|---|---|---|
| after_norm | 2e-3 | 1,024 |
| block_11 (last) | 2e-3 | 3,416,576 |
| block_10, 9, 8, 7, 6, 5, 4 | 1e-3, 5e-4, 2.5e-4, 1.25e-4, 6.25e-5, 3.1e-5, 1.6e-5 | 3,416,576 each |
| block_3, 2, 1, 0 | 1e-5 (decayed to 7.8e-6 ... 9.8e-7, floored) | 3,416,576 each |
| embed (input linear) | 1e-5 (floored) | 262,656 |
| frontend (ResNet1D) | 1e-5 (floored) | 3,848,576 |
| ctc head | 6e-3 | 514,026 |
| decoder (group "other") | 6e-3 | 10,118,378 |

**31.9 % of the parameters (17.8 M: the whole front-end, the input embedding and blocks 0-3) have a peak LR of
1e-5 under R-A.** That is sensible for fine-tuning a pre-trained encoder and doubtful for random initialisation;
it is why R-B (one LR, no decay) exists. R-A is implemented as written, not tuned.

### 4.6 Checkpoints, resume, selection

* `last.pt` (model, AdamW state, completed epochs, global step, torch CPU/CUDA RNG states, sampler position,
  metric history, provenance hashes, the recipe) is written at the end of every epoch to a temp file in the same
  directory, fsynced and renamed (`atomic_torch_save`); a failed write leaves the previous file and no debris
  (tested by making `torch.save` fail mid-write). It is loaded with `weights_only=True`.
* `weights_epochNNNN.pt`: plain `state_dict`, the last `--keep-last-weights` (default 10) epochs, older ones deleted.
  `final.pt`: the selected weights, also a plain `state_dict` that `scripts/parity_eval.py` loads. A run directory
  holds 15 files (cap 20 asserted each epoch, PROTOCOL §5.2) and about 3.1 GB (`last.pt` 0.67 GB, 11 weight files of 0.22 GB).
* `--resume` restores model, optimizer, RNG and counters and refuses a checkpoint whose recipe (including the
  recipes-file hash), batching or data/tokenizer/architecture hashes differ. Resumption is from an epoch boundary
  only. **Test: N epochs straight vs 1 epoch, stop, resume in a fresh trainer built from a different random
  init, N - 1 more, for a uniform-LR recipe and a layer-decay recipe with drop-path 0.2: every model tensor
  (BatchNorm statistics included), every AdamW state tensor, every per-epoch loss and WER and the final weights are
  EXACTLY equal (difference 0.0, CPU).** Mutation check: disabling the RNG restore or the optimizer restore makes the
  test fail. On GPU, CTC backward has no deterministic kernel (PROTOCOL §10 item 1) so bit-equality is not claimed.
* Selection: `final_epoch` (default of both recipes) or `average_last_k` (floats mean, integer buffers floor of the
  mean, BatchNorm running statistics averaged not re-estimated), proven bit-identical to the vendored
  `average_checkpoints` on Lightning-layout files.

### 4.7 Validation and what is logged

Per epoch, on the 1,200 val clips in eval mode: teacher-forcing loss (total, CTC, attention, token accuracy; per-utterance
means) and CTC-greedy corpus WER (argmax of the CTC head, collapse repeats, drop blanks, pieces to text as
`finetune_learner.calculate_wer`). At the end (and every `--beam-val-every` epochs) the full joint CTC/attention
beam-40 corpus WER through the unchanged `parity.decode` / `parity.metrics`: slow (one utterance at a time, 0.25-0.6 s
per audio second estimated in Stage 1, i.e. 20-45 min for 1.15 h). The trainer never opens the test split: records
must come from the index's `trainval` split (tested), train and val must be disjoint, the script has no unlock and
never names the test id file; the sbatch never sets `SPELL_FINAL_EVAL`.

W&B (`job_type` `parity_train`, run `parity-train-<recipe>-<job id>`, series `parity_train/*` only, opened first,
finished from a `finally`, step metric `parity_train/update`): per update `loss`, `loss_ctc`, `loss_att`, `acc`,
`grad_norm`, `update_s`, `data_wait_s`, `audio_s_per_s`, `audio_s`, `padded_audio_s`, `n_utts`, `n_micro_batches`,
`lr/<group>`, `gpu_mem_peak_gb`; `progress/*` at wall-clock cadence (fraction, last/slowest update, rate, ETA);
`epoch/*` (val loss, CTC-greedy WER, epoch time, ETA to completion); `beam_val/*` live during the slow pass; `final/*`.

### 4.8 Precision and determinism

fp32 only: TF32 is switched off for matmul and cuDNN (the development GPU is Ada, the target Turing; PyTorch's default
would silently use TF32 convolutions on the former). Deterministic kernels with `warn_only` are the default
(PROTOCOL §10 item 1) and cost about **1.5x in throughput** on the development GPU for this model (154 vs 105
audio-seconds per second in a measurement with 140 synthetic utterances), so `--no-deterministic-kernels` /
`PARITY_NONDETERMINISTIC=1` is offered and recorded in `summary.json`. Seeds: 42 for torch, numpy, python, set before the
model is built (initial weights are a function of the seed) and again at the start of `Trainer.fit`; the sampler, the
time mask and the DataLoader generator are pure functions of `(seed, epoch, ...)`, so the loader's worker count does not
change the result (tested: 0 and 2 workers give identical weights, two CLI runs give identical weights).

## 5. Assumptions and known unknowns, in order of expected impact on parity

1. **Effective batch size and hence the number of updates.** The paper states none for fine-tuning. R-A and R-B
   assume 198 s per update (USR's 155 frames per GPU x 32 GPUs), 530 updates per epoch. App. A.3 gives
   pre-training batches of 96 s per GPU (= USR's 2,400 frames), so a fine-tuning batch of the same order per GPU
   is not excluded: 32 x 96 s would be 3,072 s per update, about 35 updates per epoch (15x fewer; 1,750 updates in
   50 epochs and a 700-update warm-up). That would change optimisation dynamics a lot; it is the single largest
   uncertainty. Mitigation: none in the recipes (fixed before any run); it is the first candidate for a third recipe
   under §3.34(f). Asking the first author remains the cleanest fix.
2. **Layer-wise decay and LRs for from-scratch training.** Table 7 is the fine-tuning table (pre-trained
   encoder, random decoder). R-A applies it literally: 31.9 % of the parameters train at <= 1e-5 (section 4.5).
   R-B drops the decay and uses USR's single LR. Which of them (if either) is what RAVEn used from scratch is unknown.
3. **BatchNorm under a different micro-batch.** RAVEn's per-GPU batch was about 6.2 s (155 frames) if USR's value
   holds; here 60 padded seconds (about 18 utterances), so the statistics are smoother and the stochastic
   regularisation from BatchNorm is smaller. Padded frames enter the statistics in train mode (section 4.1) and the
   running statistics used at decoding are accumulated per micro-batch (momentum 0.1 per micro-batch, so with 3-4
   micro-batches per update the running statistics move faster per update than with one batch per update).
   `--micro-batch-seconds 6.2` reproduces the per-GPU size at 1.5x the run time (measured on the development GPU with the real trainer,
   300 synthetic utterances: 72 vs 110 audio-seconds/s end to end, peak 1.5 GB instead of 6.0 GB); it is not a recipe and would be a logged deviation.
4. **SentencePiece model.** RAVEn's is not released. Ours is trained on the parity train text only (section 6:
   864 of 1,000 pieces are also in RAVEn's 1,047-entry list, 55 % of RAVEn's test references tokenise to the same
   piece sequence). The 1,047 list the released checkpoint uses contains 30 digit-string pieces (`46664`, `7864`, `888`) that a
   trainval-only vocabulary lacks (the 1,017-entry `unigram1000trainval_units.txt` has 2; section 6), so RAVEn's tokenizer was probably
   trained on more text than trainval; its training text is unknown. Effect on WER: unknown, probably small. LRS3 noise tags (`{LG}`, `{NS}`)
   are kept in the train text (20 train utterances) and become pieces.
5. **Checkpoint averaging.** Unknown for RAVEn (Ma et al. 2022 averaged the last 10 checkpoints). Both recipes use the
   final epoch; averaging is implemented and tested, so it is one config value away (a third recipe).
6. **Weight decay on which parameters.** All of them (AdamW over every parameter). If RAVEn excluded norms and biases
   the effective regularisation differs slightly.
7. Smaller: the cosine ends at `min_lr` (USR's end value is unstated); dropout 0.1 in training (upstream default);
   fp32 (not stated); label smoothing 0.1 (upstream config, not in the paper); LRS3 trainval audio is AAC-derived and
   test audio is PCM from a mirror (Stage-1 unknown 4); the val clips are not video-disjoint from train, so val WER is
   optimistic as a generalisation estimate; one seed per recipe (the plan calls for 2 seeds for the final number).
8. Gradient weighting. Each utterance weighs the same in an update (as Auto-AVSR's rescaling does). A DDP loop that averages per-GPU batch
   means without that rescaling weighs each GPU batch equally instead; RAVEn's unreleased loop is not known to do either.
9. The sampler groups by 25-fps frame count with random tie-breaks as RAVEn's does, but RAVEn shards batches over 32
   GPUs and shuffles them with a distributed wrapper; ours is one process. Same data per epoch, different grouping into updates.

## 6. Numbers obtained in this session

Hardware for every timing: the development laptop, NVIDIA GeForce RTX 4060 Laptop (8 GB, Ada), torch 2.14.1+cu130,
Python 3.12.3, fp32, TF32 off. **Not an Ada 2080 Ti.** Nothing in this section is an Ada or LRS3-audio measurement.

* **Tokenizer** (`scripts/parity_train_spm.py` on the real index and the real `lrs3-valid.id`; 4 s; two runs are
  byte-identical): 1,000 units; trained on the 30,782 train references only; encode/decode round-trip mismatches
  0 / 0 / 0 (train / val / test); piece overlap with RAVEn's `unigram1000_units.txt` (1,047 entries): 864 common =
  86.4 % of ours, 82.5 % of RAVEn's, Jaccard 0.730; on RAVEn's 1,321 test rows, **726 (55.0 %)** have exactly the
  same piece sequence under our model as RAVEn's own ids spell; mean tokens per test row 11.43 (ours) vs 11.30
  (RAVEn's); 127 distinct RAVEn test pieces (727 occurrences) are absent from our vocabulary. Exact equality was not
  expected (different list size) and does not hold. **Second RAVEn list**: upstream's `labels/` also holds
  `unigram1000trainval_units.txt` (1,017 entries, sha256 `0a0deb83...`), which no upstream code reads and which is NOT vendored (the
  vendored `unigram1000_units.txt`, 1,047 entries, is the one `UNIGRAM1000_LIST`, the released checkpoint's 1,049-way head and
  the test csv use). One-off comparison against the pristine clone: 916 of our 1,000 pieces are in the 1,017 list (91.6 % of ours,
  90.1 % of theirs, Jaccard 0.832); the two RAVEn lists share only 888 pieces. The 1,047 list has 30 digit-string pieces
  (`46664`, `7864`, ...), the 1,017 list 2 and ours 1: those pieces come from text that is not trainval-only, which
  explains most of the gap to the 1,047 list. A sequence-equality check against the 1,017 list is impossible (the csv ids index the
  1,047 list).
* **Sampler on the real train split**: 530 updates/epoch, 1,815 micro-batches, padding 1.0005 (opt-in test).
* **Memory and speed of the real trainer** (full-size model, R-A or R-B, synthetic LRS3-like FLAC data, 4-6 loader
  workers, deterministic kernels): at 60 padded seconds per micro-batch, peak allocated 5.9-6.1 GB, reserved 6.5 GB
  (7.2 GB without `expandable_segments`); 110 audio-seconds/s end to end (900 utterances, 3,000 s, 15 updates of
  about 200 s, 27 s per epoch); 154 vs 105 audio-seconds/s without vs with deterministic kernels in the 140-utterance measurement. Forward+backward alone
  (no optimizer state): peak 2.0, 3.8, 5.5, 7.2 GB at 20, 41, 61, 82 padded seconds, i.e. 0.40 GB + 0.0858 GB per second, OOM at
  100 s on 8 GB; AdamW adds 0.42 GB (60 s predicted 5.97 GB, measured 5.9-6.1 GB). 80 s per micro-batch does not fit 8 GB with AdamW state. For the 11 GB
  2080 Ti the 60 s choice leaves about 3.5 GB margin. Extrapolated to the real split (105,300 s): about 16 min per
  epoch, R-B about 20 h, R-A about 14 h on this GPU: an estimate to be replaced by the 1-epoch timing job.
* **Smoke overfit** (`tests/test_parity_smoke_overfit.py`, 4 real LibriSpeech clips of 1.6-1.9 s, tiny
  transformer, character tokens, 140 updates, about 90 s on 4 CPU threads): train loss 46.7 -> 7.7; val loss 47.1 -> 6.0
  (CTC 102.9 -> 20.5, attention 40.9 -> 4.4); teacher-forcing token accuracy 0.08 -> 0.91; greedy CTC character
  error rate 0.983 -> 0.466 (with torch 2.6.0+cpu, the Ada pin: 0.983 -> 0.345) (e.g. `WHERE IS THAT` -> `ERE IS`, `THE TWENTIES` -> `E WENTIES`, `HE DOESN'T WORK AT ALL`
  -> `EN'T W RK AL`). Greedy decoding is not yet readable after 140 updates; this shows the loop learns, not how well.
* **Batch invariance, resume equivalence**: section 4.1 and 4.6.
* **Ada pins**: the Stage-2 tests were also run in the lead's throwaway environment with torch 2.6.0+cpu, numpy
  2.5.2 and no sentencepiece / torchaudio / lightning (section 8).

## 7. Commands for the user

**a. Tokenizer step (once, on the laptop; Ada has no sentencepiece and a second venv is forbidden, PROTOCOL §5.5).**

```bash
uv venv /tmp/spm_venv --python 3.12 && VIRTUAL_ENV=/tmp/spm_venv uv pip install sentencepiece pandas pyarrow pyyaml numpy
/tmp/spm_venv/bin/python scripts/parity_train_spm.py --valid-id-file <lrs3-valid.id> --out-dir parity/spm
rm -rf /tmp/spm_venv
# read parity/spm/tokenizer.json (fidelity_vs_raven), then commit parity/spm (3 MB: model, units, token ids,
# provenance) and push it to Ada like any other change (git-only transport, §5.7). The token-id file covers ALL
# trainval utterances, so it does not reveal the validation membership; whether to commit the generated split
# lists is still the licence decision of REVIEW.md item 9 (the Ada job builds them from PARITY_VALID_ID_FILE).
```

**b. 1-epoch timing run on a real 2080 Ti** (after `git pull`, `squeue --me` empty, `mkdir -p logs`):

```bash
sbatch --time=02:00:00 --export=ALL,PARITY_RECIPE=R-B,PARITY_STOP_AFTER_EPOCHS=1,PARITY_BEAM_VAL_EVERY=1,PARITY_BEAM_VAL_LIMIT=20,PARITY_VALID_ID_FILE=$HOME/spell/external/lrs3-valid.id slurm/parity_train.sbatch
# read outputs/parity/train_R-B_<job>/summary.json: history[0].epoch_wall_s, history[0].val_beam_decode_s (20 utterances),
# environment.gpu, and the W&B gpu_mem_peak_gb series; re-size --time in slurm/parity_train.sbatch from them.
```

**c. The two full runs** (one job each; they can run side by side within the `medium` QOS limits):

```bash
sbatch --export=ALL,PARITY_RECIPE=R-A slurm/parity_train.sbatch
sbatch --export=ALL,PARITY_RECIPE=R-B slurm/parity_train.sbatch
```

**d. Continue a run** that hit the wall limit or died (`<RUN_ID>` = the FIRST job's id; at most one epoch is lost):

```bash
sbatch --export=ALL,PARITY_RECIPE=R-B,PARITY_RESUME=1,PARITY_RUN_ID=<RUN_ID> slurm/parity_train.sbatch
```

**e. Evaluate the produced checkpoint** (val first; the recipe is chosen on val WER only):

```bash
sbatch --export=ALL,PARITY_SPLIT=val,PARITY_CKPT=outputs/parity/train_R-B_<RUN_ID>/final.pt,PARITY_UNITS=parity/spm/unigram1000_units.txt,PARITY_EXPECT_SHA256=none slurm/parity_eval.sbatch
# the test split, ONCE, for the chosen recipe, after the Phase-1 gate (PROTOCOL §3.33(e)) has passed and the professor
# has confirmed the band (§3.34(d)):
SPELL_FINAL_EVAL=1 sbatch --export=ALL,PARITY_SPLIT=test,PARITY_CKPT=outputs/parity/train_<chosen>_<RUN_ID>/final.pt,PARITY_UNITS=parity/spm/unigram1000_units.txt,PARITY_EXPECT_SHA256=none slurm/parity_eval.sbatch
```

**f. Re-running the opt-in tests** (laptop): `LRS3_VALID_ID_FILE=<id file>` (real split sizes, 530 updates, the real tokenizer
pipeline), `LIBRISPEECH_SMOKE_PARQUET=<parquet>` (the smoke overfit), `RAVEN_ASR_CKPT=<checkpoint>` (Stage 1).

## 8. Independent audit of Stage 1 (by the lead, before Stage 2)

Stated to the Stage-2 author and relied on; not re-derived here. The decode path matches upstream
`finetune_learner.py`; the splits were re-derived independently and are equal; `pair_edits` equals jiwer on 3,000
random pairs; the vendored model is numerics-identical to pristine upstream; the decoder attention-head count of 4 is
empirically the best on 24 LibriSpeech clips with the released checkpoint (CPU, beam 40: 4 heads -> WER 27.5 % /
teacher-forced decoder NLL 1.58 nats per token; 8 -> 32.7 % / 1.94; 2 -> 44.2 % / 2.65; 16 -> 47.0 % / 2.85).

## 9. Test results (2026-10-19, development laptop)

* **Full repository suite**, default environment (`~/bin/pymax`, Python 3.12.3, torch 2.14.1; no opt-in variables):
  `pytest -q -p no:cacheprovider --basetemp=<scratch>/pt` -> **794 passed, 15 skipped** (139 s). The Stage-1 baseline in this
  worktree was 616 passed, 7 skipped; Stage 2 adds 178 passing tests and 8 skips (6 tests that need sentencepiece, the
  real-split sampler test, the smoke overfit).
* **Parity modules with every opt-in variable set** (`LRS3_VALID_ID_FILE`, `RAVEN_ASR_CKPT`, `RAVEN_UPSTREAM_CLONE`,
  `LIBRISPEECH_SMOKE_PARQUET`), one pytest process per file: **403 passed, 6 skipped** (the 6 are the sentencepiece tests, which
  run in the throwaway venv: 18 passed there with the id file, 17 passed + 1 skipped without it). Stage 1 had 223 parity tests.
* **Ada pins** (the lead's throwaway environment: torch 2.6.0+cpu, numpy 2.5.2, pandas 3.0.5, no sentencepiece / torchaudio /
  lightning / hydra), parity files one process each with the opt-in variables: all pass, about **392 passed, 9 skipped**
  (skips: torchaudio-only comparisons, sentencepiece tests, the unpatched-upstream equivalence test). One failure was found
  here and fixed: Stage 1's import-hygiene test forbade `sentencepiece` under `parity/`; it now allows it, function-level only, in the two
  offline-tokenizer files (section 1). Not run on Ada itself (no access).
* Smoke overfit character error rate, same test in three runs (greedy CTC, 4 real clips): 0.983 -> 0.466 (4 threads), 0.534
  (6 threads), 0.345 (torch 2.6.0): the loop learns in each, the numbers differ with thread count and library version, and the test
  asserts only `< 0.8` and `< 0.85 x` the initial value. The slowest tests: smoke overfit (about 90 s), trainer (65 s), CLI (20 s), batching (12 s).
* Disk: the full suite's pytest temp directory is 1.6 GB, almost all Stage-1 checkpoint tests (214 MB per synthetic checkpoint); the
  Stage-2 training tests delete their temp directories after each test (without that they kept 2.8 GB).
* Provenance: `tests/test_parity_vendor_provenance.py` (27 tests) passes, so no vendored file changed; `git status` shows only
  `PROTOCOL.md` modified among tracked files.

## 10. Deviations from the implementer brief

* The brief names `parity/vendor/raven/labels/unigram1000trainval_units.txt`; the vendored tree has only `unigram1000_units.txt` (the list
  the code, the checkpoint and the test csv use). The overlap with the 1,047-entry vendored list is what `tokenizer.json` records; the
  overlap with upstream's 1,017-entry `unigram1000trainval_units.txt` was measured once against the pristine clone (section 6).
* The token-id file is ONE file for all trainval utterances (`token_ids_trainval.tsv`) instead of separate train and val
  files, so it does not disclose the licensed validation membership; every train and val utterance is in it.
* The smoke overfit uses 4 clips (the four shortest, 1.6-1.9 s) instead of 6-8 and 140 updates: on CPU the real ResNet
  front-end dominates the cost (about 0.6 s per update for 7 s of audio); six clips did not fit the 2-minute budget.
  Its parquet is given by `LIBRISPEECH_SMOKE_PARQUET` (skipped when unset) instead of a hard-coded scratch path, and
  its time-mask window is 0.05 s (light masking) so that 140 updates can memorise (the 0.4 s mask is tested elsewhere), it pins
  4 CPU threads (the default oversubscribed a busy machine: over 10 minutes instead of 90 s) and it skips the per-epoch
  `last.pt` writes (54 MB fsynced each; checkpointing is tested in `test_parity_trainer.py`).
* `min_lr` is both the per-group floor of `get_param_groups` (what the vendored function does) and the end of the
  cosine (what the brief asked); the brief mentioned only the second.
* `precision` and `label_smoothing` are tagged `assumed` / `inferred` rather than `paper-stated`: the PDF's Table 7 has
  no precision row and the paper never mentions label smoothing (the brief listed both under Table 7).
* Extra options not in the brief: `--no-deterministic-kernels`, `--beam-val-limit`, `--stop-after-epochs`,
  `--micro-batch-seconds` (recorded as a deviation), `PARITY_RUN_ID` (a resumed job has a new SLURM id, the output
  directory keeps the first one), a periodic-beam timing row (`val_beam_decode_s`).
* Weights-only checkpoints and `last.pt` use `weights_only=True` loading; W&B runs are per submission (a resumed job is a
  new W&B run; the `parity_train/update` step metric lines them up).
* One Stage-1 test was generalised: the import-hygiene scan of `tests/test_parity_upstream.py` forbade `sentencepiece` everywhere
  under `parity/` and `scripts/parity_*.py`; it now allows it in the two offline-tokenizer files only, function-level only.
* The throwaway venv for the tokenizer step was made with `uv` (sentencepiece 0.2.2, pandas, pyarrow, pyyaml, numpy) and
  deleted afterwards; `uv`'s global cache keeps its wheels.

## 11. Things in the brief that looked wrong or unconfirmed

* "Table 7 ... label smoothing 0.1 ... fp32": not in Table 7 (section 2).
* "Seed 42, recipe-overridable": the recipes file has the seed in `common`; changing it means editing the pinned file.
* "Weights-only checkpoints of the last 10 epochs ... within the bundle conventions": 11 weight-sized files plus `last.pt` make 3.1 GB
  per run, 6.2 GB for both recipes against a 23 GB `$HOME` abort gate; fine, but not small.
* The brief's expected 8-14 h for 75 epochs on a 2080 Ti is lower than this session's laptop extrapolation (about 20 h);
  the 1-epoch timing job decides.
