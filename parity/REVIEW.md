# RAVEn-parity track, Stage 1: review document

For the professor and the independent code reviewer. Stage 1 builds the verifiable
foundation and the evaluation-harness parity check; it contains no training loop, no
optimiser, no scheduler and no SentencePiece training. PROTOCOL §3.33 is the binding rule;
`parity/CLAIMS.md` is the evidence ledger; `parity/vendor/raven/PATCHES.md` explains every
vendored change. Nothing here claims parity or any WER: no LRS3 audio exists on the
development laptop, so every measured number is `PENDING` until the user runs Phase 0 and
Phase 1 on Ada.

## 1. What exists

| path | purpose |
|---|---|
| `parity/vendor/raven/` | upstream RAVEn @ `9fd9a37` (MIT), minimal closure for model + beam search + test data, with provenance proof |
| `parity/upstream.py` | the one explicit way to put the vendored tree on `sys.path` (refuses a shadowing `espnet`) |
| `parity/vendor_provenance.py`, `scripts/parity_vendor_manifest.py` | manifest/patch tooling and the strict reverse-patch applier |
| `parity/model.py`, `configs/parity/arch_base.yaml` | `build_model`, `load_released_checkpoint` (strict, `weights_only`), architecture assertions |
| `parity/decode.py` | upstream's joint CTC/attention `BatchBeamSearch` recipe minus Lightning |
| `parity/text.py`, `parity/metrics.py` | reference text, units list, corpus WER (+ macro mean, quartiles) |
| `parity/data.py` | FLAC eval dataset (bit-identical to `torchaudio.load(normalize=True)` for PCM16) |
| `parity/splits.py`, `scripts/parity_build_splits.py` | train/val/test id lists + `provenance.json` |
| `scripts/parity_rescore_track_b.py` | Phase 0: macro vs corpus WER of an existing Track-B `predictions.jsonl` |
| `scripts/parity_eval.py`, `slurm/parity_eval.sbatch` | Phase 1: released checkpoint over a parity split |
| `parity/wandb_run.py`, `parity/progress.py`, `parity/runinfo.py` | §3.23-§3.25 logging helpers, wall-clock reporter, provenance fields |
| `tests/test_parity_*.py` | 14 test modules; the full suite stays green |

## 2. Per-file provenance (upstream file -> vendored file)

Three upstream root modules are renamed on vendoring (reasons in `PATCHES.md`); everything
else keeps its upstream path. Generated from `UPSTREAM_MANIFEST.json` by
`python scripts/parity_vendor_manifest.py table`; `tests/test_parity_docs.py` fails if this
table drifts from the manifest.

<!-- provenance-table:start -->
| upstream file | vendored file | upstream sha256 | status |
|---|---|---|---|
| `LICENSE` | `LICENSE` | `a853e9e1620d9c176a101e6f321e1cbd13055fd202c10f265e5b4437b3bec1b5` | identical |
| `conf/model/audio_backbone/resnet_transformer_base.yaml` | `conf/model/audio_backbone/resnet_transformer_base.yaml` | `7e1f209abb2ba7bf156f9bc6f437a99bc75e8a61a3fb140345d7bd37b604c25c` | identical |
| `data_paths/test_with_tags_counts_unigram1000.csv` | `data_paths/test_with_tags_counts_unigram1000.csv` | `3be424dfc77b5fbabd7119cdeb7820dd719d5b62da76d511f7e6c316a1fe3212` | identical |
| `espnet/asr/asr_utils.py` | `espnet/asr/asr_utils.py` | `380300e458ecd7ba05a47bcc5ad9d34d2e37ebf9f4cf82562820d4cbf31fafc8` | identical |
| `espnet/nets/batch_beam_search.py` | `espnet/nets/batch_beam_search.py` | `9a97a36238ca38df3050b1b9dbd6466c578e01af2e82e5fe35deffcbd61dc185` | identical |
| `espnet/nets/beam_search.py` | `espnet/nets/beam_search.py` | `5be747eb32dc052e9dbe42383073c2ea598e55ecb5995edfa4c86cbd707811ec` | identical |
| `espnet/nets/ctc_prefix_score.py` | `espnet/nets/ctc_prefix_score.py` | `867b0179ec6230e9e69a32593c0bf96b12722cb184abe866a67e2042022d7fef` | patched (vendored sha256 `d74abd58969800d8575db22c6a11f77b65296487a4501bae26b1e1e1a093acb4`) |
| `espnet/nets/e2e_asr_common.py` | `espnet/nets/e2e_asr_common.py` | `cfddfe2b403e9b73743633f1deaaec2d6e67c08887414c8e7745882d136a8a2d` | patched (vendored sha256 `b8846ee45dc26b1f393565ba408f9939fffa0255767ab506ca6a2a53c6ec0299`) |
| `espnet/nets/pytorch_backend/backbones/conv1d_extractor.py` | `espnet/nets/pytorch_backend/backbones/conv1d_extractor.py` | `9569c1b09947253711b9190551d9c0e01d1b1decf497414483150b0676c3663c` | identical |
| `espnet/nets/pytorch_backend/backbones/conv3d_extractor.py` | `espnet/nets/pytorch_backend/backbones/conv3d_extractor.py` | `3f4bab6a04b59aed37a22f9253a17350a2302535b62500642db9446dbbf876d4` | identical |
| `espnet/nets/pytorch_backend/backbones/modules/resnet.py` | `espnet/nets/pytorch_backend/backbones/modules/resnet.py` | `bd2e709f79301f4f903c8d36c7843d899a032157aa05517256bdcb1bc41b88f0` | identical |
| `espnet/nets/pytorch_backend/backbones/modules/resnet1d.py` | `espnet/nets/pytorch_backend/backbones/modules/resnet1d.py` | `87987c42b31740679e3511cbe105e9fae45b95557be3f1a3dc8f9208f47692ec` | identical |
| `espnet/nets/pytorch_backend/ctc.py` | `espnet/nets/pytorch_backend/ctc.py` | `4929e40ec38c434080e13fbb8a1d2c9d69c43ce981d5b65125a1f12e12394f40` | patched (vendored sha256 `2d4ef4d9346c4302aff7afa94e3ec6b5d07b6fb2530c47da91c74c5450fbb663`) |
| `espnet/nets/pytorch_backend/e2e_asr_transformer.py` | `espnet/nets/pytorch_backend/e2e_asr_transformer.py` | `f8ed66cfc0f01c5b63333b208e79114bc83e48316e70b8c618b6641e95a96270` | identical |
| `espnet/nets/pytorch_backend/nets_utils.py` | `espnet/nets/pytorch_backend/nets_utils.py` | `53a66c3ed95de0fdd59677df1682dcf8bf3f109a62f6d2cbf284d7eb948f4efb` | patched (vendored sha256 `8fa0a29ce7bb6e244a825be58702e2958a3710097d8bf4b37e2751dfe9e498bf`) |
| `espnet/nets/pytorch_backend/transformer/__init__.py` | `espnet/nets/pytorch_backend/transformer/__init__.py` | `aba1b9ff25a1b69d8a6905e7ed877205b5b8af6f3d1d9e82bb28b8aaed9aaeee` | identical |
| `espnet/nets/pytorch_backend/transformer/add_sos_eos.py` | `espnet/nets/pytorch_backend/transformer/add_sos_eos.py` | `e73c11b60af9a67467bde1e53b031d43ccd00cd55e5f85813c4e6d757114c7ce` | identical |
| `espnet/nets/pytorch_backend/transformer/attention.py` | `espnet/nets/pytorch_backend/transformer/attention.py` | `bfea4fa020b93bcf096ccd960635144ee5eb7f9683a8537ed0f637ed556be645` | identical |
| `espnet/nets/pytorch_backend/transformer/convolution.py` | `espnet/nets/pytorch_backend/transformer/convolution.py` | `ce9e9f1dc6bf22ff51ac253a1554120181f99f63fea212030ad837f975ebbf7b` | identical |
| `espnet/nets/pytorch_backend/transformer/decoder.py` | `espnet/nets/pytorch_backend/transformer/decoder.py` | `4899bb6e9c0b77f4fe1dcff353433b9428b549eb4ca44ee5fd11aa7ceeca4c4e` | identical |
| `espnet/nets/pytorch_backend/transformer/decoder_layer.py` | `espnet/nets/pytorch_backend/transformer/decoder_layer.py` | `11c4ca35005150e1ff776ea750a3f30874f366279f728470cbdd80eb539c9f1a` | identical |
| `espnet/nets/pytorch_backend/transformer/embedding.py` | `espnet/nets/pytorch_backend/transformer/embedding.py` | `a5a1fb01222a7b8e6117daa333e1df1847c35cdb56d07a5a70fc1fb044cb327a` | identical |
| `espnet/nets/pytorch_backend/transformer/encoder.py` | `espnet/nets/pytorch_backend/transformer/encoder.py` | `878dd2a4441755a7d68a6e9e5532a9d35b253816f639fe5d41a25af0a9072d0d` | identical |
| `espnet/nets/pytorch_backend/transformer/encoder_layer.py` | `espnet/nets/pytorch_backend/transformer/encoder_layer.py` | `854efa9167791365ab6b20fe2eebade029c5fd82f1e6d4473d8fc3bd1db15ae4` | identical |
| `espnet/nets/pytorch_backend/transformer/label_smoothing_loss.py` | `espnet/nets/pytorch_backend/transformer/label_smoothing_loss.py` | `70f6a14a1f8749c1dfa39a96469d9633246856572d2140cddd0378857598e265` | identical |
| `espnet/nets/pytorch_backend/transformer/mask.py` | `espnet/nets/pytorch_backend/transformer/mask.py` | `98247deea6952704379537e22ece50c956b40257a124359a575b54c7073e763d` | patched (vendored sha256 `2c7bad3e47377bbda6d6e79961be51c23112ddf0a0b3db5b52fdc7c9b2fdc2b6`) |
| `espnet/nets/pytorch_backend/transformer/multi_layer_conv.py` | `espnet/nets/pytorch_backend/transformer/multi_layer_conv.py` | `00103ea8b18d759dabcc80b846559bd49fe9348308ad46ff41aa34ed0bfbe3fd` | identical |
| `espnet/nets/pytorch_backend/transformer/norm.py` | `espnet/nets/pytorch_backend/transformer/norm.py` | `888715d18ee4866d8b01c45f218f389e6a2d513374b312a3a2250f5717b4ff30` | identical |
| `espnet/nets/pytorch_backend/transformer/positionwise_feed_forward.py` | `espnet/nets/pytorch_backend/transformer/positionwise_feed_forward.py` | `c4db61e228ba3c55158d20ce53a1d5ec13e87a34148d38164230d4ac93d506b2` | identical |
| `espnet/nets/pytorch_backend/transformer/repeat.py` | `espnet/nets/pytorch_backend/transformer/repeat.py` | `88885f45acfb4675fe35db9e1f7ced655281e0b2d2958583f4b7b638e21e7577` | identical |
| `espnet/nets/scorer_interface.py` | `espnet/nets/scorer_interface.py` | `6502f5901d0cd7b4fae768826c87c3374d3f843de9943ef4a3db3f5f5ae302a4` | identical |
| `espnet/nets/scorers/__init__.py` | `espnet/nets/scorers/__init__.py` | `aba1b9ff25a1b69d8a6905e7ed877205b5b8af6f3d1d9e82bb28b8aaed9aaeee` | identical |
| `espnet/nets/scorers/ctc.py` | `espnet/nets/scorers/ctc.py` | `110110ee014c39d6d90f4e227d60e77d394ecdfa3a2afe38f7a244ebcb2ecb5e` | identical |
| `espnet/nets/scorers/length_bonus.py` | `espnet/nets/scorers/length_bonus.py` | `3a022a50d0143b0f9bd631d05e0a256252f03afe4e5649ffe674d91c65bc3b08` | identical |
| `labels/unigram1000_units.txt` | `labels/unigram1000_units.txt` | `ead1469024d34194e1c7abfe2ca0457f06d5afee78267af2a0cbdea946f359f4` | identical |
| `data/transforms.py` | `raven_data_transforms.py` | `4d2bbc5a2c611d3a297393eafc0cc5ad76d74b7ddff5b4442085a8babe1cbe21` | patched (vendored sha256 `f5b9a0413f017d646ea3a832a4c2b285d61e4aeca1d7c2a6931a6e46a3da35e5`) |
| `metrics.py` | `raven_metrics.py` | `730f88560cd0ecf382ddf476feb6cce69d62078a8026093324e306198cf3d391` | patched (vendored sha256 `254b37f158670a6b97d9629d09457c7be15313fd0af5075cba2a0013033e593b`) |
| `utils.py` | `raven_utils.py` | `1c74937914563c31a34bcddc9c3d02472ccce3de842f09550cc3a7135acfde1a` | identical |
<!-- provenance-table:end -->

## 3. How to check it yourself

```bash
# a. clone upstream at the pinned commit
git clone https://github.com/ahaliassos/raven.git /tmp/raven && git -C /tmp/raven checkout 9fd9a3743c41ba93532258b340d123c20fe25432

# b. per-file diff of every vendored file against upstream (renames handled explicitly); run from the repo root
(cd parity/vendor/raven \
 && for f in $(find . -type f ! -path '*/__pycache__/*' ! -name PATCHES.md ! -name UPSTREAM_MANIFEST.json ! -name vendor.patch | sed 's|^\./||' | LC_ALL=C sort); do
      case "$f" in
        raven_utils.py)           up=utils.py ;;
        raven_metrics.py)         up=metrics.py ;;
        raven_data_transforms.py) up=data/transforms.py ;;
        *)                        up="$f" ;;
      esac
      diff -u --label "a/$up" --label "b/$f" "/tmp/raven/$up" "$f"
    done)      # prints exactly the contents of vendor.patch (same file order)

# c. recompute manifest + patch from the clone and require byte equality with the stored ones
python scripts/parity_vendor_manifest.py check --upstream /tmp/raven
# d. offline proof (no clone): reverse-apply the patch, compare sha256 + git blob sha1
python scripts/parity_vendor_manifest.py verify
pytest -q tests/test_parity_vendor_provenance.py
# e. independent applier: GNU patch -R must also give back upstream bytes (run from the repo root)
t=$(mktemp -d) && cp -r parity/vendor/raven/. "$t" && (cd "$t" \
  && mkdir data && mv raven_data_transforms.py data/transforms.py && mv raven_metrics.py metrics.py && mv raven_utils.py utils.py \
  && patch -R -p1 < vendor.patch \
  && for f in $(find . -type f ! -path '*/__pycache__/*' ! -name PATCHES.md ! -name UPSTREAM_MANIFEST.json ! -name vendor.patch | sed 's|^\./||'); do
       cmp "$f" "/tmp/raven/$f" || echo "DIFFERS: $f"; done)
# f. patches change no numerics: unpatched upstream vs vendored, bit-identical (needs six + distutils)
RAVEN_UPSTREAM_CLONE=/tmp/raven RAVEN_ASR_CKPT=<checkpoint> pytest -q tests/test_parity_upstream_equivalence.py
# g. the checkpoint and the architecture
sha256sum <checkpoint>     # 78212d2b008a3636f5798745c8c0f60e86b9bed71f280fcd2a603102c4a01f4f
RAVEN_ASR_CKPT=<checkpoint> pytest -q tests/test_parity_model.py
# h. the splits (needs AV-HuBERT's lrs3-valid.id, sha256 4e24da96...; not stored in the repo)
LRS3_VALID_ID_FILE=<lrs3-valid.id> pytest -q tests/test_parity_splits.py
```

## 4. The patches, in one paragraph each

Seven vendored files differ from upstream; each change is listed with line numbers in
`PATCHES.md`. Five files get import-level edits only: `distutils.version.LooseVersion`
(removed in Python 3.12) becomes `packaging.version.Version` under the same name in
`mask.py` and `pytorch_backend/ctc.py`; `import six` and `six.moves.range` become plain
`range` in `pytorch_backend/ctc.py`, `ctc_prefix_score.py` and `e2e_asr_common.py`;
`dtype=np.int` (a numpy alias removed in 1.24) becomes `dtype=int` in four RNN-only lines of
`nets_utils.py` that the transformer path never reaches. Two files are cut down by deletion
only: `raven_metrics.py` loses the `torchmetrics`-based `WER` class (its logic is
re-implemented and cross-checked in `parity/metrics.py`) and `raven_data_transforms.py`
keeps only `AdaptiveLengthTimeMask`. A whitelist test accepts no other changed line, and
the equivalence test (step f) shows identical encoder output and beam-search scores with and
without the patches.

## 5. Where each setting comes from

| setting | value | class | source |
|---|---|---|---|
| encoder / decoder widths, depths, FF sizes, LayerScale, BatchNorm-before-FF, relative MHA, no macaron / conv module | idim = adim 512, 8 heads, FF 2048, 12 blocks; decoder 256-d, FF 2048, 6 blocks | checkpoint-derived | tensor shapes of the 577-tensor `state_dict`; fingerprint test |
| decoder attention heads | 4 | paper-stated, not checkpoint-checkable | paper Table 7 (low-resource column); upstream `*_trainval` scripts |
| `relu_type` swish, `rel_pos_type` latest, `zero_triu` False, `a_upsample_ratio` 1 | as listed | upstream-default, not checkpoint-checkable | upstream `resnet_transformer_base.yaml` |
| decoder MLP size | 2048 (paper table says 1024) | checkpoint authoritative | `decoder.decoders.*.feed_forward.w_1.weight` 2048 x 256 |
| dropout / drop-path at evaluation | irrelevant (`eval()`) | n/a | model is put in eval mode; `transcribe` refuses a train-mode model |
| beam 40, CTC weight 0.1, decoder weight 0.9, no LM, no length bonus, max/min length ratio 0 | as listed | paper-stated and upstream-code | paper App. A.4; upstream `conf/decode/default.yaml`, `finetune_learner.py` |
| audio: 16 kHz mono float32 in [-1, 1], no normalisation | | paper-stated and upstream-code | paper App. A.2; upstream identity audio transform |
| encoder input shape `[1, T, 1]`, truncation to a multiple of 640 samples | | upstream-code | `finetune_learner.calculate_wer`, `conv1d_extractor.py` |
| token list, sos = eos = 1048, blank 0 | | checkpoint-derived and upstream-code | `ctc.ctc_lo.weight` 1049 x 512; units file; `UNIGRAM1000_LIST` |
| hypothesis post-processing (drop `<eos>`, `▁` -> space, strip) | | upstream-code | `finetune_learner.calculate_wer` |
| references: upper-case `text_raw`, digits kept | | upstream-code and data | upstream labels are upper-case units; our test texts equal RAVEn's csv labels as a multiset |
| corpus WER as the headline metric | | upstream-code | `metrics.WER` (errors weighted by reference length / total words) |
| train / val / test composition | 30,782 / 1,200 / 1,321 | partly inferred | val ids and test labels come from pinned public sources; that RAVEn trained on trainval minus those ids is a reconstruction (their train/val csv files are not released) |
| Phase-1 pass band 4.7 +/- 0.2 | | assumed (project decision, PROTOCOL §3.33(e)) | paper and README both list 4.7; the README says re-trained models "may differ slightly" |
| `--time=04:00:00`, 4 CPUs, 16 GB | | assumed | no Ada measurement; see the sbatch comment |

## 6. What the Phase-1 gate does and does not show

The released checkpoint is RAVEn-pre-trained (433 h unlabelled LRS3) and then fine-tuned on
30 h; it is not a from-scratch model. Reproducing 4.7 % corpus WER on the official test set
shows that this harness (model code, checkpoint loading, audio front-end, beam search,
token mapping, text handling, corpus WER, our test audio and references) is equivalent to
RAVEn's. It says nothing about the from-scratch training recipe. The val run is a pipeline
sanity number, not a held-out measurement, unless RAVEn really excluded those 1,200 clips
from fine-tuning (unconfirmed): a good val WER proves little, a bad one proves the harness
is broken. Decoding is never tuned to make a gate pass.

## 7. Commands for the user

Once, after committing and pushing (git-only transport, §5.7), on Ada:

```bash
cd ~/spell/repo && git pull origin main && mkdir -p logs ~/spell/ckpts ~/spell/external
# checkpoint (223,444,287 bytes) and the licensed id file, both kept OUTSIDE the repo:
curl -L "https://drive.usercontent.google.com/download?id=1UJXGo9qUZ0VxPNlJfL2-JD5_428JqxNv&export=download&confirm=t" \
     -o ~/spell/ckpts/raven_base_lrs3_asr.pth
echo "78212d2b008a3636f5798745c8c0f60e86b9bed71f280fcd2a603102c4a01f4f  $HOME/spell/ckpts/raven_base_lrs3_asr.pth" | sha256sum -c -
# ~/spell/external/lrs3-valid.id: avhubert/preparation/data/lrs3-valid.id from
# facebookresearch/av_hubert @ b4fa4af9b437757b8649dc8a226e6d4d8e780f6f (sha256 4e24da96...)
squeue --me     # must be empty (PROTOCOL §5.8)
```

Phase 0 (minutes, CPU; compute nodes cannot see /share1, so copy the Track-B eval output under
`$HOME` first). Needs an interactive CPU allocation because the login node runs no Python:

```bash
srun -p u22 -A research --qos=medium --constraint=2080ti --exclude=gnode066 --gres=gpu:0 -N1 -n1 -t 10 --pty bash
module purge && module load u22/python/3.12.4 && source ~/envs/spell/bin/activate && cd ~/spell/repo
python scripts/parity_rescore_track_b.py --predictions <eval_dir>/predictions.jsonl \
    --summary <eval_dir>/summary.json --out outputs/parity/rescore_ceiling.json
exit
```

Phase 1, the two commands, in this order:

```bash
# 1) val first (no unlock needed; the split lists are built in the job from the id file if absent)
sbatch --export=ALL,PARITY_SPLIT=val,PARITY_VALID_ID_FILE=$HOME/spell/external/lrs3-valid.id slurm/parity_eval.sbatch
# 2) only after the val run looked sane (read outputs/parity/eval_val_<jobid>/summary.json and the W&B run):
SPELL_FINAL_EVAL=1 sbatch --export=ALL,PARITY_SPLIT=test slurm/parity_eval.sbatch
```

Pass criterion for command 2 (PROTOCOL §3.33(e)): `corpus_wer` in `summary.json` between 0.045 and
0.049. Then fill the `PENDING` rows of `parity/CLAIMS.md`. If it fails, stop; do not tune decoding.
First diagnostics if the number is far off: the decoder head count in `configs/parity/arch_base.yaml`
(4 vs 8), then the test-audio provenance (known unknown 4 below).

## 8. Test results (as run during Stage 1)

Laptop (`~/bin/pymax`, Python 3.12.3, torch 2.14.1), `pytest -q` over the whole repository:

* default environment: **616 passed, 7 skipped** (before Stage 1: 399 passed, 1 skipped; the six
  new skips are the opt-in tests that need an external file: 3 checkpoint tests, 2 tests needing
  the licensed id file, 1 equivalence test).
* with `RAVEN_ASR_CKPT`, `LRS3_VALID_ID_FILE` and `RAVEN_UPSTREAM_CLONE` set: **622 passed,
  1 skipped** (the remaining skip is the pre-existing `SPELL_DATA_ROOT` layout probe).

Throwaway venv with Ada's pins (Python 3.12.3, torch 2.6.0+cpu, numpy 2.5.2, pandas 3.0.5, no `six`,
`torchaudio`, `torchmetrics`, `hydra` or `lightning`), the 14 parity modules with the checkpoint and
id-file env vars set: **220 passed, 3 skipped** (torchaudio absent for 2 comparison tests; the
unpatched-upstream equivalence test was run on the laptop only, it needs `six` + `distutils`).

Other one-off runs (details in `CLAIMS.md`): `strict=True` load of the real checkpoint (577 keys,
55,791,794 parameters); `scripts/parity_eval.py` end to end with the real checkpoint on a synthetic
two-utterance staged tree (CPU, beam 40, sha256 verified); bit-identity of the vendored and the
unpatched upstream encoder/beam search on the real weights; GNU `patch -R` reconstruction of all
seven patched files; splits built from the real id file (30,782 / 1,200 / 1,321).

## 9. Known unknowns

1. **Nothing ran on Ada.** Ada's env is torch 2.6.0+cu124 on 2080 Ti GPUs; verification used a
   CPU wheel of torch 2.6.0 (Python 3.12.3, not 3.12.4) and the laptop's torch 2.14.1. GPU
   beam-search tie-breaking can differ from CPU in the last bits of a score.
2. **Decoder head count (4)** is paper-stated but no tensor shape depends on it; only the
   Phase-1 WER can confirm it end to end.
3. **Whether the released checkpoint saw the 1,200 val clips** (see section 6).
4. **Test-audio provenance.** Our 1,321 test clips are PCM wav from a Hugging Face mirror
   (PROTOCOL §4) converted to 16 kHz FLAC; RAVEn decoded `.wav` files it extracted itself from
   the official LRS3 `.mp4`s. The transcripts are verified equal; the audio paths are not.
5. **Staged FLAC bit depth** is unverified (no audio on the laptop). `parity_eval.py` records the
   libsndfile subtype of the first utterance in `summary.json`; any integer PCM subtype gives
   the same normalised values as torchaudio.
6. **Released-checkpoint training details beyond shapes** (the README warns that some models were
   re-trained, so WER may differ slightly from the paper's 4.7).
7. **W&B from compute nodes** is exercised only through mocks here; the older scorers log online
   the same way.
8. **Run time on Ada** is estimated from the paper's A100 figure and laptop-CPU timing.
9. **Licence of the generated split lists**: `parity_val_ids.txt` is, in effect, the content of the
   Meta-licensed `lrs3-valid.id` and the train list is its complement. They are generated, not
   committed; whether to commit them is the user's licence decision. Until then Ada builds them
   in the job from the id file (`PARITY_VALID_ID_FILE`).
10. **New location `$HOME/spell/ckpts/`** for the checkpoint is a convention introduced here, not a
    PROTOCOL §5.0 constant; override with `PARITY_CKPT`.
11. **LRS3 noise tags in references.** Three of the 1,200 val references end in `{LG}` or `{NS}` (3 of 13,477 reference words) and the units list has no brace pieces, so each costs at most one deletion on the val run (<= 0.03 % absolute); the test references contain none.
12. A real-speech smoke decode (a TED excerpt and the JFK clip, not LRS3) produced readable
    English with the released checkpoint, which supports but does not replace the Phase-1 gate.

## 10. Deviations from the implementer brief

* `espnet/asr/asr_utils.py` is vendored whole and verbatim (990 lines) instead of being cut
  down to `add_results_to_json`: it imports only numpy/torch/stdlib at module level, so
  extraction would only have added a large deletion patch.
* The brief called `import pdb` a Python 3.12 incompatibility; `pdb` is standard library, so
  those lines are left as upstream wrote them.
* The brief says `six`, `hydra` and `torchmetrics` are absent on Ada; `requirements.lock`
  lists `six==1.17.0`, `hydra-core==1.3.5`, `torchmetrics==1.9.0`. The parity code avoids them
  anyway, as instructed (only `torchaudio`, `sentencepiece` and `fairseq` are truly outside the lock).
* The brief's `x = wav[None]` is `[1, T, 1]` in upstream (batch, samples, channel); that is what
  `parity/decode.py` feeds the encoder.
* Three upstream root modules are vendored under new names (`raven_*.py`), one of them
  because `.gitignore` contains `data/`.
* `numpy` aliases `np.int` in four dead lines of `nets_utils.py` were patched too (brief asked to
  check numpy-2 compatibility of everything vendored).
* `parity_eval.py` calls `hardware_guard.enforce_gpu_policy()` (which calls `assert_gpu()` unless
  the laptop-only `SPELL_DEV_GPU` bypass is set), like `evaluate_track_b.py`; `--device cpu` skips it.
* Extra, unrequested but cheap safeguards: the splits' `provenance.json` records the sha256 of
  `data_index.parquet` and `parity_eval.py` refuses lists built from a different index; an
  existing `summary.json` is never overwritten without `--overwrite`; a `--limit` run is flagged
  `is_subset_smoke` in its summary.
