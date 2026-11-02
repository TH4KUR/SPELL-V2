# Vendored RAVEn: provenance and patches

Upstream: <https://github.com/ahaliassos/raven>, commit
`9fd9a3743c41ba93532258b340d123c20fe25432` ("fix baseplus model configs",
2025-02-27). Licence: MIT (`LICENSE`, kept). The ESPnet-derived files under
`espnet/` keep their Apache-2.0 headers untouched (a test pins this).

This directory holds 38 upstream files (34 Python modules, the units list, the test
csv, one yaml config and `LICENSE`) and three documents we wrote: this file, `UPSTREAM_MANIFEST.json` and `vendor.patch`.
Nothing else may live here (`tests/test_parity_vendor_provenance.py` fails on any
extra file).

## How "vendored == upstream + these patches" is proven

* `UPSTREAM_MANIFEST.json` records, per file, the upstream sha256 and git blob sha1
  at the pinned commit, and the sha256 of the vendored file.
* `vendor.patch` is the concatenated `diff -u` of every file that differs
  (headers `a/<upstream path>` / `b/<vendored path>`).
* `pytest tests/test_parity_vendor_provenance.py` (offline) reverse-applies the patch
  to the vendored bytes and requires the result to hash to the recorded upstream
  sha256 *and* git blob sha1; unpatched files must hash to them directly. It also
  proves that every changed line is one of the whitelisted numerics-neutral rewrites
  listed below, that no unlisted file exists, and that no vendored file is git-ignored.
* With a clone of upstream at the pinned commit:
  `python scripts/parity_vendor_manifest.py check --upstream <clone>` recomputes the
  manifest and the patch from scratch and requires byte equality with the stored ones.
* `RAVEN_UPSTREAM_CLONE=<clone> pytest tests/test_parity_upstream_equivalence.py`
  runs the UNPATCHED upstream tree and the vendored tree in separate processes on the
  same weights and input and requires bit-identical encoder features, 1-best token ids
  and beam-search scores (needs an environment that still has `six` and `distutils`).

## Renamed files (content verbatim unless listed under "Patches")

| upstream | vendored | why |
|---|---|---|
| `utils.py` | `raven_utils.py` | generic top-level names would shadow unrelated modules once this directory is on `sys.path` |
| `metrics.py` | `raven_metrics.py` | same |
| `data/transforms.py` | `raven_data_transforms.py` | same, and the repo's `.gitignore` contains `data/`, which would silently drop the file from a commit |

## Import mechanism

Upstream imports itself absolutely (`from espnet.nets... import ...`), so this
directory must be on `sys.path`. `parity/upstream.py::activate()` is the single,
explicit way to do that; it also refuses to continue if `espnet` resolves anywhere
other than here (a pip-installed `espnet` is a regular package and would beat this
namespace package). Tests assert the imported `espnet` comes from this directory.

## Patches (all numerics-neutral)

Line numbers are upstream line numbers.

| file | lines | change | reason |
|---|---|---|---|
| `espnet/nets/pytorch_backend/transformer/mask.py` | 8 | `from distutils.version import LooseVersion` -> `from packaging.version import Version as LooseVersion` | `distutils` was removed in Python 3.12. Only used for `LooseVersion(torch.__version__) >= LooseVersion("1.2.0")` style comparisons, which are the same under `Version` (`packaging` is in `requirements.lock`). |
| `espnet/nets/pytorch_backend/ctc.py` | 1 | same import swap | same; used for `LooseVersion(torch.__version__) < LooseVersion("1.7.0")`, false on every torch we run, so `CTC.ctc_type` is `"builtin"` exactly as before. |
| `espnet/nets/pytorch_backend/ctc.py` | 4, 195, 196, 219, 223 | remove `import six`; `six.moves.range(` -> `range(` (`CTC.forced_align`, never called for decoding) | `six` is not part of the parity runtime; on Python 3 `six.moves.range` *is* `range`. |
| `espnet/nets/ctc_prefix_score.py` | 9, 267, 300, 331, 340 | remove `import six`; `six.moves.range(` -> `range(` | same (these loops ARE executed by CTC prefix scoring; semantics identical). |
| `espnet/nets/e2e_asr_common.py` | 15, 34 | remove `import six`; `six.moves.range(M)` -> `range(M)` in `end_detect` | same (executed by beam search end detection). |
| `espnet/nets/pytorch_backend/nets_utils.py` | 410, 420, 435, 454 | `dtype=np.int` -> `dtype=int` in `get_subsample` | `np.int` was removed in numpy 1.24; it was an alias of builtin `int`. These four lines are in the RNN branches that the transformer path (`return np.array([1])`) never reaches; patched so the file is numpy-2 clean. |
| `raven_metrics.py` (upstream `metrics.py`) | 3, 39-58 | delete `from torchmetrics import Metric` and `class WER(Metric)` | `torchmetrics` is not part of the parity runtime. `get_wer`/`get_cer`/`get_er` (the Levenshtein implementation, the oracle for `parity/metrics.py`) are untouched. RAVEn's `WER` logic (sum of `get_wer * n_ref_words` over sum of `n_ref_words`) is re-implemented in `parity/metrics.py` and cross-checked in `tests/test_parity_metrics.py`. |
| `raven_data_transforms.py` (upstream `data/transforms.py`) | 1, 5-430 | deletion only: everything except `import random`, `import torch` and `class AdaptiveLengthTimeMask` | the rest needs `torchvision` (not in the env) and is video-only. `AdaptiveLengthTimeMask` is vendored verbatim (file keeps upstream's CRLF line endings) for Stage 2; unused in Stage 1. |

Statement on numerics: none of these edits touches arithmetic, tensor shapes, dtypes,
control flow or random-number consumption. (1) The `LooseVersion` comparisons evaluate
identically for every torch version in use. (2) `six.moves.range` is `range`. (3) The
`np.int` lines are unreachable for this model. (4) The two extractions only delete
code that nothing in the parity path imports. The whitelist test mechanically rules out
any other changed line, and the opt-in equivalence test shows bit-identical encoder
output and beam search against the unpatched tree.

## Not patched (known, inert)

* `espnet/asr/asr_utils.py` is vendored whole and verbatim (990 lines) instead of
  being cut down to `parse_hypothesis` / `add_results_to_json`: its module-level
  imports are only `numpy`/`torch`/stdlib, so extracting would only add a large
  deletion patch. Its trainer-extension helpers import `chainer` lazily and are never
  called; so are `matplotlib` (plotting), `editdistance` (`ErrorCalculator` in
  `e2e_asr_common.py`) and `warpctc_pytorch` (the `warpctc` branch of `CTC.__init__`,
  disabled on torch >= 1.7). `tests/test_parity_upstream.py` pins that these four appear
  only as function-level imports in vendored files and that importing the whole closure
  works with them blocked.
* `import pdb` (unused) in `resnet.py` / `resnet1d.py`: `pdb` is standard library in
  Python 3.12, so it is left as upstream wrote it.
* `conv3d_extractor.py` references `ShuffleNetV2`, which upstream never defines; it is
  only reachable with `backbone_type="shufflenet"` and is dead for audio.
* `nets_utils.get_activation` lazily imports a package that does not exist here; unused.

## Upstream inconsistencies found while pinning the architecture

* The upstream audio test scripts (`scripts/testing/asr/**`) set the decoder shape
  (`ddim=256 dheads=4 dunits=2048 dlayers=6`) on `model.visual_backbone.*`, which an
  `data.modality=audio` run never reads (it uses `model.audio_backbone`, whose `ddim` /
  `dheads` / `dunits` interpolate to 512 / 8 / 2048). The released checkpoint has a
  256-d decoder, so the audio configuration as written upstream would not load it.
* The paper's Table 7 (low-resource column) lists decoder hidden size 256 and 6 Base
  blocks, which match the checkpoint, but decoder MLP size 1024; the checkpoint's
  `decoder.decoders.*.feed_forward.w_1.weight` is 2048 x 256.
* The checkpoint cannot confirm the decoder head count (it changes no tensor shape);
  `configs/parity/arch_base.yaml` uses 4, which is what Table 7 (low-resource) and the
  same upstream scripts say.

The released-checkpoint shapes are authoritative for everything they can determine.
