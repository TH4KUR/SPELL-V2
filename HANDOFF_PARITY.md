# HANDOFF: RAVEn-parity track (written 2026-10-29)

You are resuming with no conversation memory. This file covers ONLY the parity track.
`HANDOFF.md` (repo root of the main checkout, untracked, dated 2026-08-28) describes the
older SPELL-RQ2 / Track B work and predates this track; do not overwrite it.
Verify every claim below against the repo before repeating it. Claims are tagged
**[verified]** (re-checked by the lead in a session), **[agent-reported]** (an implementer
subagent said so; not independently re-run) or **[unverified]**.

## 1. Why this track exists

The advisor (Oct 2026) said our Track-B ceiling WER of 49.6% is high for the same data and
asked us to reproduce RAVEn's from-scratch audio-only ASR baseline first (same architecture,
same 30 h LRS3 data), and only then run selectors (DNSMOS, K-means, ...) on that setup. The
5%/10% budget sweep is on hold.

Target: RAVEn (arXiv 2212.06246) Table 1, Base, no pre-training, no LM, 30 h trainval,
official test, **18.5% corpus WER**. Upstream `ahaliassos/raven` @9fd9a374 has NO training
loop (test-only), so the loop is written here from indirect evidence and every unknown is a
pre-registered recipe variable.

Do not report any parity WER to the advisor unless it has a row in `parity/CLAIMS.md`.

## 2. State at a glance

| Item | State |
|---|---|
| Stage 1: vendored RAVEn + eval harness + Phase-0 rescore | built, audited by the lead, **uncommitted** |
| Stage 2: tokenizer prep, train data, trainer, recipes, sbatch, docs | built by a subagent, partially audited by the lead, **uncommitted** |
| Anything run on Ada or on LRS3 audio | **nothing** |
| Phase 0 (rescore Track B with corpus WER) | not run |
| Phase 1 gate (released checkpoint, WER 4.7 +- 0.2, val then test) | not run |
| Tokenizer step, 1-epoch timing run, full runs | not run |
| Open decision | R-A layer-decay (section 5) |

## 3. Where everything lives

- Worktree (all Stage 1 + 2 code, UNCOMMITTED): `/home/th4kur/Desktop/SPELL-V2/.claude/worktrees/agent-af1832bf6d3e0c256`,
  branch `worktree-agent-af1832bf6d3e0c256`, based on `bbbf2aa`. Only tracked change:
  `PROTOCOL.md` (+99 lines: Stage 1's 55 + new item 3.34). Everything else is untracked.
  **Do not remove this worktree or `git clean` it: the work exists nowhere else.**
- Main checkout: `/home/th4kur/Desktop/SPELL-V2` (`main`, 2 commits ahead of the `ada` remote).
- Plan: `/home/th4kur/.claude/plans/ok-then-make-a-staged-turtle.md`.
- Memory: `/home/th4kur/.claude/projects/-home-th4kur-Desktop-SPELL-V2/memory/spell-v2-raven-parity.md`.
- Code: `parity/` (vendor/raven, upstream, model, decode, text, metrics, data, splits,
  wandb_run, progress, runinfo, recipes, optim, train_data, trainer, checkpoint_avg,
  spm_tokens), `scripts/parity_*.py` (build_splits, eval, rescore_track_b, train, train_spm,
  vendor_manifest), `slurm/parity_eval.sbatch`, `slurm/parity_train.sbatch`,
  `configs/parity/{arch_base,recipes}.yaml`, 24 `tests/test_parity_*.py` files + `tests/parity_train_helpers.py`.
- Docs to read in this order: `parity/REVIEW.md` (Stage 1; section 7 has the Phase 0/1
  commands), `parity/REVIEW_STAGE2.md` (section 3 recipes value by value, section 5
  assumptions ranked by expected impact, section 7 user commands, section 8 Stage 1 audit,
  section 11 problems in the brief), `parity/CLAIMS.md` (PENDING rows), `PROTOCOL.md` 3.33 and 3.34.

## 4. What was checked, and by whom

**[verified by the lead]**
- Stage 1 decode path matches upstream `finetune_learner.py`; vendored model is
  numerics-identical to pristine upstream; splits re-derived independently and equal
  (train 30,782 utts / 29.25 h, val 1,200 / 1.15 h, test 1,321, all disjoint; all 1,321 test
  texts equal RAVEn's CSV); `pair_edits` equals jiwer on 3,000 random pairs; decoder heads = 4
  is empirically best (24 LibriSpeech clips: 4 -> 27.5% WER, 8 -> 32.7%, 2 -> 44.2%, 16 -> 47.0%).
  223 parity tests passed with the real checkpoint and pristine clone (2026-10-19).
- Stage 2, on 2026-10-21: R-A has 17 param groups and R-B one; group sizes sum to the model's
  55,791,794 parameters; in R-A the front-end, embed and blocks 0-3 (31.9% of parameters) sit
  at the 1e-5 floor; `lr_at` matches an independent warm-up + cosine formula at 6 points per
  recipe and ends at 1e-5; `time_mask` is identical to the vendored `AdaptiveLengthTimeMask`
  in 140/140 seeded comparisons (mean masked fraction 25%, zeros only); collate matches what
  `E2E.forward` consumes (valid = True mask, lengths in 640-sample frames); the trainer
  refuses records outside the trainval split; no frozen file changed; 530 updates/epoch is
  consistent with 29.25 h / 198 s.

**[agent-reported, NOT re-run by the lead]**
- Full suite 794 passed / 15 skipped (Stage 1 baseline 616 / 7); parity modules with all
  opt-in variables 403 passed / 6 skipped; sentencepiece tests 18 passed in a throwaway venv.
  The user declined the lead's attempt to rerun the full suite, so these counts are unconfirmed.
- Tokenizer fidelity: 1,000 units trained on train text only. Overlap with RAVEn's vendored
  1,047-piece list: 864 (82.5% of theirs, Jaccard 0.730). 726/1,321 (55.0%) RAVEn test
  references tokenize identically under our model. Upstream also has an unvendored 1,017-piece
  `unigram1000trainval_units.txt` (overlap 916, Jaccard 0.832).
- Batch invariance in eval mode: loss rel diff 7.8e-8 / grad 5e-5 rel L2 when the front-end BN
  maps 0 to 0; 6.9e-6 / 3.8e-4 with ordinary BN; train mode couples utterances (0.8% loss diff).
- Resume equivalence: model, AdamW state, losses and WERs exactly equal (0.0) on CPU for
  uniform-LR and layer-decay recipes.
- Laptop RTX 4060 8 GB (NOT Ada), full-size model, 60 s micro-batches: peak 5.9-6.1 GB, ~110
  audio-s/s end to end with deterministic kernels (154 without; 72 at 6.2 s micro-batches);
  extrapolated ~16 min/epoch, R-B ~20 h, R-A ~14 h; `--time=48:00:00` flagged UNMEASURED ON ADA.
- Smoke overfit on 4 real LibriSpeech clips, 140 updates: train loss 46.7 -> 7.7, token acc
  0.08 -> 0.91, greedy CER 0.983 -> 0.466.

**[unverified]** anything on Ada or LRS3 audio; GPU memory/speed on the 2080 Ti; the BatchNorm
effect of 60 s vs ~6 s micro-batches; the effect of our SentencePiece model on WER;
index-vs-FLAC duration agreement; GPU bit-reproducibility (not claimed; CTC backward).

## 5. Open decision: should R-A stay as the literal RAVEn Table 7 reading?

`configs/parity/recipes.yaml` (sha256 `4f0ea8572664c7fa9066831c450036ebd2d361e7173c982ed123de08c9a8dcdd`,
pinned in `parity/recipes.py::RECIPES_SHA256` and PROTOCOL 3.34) holds R-A (50 epochs, 20
warm-up, encoder LR 2e-3, decoder/CTC LR 6e-3, wd 0.1, layer decay 0.5, min LR 1e-5, clip 5.0,
drop path 0.2) and R-B (USR Table 8: 75 epochs, 20 warm-up, LR 3e-3 uniform, wd 0.04, no layer
decay, min LR 1e-5 assumed, clip 3.0, drop path 0.1). Table 7 is the FINE-TUNING table; layer
decay 0.5 suits a pre-trained model. From random init R-A leaves the front-end and blocks 0-3
at 1e-5 against a 2e-3 peak, so it will probably score far above 18.5%, costing about 14 GPU-hours
to learn that. Nothing has run, so changing the recipes file is still allowed by our own rule,
but it needs the hash constant, PROTOCOL 3.34 and CLAIMS updated together. Options:
1. keep R-A unchanged (records both the literal reading and its failure);
2. set R-A `layer_decay: null` with `lr_decoder == lr_encoder` (validation requires that) and
   keep everything else, which makes it close to R-B (so distinguish it by LR 2e-3 vs 3e-3, wd, clip);
3. ask the professor which reading RAVEn most plausibly used (the lead's recommendation, then 1).
No edit has been made. Decide with the user.

## 6. Next steps, in order

1. User decides section 5; if the recipes change, update the hash constant, PROTOCOL 3.34, CLAIMS, REVIEW_STAGE2 section 3 and rerun the recipe tests.
2. User/professor: confirm the parity band (proposed: test corpus WER within 2.0 absolute of 18.5, i.e. 16.5-20.5; PROTOCOL 3.34(d)). Do not commit before the professor has seen it, unless the user says otherwise.
3. Re-fetch the verification resources (section 7) if you need to rerun the opt-in tests locally.
4. When the user asks: commit the worktree on its branch, merge/cherry-pick to `main`, push (git-only transport, PROTOCOL 5.7). Never commit or push unasked.
5. On Ada (user submits jobs; commands in the review docs): Phase 0 rescore -> tokenizer step (`scripts/parity_train_spm.py`, then commit `parity/spm`, about 3 MB) -> Phase 1 gate (released checkpoint, val first, then test only with the unlock and the professor's sign-off) -> 1-epoch timing run to replace the extrapolated `--time` -> the full runs (`PARITY_RECIPE=R-A|R-B`, resumable with `PARITY_RESUME=1`).
6. Carried over from the earlier wrap-up (details live outside this repo, not re-verified here): send the advisor the one-line correction that loss-ranking keeps the lowest-loss, easiest examples; fix the Ada gateway `ada-gw1` and push the two commits that never reached the `ada` remote (`git log ada/main..main` shows bbbf2aa and 5106847).

## 7. Resources that were lost (scratch was wiped) and how to recover them

On 2026-10-29 the session scratch directory
`/tmp/claude-1000/-home-th4kur-Desktop-SPELL-V2/46182985-7fa9-48c6-bc24-b249734af901/scratchpad`
was empty: the pristine upstream clone, the released checkpoint, the LibriSpeech clips and the
AV-HuBERT id file are gone. Until they are re-fetched, the opt-in parity tests and the smoke
overfit skip or fail to find them. Recipes: `parity/REVIEW.md` section 3 (a-h) and section 7.
- Upstream: clone `ahaliassos/raven` at the pinned commit 9fd9a374 (full hash in `parity/vendor/raven/PATCHES.md`).
- Checkpoint: released RAVEn ASR Base, 223,444,287 bytes, sha256 `78212d2b...` (source in REVIEW.md section 7). Keep it OUTSIDE the repo; do not copy it into scratch twice.
- Id file: `avhubert/preparation/data/lrs3-valid.id` from `facebookresearch/av_hubert` @ b4fa4af9b437757b8649dc8a226e6d4d8e780f6f, sha256 `4e24da96...`. Meta licence: never copy it into the repo.
- LibriSpeech smoke clips: the origin of the 73-clip parquet was not recorded here; the smoke test skips cleanly when it is absent.

## 8. Constraints that still apply

- No commit, push, stash or git config changes unless the user asks. No SSH. No `sbatch`/`srun`/`scancel`: the user alone submits jobs. No code sets `SPELL_FINAL_EVAL`; the test split is scored once, for the chosen recipe, after the Phase-1 gate and the professor's confirmation (the user explicitly granted the test unlock for this track).
- Frozen, do not edit: `text_norm.py`, `subsets/splits/*`, `configs/protocol.yaml`, `slurm/template.sbatch`, `scripts/gen_run_plan.py`, `slurm/run_plan_*.tsv`, Track A/B model and trainer code, `requirements.lock`, anything under `parity/vendor/raven/` (a provenance test checks it byte for byte).
- Production code may import only what `requirements.lock` has (no torchaudio, no sentencepiece, no fairseq, no Lightning); only `scripts/parity_train_spm.py` and `parity/spm_tokens.py` may use sentencepiece, inside functions.
- The user lost a grade to a wrong AI baseline: primary-source verification, a claims ledger, an independent reviewer, no unverified numbers in prose.
- Laptop env: `~/bin/pymax/bin/python`; Ada: `~/envs/spell` (frozen, never a second venv).
- DISK IS TIGHT (root filesystem 96% used, about 4 GB free on 2026-10-29; it was 100% on 2026-10-18). Run pytest only as
  `~/bin/pymax/bin/python -m pytest -q -p no:cacheprovider --basetemp=<scratch>/pt` and `rm -rf` that directory afterwards
  (about 1 GB per full run otherwise). Check `df -h /` before heavy steps.

## 9. Gotchas

- The advisor's mail draft `mail.md` (main checkout root) is untracked: never `git add` it. The generated split lists are intentionally not left in the repo.
- RAVEn's unit list has 1,047 entries and the checkpoint pins `odim`; our tokenizer has 1,000 units plus blank/eos, so exact tokenization equality with RAVEn is not expected.
- The decoder FF dimension is 2048 (checkpoint-derived), not Table 7's 1024.
- Our Track-B WER was a macro mean of per-utterance WERs; RAVEn and Auto-AVSR report corpus WER. Part of the gap to 18.5% may be the metric: Phase 0 measures that.
