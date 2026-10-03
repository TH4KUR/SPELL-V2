# SPELL-RQ2

Do data-selection methods rank the **same** when identical subsets of discrete speech
tokens are consumed by a generative vocoder (Track A) versus an ASR model (Track B)?
This repo implements a controlled dual-track benchmark on LRS3: selectors produce subset
manifests once, and both tracks train from scratch on each manifest under a locked
protocol (`PROTOCOL.md` — binding, read it before changing anything).

Docs: `PLAN.md` (phased spec) · `RESEARCH.md` (study design) · `PROTOCOL.md` (locked rules)
· `docs/ada_guide.md` (cluster operations).

---

## The one-paragraph version

If you can't afford to train on all your data, which subset should you keep? We tested
six different, genuinely distinct strategies for picking 25% of a 29k-utterance speech
corpus — audio-quality filtering, distribution matching, diversity clustering, two
flavors of hard-example mining, and a gradient-influence method — against the simplest
possible baseline: picking 25% at random. The headline result, checked with real
statistical rigor (not just eyeballing numbers): **only one of six methods actually beat
random**, and it was the simplest one. The other five either showed no real difference
or made training measurably *worse*, with one method's harm rivaling a deliberately
sabotaged "pick the worst data on purpose" control.

---

## 1. Motivation and research question

**RQ2**: Do data-selection methods produce the same effectiveness ranking on a
generative task (a vocoder resynthesizing speech) and a discriminative task (an ASR
model transcribing speech), when both consume *identical* subsets of the same frozen
discrete speech tokens, at the same budget, under a locked protocol?

Why this matters: data selection ("pick the best K% of your training data") is a
common technique across ML, often justified by intuitive heuristics — keep the hardest
examples, keep what's most "informative," keep what matches your target distribution.
Those heuristics are usually validated on one task/model and then assumed to transfer.
This project tests that assumption directly, on discrete speech representations
specifically (an increasingly common input format for modern speech/audio models, via
neural audio codecs), where none of these selection heuristics have much track record.

## 2. Setup

- **Corpus**: LRS3-TED, 31,071 selectable utterances (30.18h) after filtering out
  utterances too short to produce valid training targets. Split into a 29,064-utterance
  train pool (28.16h) and a 2,007-utterance internal validation set (2.02h, video-disjoint,
  frozen). The official LRS3 test set (1,321 clips) is fully quarantined — read only once,
  by a final-evaluation module, after every run finishes.
- **Representation**: every utterance is pre-tokenized once by a frozen, pretrained
  SpeechTokenizer into 8 parallel discrete token streams (residual vector quantization,
  50 Hz). Track B (ASR) reads only stream 1 (the "semantic" layer); Track A (vocoder,
  not yet built) will read all 8 — same frozen tokens into two different consumers is
  the entire point of the comparison.
- **Track B model**: a small (~6.6M param) Conformer encoder + CTC head, trained from
  scratch, greedy decoding, no external language model. Deliberately small/simple so
  that "did the subset matter" stays a legible question, not one confounded by a huge
  model memorizing everything regardless of what it saw.
- **Operating point**: a single, well-powered budget (25% of the pool, 7,768 utterances)
  rather than a thin sweep across many budget sizes — chosen because 25% gives low-noise
  WER measurements (repeating random selection 5 times gives nearly identical results)
  while still being small enough that *which* utterances you keep can plausibly matter.
  A 5%/10% follow-up sweep is pre-registered, but only triggers for a method that
  actually shows a real effect at 25% (no point re-testing a null result at a noisier,
  smaller scale).

## 3. The six selection methods tested

| Method | What it does |
|---|---|
| **DNSMOS** | Keep the best-*sounding* audio (a neural network predicts what a human audio-quality rating would be, no model training involved) |
| **DSIR** | Reweight/resample the pool to better match the validation set's distribution (Xie et al. 2023) |
| **K-means** | Keep a diverse, cluster-balanced spread of token patterns rather than a skewed sample |
| **Loss-ranking** | Keep the *easiest*, cleanest examples (lowest loss under a small "proxy" model trained on a 10% subset) |
| **EL2N** | Keep the *hardest* examples by a gradient-norm-based difficulty score (Paul et al. 2021) |
| **LESS** | Keep examples whose training gradient best "points toward" reducing validation loss (Xia et al. 2024), estimated from the same proxy model |
| *Worst-on-purpose (control)* | *Deliberately keep the worst-by-proxy-loss data — a sanity check, not a real candidate* |

All six (plus 5 repeated random-selection runs as the baseline) were trained under
*identical* hyperparameters, epochs, and evaluation — the subset manifest is the only
thing that differs between runs.

## 4. Results (Track B, 25% budget)

| Condition | WER | vs. random |
|---|---|---|
| 100% of the data (ceiling) | 49.56%* | — |
| Random 25% (mean of 5 seeded runs) | 63.28% | baseline |
| **DNSMOS** | **62.44%** | ✅ **significantly better** |
| DSIR | 63.17% | ➖ ties (no real difference) |
| K-means | 64.33% | ❌ significantly worse |
| Loss-ranking | 65.02% | ❌ significantly worse |
| EL2N | 65.96% | ❌ significantly worse |
| LESS | 67.04% | ❌ significantly worse |
| *Worst-on-purpose (control)* | *67.21%* | *worse, as designed* |

\* Independently confirmed via an official post-hoc re-decode (not just the live
training-time number) to rule out a degenerate/collapsed model — matched to within
0.0001 across all 18 completed runs.

**This isn't just "DNSMOS looked best of six."** Every verdict above is from a proper
significance test against the random floor's measured natural variation, with
**Holm-Bonferroni correction applied across all 6 methods** — the standard fix for the
fact that testing 6 things at once gives chance more opportunities to produce a
misleadingly impressive result. DNSMOS's result survives that correction comfortably
(p ≈ 0.0004 against a corrected bar of 0.025); DSIR's near-miss (p = 0.457) is nowhere
close to significant, confirming it as a genuine tie, not a borderline case.

## 5. What this means (inferences and impact)

- **Simple, model-agnostic signals beat model-derived "cleverness" here.** DNSMOS
  doesn't know anything about the downstream task or model — it just filters by raw
  audio quality. Every method that tried to be "smarter" by reasoning about what the
  *model* finds hard, useful, or informative either did nothing or actively hurt. That's
  a concrete, actionable finding for anyone selecting training data for speech models
  built on discrete tokens: don't assume a sophisticated, model-derived heuristic beats
  a dumb quality filter — check first.
- **A method can be internally consistent and still be wrong.** LESS passed its own
  reproducibility check convincingly (0.998 rank correlation across random seeds) before
  training even started — yet it was one of the worst performers, nearly as harmful as
  the deliberately-sabotaged control. Best-supported explanation: LESS's gradient
  signal came from a smaller proxy model, not the model that actually consumes the
  selected data, and gradient-based influence estimates are known in the literature to
  transfer poorly across differently-scaled models. The lesson: a selector's internal
  consistency proves it's *measuring something repeatably* — it doesn't prove that thing
  is useful.
- **Borrowed methods carry hidden assumptions.** DSIR was originally designed to exploit
  a *mismatch* between a large, heterogeneous data pool and a narrow target domain
  (its original use case: picking web-scraped text that matches a specific downstream
  domain). Our corpus has no such mismatch — train and validation are drawn from the
  same homogeneous distribution. DSIR tying random isn't a bug; it's DSIR correctly
  finding nothing to exploit, because the condition it needs wasn't present. Applying a
  published method outside the regime it was validated in is a real, common pitfall.
- **A training-set statistic isn't the same as held-out validation.** An earlier
  diagnostic (DSIR's "weight spread") looked like a red flag when read as a raw
  training-set number, swinging 100x across a hyperparameter sweep. A proper held-out
  cross-validation check showed the real, generalizable signal was flat and weak at
  every setting — the dramatic swing was overfitting noise, not a sign of something
  broken. Standing lesson: judge a selector by held-out discriminative performance,
  never by a training-set statistic alone.
- **What's still open**: this is the discriminative (ASR) side only. The actual
  research question — does this exact ranking hold, invert, or diverge on the
  generative (vocoder) side — can't be answered until Track A exists. Until then, this
  is a solid, rigorously-checked result about *one* consumer task, not yet the
  cross-task comparison the project is ultimately built to make.

## 6. Engineering built to support this

A fair, reproducible comparison across 6+ selection methods and two consumer tasks
needed real infrastructure, not just one-off scripts: a shared selector interface
(`selection/*.py`, one `select()` function per method writing through a single
sanctioned manifest tool with built-in leakage/duplicate checks); a scoring pipeline per
method with live progress + error logging to W&B (so a multi-hour cluster job is
observable and resumable instead of an opaque black box); a protocol doc (`PROTOCOL.md`)
that locks every hyperparameter, naming convention, and methodological decision *before*
results exist, specifically to prevent post-hoc rationalization; and a statistical
framework (random-floor variance → per-selector z-test → Holm correction) built once and
reused for every comparison. None of this changes the science, but it's what makes the
results in §4 something you can actually trust rather than just report.

## 7. Current status

- All 6 selectors + random floor + ceiling: trained and verified at the 25% operating
  point (§4).
- Pending: an advisor-requested 50%/75% ceiling extension and a 5%/10% sweep for
  DNSMOS (the one selector that cleared the "beats random" bar) to confirm the effect
  holds across budgets, not just at 25%.
- Not yet started: Track A (the vocoder) — the piece that turns this from "one
  consumer task's result" into the actual cross-task RQ2 answer.

---

## Dataset provenance

- Corpus: **LRS3-TED** — Afouras, Chung, Senior, Vinyals, Ma, Zisserman,
  *Deep Audio-Visual Speech Recognition*, ICASSP 2018.
- Layout: CANONICAL STAGED TREE per `docs/layout.md` — `<data_root>/<video_id>/<stem>.{tokens.pt,txt,flac}`
  (trainval), `<data_root>/<stem>.{tokens.pt,flac}` bare (test); split membership is index metadata
  only. Raw audit-era SOURCE (laptop, historical): `datasets/LRS3/trainval/<video_id>/<stem>{.mp4,.txt,.tokens.pt}`
  (per-video folders; numeric stems restart per folder, hence `<video_id>/<stem>` utterance IDs) and
  `datasets/LRS3/test/<id>{.wav,.tokens.pt}` (flat).
- `.tokens.pt`: frozen SpeechTokenizer output, `[8, T]` int64 RVQ codes @50 Hz over
  [0,1024); codebook size 1024; stream index 0 = RVQ₁ (semantic). Verified in Phase 0 to
  be full-utterance streams: `T == ceil(n_samples/320)`.
- **Official-test transcripts** were not shipped with the flat test split; they belong to
  the official LRS3-TED test split and were recovered in Phase 0 from the
  [`mattymchen/lrs3-test`](https://huggingface.co/datasets/mattymchen/lrs3-test) HF mirror
  (1,321 rows ↔ 1:1 with local wavs) via `scripts/recover_test_transcripts.py`.
  Cite Afouras et al. for both the corpus and the test split.

## Setup

```bash
source ~/bin/pymax/bin/activate   # required venv for every command
pip install pytest num2words jiwer # already satisfied in pymax as of Phase 0
```

Commands (from repo root; see PROTOCOL.md §8):

```bash
python scripts/audit_data.py      # rebuild data_index.parquet + audit_report.json
python scripts/build_splits.py    # rebuild FROZEN internal splits (byte-stable)
python scripts/check_storage.py   # storage gate (--strict before cluster launches)
python scripts/spot_check.py      # decode 10 trainval utts for manual ear verification
pytest -q                         # unit tests (incl. crop alignment proof)
bash scripts/debug.sh             # single-GPU debug launcher
```

## Key locked facts (Phase 0 audit)

| Item | Value |
| --- | --- |
| trainval | 31,982 utts / 30.40 h / 4,004 videos |
| Selectable universe | **31,071 utts ≥ 50 frames** (911 short ones excluded everywhere) |
| Budget basis | BY UTTERANCE COUNT over the universe: 25 % ⇒ 7,768 utts |
| Internal val | video-disjoint, ≥2000 utts, seed 20260825 (`subsets/splits/`, sha256-frozen) |
| Official test | quarantined until the final-eval module |

## Ada cluster (compute target)

Formal runs run on Ada; this laptop is the authoring/pilot box. See `docs/ada_guide.md`
for a short operational guide (environment activation, scheduling string, QOS limits,
common gotchas); PROTOCOL.md §5 for the full, authoritative policy.

| Constraint | Value |
| --- | --- |
| GPU | **RTX 2080 Ti, permanent** — `hardware_guard.assert_gpu()` aborts any drifted card (a 3080 was observed in the pool, gnode077) |
| Host | CentOS 7 / GLIBC 2.17 → manylinux2014-compatible wheels only |
| Modules | `u22/python/3.12.4`, venv in `$HOME` |
| pip policy | only inside srun sessions carrying the FULL §5.8 constants (`-p u22 -A research --qos=medium --constraint=2080ti --exclude=gnode066`; CPU jobs add `--gres=gpu:0 --mem=16G`; **never `u22-cpu`**, devalab-restricted), and never on the login node; always `--no-cache-dir`. Env is BUILT AND PINNED (torch 2.6.0+cu124, GLIBC gate passed) — no second venv, ever. **`requirements.lock` IS the env description — a venv not described by it is BROKEN** (§5.5): manual installs must be lock-regenerated + committed immediately; missing deps belong in `scripts/setup_env.sbatch`, which reconciles the env exactly against the lock |
| partitions | before adopting any partition, check access: `scontrol show partition <name>` → `AllowAccounts` must include our account |
| Storage | dataset staged at `$HOME/spell/data` (FLAC+tokens+transcripts, sha256 manifest); runs bundle in `$HOME/spell/runs/<id>/` (≤~20 files); HOME gates warn 20 GB / abort 23 GB |
| Archive | compute nodes can't see /share1; `scripts/drain_runs.sh` copies bundles to the archive from a mounted node (relay copy kept by default — see PROTOCOL §5 item 2) |

**Phase 0b (blocking):** stage + verify the dataset on Ada before any Phase-1 training:

```bash
python scripts/stage_to_ada.py --dest ~/spell/data   # FLAC + tokens + transcripts + sha256 manifest
bash scripts/push_data_to_ada.sh                     # laptop -> Ada incremental rsync (resumable; -n = dry-run)
python scripts/stage_to_ada.py --verify              # on Ada; must be clean before Phase 1
```
