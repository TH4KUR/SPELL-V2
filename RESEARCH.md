# Project SPELL — Phase 1 Plan (RQ2 Only)

## 1. Scope Statement

**In scope:** One question — _is the ranking of data-selection algorithms stable when identical discrete-token subsets are consumed by a generative task (vocoder) versus a discriminative task (ASR)?_

**Out of scope (deferred, not abandoned):** RQ1 layer-view ablation, RQ3 full transfer matrix, budget sweep beyond the single operating point, mismatched corpus. Cheap insurance actions that keep these revivable are listed in §8.

**Honest expectation-setting:** RQ2 alone supports an Interspeech/ICASSP-scale paper (a controlled dual-task selection benchmark + rank-stability analysis). A\* claims need the deferred RQs — but every run below is designed so nothing is wasted if the project expands later.

---

## 2. Research Question & Hypotheses

> **RQ2:** Do data-selection methods produce the same effectiveness ranking on a vocoder and an ASR model trained on the same tokenized corpus, at the same subset fraction, under locked protocols?

- **H2a (task-universal):** Rankings correlate strongly (Kendall τ ≥ 0.6) — selection value is intrinsic to the data, not the consumer.
- **H2b (task-relative):** Rankings decorrelate or invert — e.g., prosody-rich subsets help STOI/PESQ but not WER.
- **H2c (null):** All selectors tie within noise of random — itself a publishable negative result about 25%-budget selection on homogeneous corpora.

All three outcomes are informative _because_ the experiment measures a rank relationship, not a single winner.

---

## 3. Core Design: Two Subset Classes

This is the structural decision everything else hangs on:

| Class                | Selectors                                        | Subsets generated                                                        | Consumed by     |
| -------------------- | ------------------------------------------------ | ------------------------------------------------------------------------ | --------------- |
| **Task-agnostic**    | Random, DNSMOS filter, Diversity heuristic, DSIR | **One subset each**, computed once                                       | **Both tracks** |
| **Task-conditioned** | Loss-ranking, LESS, (opt. Oracle-RHO)            | **One subset per track** (gradients/proxies from that track's objective) | Native track    |

Why this split matters: the task-agnostic class gives the _cleanest_ form of RQ2 — literally identical subsets, differing only in consumer. If even these ranks diverge across tracks, that's the strongest possible evidence of task-relativity. The conditioned class adds the practical question ("does the best method for TTS look different from the best for ASR?").

**Operating point:** 25% of LRS3 (~7.5h), pending the floor-run sanity gates in P1.

---

## 4. The Two Consumer Tracks

|          | **Track A — Generative**                                                      | **Track B — Discriminative**                                                                                                    |
| -------- | ----------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Model    | Vocos-style vocoder (ConvNeXt + iSTFT), HiFi-GAN discriminators, from scratch | **Token-input CTC**: embedding(RVQ₁ codes) → 4–6-layer Conformer/Transformer → linear → CTC, from scratch                       |
| Input    | Full 8-layer RVQ stack                                                        | RVQ₁ stream only (semantic layer) — keeps the "identical tokens/files" premise airtight; both read the same `.tokens.pt` shards |
| Metric   | STOI, PESQ (fixed test crops, generator EMA)                                  | WER via `jiwer`, greedy decoding, char vocab, fixed normalization                                                               |
| Cost/run | ~8–12 GPU·h @25%                                                              | ~1–2 GPU·h                                                                                                                      |

Optional appendix anchor: one wav2vec2+LoRA run to show conclusions aren't artifacts of the token-input choice. Not on the critical path.

---

## 5. Locked Protocol (written before any run in this phase)

1. Equal-epochs primary protocol (subset runs genuinely cheaper — that _is_ the efficiency claim); equal-updates control column for headline pairs.
2. Hyperparameters chosen once from 100% pilots, then frozen for every subset run — no per-subset tuning, ever.
3. Final-checkpoint evaluation for all runs (not best-val); internal val split for monitoring only; test set untouched until all runs finish.
4. Fixed SpecAugment config on Track B; fixed augmentation on Track A.
5. Per-sample loss logs every epoch; ∇ norms per batch; checkpoints every 5 epochs (feeds LESS now, Oracle-RHO/RQ1 later).
6. Seeds: 5 for random floors, 3 for all other conditions, matched across tracks.

---

## 6. Phased Plan with Gates

| Phase                          | Weeks | Work                                                                                                                                                                                                                        | Gate to proceed                                                                                                                                                                                                     |
| ------------------------------ | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **P0 — Foundations**           | 1–2   | Protocol doc committed to repo; transcript extraction + normalization; char vocab; internal val split; Track B learning pilot (100%, short run); Track A pipeline re-verified against new 50-token crops                    | Learning pilot reaches sane WER; both pipelines reproduce known behavior                                                                                                                                            |
| **P1 — Sensitivity + Anchors** | 2–4   | **Sensitivity pilot:** 25%-random vs 100%, 2 seeds, _both tracks_. Then full anchors: 100% ceilings, 5-seed random floors, anti-selection, DNSMOS top-25% — both tracks                                                     | **G1 (Track A):** floor ≥2σ below ceiling (headroom exists). **G2 (Track B):** ΔWER(floor→ceiling) > seed scatter, AND anti-selection hurts WER. Failure ⇒ adjust capacity/budget _now_, before any selector exists |
| **P2 — Selectors**             | 4–6   | Implement: DSIR (hashed n-grams over token streams), Diversity k-means heuristic, loss-ranking (small proxy models), LESS (Mel-L1 gradients for A; CTC gradients for B; 5-epoch trajectory snapshots). Generate all subsets | Each selector emits a subset + a characterization stat sheet (duration, speaker count, codebook usage entropy)                                                                                                      |
| **P3 — Main Grid**             | 6–9   | Train every subset × both tracks × 3 seeds under locked protocol (~36–42 runs; SLURM queue, mostly unattended)                                                                                                              | All runs converge; no protocol deviations logged                                                                                                                                                                    |
| **P4 — Analysis**              | 9–10  | Per-track league tables with bootstrap CIs; **primary result: Kendall τ over the four shared (task-agnostic) conditions**, secondary τ including conditioned selectors; paired seed tests; per-utterance paired bootstrap   | —                                                                                                                                                                                                                   |
| **P5 — Writing**               | 10–12 | Paper draft + toolkit release                                                                                                                                                                                               | —                                                                                                                                                                                                                   |

---

## 7. Run & Compute Budget

| Block                                                        | Runs    | Est. GPU·h     |
| ------------------------------------------------------------ | ------- | -------------- |
| Ceilings (A+B)                                               | 2       | ~45            |
| Floors: 5 seeds × 2 tracks                                   | 10      | ~55            |
| Anti-selection × 2 tracks                                    | 2       | ~22            |
| Shared selectors (DNSMOS, Div, DSIR): 3 × 3 seeds × 2 tracks | 18      | ~190           |
| Conditioned: LESS_A, LESS_B (3 seeds each, native track)     | 6       | ~35            |
| **Total**                                                    | **~38** | **~350 GPU·h** |

Comfortably within your Ada allocation history; blocks are sequential-gated so P3 spend only happens after G1/G2 pass.

---

## 8. Cheap Insurance for Deferred RQs (do these now, cost ≈ 0)

- Keep `.tokens.pt` organized by layer view → RQ1 revival needs no retokenization.
- Save per-sample losses + 5-epoch checkpoints for _every_ grid run → Oracle-RHO and RQ5-style validity checks become offline notebooks.
- Store every subset as an index file with its characterization stats → RQ3 transfer cells just point existing trainers at existing subsets.
- One paragraph in the protocol doc noting the deferred questions, so reviewers/co-authors see them as roadmap, not omission.

---

## 9. Analysis Pre-Registration (write this down before P3 finishes)

- Primary endpoint: τ (shared-class methods) between Track A and Track B rankings.
- With only 4 shared conditions, τ granularity is coarse — mitigate by (i) treating random's 5 seeds as one condition, (ii) reporting pairwise win-rate matrix with paired tests alongside τ, (iii) adding Oracle-RHO to the shared pool if implemented, widening to 5–6 points.
- Decision language fixed in advance: τ ≥ 0.6 → "stable"; τ ≤ 0.2 → "task-relative"; middle → inconclusive, report CIs honestly.
- H2c check: if every condition sits inside the random floor band on both tracks, stop early and write the negative-result/benchmark paper — do not chase more seeds hoping for separation.

---

## 10. Expected Outcomes → Papers

| Outcome | Finding                 | Paper                                                                                                                             |
| ------- | ----------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| H2b     | Rankings diverge/invert | _"Data Selection Is Task-Relative in Discrete Speech"_ — clean Interspeech/ICASSP story, expandable with RQ1/RQ3 later toward A\* |
| H2a     | Rankings stable         | First evidence of task-universal selection value on discrete speech — still novel, weaker headline                                |
| H2c     | Everything ties         | Rigorous negative result + multi-seed benchmark + released toolkit — honest workshop/Interspeech contribution                     |

---

## 11. Immediate Next Actions (this week)

1. Write and commit the **protocol doc** (§5) — before any training.
2. Build the **transcript/text pipeline** and internal val split (day 1–2).
3. Launch **Track B learning pilot** and re-verify **Track A** on the corrected crops (day 3–4).
4. Queue the **sensitivity pilot** (day 5+) — its result decides the operating point for everything that follows.

The single most important property of this plan: **no selector is built until both tracks have proven they can measure a difference at all.** Everything expensive comes after the gates.
