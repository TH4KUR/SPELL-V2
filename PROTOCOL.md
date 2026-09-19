# PROTOCOL.md — Locked rules for SPELL-RQ2

Written in Phase 0, **before any training run**. These rules are binding for
every run in every later phase. Any deviation must be logged as a protocol
deviation in the affected `run_manifest.json` and disclosed in the paper.

## 1. Verified data facts (Phase-0 audit)

- `trainval/`: 4004 video folders; utterance triplets `<stem>.mp4` (AAC 16 kHz mono),
  `<stem>.txt` (`Text: ...` / `Conf: 1–6`), `<stem>.tokens.pt`. ~31.98k utterances.
- Utterance IDs are `<video_id>/<stem>` — stems are numeric and RESTART per folder.
- Tokens are **full-utterance streams**: `[8, T]` int64, codes in `[0, 1024)`,
  frame rate 50 Hz, and `T == ceil(n_samples / 320)` exactly (verified on both splits).
  → No re-extraction was needed; the corrected crop sampler is the alignment path.
- Official `test/` is flat `.wav` + `.tokens.pt` pairs (1321 clips), transcripts
  recovered separately by `scripts/recover_test_transcripts.py`.
- **Selectable universe** (locked): trainval utterances with `n_tokens >= 50`
  = **31,071 utts (30.18 h)**. The 911 shorter trainval utterances are excluded from
  ALL manifests, selections, and budget computations.

## 2. Frozen conventions (never change after Phase 0)

1. **Text normalization**: `text_norm.normalize_text()` is THE canonical function,
   frozen forever. Char vocab: `<pad>=0, <blank>=1`, then a-z, space, apostrophe (frozen IDs).
2. **Crop law**: frame *f* ↔ samples `[320·f, 320·(f+1))`; canonical crop = 50 frames ↔
   exactly 16 000 samples. Proven by unit test (`tests/test_crops.py`).
3. **RVQ stream indexing**: tensor dim-0 index 0 = RVQ₁ ("codebook 1", semantic);
   indices 1–7 = acoustics. Track B reads index 0 only; Track A reads all 8.
4. **Padding sentinel**: token padding uses `-1` (real codes span `[0, 1023)`).
   Every consumer must mask with returned lengths/masks; CTC never sees pad as a class.
5. **Universe & budget basis**: subset manifests may draw ONLY from the selectable
   universe (`selectable == True` in `data_index.parquet`). Budget percentages are
   defined BY UTTERANCE COUNT over this universe — 25 % ⇒
   `universe_budget(31071, 0.25) = 7768 utts` (round-half-up, see `config.universe_budget`);
   realized HOURS are reported per subset afterwards.
6. **Embedding safety**: the pad sentinel must NEVER reach an embedding lookup —
   PyTorch wraps negative indices to the LAST vocabulary row, silently corrupting
   batches. Consumers mask/slice before `nn.Embedding`. Phase 1 MUST ship a unit
   test proving pad positions never reach the lookup.

## 3. Training protocol (locked before any subset run)

1. **Equal epochs**: every run trains the same number of epochs regardless of subset size.
2. **Hyperparameters** come from the 100% pilots once, then are FROZEN for every subset
   run — no per-subset tuning, ever.
3. **Final-checkpoint evaluation only** (never best-val), uniformly across all runs.
4. Internal val split (`subsets/splits/val_ids.txt`, video-disjoint, ≥2000 utts, seed
   `20260825`) is for monitoring/model selection hygiene only; it is NOT a test set.
5. **Official test quarantine**: `datasets/LRS3/test/` is read ONLY by the final-eval
   module, after all runs finish, on identical fixed crops/wavs cached once.
6. Fixed augmentation configs for all runs of a track (see track configs).
7. Per-sample losses logged every epoch; batch-level grad L2 norms logged (∇G and ∇D for
   Track A); checkpoints every 5 epochs (feeds LESS trajectory analysis).
8. Seeds: random floors ×5 seeds; every other condition ×3 seeds; matched across tracks.
   The formal x5 random-floor cell = pilot floors (101, 102) + seeds 103–105
   (run-plan tasks 1–5, drawn 2026-08-28 via `scripts/make_random_subset.py`). The
   two-seed spread measured on 101/102 (0.0054 WER) is a PROVISIONAL dispersion
   estimate only; the final σ for the random-floor condition is computed on n=5.
9. Selection NEVER happens inside training code: trainers consume a manifest path from
   `subsets/`.
10. Deterministic data ordering per seed; identical eval batches/crops across all runs
    of a track.
11. **Seeds are two distinct things**: the `subset_seed` (e.g. 101–105) draws a random
    manifest and is part of that subset's IDENTITY (`subsets/random_25pct_seed{101..105}.txt`);
    the `train_seed` is a FIXED constant shared by every run. Track A anchors consume
    `subsets/random_25pct_seed{101..105}.txt` unchanged. AMORTIZATION LAW: the x5
    expansion (2026-08-28) drew ALL five floor manifests up front precisely so both
    tracks share the same identities — no re-draws, ever; a re-drawn manifest would
    silently change the condition under test.
12. **CTC input-length rule**: utterances whose token length is shorter than their
    normalized-text id length cannot produce valid CTC targets — they are dropped from
    training/validation batches and the drop count is logged loudly in metrics.
13. **Pad-containment mechanism (Track B)**: pad sentinel frames are re-zeroed after
    EVERY sublayer inside a Conformer block. This bounds cross-frame contamination to
    a halo of ≤ `conv_kernel // 2` frames around each true boundary per block (the
    depthwise conv is the only sublayer that mixes neighbouring positions; attention
    keys are column-masked and position-wise FFNs cannot spread values). Real frames
    within that cumulative halo legitimately see zeros at their edge — identical
    behaviour across all runs, hence ranking-valid. The no-leakage unit test asserts
    nothing beyond this bound.
14. **Normalizer choice (Track B)**: the conv module uses `GroupNorm(1, C)`, NOT
    BatchNorm1d. BN running stats depend on the batch length distribution, which
    differs structurally between 100% and 25%-budget cells and would confound the
    ranking comparison; GroupNorm has no running state, so eval-time statistics are
    identical across cells.
15. **Hyperparameters are ABSOLUTE**: every optimizer/schedule constant (lr peak,
    warmup steps, grad clip, epochs) is a fixed number applied identically to every
    run — NEVER scaled to subset size. Equal-epochs is a special case of this rule;
    if the 100% pilot's val WER is still falling steeply at epoch 20, epochs are
    EXTENDED before freezing — not compensated elsewhere.
16. **Dev-GPU bypass**: laptop bring-up (`pytest`, `overfit_one_batch`, small smoke
    runs on the RTX 4060) is permitted via env `SPELL_DEV_GPU=1`, which replaces the
    drift-guard abort with a loud banner. Formal runs must never set it; formal
    verification = one-time CPU pytest under the pinned Ada env (full §5.8
    constants incl. constraint/exclude, plus --gres=gpu:0) before pilot
    submission, then both pilots on 2080 Ti nodes only.
17. **Logging contract (W&B surface)**: every formal run streams a MINIMAL COMPLETE
    live-series set to its logger and NOTHING ELSE:
      * `train/loss_step` — step level;
      * `train/grad_l2` — every `logging.grad_norm_log_every_batches` steps;
      * an `lr*` series via `LearningRateMonitor(logging_interval="step")` in
        `scripts/train_track_b.py` — the realized warmup+decay schedule (§3.15);
      * `val/loss`, `val/wer`, `val/cer` — exactly once per validation epoch,
        emitted ONLY by the module's own epoch-end hook from module-owned
        aggregates (ownership split, §10 item 8);
      * hyperparameters — automatic at fit start via `save_hyperparameters(cfg…)`.
    Per-utterance data NEVER streams to W&B; it lives only in `metrics.parquet`
    (the bundle feed). Series NAMES are FROZEN: gate logic reads exactly these
    names (`val/wer` is THE §7 gate signal). Rules going forward:
      (a) any NEW logged series must be pinned by a logger-boundary assertion in
      `tests/test_val_wandb_series.py` BEFORE a run may depend on it;
      (b) renaming or deleting one requires a PROTOCOL revision note here;
      (c) any change touching trainer/callback/logger assembly triggers a
      FIRST-EPOCH check that each expected series exists on W&B before the rest
      of a long run is trusted — a silent miss voids gates discovered weeks late
      (2026-08-27 incident);
      (d) the run-id namespace is FROZEN (2026-09-08): every formal run's id is
      exactly `<track>/<manifest-stem>_seed<seed>_job<SLURM_JOB_ID>_t<task>` as
      derived at `slurm/template.sbatch` (mechanically pinned by
      `tests/test_wandb_naming_law.py`); backfilled historical runs adopt the
      same pattern with their original job ids. Renaming requires a protocol
      revision note first.
18. **Significance criterion v2 (user-approved amendment A, 2026-09-08)**: a
    selector BEATS random on a track iff ALL three hold:
      (a) mean ΔWER vs the n=5 random-floor mean exceeds `2·σ_diff`, where
          `σ_diff = σ·√(1/n_sel + 1/5)` with σ = the formal n=5 random-floor σ
          (§3.8) and n_sel = that selector cell's seed count — a mean-difference
          test replacing the former single-number `ΔWER > 0.005` rule, which
          conflated raw seed spread with mean-difference noise;
      (b) the result is Holm-corrected across ALL selectors in the roster
          (family = every selector compared against random, per track);
      (c) the advantage DIRECTION (better/worse than random) is consistent
          across the budget sweep {5, 10, 25}% — a sign flip at any swept
          budget voids the beats-random claim.
    PARITY claims ("selector ties random") are made via TOST (two one-sided
    equivalence tests) against the same σ_diff — never via absence of (a).
19. **Budget-sweep scheduling (user amendment, 2026-09-08 — completes §3.18(c))**:
      (a) P2 (selection) generates manifests for budgets {5, 10, 25}% × ALL
          selectors — selection is front-loaded, training is not;
      (b) training order: ALL 25% cells first;
      (c) a WRITTEN post-25% gate (§7) decides which selectors advance to
          5%/10% training cells BEFORE any such run is submitted;
      (d) random itself is part of every swept budget (×3 subset seeds per
          swept budget: 5% = seeds 111–113, 10% = seeds 121–123), and
          anti-selection (worst-by-proxy-loss) is the MANDATORY sanity column
          in every cell — its harm must GROW as the budget shrinks, or the
          whole apparatus is suspect;
      (e) a selector evaluated at a single budget is reported WITHOUT
          significance language in any paper (criterion (c) presupposes the
          sweep).
20. **Seed-law interpretation (2026-09-08, amending §3.8)**: the "every other
    condition ×3 seeds" law applies to genuinely STOCHASTIC selectors only —
    there the subset seed is the selector's internal-RNG identity (k-means
    init 201–203, DSIR resample 201–203, LESS projection 201–203).
    Deterministic selectors (DNSMOS top-k, proxy loss-ranking, EL2N,
    anti-selection) are ×1: one frozen manifest, one training run per track —
    a repeated identical subset would only re-run the same condition. Random
    floors remain ×5 at 25% and ×3 per swept budget (§3.19(d)).
21. **Val-as-target disclosure (2026-09-08)**: the target-conditioned selectors
    (DSIR, LESS) use the frozen val split (`subsets/splits/val_ids.txt`,
    2,007 utts, video-disjoint) as selection-TARGET METADATA ONLY. Val
    utterances NEVER enter any subset manifest, remain excluded from model
    selection, and the official test set stays quarantined (§3.5). Accepted
    consequence, disclosed in the paper: val-WER gate readings for
    val-targeted selector cells are optimistic-confounded; final-eval claims
    are unaffected.
22. **DSIR 72h go/no-go — RESOLVED GO, weak effect (2026-09-08 to 2026-09-17)**:
    the risk-ordered roster build (§7) carried a 72-hour go/no-go on DSIR —
    if it were not "demonstrably working" within 72h of commit b47b334, it
    would be dropped rather than debugged indefinitely. Timeline and finding:
    (a) the formal run (`l2=1e-4`) printed `weight_spread` (CV of
    `exp(logit)` over the fitted POOL rows) = 0.0295 — read at face value
    against the informal "near-uniform ⇒ drop" framing, this looked like a
    NO-GO; (b) a single-point ad-hoc check (`l2=1e-6`) raised it to 0.4571,
    proving the regularization dial mattered, but `weight_spread` is a
    TRAINING-set statistic that can rise from EITHER real signal being
    unlocked OR the fit starting to memorize idiosyncrasies of the ~31k
    fitting rows in this 66,560-dim hashed feature space — indistinguishable
    from that number alone; (c) `scripts/sweep_dsir_l2.py` (a held-out
    cross-validation sweep, NOT part of the roster pipeline — see its module
    docstring) settled it: splitting pool+val with a fixed deterministic
    80/20 rule (§3.20-style, no RNG), fitting on the 80%, and scoring the
    held-out 20% across `l2 ∈ {1e-6 … 1e-3}` gave `held_out_auc` in a narrow,
    FLAT band of **0.565–0.570 across the entire three-decade sweep** — while
    `weight_spread` swung two orders of magnitude (0.0036 to 0.4571) over the
    same range. **Conclusion: `weight_spread`'s wide swing was overfitting
    noise, not signal being unlocked — the genuine, generalizable signal
    (AUC≈0.57) was present at EVERY tested `l2`, including the original
    `l2=1e-4`.** At n_check_pool=5,813 / n_check_val=402, chance-level AUC's
    standard error is ≈0.015 (Wilcoxon/Mann-Whitney variance formula
    `(n1+n2+1)/(12·n1·n2)`), so 0.565–0.570 sits ~4.4–4.7 SE above 0.5 —
    real, reproducible, but WEAK (0.57 is far from the 0.9+ that would
    indicate a strongly distinct target distribution). **Verdict: GO, with a
    standing weak-effect disclosure** — DSIR is retained in the roster at
    its already-shipped `l2=1e-4` (`scores/dsir_weights.parquet`, unchanged,
    no re-run needed), but any paper claim must state the effect size
    plainly rather than imply strong separation; a null/near-null downstream
    Track A/B result for DSIR would be UNSURPRISING given this AUC and
    remains within the pre-registered H2c ("everything ties") outcome
    (RESEARCH.md §9), not evidence of a pipeline bug. **Standing lesson for
    any future selector go/no-go**: judge by HELD-OUT discriminative
    performance (`sweep_dsir_l2.py`'s method generalizes to any importance-
    weighting selector), never by a raw training-set statistic like
    `weight_spread` alone — the latter conflates coefficient SCALE (which
    regularization freely trades off) with ranking QUALITY (which is what
    subset selection actually consumes).

23. **W&B logging for research-crucial non-training runs (user policy,
    2026-09-18)**: §3.17's live-series contract governs FORMAL Track A/B
    TRAINING runs only — it does not exempt other runs from W&B entirely.
    Any script whose output feeds a real research decision (a selector's
    score table, a hyperparameter choice, a ranking used downstream) MUST
    log to W&B by default, not only when separately asked. This is
    STRUCTURALLY SEPARATE from §3.17: a distinct `job_type` and
    `<tool>/`-prefixed series names (never `train/*` or `val/*`), so the two
    surfaces can never collide or be mistaken for each other in the W&B UI,
    and §3.17(a)'s "pin before a run may depend on it" gate does not apply
    (nothing in the §7 phase-gate logic reads these series).
24. **Logging must be LIVE, with per-step timing, not a batch upload at the
    end (amends §3.23, 2026-09-18)**: "log to W&B" is not satisfied by
    opening the run and dumping everything just before it exits — that
    gives zero visibility for the run's entire duration and NOTHING at all
    if it crashes partway through, defeating the actual purpose. Every
    §3.23 run MUST: (a) call `wandb.init` BEFORE the expensive work starts,
    not after; (b) log progress INCREMENTALLY as each unit of work
    (checkpoint / epoch / sweep point / seed) completes, carrying at
    minimum a progress fraction (i/N) and that unit's OWN duration — not
    only cumulative elapsed time, since an average hides a single slow or
    stuck unit; (c) wrap the run body so `run.finish()` fires even on
    failure (a crashed run stays visible, not silently absent). Caught live
    (2026-09-18): the user watched `score_less.py`'s job sit on "checkpoint
    2/4" with nothing in W&B — the run had not even opened yet, because the
    original `log_to_wandb()` call sat at the very end of `main()`.
    Re-checking the ONE other §3.23-compliant script at the time
    (`scripts/sweep_dsir_l2.py`) found the IDENTICAL gap — `wandb.init` was
    also called only after its full sweep finished, despite already having
    a per-row logging loop. Both fixed the same day (`score_less.py`:
    `_wandb_init` opens early, `_log_ckpt_progress` fires per checkpoint
    with that checkpoint's own duration + cumulative elapsed;
    `sweep_dsir_l2.py`: `wandb.init` moved to the top of `main()`, each l2
    point logged as it's computed). **Compliant**: `scripts/sweep_dsir_l2.py`,
    `scripts/score_less.py`, `scripts/score_dnsmos.py`, `scripts/score_proxy.py`
    (the latter two retrofitted 2026-09-18, with explicit user sign-off,
    after job 2700964 sat at 2:40:30 of a 3:00:00 SLURM wall with zero W&B
    trace of progress — `_wandb_init` opens early; both report on a
    WALL-CLOCK cadence (`--report-every-s`, default 30s), not a row count —
    see §3.25, which amends this same day after a count-based first attempt
    turned out to be its own bug. `score_dnsmos.py`'s ProcessPoolExecutor
    pool is consumed via `as_completed()` over small submitted chunks, not
    `.map()`; `score_proxy.py` reports at BOTH the per-checkpoint level and
    the within-checkpoint utterance level, since with only 2 late
    checkpoints by default a per-checkpoint-only granularity would give
    just 2 data points for the whole job). **All six roster scorers are now
    compliant** — `scripts/score_kmeans.py` and `scripts/score_dsir.py` were
    retrofitted 2026-09-18, with explicit user sign-off, using the
    wall-clock-cadence + ETA + exit_code-aware `run.finish()` design from
    day one (§3.25's lessons, not the count-based first draft this section
    originally shipped with). `score_kmeans.py` reports during
    feature-building (the dominant per-utterance cost, before clustering);
    `score_dsir.py` reports during pool featurization (val is much
    smaller, not separately instrumented). Any NEW scorer/sweep/analysis
    script from this point on ships with §3.23, this item, AND §3.25's
    fixes from day one, not as an afterthought added after the fact.
25. **Two more W&B/scorer bugs, both caught live on the score_proxy.py
    retrofit's first real Ada run (job 2701229, 2026-09-18) — amends §3.24
    again the same day**:
    (a) *Reporting cadence must be TIME-based, not row-count-based.* The
    first cut of §3.24's retrofit used a fixed row-count threshold
    (`--report-every 2000`). That silently assumes a throughput rate: at
    `score_dnsmos.py`'s real ~3h/29k-utterance rate the FIRST log line
    didn't appear for 10+ minutes, which the user correctly called
    useless. Worse, `score_dnsmos.py` consumed its `ProcessPoolExecutor`
    via `.map()`, whose iterator yields results strictly in INPUT order —
    a single slow chunk at the front of the task list could block every
    later, already-finished chunk from being reported at all. Fixed by
    reporting on a wall-clock interval (`--report-every-s`, default 30s,
    plus a forced final report) and, for `score_dnsmos.py`, switching to
    `as_completed()` over submitted chunks so a fast chunk is visible the
    moment it finishes regardless of position in the task list. Every
    progress line also now carries an ETA (`utts_per_s`, `eta_s`), since
    "how much time will this take" was the actual question, not just a
    raw counter.
    (b) *`run.finish()` must reflect the real outcome.* Job 2701229 crashed
    (see below) but the W&B UI still showed the run as a completed
    "Finished" run — `run.finish()` called with no arguments always marks
    the run successful, even while an exception is actively propagating
    through the same `finally:` block that calls it. Every §3.23 script's
    `finally:` now calls `run.finish(exit_code=1 if sys.exc_info()[0] is
    not None else 0)` instead of a bare `run.finish()` — fixed in all four
    at once (`score_less.py`, `sweep_dsir_l2.py`, `score_dnsmos.py`,
    `score_proxy.py`) since all four shared the identical pattern, not just
    the one that happened to crash first.
    Separately (not a §3.23/§3.24 logging bug, but what job 2701229 actually
    crashed on): `score_proxy.py`'s `_score_one_lit` called
    `filter_records(recs, ids, strict=False)` with `ids` as a plain
    `list[str]`, but `dataset.filter_records`'s type contract is
    `ids: set[str]` (its body computes `ids - known`) — every OTHER caller
    in the codebase (`train_track_b.py`, `evaluate_track_b.py`) passes a
    set built from `load_id_list`, so this mismatch had never been
    exercised: `score_proxy.py` had never actually completed a real Ada run
    before this attempt. Fixed by passing `set(ids)`. The test suite missed
    this because the one existing progress test for `_score_one_lit`
    monkeypatched `filter_records` itself, bypassing the real function
    entirely — `tests/test_score_proxy.py` now also has a test that
    deliberately does NOT mock `filter_records`, exercising dataset.py's
    real implementation against a list-shaped `ids` (regression coverage
    for this exact class of bug, not just the count-vs-time behavior).
    **The shared proxy model's bundle** (§3.19's P2 proxy run, task 0 of
    `slurm/run_plan_p2_proxy.tsv`, job 2691844), which `score_proxy.py` and
    `score_less.py` both read via `--bundle`/`SPELL_PROXY_BUNDLE`/
    `SPELL_LESS_BUNDLE`, lives at:
    `runs/track_b/random_10pct_seed301_seed301_job2691844_t0`
    (per the §3.17(d) run-id namespace: `<track>/<manifest-stem>_seed<seed>
    _job<job_id>_t<task>`) — recorded here because it is not derivable from
    anything committed to this repo (`runs/` is gitignored) and had to be
    looked up by hand on Ada after this same incident.
26. **P3 main grid, first wave queued (2026-09-18)**: `scripts/proxy_scores.parquet`
    landed (job on Ada), unblocking `el2n`/`lossrank`/`anti` manifests —
    12/13 roster selector cells now exist (`dnsmos`'s is still blocked on
    its own still-running scoring job). `slurm/run_plan_p3_selectors.tsv`
    (new `gen_run_plan.py` PLANS entry) queues all 12 as one array: same
    frozen `configs/pilot_25pct_random_x2seeds.yaml` as the random floors
    (no per-subset tuning, rule 2); stochastic selectors (dsir/kmeans/
    less_ctc) use their manifest's own §3.20 seed, deterministic ones
    (el2n/lossrank/anti, whose characterization JSON carries
    `subset_seed: null`) use `0`, matching task 0's (100% ceiling)
    existing precedent for "no real subset-seed identity" in this schema.
    `keep_local_traj=0` for all 12 — unlike the P2 proxy run, nothing
    downstream consumes these runs' own trajectories (LESS already ran off
    the proxy run's checkpoints), so there is no reason to spend the run
    archive's tight inode budget (§5.0) keeping them. This wave is
    independent of the DNSMOS/proxy scoring jobs and needed no additional
    phase-gate clearance beyond P1's already-passed G1/G2 anchor cells
    (§7's "advancing-selector" recording requirement applies to the
    LATER 5%/10% promotion gate, not this initial 25%-cell submission).
    **Update (2026-09-19)**: `dnsmos`'s scoring job finished (checkpoint/
    resume from §3.30/§3.31 carried it through the `DefaultTime` timeout
    and a subsequent resubmit; 29,064 rows, `ovr_mos` range 1.04–3.61,
    sane) and its manifest (`subsets/dnsmos_25pct.txt`) is generated —
    **all 13/13 roster manifests now exist.** `dnsmos`'s cell is appended
    as task 12 of `slurm/run_plan_p3_selectors.tsv`, same conventions
    (deterministic selector → seed 0, `keep_local_traj=0`). Once this
    13th cell's training finishes, the §3.29 significance table's caveat
    (a) (Holm correction needs the complete roster family) is resolved.
27. **`score_dnsmos.py`'s one-hop-at-a-time structure is NOT naive (checked,
    2026-09-18)**: the user was skeptical that scoring ~3h/29k-utterance was
    naive/suboptimal code. Checked against the actual DNS-Challenge
    reference (`dnsmos_local.py`): identical per-hop, per-file forward-call
    structure — batching was explicitly requested upstream (GitHub issue)
    and rejected `wontfix` by the maintainers, so this script is a faithful
    port, not an oversight. (Correction to an earlier assumption made THIS
    same session: most clips do NOT self-tile down to 1 hop — the tiling
    always lands in [9.01s, 18.02s), giving 1–9 hops, mean ≈5 — so the hop
    loop is a real per-utterance multiplier, not negligible for this
    dataset.) Real profiling on the vendored model DID find one genuine,
    cheap fix, applied immediately: each of the 8 `ProcessPoolExecutor`
    workers previously defaulted its own full-core intra-op thread pool, so
    8 processes fought over the same 8 allocated CPUs — measured to cost
    MORE than hop-batching would gain. `_score_all`'s worker `init()` now
    calls `torch.set_num_threads(1)` (parallelism already comes from the 8
    processes, zero numeric risk, zero behavior change). A larger,
    mechanically-safe-but-bookkeeping-heavier optimization (batching hops
    from multiple utterances into one forward call, ~1.3× additional
    speedup measured) was investigated but NOT implemented — smaller payoff
    for more review risk; revisit only if a future rerun's wall-clock still
    matters after the thread-pinning fix.
28. **P1 anchor-cell numbers — authoritative record (this section is the
    single source of truth; supersedes any figure in HANDOFF.md or other
    untracked notes)**. Track B, `configs/pilot_100pct.yaml` /
    `configs/pilot_25pct_random_x2seeds.yaml`, both equal-epochs (20),
    final-checkpoint WER:

    | condition | seed(s) | final val WER |
    |---|---|---|
    | ceiling (100%, ~29h) | train_seed only | 0.4975 |
    | random floor (25%) seed101 | 101 | 0.6316 |
    | random floor (25%) seed102 | 102 | 0.6341 |
    | random floor (25%) seed103 | 103 | 0.6303 |
    | random floor (25%) seed104 | 104 | 0.6318 |
    | random floor (25%) seed105 | 105 | 0.6364 |
    | **random floor (25%), n=5** | 101–105 | **mean 0.6328, σ (pop.) 0.0022** |

    This σ is THE formal n=5 random-floor σ referenced by §3.18's
    significance criterion (`σ_diff = σ·√(1/n_sel + 1/5)`).
29. **P3 first-wave selector-cell results vs. the §3.18 significance
    criterion (2026-09-19, PRELIMINARY)**. All 12/13 roster manifests'
    25% Track B cells (same config as item 28's floor), final val WER,
    Δ = mean WER − 0.6328 (random floor mean), tested against
    `2·σ_diff` (σ=0.0022 from item 28):

    | selector | seeds | mean WER | Δ vs random | 2·σ_diff | verdict |
    |---|---|---|---|---|---|
    | dsir | 201,202,203 (0.6343, 0.6300, 0.6307) | 0.6317 | −0.0012 | 0.0032 | ties random |
    | kmeans | 201,202,203 (0.6471, 0.6401, 0.6426) | 0.6433 | +0.0104 | 0.0032 | significantly WORSE |
    | lossrank | — (0.6502) | 0.6502 | +0.0174 | 0.0047 | significantly WORSE |
    | el2n | — (0.6596) | 0.6596 | +0.0268 | 0.0047 | significantly WORSE |
    | less_ctc | 201,202,203 (0.6672, 0.6729, 0.6710) | 0.6704 | +0.0375 | 0.0032 | significantly WORSE |
    | anti (intentional worst) | — (0.6721) | 0.6721 | +0.0393 | 0.0047 | significantly WORSE (as designed) |

    **Headline**: no selector beats random at 25% on Track B; `dsir` ties,
    every other selector is significantly WORSE, and `less_ctc`'s harm
    (+0.0375) rivals the deliberately-adversarial `anti` baseline
    (+0.0393) despite `less_ctc` passing its own internal consistency
    check (0.998 cross-seed Spearman, §3.20-era result) — a selector can
    be self-consistent in WHAT it ranks and still hurt training when
    acted on. Per §7's gate, "beats random" is the promotion bar for the
    5%/10% sweep — nothing currently clears it, so there is no candidate
    to advance yet.
    **PRELIMINARY — three things still needed before this is final**:
    (a) `dnsmos`'s cell is still blocked on its own scoring job — the
    Holm correction (§3.18(b)) needs the COMPLETE roster family, not this
    partial one, though margins here are large enough that the verdicts
    above are unlikely to flip; (b) `scripts/evaluate_track_b.py`'s
    official post-hoc cross-check has not been run on ANY of these bundles
    (including the item-28 floor/ceiling) — `verified` is `NO` across the
    board; the numbers above are the live/primary signal per the §7
    gate-signal policy, not yet the official one; (c) this is Track B
    ONLY — RQ2's actual cross-track question (does this ranking pattern
    hold on Track A) cannot be answered until Track A exists.
30. **Checkpoint/resume for score_dnsmos.py + error logging to W&B for all
    six scorers (2026-09-19)**. Job 2701385 hit `slurm/score_dnsmos.sbatch`'s
    `--time=06:00:00` wall at 82.1% done (23,856/29,064 utterances,
    `sacct`-confirmed `TIMEOUT`) with NOTHING written to disk — the entire
    job wrote `scores/dnsmos_scores.parquet` only once, at the very end, so
    the kill discarded ~5 hours of already-completed work outright. Fixed:
    (a) `_score_all` now writes a checkpoint (`<out>.partial.parquet`, atomic
    write-then-`os.replace`) on the SAME live cadence as its W&B progress
    reports; `main()` loads it on startup (unless `--fresh`), skips
    already-scored utterance_ids, and merges checkpoint + freshly-scored
    rows into the final `--out` before deleting the checkpoint — a
    resubmission of the identical command now resumes instead of
    restarting at utterance 0. Progress-callback rate/ETA math stays
    SESSION-LOCAL (not prefix-inclusive) deliberately — mixing a resumed
    prefix count into `done/elapsed_s` would make both wildly wrong (e.g.
    dividing 23,856 resumed + 50 new rows by 5 session-elapsed seconds).
    (b) **SUPERSEDED same day, see item 31** — `--time` was first OMITTED
    entirely on the (wrong) assumption that this falls back to the
    medium QOS's generous `MaxWall` (4 days, §5 item 8a); it does not. The
    checkpoint above is still what actually fixes the LOSS-of-work
    problem, not the wall itself — a node failure, preemption, or
    `scancel` would bypass any `--time` value entirely and still need the
    checkpoint to avoid losing completed work.
    (c) all six W&B-logging scripts' `finally:` blocks (`score_dnsmos.py`,
    `score_proxy.py`, `score_kmeans.py`, `score_dsir.py`, `score_less.py`,
    `sweep_dsir_l2.py`) now log the actual exception onto `run.summary`
    (`error`, `traceback`) before `run.finish(exit_code=1)` — previously
    exit_code=1 told you a run failed but not WHAT failed; that used to
    require grepping the SLURM `.err` log. Uses the LOCAL `run` object
    returned by `_wandb_init`, not the global `wandb.run` — the latter
    would be `None` in any test that mocks `_wandb_init` with a fake run
    object rather than calling real `wandb.init()`.
31. **Omitting `--time` is WRONG — it does NOT fall back to the QOS's
    `MaxWall` (corrects item 30(b), same day, 2026-09-19)**: job 2704082
    was submitted with no `--time` and was killed after only ~59 minutes
    at 24.6% done — `scontrol show partition u22 | grep DefaultTime` shows
    `DefaultTime=01:00:00`. **The mechanism**: when `--time` is unset,
    SLURM applies the PARTITION's `DefaultTime`, not the QOS's `MaxWall` —
    these are two independent settings; `MaxWall` only bounds what you are
    ALLOWED to request via `--time`, it is not what you get by leaving
    `--time` unset. `slurm/score_dnsmos.sbatch` now sets an EXPLICIT
    `--time=08:00:00`, sized off real observed throughput WITH the
    thread-pinning fix (§3.27) active: ~2.02 utt/s → ~4h for the full
    29,064-utt pool, so 8h gives 2x margin. **Standing rule for every
    sbatch file in this project: always pass an explicit `--time` sized
    from real observed throughput (with margin) — never omit it and never
    assume a QOS-level setting fills the gap.** Job 2704082's ~59 minutes
    of progress was NOT lost this time — the checkpoint/resume feature
    (item 30(a)) already existed when this job ran, so a resubmission of
    the identical command resumed from `scores/dnsmos_scores.partial.parquet`
    instead of restarting at utterance 0.

## 4. Known caveats (accepted, uniform ⇒ ranking-valid)

- Trainval audio decodes from AAC mp4; official test wavs are PCM. All subsets share the
  same decode path, so selection rankings are unaffected; noted for absolute-metric claims.
- Transcript confidence (`Conf: 1–6`) is stored in `data_index.parquet` as metadata;
  it is not used by any Phase-0 decision.

## 5. Storage, names & environment policy (Ada revision — supersedes NAS write-through)

**Quotas:** `/home` = 30 GB + 300k inodes; `/share1` = 100 GB but ~3200-inode cap.
Host is CentOS 7 / GLIBC 2.17 despite u22 module names — binary deps must be
manylinux2014-compatible or cluster-module-provided.

### 5.0 Canonical names — FROZEN project constants

These names/paths are FROZEN: do not rename, do not accept variants in scripts,
configs, docs or handoffs. Laptop dir name `SPELL-V2` NEVER appears outside the
laptop filesystem.

| Role | Frozen constant |
|---|---|
| Ada working repo | `~/spell/repo` |
| Ada bare git remote | `~/spell/repo.git` (laptop reaches it via its `ada` remote — see §5.7) |
| Ada python env | `~/envs/spell` (torch 2.6.0+cu124) |
| Data root on Ada | `$HOME/spell/data` (staged Phase-0b tree; verified 99,909 files). Runtime selection via `SPELL_DATA_ROOT` — template self-defaults it (§10 item 7) |
| Canonical archive | `/share1/$USER/spell/runs/` |
| Relay directory | `$HOME/spell/runs/` (relay `runs_dir`; `drain_runs.sh` copies relay → archive by default, §5 item 2 amendment — `--prune-relay` restores the old delete-after-verify behavior) |

`~/pymax` is a LAPTOP-ONLY fossil — its appearance anywhere Ada-facing is a bug.

1. **Dataset location**: staged dataset lives ONLY in `$HOME/spell/data`
   (`scripts/stage_to_ada.py`: tokens + transcripts + 16 kHz mono FLAC, sha256
   manifest). No per-utterance files anywhere else; NEVER on /share1.
2. **Relay archiving, LOCKED**: compute nodes CANNOT see /share1 (verified; mounts vary
   per node). Runs bundle under `$HOME/spell/runs/<run_id>/` (ckpts + metrics.parquet,
   ≤~20 files per run) with a `COMPLETED` marker written IN-PROCESS by the
   training entrypoint on trainer-attested success (single-writer law, §10
   item 9) — shell traps are cleanup-only;
   `scripts/drain_runs.sh` independently validates content-provenance (marker +
   metrics for both splits + loadable last.ckpt + manifest keys; refusals name
   the failed check, `--verify-only` gates archiving) and COPIES validated
   bundles to `/share1/$USER/spell/runs/` from a mounted node, KEEPING the
   relay copy by default (amended 2026-09-19: other scripts, e.g.
   `summarize_track_b.py` run from a compute node that cannot see `/share1`
   at all, sometimes need the relay bundle to still be there post-drain) —
   `--prune-relay` opts into the old delete-after-verify behavior for a given
   invocation. Copy-by-default means the relay does NOT shrink on its own, so
   it will climb toward the $HOME quota (item 3 below) as more runs drain
   over the project's lifetime; `check_storage.py --strict`'s existing gate
   (loud refusal at job start, never silent) is the backstop, and
   `--prune-relay` (or a manual `rm -rf` of an already-archived, no-longer
   locally-needed bundle) is the release valve when that matters. Drainer probes
   run under the FROZEN env python (`~/envs/spell/bin/python`, §5.0) — bare
   `python` on Ada login shells is CentOS-7 2.7 and cannot parse them; a probe
   that cannot run NEVER validates (empty verdict ⇒ refusal). Because the
   venv's base interpreter (`/usr/local/apps/python-3.12.4`) is mounted ONLY
   on compute nodes (2026-08-28), the login node itself runs NO python: the
   content checks execute as ONE CPU `srun` job carrying the full §5.8
   scheduling string (`scripts/drain_probe.py` under the frozen venv — the
   interpreter that trained the runs), while discovery, structural checks and
   rsync/diff/archive stay shell-only on the login node. Direct writes outside $HOME are rejected by policy everywhere
   (`archive_mode != "relay"` raises).
13. **sbatch startup ordering, LOCKED (2026-09-09, two incidents same day)**:
    (a) every sbatch script MUST derive `REPO_ROOT="${SLURM_SUBMIT_DIR}"` —
    NEVER `$(cd "$(dirname "$0")/.." && pwd)`. This cluster's slurmd executes
    batch scripts from a per-job spool copy, so `$0` does not resolve back
    into the submitter's clone; the derived path is unwritable and `set -e`
    aborts on the first `mkdir -p logs` (job 2691902). (b) every `python`
    invocation MUST come AFTER `source ~/envs/spell/bin/activate` — PATH
    carries no python at all beforehand unless the u22 module is also loaded,
    so a bare `python scripts/check_storage.py` run first dies with
    `command not found` (job 2691954, the very next resubmit after fixing
    (a)). Both were latent in `score_{dnsmos,kmeans,dsir}.sbatch` and copied
    into the first `score_proxy.sbatch` draft; all four were fixed together
    and are pinned by `tests/test_sbatch_repo_root.py`. `template.sbatch` and
    `setup_env.sbatch` never had either bug — they are the reference order to
    copy from for any future sbatch script.
3. **$HOME gates before every launch**: usage warn ≥20 GB, abort ≥23 GB;
   inode warn at 240k. `scripts/check_storage.py --strict` runs at job start in the
   SLURM template; soft mode locally.
4. **Hardware drift guard (locked)**: a physically swapped RTX 3080 was observed inside
   the 2080 Ti pool (gnode077). Every train/eval entrypoint calls
   `hardware_guard.assert_gpu()` — aborts with hostname unless device 0 is an
   "RTX 2080 Ti". Formal runs are 2080 Ti-constrained PERMANENTLY (ihub/3080 partition
   inaccessible — do not reference it in configs). Driver range 570–580 is fine for
   cu124 wheels; logged in manifests, not gated.
5. **Environment policy**: module `u22/python/3.12.4` + env `~/envs/spell`; ALL pip
   installs run INSIDE allocated srun sessions with `--no-cache-dir` — never on the
   RAM-limited login node. CPU-only jobs (env builds, preprocessing, selection
   scripts) add `--gres=gpu:0`; **`u22-cpu` is devalab-restricted and must not be used**.
   Activation is ALWAYS `source ~/envs/spell/bin/activate`, after which invoke
   `python` (never `python3`).
   **Environment status: BUILT AND PINNED — torch 2.6.0+cu124, GLIBC gate passed
   (Ada env commit 8c4c340). Never create a second venv under any circumstances.**
   **Lockfile discipline (§5.5)**: `requirements.lock` at the repo root IS the
   description of `~/envs/spell` — **a venv not described by the lockfile is
   considered BROKEN.** ANY interactive/manual pip install into the env must be
   IMMEDIATELY followed by regenerating the lock (`pip freeze > requirements.lock`),
   committing it to the repo. Better still: when a test run reveals a missing
   dependency, that dependency belongs in `scripts/setup_env.sbatch` /
   requirements.lock so the env stays reproducible from scratch — hand-installs
   are the exception, not the workflow.
   **`pip freeze` is a DUMP, not a validation (2026-09-09 incident, TWO acts)**:
   act 1 — a bare `pip install onnxruntime==1.18.1` (no other constraints on
   the command line) let pip's resolver silently downgrade numpy 2.5.2 ->
   1.26.4 to satisfy onnxruntime's OWN declared `numpy<2.0,>=1.21.6` pin —
   pip had no reason to preserve numpy's prior version since nothing on that
   command line asked it to. `pip freeze > requirements.lock` then faithfully
   committed that downgraded numpy alongside scipy==1.18.1/lightning/
   torchmetrics, which need numpy>=2.0 at import time — breaking every job
   that imports Lightning (job 2691962). Act 2 — reverting ONLY the numpy pin
   back to 2.5.2 in the same lock, still installed via one flat
   `pip install -r requirements.lock`, surfaced the REAL conflict pip had
   been quietly resolving around: asked for numpy==2.5.2 AND
   onnxruntime==1.18.1 explicitly together, pip refuses outright
   (`onnxruntime 1.18.1 depends on numpy<2.0`). The two packages' pins are
   FUNDAMENTALLY incompatible in one resolver pass — no version juggling
   fixes this, only splitting the install does: `scripts/setup_env.sbatch`
   installs everything else via the normal dependency-checked `-r` pass, then
   onnxruntime SEPARATELY via `--no-deps` (skips onnxruntime's own resolver
   check; numpy stays at the lock's pin). Act 3 — the resulting env's
   smoke-check CONFIRMED onnxruntime==1.18.1 is broken at the binary level
   too, not just by its declared pin: `import onnxruntime` raises
   `AttributeError: _ARRAY_API not found` — the canonical numpy-2.0 C-ABI
   break for a compiled extension built against numpy 1.x and never
   rebuilt. **Resolution (user decision, 2026-09-09): onnxruntime is DROPPED
   ENTIRELY.** `scripts/score_dnsmos.py` now runs `dnsmos_model.DNSMOSTorch`,
   a pure-PyTorch reimplementation of the vendored ONNX graph, extracted and
   verified bit-equivalent (<1e-6 max abs diff) by
   `scripts/port_dnsmos_to_torch.py` against the ONNX graph directly (run in
   a throwaway venv, never `~/envs/spell` — see that script's docstring).
   `requirements.lock` no longer carries onnxruntime (nor its
   onnxruntime-only transitive deps coloredlogs/flatbuffers/humanfriendly);
   `scripts/setup_env.sbatch` actively uninstalls any leftover copy so the
   live env matches the lock exactly. `tests/test_lockfile_pins.py::
   test_onnxruntime_never_reenters_the_lock` guards against a future
   hand-install silently resurrecting this. See
   `models/dnsmos/PROVENANCE.md` for the full graph trace and verification.
   Going forward: after ANY hand-install, diff `pip freeze` output against
   the CURRENT lock before overwriting it, and re-run the suite-critical
   smoke imports (`scripts/setup_env.sbatch`'s own check) BEFORE committing —
   a changed pin on a package you did not ask to install is the signal to
   stop and look, not to freeze through it. Pinned by
   `tests/test_lockfile_pins.py` (numpy must stay >=2.0).
   `scripts/setup_env.sbatch` builds fresh or
   reconciles the existing env EXACTLY against the lock (this is also how newly
   locked deps — pytest, lightning — reach the live env); pip's exact-pin install
   IS the version guarantee, and the script's post-build step just import-smoke-
   checks the suite-critical packages.
6. **Partition adoption rule**: before adopting any partition, verify access —
   `scontrol show partition <name>` and confirm `AllowAccounts` includes our account.
   Never build configs or workflows around an unverified partition (ihub was rejected
   this way; u22-cpu-style surprises cost a revision).
7. **Git transport (single rule)**: the repo moves between laptop and Ada ONLY via
   git. `rsync --delete` (or any bulk-copy) into a git working tree is FORBIDDEN:
   it silently clobbers divergence between the two checkouts.
   **ACHIEVED STATE (since Phase 0b)**: the bare remote, the laptop's `ada`
   remote, and the canonical clone `~/spell/repo` all exist. The working flow is:
   laptop commit → `git push ada main` → on Ada, `cd ~/spell/repo && git pull`.
   Handoffs therefore NEVER re-specify bootstrap steps (`git init --bare`,
   `git clone`, `git remote add`) — they assume this state and contain only
   INCREMENTAL commands. Remote examples in handoffs use full SSH URLs or the
   `ada`/`ada:` alias form, never local paths.
   **GitHub mirror (user-approved amendment B, 2026-09-08)**: the laptop also
   carries `origin` = `github.com/TH4KUR/SPELL-V2` (PRIVATE). `ada` remains the
   SOLE CANONICAL remote — Ada pulls only from its local bare repo and never
   from GitHub; `origin` is a PUSH-ONLY mirror, updated at sync points from the
   laptop (both pushes carry the same commit). Compute nodes never touch GitHub
   and no GitHub credentials exist on Ada. Before ANY public release: scrub
   data-derived artifacts (`data_index.parquet`, split lists), cluster topology
   (`KNOWN_BAD_NODES.md`, quotas), and W&B entity references.
8. **Scheduling constants for EVERY sbatch/srun block, GPU or CPU alike** (no drift):
   `-p u22 -A research --qos=medium --constraint=2080ti --exclude=gnode066`.
   The FULL string — constraint and exclude included — applies to CPU-only jobs
   too (those add `--gres=gpu:0`). Rationale: uniform blocks are copy-paste safe,
   they insure against accidental CUDA touchpoints landing on a drifted card or
   gnode066, and carrying unused filters costs nothing in the queue.
   Interactive GPU allocations are never parked idle — release with `exit`;
   before submitting, `squeue --me` must show zero stale rows.
   `SLURM_JOB_GPUS` is unreliable on this cluster:
   `CUDA_VISIBLE_DEVICES` / `torch.cuda.get_device_name()` are the source of
   GPU truth (the drift guard reads exactly those).
8a. **`medium` QOS per-user caps (checked 2026-09-19 via `sacctmgr show qos
    medium format=Name,MaxSubmitPU,MaxJobsPU`)**: `MaxSubmitPU=8` (running +
    PENDING jobs combined, per user, across the whole QOS) and `MaxJobsPU=4`
    (of those, at most 4 may actually be RUNNING at once — the rest queue as
    PENDING, which is fine and does not violate MaxSubmitPU). Each SLURM
    ARRAY TASK counts as its own job against both caps — an `sbatch
    --array=0-11` (12 tasks) is REJECTED outright with
    `QOSMaxSubmitJobPerUserLimit` if it would push the user's total
    (existing jobs + new array tasks) over 8, even if fewer than 8 would
    ever run concurrently. Caught live: `sbatch --array=0-11 ...` failed
    while a single DNSMOS scoring job was already running (1 + 12 = 13 > 8).
    **Before submitting an array of size N, check `squeue -u $USER | wc -l`
    (existing jobs) and keep N + existing ≤ 8** — split a larger batch into
    multiple `sbatch --array=...` calls (e.g. tasks 0-6 now, 7-11 once
    something else finishes) rather than one oversized submission.
    **`MaxWall=4-00:00:00`** (4 days, checked the same way via
    `sacctmgr show qos medium format=Name,MaxWall`) is the QOS's own wall-clock
    ceiling on what a job may REQUEST via `--time` — it is NOT applied when
    `--time` is left unset. **Corrected, §3.31**: omitting `--time` instead
    falls back to the u22 PARTITION's `DefaultTime` (`scontrol show
    partition u22`), a separate, independent, and much shorter setting
    (`01:00:00`) — job 2704082 was killed after only ~59 minutes finding
    this out live. Always pass an explicit `--time`; never omit it on the
    assumption that a generous QOS `MaxWall` fills the gap.
9. **Adoption rule going forward**: every future sbatch/srun/command block emitted
   in this project MUST use exactly the §5.0 names and §5.8 scheduling constants;
   if a handoff contains a stale path/name (`~/pymax`, missing `-p u22`,
   `/share1/NAS/...`), a re-emitted bootstrap step, or a local-path git remote,
   treat it as a bug and self-correct BEFORE presenting (§5.7 governs the flow).
10. **Run-plan TSVs — schema FROZEN, generator-sanctioned**: 6 TAB-separated
    fields `task|track|subset_manifest|seed|config_yaml|keep_local_traj`; the
    first non-comment row is the literal header; `#`/blank lines are ignored
    anywhere. **The task id lives IN THE DATA**: readers match column 1 against
    `SLURM_ARRAY_TASK_ID` via `scripts/plan_reader.awk` — position/offset reads
    (physical line = array id + N) are FORBIDDEN; they broke silently when a
    regenerated file's banner length changed (§10, pilot-A incident).
    `scripts/gen_run_plan.py` is the ONLY sanctioned writer; hand-editing a
    committed `slurm/*.tsv` is a protocol violation, and its `--check` mode must
    stay green in the test suite. `template.sbatch` pins NO array range (always
    passed on the sbatch CLI) and carries flags-only #SBATCH lines.
11. **No live edits on cluster clones**: tracked files on Ada change ONLY via the
    §5.7 git flow. Patching remote copies with sed/python `.replace()` is
    FORBIDDEN — exact-string patches no-op silently when whitespace drifts
    (observed: the index-free reader patch never landed). Cluster-side change ⇒
    edit in repo → tests green → push → pull → resubmit. A temporary bypass
    script requires explicit user authorization AND post-run reconciliation of
    anything it produced (see §10 pilot-A entry).
12. **Score tables (2026-09-08)**: per-utterance selector scores (DNSMOS,
    k-means assignments, DSIR weights, proxy loss/EL2N, LESS influences) live
    in `scores/*.parquet`, keyed by `utterance_id`, TRAIN-universe rows only
    (29,064 — val/reference fits happen INSIDE the Ada scoring job and are
    never committed as rows). They are committed to the repo (data-derived ⇒
    added to the §5.7 public-release scrub list; `.gitignore` carries explicit
    `!scores/*.parquet` and `!data_index.parquet` negations over the blanket
    `*.parquet`). Because score files are BORN on Ada, committing them FROM the
    Ada clone (`git add scores && git commit && git push`; laptop pulls) is
    SANCTIONED; §5 item 11 (no live edits to tracked files) still applies to
    everything tracked.

## 6. Run manifest contract

Every training run writes `run_manifest.json` next to its checkpoints:
subset manifest path, seed, config hash (`config.config_hash`, sha256 over canonicalized
YAML), git SHA, gpu_name + driver version, protocol version (= this file's revision),
start/end timestamps.

## 7. Phase gates

- **Phase 0b (BLOCKING)**: stage the dataset to Ada (`$HOME/spell/data`) via
  `scripts/stage_to_ada.py`, sha256-manifested, then `--verify` clean. No Phase-1
  training code runs before this gate passes.
  **Status: PASSED 2026-08-26** — transfer complete, on-Ada verify returned
  `99909 expected | missing=0 extra=0 hash_mismatch=0`.
- Phase 1+ follow PLAN.md's phase order with the pilot gates defined there. Track B
  pilot unlock gate: val WER decreasing by epoch ~5; if still falling steeply at the
  provisional epoch budget, EXTEND epochs before freezing hyperparameters (rule 15).
  **Gate signal policy**: the PRIMARY signal is the LIVE `val/{wer,cer,loss}` series
  emitted by the module to its logger each epoch; `scripts/evaluate_track_b.py`
  post-hoc decoding remains the fallback cross-check and the source of official
  final numbers (policy set 2026-08-27 after fixing dead live-val logging).
- **Post-25% sweep gate (written 2026-09-08, implements §3.19(c))**: 5%/10%
  training cells may be submitted ONLY when (i) every 25% selector cell is
  COMPLETED, drained, and summarized (`scripts/summarize_track_b.py`), (ii)
  zero protocol deviations are logged in those runs, (iii) the across-selector
  mean-ΔWER spread exceeds 1×σ_diff — otherwise convene on the H2c early-stop
  (RESEARCH §9) BEFORE spending on the sweep, and (iv) the advancing-selector
  list is recorded in THIS section's revision history BEFORE submission.
  Random is swept at every budget; anti-selection rides along as the sanity
  column. If the 25% picture is ambiguous (|Δ| < 2σ_diff for the leading
  methods), the 5% cell promotes INTO ICASSP scope — a single-budget ambiguous
  result is unpublishable; §3.19(e) protects single-budget claims meanwhile.

## 8. Deferred-but-revivable research questions

RQ1 (layer-view ablation) and RQ3 (transfer matrix) stay out of scope; cheap insurance:
tokens are already organized per-layer-view, per-sample losses + 5-epoch checkpoints are
kept for every grid run, and every subset carries characterization stats.

## 9. How to run things

Laptop: pymax venv (`source ~/bin/pymax/bin/activate` — laptop-only).
Ada: `source ~/envs/spell/bin/activate`, inside the working repo `~/spell/repo`
(synced ONLY via git per §5.7: the bootstrap is DONE — laptop commits and
pushes to its `ada` remote, Ada runs `cd ~/spell/repo && git pull`).
All commands run from the repo root so flat root modules import cleanly; every
scheduler invocation carries §5.8's frozen flags.

```bash
sbatch scripts/setup_env.sbatch              # build/reconcile ~/envs/spell vs requirements.lock
python scripts/gen_run_plan.py [--check]     # the ONLY sanctioned run-plan TSV writer
python scripts/audit_data.py                 # rebuild data_index.parquet + audit report
python scripts/build_splits.py               # rebuild frozen split files (byte-stable)
python scripts/check_storage.py [--strict]   # $HOME quota gate (warn 20G / abort 23G)
python scripts/stage_to_ada.py               # PHASE 0b: stage dataset + sha256 manifest
python scripts/stage_to_ada.py --verify      # re-hash staged tree vs manifest
bash scripts/push_data_to_ada.sh [-n]        # push staged dataset to Ada (preflight/transfer/postflight)
bash scripts/drain_runs.sh -n                # preview relay-archive drain
python scripts/spot_check.py                 # decode 10 utts -> outputs/spotcheck/ (ear check)
pytest -q                                    # unit tests incl. crop alignment proof
bash scripts/debug.sh                        # single-GPU debug launcher (tests + smoke)
```

Revision history: `universe-v2` re-froze the internal split over the selectable universe;
`ada-storage-rev1` superseded NAS write-through with relay archiving + HOME gates +
hardware drift guard; `sig-v2-github-mirror` (2026-09-08) added the significance
criterion v2 (§3.18) and the GitHub mirror law (§5.7); `p2-selection-law`
(2026-09-08) added §3.19–3.21, §5 item 12, §3.17(d), and the §7 post-25% sweep
gate; `sbatch-repo-root-fix` (2026-09-09) added §10 item 13 and fixed all
scorer sbatch scripts after job 2691902 died on a spool-relative `$0`, then
job 2691954 caught a second latent ordering bug (python invoked before venv
activation) in the same four scripts, fixed in the same item; `lock-numpy-fix`
(2026-09-09) restored numpy==2.5.2 in requirements.lock after job 2691962
showed the onnxruntime install's silent numpy downgrade had broken
scipy/lightning imports, discovered the numpy==2.5.2/onnxruntime==1.18.1 pins
are irreconcilable in one pip resolve, and amended §5.5 with the
`pip freeze`-is-not-validation lesson; `dnsmos-torch-port` (2026-09-09)
confirmed onnxruntime==1.18.1 crashes at runtime under numpy>=2.0 (not just
its declared pin) and resolved it by dropping onnxruntime entirely —
`dnsmos_model.DNSMOSTorch` is a verified pure-PyTorch port of the vendored
ONNX graph, `requirements.lock` no longer carries onnxruntime at all;
`dsir-l2-goNoGo` (2026-09-17) added §3.22, resolving the DSIR 72h go/no-go
GO with a weak-effect disclosure — `scripts/sweep_dsir_l2.py`'s held-out
cross-validation showed `weight_spread`'s dramatic swing across `l2` was
overfitting noise, not real signal, while the genuine held-out AUC
(≈0.57, statistically real but weak) held flat at every tested `l2`
including the original formal `l2=1e-4` (no re-run needed);
`wandb-crucial-runs` (2026-09-18) added §3.23 after `score_less.py` shipped
without W&B logging — any research-crucial scorer/sweep/analysis script
now logs by default, structurally separate from §3.17's formal-run
contract; `wandb-live-progress` (same day) added §3.24 after the user
caught, live, that even the fixed `score_less.py` only opened its W&B run
at the very end — logging must open the run before the expensive work
starts and report progress incrementally with each unit's own duration,
not a batch upload at exit; `sweep_dsir_l2.py` had the identical gap and
was fixed the same day. The four pre-existing scorers remain flagged
non-compliant with both items pending explicit sign-off to retrofit them.
Earlier wording remains in git history.

## 10. Known issues & permanent policies

1. **CTC backward has no deterministic CUDA kernel** → strict
   `torch.use_deterministic_algorithms(True)` aborts on GPU. PERMANENT policy:
   training uses Lightning `Trainer(deterministic="warn_only")`. Seeding, data
   order and init remain fully pinned by `train_seed=20260826` (§3.11) — only
   bit-exact CUDA reproducibility is waived (impossible for this op class).
2. **gnode066**: CUDA-init failure documented on that node. Excluded from every
   job via `--exclude=gnode066`. The exclusion list grows ONLY through
   `KNOWN_BAD_NODES.md` plus a manual PROTOCOL.md note here — never ad-hoc
   command-line exclusions without the bookkeeping entry.
3. **Pilot-A bypass incident (2026-08-27, reconciled)**: a run-plan reader keyed
   on PHYSICAL LINE (`array_id + 2`) broke the moment a regenerated TSV changed
   its banner length; jobs launched against non-data lines died instantly. An
   index-free fix was then attempted as an in-place patch ON ADA only — the
   exact-string `.replace()` no-opped silently (whitespace drift), so neither
   version reached git and training went out via an ad-hoc bypass sbatch
   (/tmp/pilotA.sbatch). Consequences locked in: §5 item 10 (data-keyed schema +
   sanctioned generator), §5 item 11 (no live edits on cluster clones), full
   template rewrite + regression suite (`tests/test_run_plan.py`). The running
   pilot's bundle lacks `--run-manifest-out` provenance — reconcile post-run if
   it matters downstream (see handoff); Phase-2+ must never need this class of
   bypass again because the template now reads plans by ID with loud fatals.
4. **DATA LAYOUT LAW**: `docs/layout.md` declares the canonical staged tree
   (`<root>/<VIDEO_ID>/<stem>.{tokens.pt,txt,flac}`; `<root>/<stem>.*` bare for
   test); the loader (`paths.py`, the single path authority) validates against
   it at startup and deviations ABORT LOUDLY before epoch 1 (preflight).
   NO module may concatenate `dataset_root + split + folder` heuristics;
   split membership is never a filesystem segment. (Birthed by Pilot A failing
   three times on an imagined `<root>/trainval/<VIDEO_ID>/…` layout.)
5. **INDEX IS TRUTH**: where an utterance's bytes live comes ONLY from
   `data_index.parquet` identity columns (video_id, stem, split) plus the
   declared filename pattern; a new staging pass must regenerate the index AND
   pass `stage_to_ada.py --verify` before anything consumes it. There are never
   two parallel truths about locations; stored `*_path` strings are raw-source
   provenance, not runtime addresses.
6. **ATOMIC LAYOUT CHANGES**: any future staging/layout change lands in ONE
   commit containing ALL of: index regeneration + loader adjustment
   (`paths.py` + consumers) + `docs/layout.md` update + tests (layout suite +
   regression UID `0D9QIG36J9Q/50001`). Half-migrated states are protocol
   violations even when each half is individually correct.
7. **RUNTIME DATA ROOT IS EXPLICIT**: `configs/paths.yaml`'s `dataset_root`
   default (`datasets/LRS3`) is the Phase-0 RAW audit source and exists for
   audit tooling ONLY — it is never a legal training-time root under staged
   layout. Therefore: staged mode REFUSES any root shaped `…/datasets/LRS3`
   at construction (`paths.LayoutError` with the remedy attached); runtime
   selection happens via `SPELL_DATA_ROOT`, which `slurm/template.sbatch`
   defaults to the §5.0 constant (`$HOME/spell/data`), exports, existence-gates,
   and echoes in its banner — interactive shells must export it themselves.
   A staged run WITHOUT an explicit root is a configuration error, not a
   fallback. (Birthed 2026-08-27 when the first template-based resubmit ran
   without the env var and aborted cleanly at the new preflight — the gate
   working as designed; pinned by tests incl. template grep-pins.)
8. **LIVE VAL LOGGING SILENT-MISS (2026-08-27)**: the sole-consumer refactor that
   fixed the epoch-shift bug left the lit module's post-drain snapshot permanently
   EMPTY (callback hook drains rows before the module hook) — so every
   `val/*` W&B series was silently dead while metrics.parquet stayed complete;
   training itself unaffected. Fix = OWNERSHIP SPLIT: per-utterance rows remain
   callback-drained; module-owned scalar aggregates are consumed only by the
   module's own (later-firing) hook, which is the ONLY sanctioned W&B emitter.
   Any future logging path must prove itself against the spy-logger fit test
   (`tests/test_val_wandb_series.py`: exactly-once emission, sanity suppressed)
   — a stub-mocked Trainer cannot catch logger-boundary regressions.
9. **COMPLETED = trainer-attested success, written in-process, never by shell
   traps** (2026-08-27): crashed bypass bundles carried EXIT-trap-written
   markers that the drainer accepted as valid — process exit ≠ training
   success. The fix is a SINGLE-WRITER LAW: `train_track_b.attest_completed()`
   (the training entrypoint's last statement, after clean fit + durable
   metrics) is the ONLY writer repo-wide, mechanically enforced by
   `tests/test_single_writer_law.py` (exactly one writer context allowed).
   History: THREE writers existed — bash traps rc-gated at best, a callback
   `on_train_end` writer safe only by accident (Lightning's fit loop has no
   finally around `on_run_end`, so exceptions skip the hook by design) whose
   status guard was dead code (`TrainerStatus` has no STOPPED), and now the
   entrypoint. `drain_runs.sh` additionally re-validates content-provenance
   independently (marker alone insufficient; ≥1 epoch row per split,
   weights_only-loadable ckpt, manifest keys) with named refusals and a
   `--verify-only` mode — auto-discovery depth was also silently wrong
   (mindepth/maxdepth 2 can never see `<runs>/<track>/<id>/`) and is pinned
   exact. Archiving of Pilot A's bundle clears only via `--verify-only`.
   Probe-env corollary (same day): the drainer defaulted to bare `python`,
   which on Ada login shells is CentOS-7 Python 2.7 — the probe died at parse
   time, and an empty probe verdict used to fall through as VALIDATE. Fixed
   both: PYTHON_BIN anchors the frozen env + startup gate refuses non-python3;
   empty verdict ⇒ named refusal, pinned by tests. Node-class corollary
   (2026-08-28): the venv base interpreter is compute-only by mount, so the
   drainer split — login node does shell-only work; ONE CPU srun job (§5.8
   string) runs `scripts/drain_probe.py` under the frozen venv; a failed or
   under-reporting probe job refuses EVERYTHING loudly (never silently
   passes), pinned by srun-shim subprocess suites on the laptop.
