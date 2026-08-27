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
9. Selection NEVER happens inside training code: trainers consume a manifest path from
   `subsets/`.
10. Deterministic data ordering per seed; identical eval batches/crops across all runs
    of a track.
11. **Seeds are two distinct things**: the `subset_seed` (e.g. 101/102) draws a random
    manifest and is part of that subset's IDENTITY (`subsets/random_25pct_seed{101,102}.txt`);
    the `train_seed` is a FIXED constant shared by every run. Track A anchors consume
    `subsets/random_25pct_seed{101,102}.txt` unchanged.
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
      (2026-08-27 incident).

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
| Relay directory | `$HOME/spell/runs/` (relay `runs_dir`; `drain_runs.sh` moves relay → archive) |

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
   the failed check, `--verify-only` gates archiving) and moves validated
   bundles to `/share1/$USER/spell/runs/` from a mounted node. Drainer probes
   run under the FROZEN env python (`~/envs/spell/bin/python`, §5.0) — bare
   `python` on Ada login shells is CentOS-7 2.7 and cannot parse them; a probe
   that cannot run NEVER validates (empty verdict ⇒ refusal). Direct writes outside $HOME are rejected by policy everywhere
   (`archive_mode != "relay"` raises).
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
   are the exception, not the workflow. `scripts/setup_env.sbatch` builds fresh or
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
hardware drift guard. Earlier wording remains in git history.

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
   empty verdict ⇒ named refusal, pinned by tests.
