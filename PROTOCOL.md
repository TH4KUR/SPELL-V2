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

## 4. Known caveats (accepted, uniform ⇒ ranking-valid)

- Trainval audio decodes from AAC mp4; official test wavs are PCM. All subsets share the
  same decode path, so selection rankings are unaffected; noted for absolute-metric claims.
- Transcript confidence (`Conf: 1–6`) is stored in `data_index.parquet` as metadata;
  it is not used by any Phase-0 decision.

## 5. Storage & environment policy (Ada revision — supersedes NAS write-through)

**Quotas:** `/home` = 30 GB + 300k inodes; `/share1` = 100 GB but ~3200-inode cap.
Host is CentOS 7 / GLIBC 2.17 despite u22 module names — binary deps must be
manylinux2014-compatible or cluster-module-provided.

1. **Dataset location**: staged dataset lives ONLY in `$HOME/spell/data`
   (`scripts/stage_to_ada.py`: tokens + transcripts + 16 kHz mono FLAC, sha256
   manifest). No per-utterance files anywhere else; NEVER on /share1.
2. **Relay archiving, LOCKED**: compute nodes CANNOT see /share1 (verified; mounts vary
   per node). Runs bundle under `$HOME/spell/runs/<run_id>/` (ckpts + metrics.parquet,
   ≤~20 files per run) with a `COMPLETED` marker written by an EXIT trap;
   `scripts/drain_runs.sh` verifies and moves bundles to `/share1/NAS/spell-rq2` from a
   mounted node. Direct writes outside $HOME are rejected by policy everywhere
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
5. **Environment policy**: module `u22/python/3.12.4` + venv in $HOME; ALL pip installs
   run INSIDE srun sessions — never on the RAM-limited login node. CPU-only jobs (env
   builds, preprocessing, selection scripts) go to partition `u22` with
   `--gres=gpu:0 --mem=16G`; **`u22-cpu` is devalab-restricted and must not be used**.
   `--no-cache-dir` always.
   **Environment status: BUILT AND PINNED — torch 2.6.0+cu124, GLIBC gate passed
   (Ada env commit 8c4c340). Never create a second venv under any circumstances.**
6. **Partition adoption rule**: before adopting any partition, verify access —
   `scontrol show partition <name>` and confirm `AllowAccounts` includes our account.
   Never build configs or workflows around an unverified partition (ihub was rejected
   this way; u22-cpu-style surprises cost a revision).

## 6. Run manifest contract

Every training run writes `run_manifest.json` next to its checkpoints:
subset manifest path, seed, config hash (`config.config_hash`, sha256 over canonicalized
YAML), git SHA, gpu_name + driver version, protocol version (= this file's revision),
start/end timestamps.

## 7. Phase gates

- **Phase 0b (BLOCKING)**: stage the dataset to Ada (`$HOME/spell/data`) via
  `scripts/stage_to_ada.py`, sha256-manifested, then `--verify` clean. No Phase-1
  training code runs before this gate passes.
- Phase 1+ follow PLAN.md's phase order with the pilot gates defined there.

## 8. Deferred-but-revivable research questions

RQ1 (layer-view ablation) and RQ3 (transfer matrix) stay out of scope; cheap insurance:
tokens are already organized per-layer-view, per-sample losses + 5-epoch checkpoints are
kept for every grid run, and every subset carries characterization stats.

## 9. How to run things

Laptop: pymax venv (`source ~/bin/pymax/bin/activate`). Ada: venv in $HOME built per §5.5.
All commands run from the repo root so flat root modules import cleanly:

```bash
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
