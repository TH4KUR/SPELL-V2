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

## 4. Known caveats (accepted, uniform ⇒ ranking-valid)

- Trainval audio decodes from AAC mp4; official test wavs are PCM. All subsets share the
  same decode path, so selection rankings are unaffected; noted for absolute-metric claims.
- Transcript confidence (`Conf: 1–6`) is stored in `data_index.parquet` as metadata;
  it is not used by any Phase-0 decision.

## 5. Storage policy

- Trajectory checkpoints **stream to the NAS on write** (`nas_root`, default
  `/share1/NAS/spell-rq2`, override with `SPELL_NAS_ROOT`); local run directories keep
  ONLY `last.ckpt`.
- **df gate before every array launch**: `scripts/check_storage.py --strict` runs at job
  start in `slurm/template.sbatch`; any failure aborts the task before training.
  Locally the gate runs soft via `scripts/debug.sh` (dev boxes have no `/share1`).
- Only **LESS-designated runs** (`keep_local_traj=1` column in the run plan TSV) retain
  full trajectories locally.

## 6. Run manifest contract

Every training run writes `run_manifest.json` next to its checkpoints:
subset manifest path, seed, config hash (`config.config_hash`, sha256 over canonicalized
YAML), git SHA, protocol version (= this file's revision), start/end timestamps.

## 7. Deferred-but-revivable research questions

RQ1 (layer-view ablation) and RQ3 (transfer matrix) stay out of scope; cheap insurance:
tokens are already organized per-layer-view, per-sample losses + 5-epoch checkpoints are
kept for every grid run, and every subset carries characterization stats.

## 8. How to run things

All commands run from the project root inside the pymax venv
(`source ~/bin/pymax/bin/activate`) so flat root modules import cleanly:

```bash
python scripts/audit_data.py                 # rebuild data_index.parquet + audit report
python scripts/build_splits.py               # rebuild frozen split files (byte-stable)
python scripts/check_storage.py              # storage gate (--strict for launches)
python scripts/spot_check.py                 # decode 10 utts -> outputs/spotcheck/ (ear check)
pytest -q                                    # unit tests incl. crop alignment proof
bash scripts/debug.sh                        # single-GPU debug launcher (tests + smoke)
```

Revision history: `universe-v2` (post-audit amendment) re-froze the internal split over
the selectable universe; the original Phase-0 lists remain in git history.
