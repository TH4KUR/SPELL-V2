# SPELL-RQ2

Do data-selection methods rank the **same** when identical subsets of discrete speech
tokens are consumed by a generative vocoder (Track A) versus an ASR model (Track B)?
This repo implements a controlled dual-track benchmark on LRS3: selectors produce subset
manifests once, and both tracks train from scratch on each manifest under a locked
protocol (`PROTOCOL.md` — binding, read it before changing anything).

Docs: `PLAN.md` (phased spec) · `RESEARCH.md` (study design) · `PROTOCOL.md` (locked rules).

## Dataset provenance

- Corpus: **LRS3-TED** — Afouras, Chung, Senior, Vinyals, Ma, Zisserman,
  *Deep Audio-Visual Speech Recognition*, ICASSP 2018.
- Layout: `datasets/LRS3/trainval/<video_id>/<stem>{.mp4,.txt,.tokens.pt}` (per-video
  folders; numeric stems restart per folder, hence `<video_id>/<stem>` utterance IDs) and
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

Formal runs run on Ada; this laptop is the authoring/pilot box.

| Constraint | Value |
| --- | --- |
| GPU | **RTX 2080 Ti, permanent** — `hardware_guard.assert_gpu()` aborts any drifted card (a 3080 was observed in the pool, gnode077) |
| Host | CentOS 7 / GLIBC 2.17 → manylinux2014-compatible wheels only |
| Modules | `u22/python/3.12.4`, venv in `$HOME` |
| pip policy | only inside srun sessions carrying the FULL §5.8 constants (`-p u22 -A research --qos=medium --constraint=2080ti --exclude=gnode066`; CPU jobs add `--gres=gpu:0 --mem=16G`; **never `u22-cpu`**, devalab-restricted), and never on the login node; always `--no-cache-dir`. Env is BUILT AND PINNED (torch 2.6.0+cu124, GLIBC gate passed) — no second venv, ever. **`requirements.lock` IS the env description — a venv not described by it is BROKEN** (§5.5): manual installs must be lock-regenerated + committed immediately; missing deps belong in `scripts/setup_env.sbatch`, which reconciles the env exactly against the lock |
| partitions | before adopting any partition, check access: `scontrol show partition <name>` → `AllowAccounts` must include our account |
| Storage | dataset staged at `$HOME/spell/data` (FLAC+tokens+transcripts, sha256 manifest); runs bundle in `$HOME/spell/runs/<id>/` (≤~20 files); HOME gates warn 20 GB / abort 23 GB |
| Archive | **relay-locked** — compute nodes can't see /share1; `scripts/drain_runs.sh` moves bundles from a mounted node |

**Phase 0b (blocking):** stage + verify the dataset on Ada before any Phase-1 training:

```bash
python scripts/stage_to_ada.py --dest ~/spell/data   # FLAC + tokens + transcripts + sha256 manifest
bash scripts/push_data_to_ada.sh                     # laptop -> Ada incremental rsync (resumable; -n = dry-run)
python scripts/stage_to_ada.py --verify              # on Ada; must be clean before Phase 1
```

