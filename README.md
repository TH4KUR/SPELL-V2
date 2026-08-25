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

## Storage

Trajectory checkpoints stream to the NAS on write (`SPELL_NAS_ROOT`, default
`/share1/NAS/spell-rq2`); local dirs keep only `last.ckpt`; only LESS-designated runs keep
full local trajectories. See PROTOCOL.md §5.
