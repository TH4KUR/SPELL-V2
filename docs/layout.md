# Data layout — canonical declaration

**Status: LOCKED (PROTOCOL §10 item 4, DATA LAYOUT LAW).** This document declares the
one canonical on-disk arrangement of utterance files. The loader (`paths.py`) validates
against it at startup; deviations abort loudly. No code may construct data paths any
other way.

## The canonical layout: STAGED flat tree

This is what **physically exists** on Ada under `$HOME/spell/data` (verified Phase 0b:
99,909 files hash-matched, inode-exact). No re-staging or move is needed — it already
is the canonical form.

```
<root>/<VIDEO_ID>/<stem>.tokens.pt     # split = "trainval"
<root>/<VIDEO_ID>/<stem>.txt
<root>/<VIDEO_ID>/<stem>.flac          # audio is ALWAYS 16 kHz mono FLAC on the staged tree
<root>/<stem>.tokens.pt                # split = "test": BARE files at the root
<root>/<stem>.flac                     #   (test video_id is "" in the index)
```

Worked examples (real IDs):

- trainval, the exact utterance that failed Pilot A under the old buggy assumption:
  `$HOME/spell/data/0D9QIG36J9Q/50001.tokens.pt` ✓ exists
- test: `$HOME/spell/data/<uid>.tokens.pt` (no folder level)

### Split membership is METADATA ONLY

The filesystem does **not** encode which split an utterance belongs to — trainval
utterances sit in per-video folders, test utterances sit bare at the root, but nothing
labels either, and that distinction is an artifact of the staging shape, not a rule.
Split membership lives exclusively in `data_index.parquet`'s `split` column and the
frozen split ID lists (`subsets/splits/*`). A directory walk can tell you *that* a file
is staged; only the index can tell you *which* split it serves.

## The laws (PROTOCOL §10 items 4–6, binding)

1. **DATA LAYOUT LAW** — this file declares the tree; `paths.py` is the single path
   authority (`DataPaths.tokens_relpath / transcript_relpath / flac_relpath /
   resolve_tokens / resolve_audio / preflight`). No module may concatenate
   `dataset_root + split + folder` heuristics; every such construction found is a bug —
   delete it and route through `paths`.
2. **INDEX IS TRUTH** — where an utterance's bytes live comes only from
   `data_index.parquet` (identity, video_id, stem, split) plus the declared filename
   pattern above. Any new staging pass regenerates the index and passes
   `stage_to_ada.py --verify` before anything consumes it. There are never two parallel
   truths. (The committed index's `tokens_path/txt_path/audio_path` string columns are
   *provenance* of the raw container; runtime consumers derive locations from
   `(video_id, stem)` + suffix, never by parsing those strings.)
3. **ATOMIC LAYOUT CHANGES** — any future staging/layout change lands in ONE commit
   containing all of: index regeneration + loader/`paths.py` adjustment +
   `docs/layout.md` (this file) + tests. Half-migrated states are protocol violations.

## Startup preflight

Every runtime entrypoint (trainer, evaluator, overfit smoke) prints its resolved root
and samples random active-manifest IDs through the authority BEFORE epoch 1:

```
[spell] data_root=$HOME/spell/data layout=staged
[spell] preflight: sampled 50/N manifest utts -> 50/50 token files present
```

A miss raises `paths.LayoutError` printing data_root, the resolved path, the expected
tree pattern from this file, and the regen+verify guidance. Training never starts on a
mismatched tree.

## Legacy layout (quarantined)

The Phase-0 raw source `datasets/LRS3/trainval|test/…` with `.mp4`/`.wav` containers
survives ONLY behind `SPELL_DATA_LAYOUT=legacy` for pre-restage tooling (e.g. auditing,
resynthesis). It is never a training-time layout and exists only until the next audit
regen deletes the migration shim in `scripts/stage_to_ada.py`.

## History (why this document exists)

Pilot A failed three times on path mismatches, all one class: loaders assuming a
`trainval/` segment that the staged tree does not carry. Root cause was path
construction scattered across modules instead of one declared authority. Item (a) of
the reconciliation decision was adopted: resolve `<root>/<VIDEO_ID>/<stem>.*` ALWAYS;
split stays pure index metadata. The on-disk Ada tree already matched this — only the
code moved.
