#!/usr/bin/env python3
"""Track B checkpoint evaluator — greedy decode + per-utterance WER/CER to disk.

    python scripts/evaluate_track_b.py --ckpt runs/track_b/.../last.ckpt \
        --split val --output-dir outputs/eval_pilot100

Emits:
  predictions.jsonl  one row per utterance: {utterance_id, ref, hyp, wer, cer, loss}
  summary.json       aggregate WER/CER/loss, drop counts, hardware provenance

Protocol enforcement here:
- **Official-test quarantine (§3.5)**: ``--split test`` is refused outright unless
  env ``SPELL_FINAL_EVAL=1`` (final-eval module context only).
- Drift guard / dev bypass identical to training (``enforce_gpu_policy``).
- §3.12 input-length rule rows are skipped and COUNTED into summary.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch                                            # noqa: E402
from torch.utils.data import DataLoader                 # noqa: E402

import ctc as ctc_lib                                   # noqa: E402
from config import (                                    # noqa: E402
    PROJECT_ROOT,
    PROTOCOL_REVISION,
    load_paths,
)
from dataset import (                                   # noqa: E402
    TokenDataset,
    collate_token_batch,
    filter_records,
    load_id_list,
    load_records,
)
from hardware_guard import enforce_gpu_policy           # noqa: E402
import paths as data_paths                              # noqa: E402  DATA LAYOUT LAW
from lit_track_b import LitConformerCTC                 # noqa: E402
from manifest import _utc_now_iso                       # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # Official-test quarantine — hard refusal (§3.5).
    if args.split == "test" and os.environ.get("SPELL_FINAL_EVAL") != "1":
        print(
            "REFUSED: official LRS3 test split is quarantined until every run has "
            "finished (PROTOCOL §3.5). Re-run with SPELL_FINAL_EVAL=1 from the "
            "final-eval module only.",
            file=sys.stderr,
        )
        return 3

    gpu_info = enforce_gpu_policy()

    ckpt_path = args.ckpt.resolve()
    if not ckpt_path.exists():
        print(f"FATAL: checkpoint not found: {ckpt_path}", file=sys.stderr)
        return 2

    lit = LitConformerCTC.load_from_checkpoint(str(ckpt_path), map_location="cpu")
    lit.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lit.to(device)
    cfg = lit.cfg
    vocab = lit.vocab

    paths = load_paths()
    index_split = "trainval" if args.split == "val" else "test"
    recs = load_records(paths.index_path, split=index_split)
    if args.split == "val":
        val_ids = load_id_list(paths.splits_dir / "val_ids.txt")
        recs = filter_records(recs, val_ids, strict=False)

    # PREFLIGHT before any batch runs (§10 item 4) — resolve k random utts.
    dp = data_paths.current()
    print(f"[spell] data_root={dp.root} layout={dp.layout}", flush=True)
    data_paths.preflight_resolve(recs, k=min(50, len(recs)))

    collate = partial(collate_token_batch, token_pad_id=-1, text_pad_id=vocab.pad_id)
    dl = DataLoader(
        TokenDataset(recs, vocab=vocab, include_text=True),
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        collate_fn=collate,
    )

    blank = vocab.blank_id
    input_stream = int(cfg["model"].get("input_stream", 0))
    out_rows: list[dict] = []
    dropped_rule_312 = 0
    with torch.no_grad():
        for batch in dl:
            keep = ctc_lib.input_length_keep_mask(batch["lengths"], batch["text_lengths"])
            idx = torch.nonzero(keep, as_tuple=False).flatten()
            dropped_rule_312 += int((~keep).sum().item())
            if idx.numel() == 0:
                continue
            tokens = batch["tokens"][idx].to(device)
            lengths = batch["lengths"][idx].to(device)
            text_ids = batch["text_ids"][idx]
            text_lengths_dev = batch["text_lengths"][idx].to(device)
            uids = [batch["utterance_ids"][i] for i in idx.tolist()]

            stream = tokens[:, input_stream, :]
            log_probs, out_lengths = lit.model(stream, lengths)
            loss_vec = ctc_lib.ctc_loss_per_utt(
                log_probs.transpose(0, 1), text_ids.to(device),
                out_lengths, text_lengths_dev, blank,
            )
            hyps = ctc_lib.greedy_decode(log_probs, lengths, vocab)
            refs = [vocab.decode(t.tolist(), collapse_repeats=False) for t in text_ids]

            for uid, hyp, ref, lv in zip(uids, hyps, refs, loss_vec.cpu().tolist()):
                scoreable = bool(ref.strip())
                out_rows.append({
                    "utterance_id": uid, "ref": ref, "hyp": hyp, "loss": float(lv),
                    "wer": ctc_lib.wer_one(ref, hyp) if scoreable else None,
                    "cer": ctc_lib.cer_one(ref, hyp) if scoreable else None,
                })

    scored_wer = [r["wer"] for r in out_rows if r["wer"] is not None]
    scored_cer = [r["cer"] for r in out_rows if r["cer"] is not None]
    summary = {
        "protocol_revision": PROTOCOL_REVISION,
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "n_utterances": len(out_rows),
        "dropped_input_length_rule": dropped_rule_312,
        "mean_wer": sum(scored_wer) / len(scored_wer) if scored_wer else None,
        "mean_cer": sum(scored_cer) / len(scored_cer) if scored_cer else None,
        "gpu_name": gpu_info.get("gpu_name"),
        "dev_gpu_bypass": os.environ.get("SPELL_DEV_GPU") == "1",
        "finished_utc": _utc_now_iso(),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"[evaluate_track_b] split={args.split} n={summary['n_utterances']} "
          f"WER={summary['mean_wer']} CER={summary['mean_cer']} -> {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
