#!/usr/bin/env python3
"""Proxy-model universe scorer (roster #4 feed) — GPU job on Ada.

    python scripts/score_proxy.py --bundle runs/track_b/<proxy_id> \
        --out scores/proxy_scores.parquet [--late-frac 0.75]

Scores ALL TRAIN-pool utterances (§5 item 12) under the shared proxy model's
LATE trajectory checkpoints (names >= ceil(max_epoch * late_frac); {15, 20}
for the 20-epoch proxy at late_frac=0.75 — PROVISIONAL window). Per utterance:
mean CTC loss and mean EL2N across late ckpts. These two columns feed the
deterministic lossrank / anti / el2n selectors (§3.20 ×1) via
scripts/make_selector_manifest.py. Seen-data bias for utterances inside the
proxy's own 10% training subset is uniform across all three selectors.
"""

from __future__ import annotations

import argparse
import math
import re
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import ctc as ctc_lib  # noqa: E402
import paths as data_paths  # noqa: E402
from config import load_paths  # noqa: E402
from dataset import (  # noqa: E402
    TokenDataset,
    collate_token_batch,
    filter_records,
    load_id_list,
    load_records,
)
from hardware_guard import enforce_gpu_policy  # noqa: E402
from lit_track_b import LitConformerCTC  # noqa: E402

SCORE_COLUMNS = ["utterance_id", "loss_mean", "el2n_mean", "n_ckpts"]
CKPT_RE = re.compile(r"^ckpt_epoch(\d{4})\.ckpt$")


def _late_ckpts(bundle: str | Path, late_frac: float) -> list[Path]:
    bundle = Path(bundle)
    traj = []
    for p in bundle.glob("ckpt_epoch*.ckpt"):
        m = CKPT_RE.match(p.name)
        if m:
            traj.append((int(m.group(1)), p))
    if not traj:
        raise FileNotFoundError(
            f"no trajectory checkpoints (ckpt_epoch*.ckpt) under {bundle} — "
            f"the proxy run must carry keep_local_traj=1 (§3.7)")
    traj.sort()
    max_epoch = traj[-1][0]
    cutoff = math.ceil(max_epoch * late_frac)
    return [p for e, p in traj if e >= cutoff]


def _load_lit(ckpt_path: str | Path) -> LitConformerCTC:
    return LitConformerCTC.load_from_checkpoint(str(ckpt_path), map_location="cpu")


def _mean_across(rows_per_ckpt: list[dict]) -> dict[str, tuple[float, float]]:
    """{uid: (loss_mean, el2n_mean)} across late ckpts."""
    acc: dict[str, list[tuple[float, float]]] = {}
    for rows in rows_per_ckpt:
        for uid, (lv, ev) in rows.items():
            acc.setdefault(uid, []).append((float(lv), float(ev)))
    return {uid: (float(np.mean([v[0] for v in vals])),
                  float(np.mean([v[1] for v in vals])))
            for uid, vals in acc.items()}


def _collect_train_records() -> list[str]:
    p = load_paths()
    recs = load_records(p.index_path, split="trainval")
    train = load_id_list(Path(p.splits_dir) / "train_ids.txt")
    return sorted(r.utterance_id for r in recs if r.utterance_id in train)


def _preflight(recs, k: int = 50) -> None:
    data_paths.preflight_resolve(recs, k=min(k, len(recs)), kind="tokens")


def _score_one_lit(lit, ids: list[str]) -> dict[str, tuple[float, float]]:
    """Inference pass over the train pool with ONE lit checkpoint."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lit = lit.to(device).eval()
    p = load_paths()
    recs = [r for r in load_records(p.index_path, split="trainval")
            if r.utterance_id in set(ids)]
    recs = filter_records(recs, ids, strict=False)
    vocab = lit.vocab
    collate = partial(collate_token_batch, token_pad_id=-1,
                      text_pad_id=vocab.pad_id)
    dl = DataLoader(TokenDataset(recs, vocab=vocab, include_text=True),
                    batch_size=32, shuffle=False, num_workers=4,
                    collate_fn=collate)
    input_stream = int(lit.cfg["model"].get("input_stream", 0))
    out: dict[str, tuple[float, float]] = {}
    with torch.no_grad():
        for batch in dl:
            keep = ctc_lib.input_length_keep_mask(batch["lengths"],
                                                  batch["text_lengths"])
            idx = torch.nonzero(keep, as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            tokens = batch["tokens"][idx].to(device)
            lengths = batch["lengths"][idx].to(device)
            text_ids = batch["text_ids"][idx].to(device)
            text_lengths = batch["text_lengths"][idx].to(device)
            uids = [batch["utterance_ids"][i] for i in idx.tolist()]
            stream = tokens[:, input_stream, :]
            log_probs, out_lengths = lit.model(stream, lengths)
            loss_vec = ctc_lib.ctc_loss_per_utt(
                log_probs.transpose(0, 1), text_ids, out_lengths, text_lengths,
                lit.blank_id)
            el2n_vec = ctc_lib.el2n_per_utt(log_probs.transpose(0, 1), out_lengths)
            for uid, lv, ev in zip(uids, loss_vec.cpu().tolist(),
                                   el2n_vec.cpu().tolist()):
                out[uid] = (float(lv), float(ev))
    return out


def _score_all(lits, ids: list[str]) -> list[dict]:
    rows_per_ckpt = [_score_one_lit(lit, ids) for lit in lits]
    means = _mean_across(rows_per_ckpt)
    rows = [{"utterance_id": uid, "loss_mean": lv, "el2n_mean": ev,
             "n_ckpts": len(rows_per_ckpt)}
            for uid, (lv, ev) in means.items()]
    return sorted(rows, key=lambda r: r["utterance_id"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--bundle", required=True, type=Path,
                    help="proxy run bundle (runs/track_b/<proxy_id>)")
    ap.add_argument("--out", type=Path, default=root / "scores" / "proxy_scores.parquet")
    ap.add_argument("--late-frac", type=float, default=0.75,
                    help="keep ckpts with epoch >= ceil(max_epoch * late_frac)")
    args = ap.parse_args(argv)

    enforce_gpu_policy()
    ckpts = _late_ckpts(args.bundle, args.late_frac)
    print(f"[proxy] late ckpts: {[Path(c).name for c in ckpts]}", flush=True)
    ids = _collect_train_records()
    recs = [r for r in load_records(load_paths().index_path, split="trainval")
            if r.utterance_id in set(ids)]
    _preflight(recs, k=50)
    lits = [_load_lit(c) for c in ckpts]
    rows = _score_all(lits, ids)
    df = pd.DataFrame(rows, columns=SCORE_COLUMNS
                      ).sort_values("utterance_id", kind="mergesort"
                                    ).reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"[proxy] wrote {args.out}: rows={len(df)} "
          f"(loss_mean min={df['loss_mean'].min():.4f} "
          f"max={df['loss_mean'].max():.4f})")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as e:
        print(f"FATAL[score_proxy] {e}", file=sys.stderr)
        raise SystemExit(2)