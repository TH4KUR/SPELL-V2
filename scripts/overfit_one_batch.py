#!/usr/bin/env python3
"""Overfit-one-batch smoke (PLAN.md hard requirement before any real run).

Takes ONE small batch of real utterances and drives the full pipeline
(data → guarded embed → Conformer → CTC → backward) with a hand-rolled loop until
the gate passes:

    exit 0  ⇔  mean CTC loss < 0.1  AND  greedy char accuracy > 99%

This is a DEV tool by definition: it force-enables the SPELL_DEV_GPU bypass with
its own loud banner (PROTOCOL §3.16) and runs augmentation OFF (use_augment=False)
so the model must memorize the exact batch.

    python scripts/overfit_one_batch.py [--steps 400] [--batch-utts 8]
"""

from __future__ import annotations

import argparse
import os
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

import ctc as ctc_lib                        # noqa: E402
from config import PROJECT_ROOT, load_config, load_paths  # noqa: E402
from dataset import (                         # noqa: E402
    TokenDataset,
    collate_token_batch,
    load_records,
)
from hardware_guard import enforce_gpu_policy          # noqa: E402
import paths as data_paths                             # noqa: E402  DATA LAYOUT LAW
from lit_track_b import LitConformerCTC        # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--batch-utts", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--loss-gate", type=float, default=0.1)
    ap.add_argument("--acc-gate", type=float, default=99.0)
    return ap.parse_args(argv)


def pick_short_batch(records, n_utts: int):
    """Real utterances of moderate length — long enough for real words, short
    enough that 200–400 steps can memorize them."""
    usable = [r for r in records if 48 <= r.n_tokens <= 320][:200]
    if len(usable) < n_utts:
        raise RuntimeError(f"only {len(usable)} usable moderate-length records found")
    return usable[:n_utts]


def main(argv=None) -> int:
    args = parse_args(argv)

    # This script exists ONLY as a dev bring-up tool → force the dev bypass on,
    # loudly, unless the user already opted in.
    if os.environ.get("SPELL_DEV_GPU", "") != "1":
        os.environ["SPELL_DEV_GPU"] = "1"
        print("[overfit_one_batch] auto-setting SPELL_DEV_GPU=1 (dev-only script)")
    enforce_gpu_policy()

    cfg = load_config(PROJECT_ROOT / "configs" / "track_b.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("[overfit_one_batch] WARNING: no CUDA device — running on CPU will be slow")

    paths = load_paths()
    recs = load_records(paths.index_path, split="trainval")
    recs = [r for r in recs if r.text_norm][:4000]
    use = pick_short_batch(recs, args.batch_utts)

    # PREFLIGHT (§10 item 4): fail here, loudly, not mid-memorization.
    dp = data_paths.current()
    print(f"[spell] data_root={dp.root} layout={dp.layout}", flush=True)
    data_paths.preflight_resolve(use, k=len(use))

    print(f"[overfit_one_batch] utterances:")
    for r in use:
        print(f"   {r.utterance_id}  T={r.n_tokens}  {r.text_norm[:60]!r}")

    from vocab import build_char_vocab
    vocab = build_char_vocab()
    ds = TokenDataset(use, vocab=vocab, include_text=True)
    collate = partial(collate_token_batch, token_pad_id=-1, text_pad_id=vocab.pad_id)
    batch = collate([ds[i] for i in range(len(ds))])

    lit = LitConformerCTC(cfg, use_augment=False).to(device)   # memorization ⇒ no augment

    tokens = batch["tokens"].to(device)
    lengths = batch["lengths"].to(device)
    text_ids = batch["text_ids"].to(device)
    text_lengths = batch["text_lengths"].to(device)

    opt = torch.optim.AdamW(lit.parameters(), lr=args.lr, weight_decay=0.0)
    history: list[float] = []

    lit.train()
    passed = False
    last_loss = float("inf")
    last_acc = 0.0
    for step in range(1, args.steps + 1):
        opt.zero_grad(set_to_none=True)
        stream = tokens[:, lit.input_stream, :]
        log_probs, out_lengths = lit.model(stream, lengths)
        loss_vec = ctc_lib.ctc_loss_per_utt(
            log_probs.transpose(0, 1), text_ids, out_lengths, text_lengths, vocab.blank_id
        )
        loss = loss_vec.mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(lit.parameters(), 5.0)
        opt.step()
        last_loss = float(loss.item())
        history.append(last_loss)

        if step % 25 == 0 or step == 1:
            hyps = ctc_lib.greedy_decode(log_probs.detach(), lengths, vocab)
            ref_ids = [t.tolist() for t in text_ids]
            last_acc = 100.0 * ctc_lib.char_accuracy(ref_ids, hyps, vocab)
            print(f"[step {step:>4}] loss={last_loss:.4f} char_acc={last_acc:.2f}%")

        # gate check rides the same cadence (decode cost is nonzero)
        if step % 25 == 0 and last_loss < args.loss_gate and last_acc > args.acc_gate:
            passed = True
            break

    verdict = (
        f"{'PASS' if passed else 'FAIL'}: final loss={last_loss:.4f} "
        f"(gate <{args.loss_gate}) char_acc={last_acc:.2f}% (gate >{args.acc_gate})"
    )
    print(f"[overfit_one_batch] {verdict}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
