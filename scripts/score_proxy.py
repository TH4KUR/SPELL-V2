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
import time
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


def _score_one_lit(lit, ids: list[str], report_every_s: float = 30.0,
                   on_progress=None) -> dict[str, tuple[float, float]]:
    """Inference pass over the train pool with ONE lit checkpoint.

    `on_progress(done, total, chunk_s, elapsed_s)` (if given) fires at
    least every `report_every_s` seconds of WALL-CLOCK time (plus once more
    at the very end) — PROTOCOL §3.24, at the finer WITHIN-checkpoint
    granularity (this is where most of a single checkpoint's wall-clock
    goes, not just at the "checkpoint done" level). A count-based threshold
    was tried first and silently assumed a throughput rate; time-based
    reporting doesn't need to know the rate in advance to stay live."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    lit = lit.to(device).eval()
    p = load_paths()
    recs = [r for r in load_records(p.index_path, split="trainval")
            if r.utterance_id in set(ids)]
    recs = filter_records(recs, set(ids), strict=False)
    vocab = lit.vocab
    collate = partial(collate_token_batch, token_pad_id=-1,
                      text_pad_id=vocab.pad_id)
    dl = DataLoader(TokenDataset(recs, vocab=vocab, include_text=True),
                    batch_size=32, shuffle=False, num_workers=4,
                    collate_fn=collate)
    input_stream = int(lit.cfg["model"].get("input_stream", 0))
    out: dict[str, tuple[float, float]] = {}
    n_total = len(recs)
    t_start = time.monotonic()
    t_prev = t_start
    t_last_report = t_start

    def _maybe_report(force: bool = False) -> None:
        nonlocal t_prev, t_last_report
        if on_progress is None:
            return
        now = time.monotonic()
        if force or now - t_last_report >= report_every_s:
            on_progress(len(out), n_total, now - t_prev, now - t_start)
            t_prev = now
            t_last_report = now

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
            _maybe_report()
    _maybe_report(force=True)
    return out


def _score_all(lits, ids: list[str], report_every_s: float = 30.0,
               on_ckpt_done=None) -> list[dict]:
    """`on_ckpt_done(ckpt_i, n_ckpts, done, total, chunk_s, elapsed_s)` (if
    given) is threaded down into each checkpoint's own within-checkpoint
    progress callback, so the caller sees BOTH which checkpoint is in
    flight and how far it has gotten through the train pool — previously
    this was `[_score_one_lit(lit, ids) for lit in lits]` with zero
    visibility across the whole pass."""
    rows_per_ckpt = []
    n_ckpts = len(lits)
    for i, lit in enumerate(lits, start=1):
        def _progress(done, total, chunk_s, elapsed_s, _i=i):
            if on_ckpt_done is not None:
                on_ckpt_done(_i, n_ckpts, done, total, chunk_s, elapsed_s)

        rows_per_ckpt.append(_score_one_lit(lit, ids, report_every_s=report_every_s,
                                            on_progress=_progress))
    means = _mean_across(rows_per_ckpt)
    rows = [{"utterance_id": uid, "loss_mean": lv, "el2n_mean": ev,
             "n_ckpts": len(rows_per_ckpt)}
            for uid, (lv, ev) in means.items()]
    return sorted(rows, key=lambda r: r["utterance_id"])


def _wandb_init(*, project: str, entity: str | None, mode: str, job_id: str,
                bundle: str, late_frac: float, n_recs: int):
    """Opens the W&B run EARLY (before the checkpoint pass starts), per
    PROTOCOL §3.23/§3.24 — this scorer's sibling (score_dnsmos.py) has
    already run into its SLURM --time wall with zero W&B trace of progress;
    this closes the identical gap here."""
    import wandb

    run = wandb.init(project=project, entity=entity, mode=mode,
                     job_type="proxy_scoring",
                     name=f"proxy-scoring-{job_id}",
                     dir=str(Path(__file__).resolve().parents[1] / "outputs" / "wandb"),
                     tags=["proxy", "roster-4"])
    wandb.config.update({"bundle": str(bundle), "late_frac": late_frac,
                        "n_recs": n_recs})
    return run


def _log_scoring_progress(ckpt_i: int, n_ckpts: int, done: int, total: int,
                          chunk_s: float, elapsed_s: float) -> None:
    import wandb

    rate = done / elapsed_s if elapsed_s > 0 else 0.0
    eta_ckpt_s = (total - done) / rate if rate > 0 else float("nan")
    wandb.log({"proxy_scoring/ckpt": ckpt_i,
              "proxy_scoring/ckpts_total": n_ckpts,
              "proxy_scoring/utts_done": done,
              "proxy_scoring/utts_total": total,
              "proxy_scoring/chunk_duration_s": chunk_s,
              "proxy_scoring/elapsed_s": elapsed_s,
              "proxy_scoring/utts_per_s": rate,
              "proxy_scoring/eta_this_ckpt_s": eta_ckpt_s})
    pct = 100.0 * done / total if total else 0.0
    eta_str = f"{eta_ckpt_s / 60:.1f} min" if eta_ckpt_s == eta_ckpt_s else "unknown"
    print(f"[proxy] ckpt {ckpt_i}/{n_ckpts}: {done}/{total} ({pct:.1f}%) scored "
          f"— +{chunk_s:.0f}s since last report, {elapsed_s / 60:.1f} min elapsed "
          f"this ckpt, ~{rate:.2f} utt/s, ETA (this ckpt) {eta_str}", flush=True)


def log_final_summary_to_wandb(df: pd.DataFrame) -> None:
    """Logged onto the ALREADY-OPEN run from _wandb_init — composes with the
    live per-chunk progress above rather than replacing it."""
    import wandb

    wandb.run.summary["proxy_scoring/loss_mean_min"] = float(df["loss_mean"].min())
    wandb.run.summary["proxy_scoring/loss_mean_max"] = float(df["loss_mean"].max())
    wandb.run.summary["proxy_scoring/el2n_mean_min"] = float(df["el2n_mean"].min())
    wandb.run.summary["proxy_scoring/el2n_mean_max"] = float(df["el2n_mean"].max())
    wandb.run.summary["proxy_scoring/n_rows"] = int(len(df))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--bundle", required=True, type=Path,
                    help="proxy run bundle (runs/track_b/<proxy_id>)")
    ap.add_argument("--out", type=Path, default=root / "scores" / "proxy_scores.parquet")
    ap.add_argument("--late-frac", type=float, default=0.75,
                    help="keep ckpts with epoch >= ceil(max_epoch * late_frac)")
    ap.add_argument("--report-every-s", type=float, default=30.0,
                    help="minimum wall-clock seconds between progress reports")
    ap.add_argument("--wandb-project", default="spell-rq2")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                    default="online")
    ap.add_argument("--job-id", default="local",
                    help="SLURM job id or other run tag for the W&B run name")
    args = ap.parse_args(argv)

    ids = _collect_train_records()

    # opened BEFORE the expensive pass, not after -- so partial progress
    # (and a crash, or a SLURM --time wall) are both visible in W&B
    run = _wandb_init(project=args.wandb_project, entity=args.wandb_entity,
                      mode=args.wandb_mode, job_id=args.job_id,
                      bundle=str(args.bundle), late_frac=args.late_frac,
                      n_recs=len(ids))
    try:
        enforce_gpu_policy()
        ckpts = _late_ckpts(args.bundle, args.late_frac)
        print(f"[proxy] late ckpts: {[Path(c).name for c in ckpts]}", flush=True)
        recs = [r for r in load_records(load_paths().index_path, split="trainval")
                if r.utterance_id in set(ids)]
        _preflight(recs, k=50)
        lits = [_load_lit(c) for c in ckpts]
        rows = _score_all(lits, ids, report_every_s=args.report_every_s,
                          on_ckpt_done=_log_scoring_progress)
        df = pd.DataFrame(rows, columns=SCORE_COLUMNS
                          ).sort_values("utterance_id", kind="mergesort"
                                        ).reset_index(drop=True)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(args.out, index=False)
        print(f"[proxy] wrote {args.out}: rows={len(df)} "
              f"(loss_mean min={df['loss_mean'].min():.4f} "
              f"max={df['loss_mean'].max():.4f})")
        log_final_summary_to_wandb(df)
        return 0
    finally:
        # exit_code reflects whether we're unwinding due to an exception --
        # run.finish() with no args always marks the run "Finished" even
        # when the body crashed (caught live, 2026-09-18: job 2701229
        # crashed on filter_records's list/set type mismatch below, but
        # still showed as a completed run in the W&B UI).
        run.finish(exit_code=1 if sys.exc_info()[0] is not None else 0)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError) as e:
        print(f"FATAL[score_proxy] {e}", file=sys.stderr)
        raise SystemExit(2)