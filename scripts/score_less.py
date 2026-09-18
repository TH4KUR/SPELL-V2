#!/usr/bin/env python3
"""LESS scorer (roster #5 — CTC-grad variant, Track B native) — GPU job on Ada.

    python scripts/score_less.py --bundle runs/track_b/<proxy_id> \
        --out-dir scores --seeds 201 202 203 [--proj-dim 512] [--batch-size 16]

Per TRAIN utterance: "influence" = alignment between that utterance's CTC-loss
gradient and a validation-set reference gradient, estimated across the shared
proxy model's FULL trajectory (ALL ckpt_epoch*.ckpt in the bundle — unlike
score_proxy.py's LATE-only window; LESS wants the trajectory, not a converged
snapshot). Writes scores/less_influence_seed{S}.parquet per seed. These feed
the deterministic-per-seed `less_ctc` selector (§3.20 ×3) via
scripts/make_selector_manifest.py.

Gradient proxy: rather than the exact per-utterance gradient of `head`'s
weight (which would need either B backward passes per batch or a hand-derived
log-softmax Jacobian), this uses the gradient of the per-utterance CTC loss
w.r.t. `feats` — the 256-dim activation immediately BEFORE `head`, captured
via a forward hook (no changes to model.py). This is exact, not approximate:
`torch.autograd.grad(loss_per_utt.sum(), feats)` never re-enters the encoder
that produced `feats` (autograd.grad halts traversal at the requested input),
and everything BETWEEN feats and the loss (`head`'s nn.Linear, per-row
log_softmax, masked_fill keyed on that row's own length, and F.ctc_loss's
per-sequence DP) is row-wise with zero cross-batch coupling — so one backward
call on the whole batch's SUMMED loss gives every row's true INDIVIDUAL
gradient, with zero cross-contamination. Proven numerically (batched vs.
naive per-example processing) in tests/test_score_less.py, not just asserted.

The per-utterance gradient is mean-pooled over valid time steps (§2.4 pad
law), then projected with a seeded RADEMACHER (+-1) matrix (§3.20: the seed's
entire job is to manufacture the required seed-to-seed diversity — 256 dims
is already tiny, so the projection is not doing compression here). Influence
= dot(projected train vector, projected val-reference vector), averaged
across trajectory checkpoints. Val vectors are aggregated into ONE mean
reference vector per checkpoint and never materialized per-utterance — never
a keyed structure — a structural guarantee (not just a filter) that no val
row can leak into the committed table (§3.21: val is reference-only).

IMPORTANT: unlike score_proxy.py's inference loop, this script's forward
passes must NOT run under torch.no_grad() — gradients are the entire point.
"""

from __future__ import annotations

import argparse
import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import ctc as ctc_lib  # noqa: E402
from config import load_paths  # noqa: E402
from dataset import TokenDataset, collate_token_batch, load_id_list, load_records  # noqa: E402
from hardware_guard import enforce_gpu_policy  # noqa: E402
from scripts.score_proxy import (  # noqa: E402
    CKPT_RE,
    _collect_train_records,
    _load_lit,
    _preflight,
)

SCORE_COLUMNS = ["utterance_id", "influence", "n_ckpts"]


def _all_ckpts(bundle: str | Path) -> list[Path]:
    """FULL trajectory (every ckpt_epoch*.ckpt, sorted by epoch) — LESS wants
    the whole training arc, not a late/converged-only window."""
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
    return [p for _, p in traj]


def _collect_val_records() -> list[str]:
    """Sorted val-split ids — symmetric to score_proxy._collect_train_records,
    filtered against val_ids.txt instead of train_ids.txt. Reference-only
    (§3.21): these ids must never appear in the committed table."""
    p = load_paths()
    recs = load_records(p.index_path, split="trainval")
    val = load_id_list(Path(p.splits_dir) / "val_ids.txt")
    return sorted(r.utterance_id for r in recs if r.utterance_id in val)


def _grad_feats_hook(lit) -> tuple[torch.utils.hooks.RemovableHandle, dict]:
    """Captures `head`'s input tensor (`feats`) on every forward call, without
    touching model.py. Asserts the calling convention it depends on, so a
    future refactor to that convention fails loudly instead of silently
    pooling garbage."""
    head = lit.model.head
    if not isinstance(head, nn.Linear):
        raise TypeError(f"expected lit.model.head to be nn.Linear, got {type(head)}")
    captured: dict = {}

    def _hook(module, inputs, output):
        feats = inputs[0]
        if feats.dim() != 3 or feats.shape[-1] != module.in_features:
            raise RuntimeError(
                f"head input shape {tuple(feats.shape)} does not match the "
                f"expected [B,T,{module.in_features}] convention this hook "
                f"depends on — model.py's forward may have changed")
        captured["feats"] = feats

    handle = head.register_forward_hook(_hook)
    return handle, captured


def _batch_pooled_grads(lit, batch: dict, captured: dict, device: str
                        ) -> tuple[list[str], torch.Tensor]:
    """One grad-ENABLED forward+backward pass over one batch -> per-utterance
    mean-pooled gradient of the CTC loss w.r.t. the pre-head activation.

    Deliberately NOT wrapped in torch.no_grad() — that is score_proxy.py's
    idiom for inference-only scoring and would silently zero out every
    gradient here if copy-pasted.
    """
    keep = ctc_lib.input_length_keep_mask(batch["lengths"], batch["text_lengths"])
    idx = torch.nonzero(keep, as_tuple=False).flatten()
    if idx.numel() == 0:
        return [], torch.empty(0, lit.model.head.in_features)

    tokens = batch["tokens"][idx].to(device)
    lengths = batch["lengths"][idx].to(device)
    text_ids = batch["text_ids"][idx].to(device)
    text_lengths = batch["text_lengths"][idx].to(device)
    uids = [batch["utterance_ids"][i] for i in idx.tolist()]

    input_stream = int(lit.cfg["model"].get("input_stream", 0))
    stream = tokens[:, input_stream, :]
    log_probs, out_lengths = lit.model(stream, lengths)
    loss_vec = ctc_lib.ctc_loss_per_utt(
        log_probs.transpose(0, 1), text_ids, out_lengths, text_lengths, lit.blank_id)

    feats = captured["feats"]
    grad_feats = torch.autograd.grad(loss_vec.sum(), feats, retain_graph=False)[0]

    tmax = grad_feats.shape[1]
    mask = torch.arange(tmax, device=device).unsqueeze(0) < out_lengths.unsqueeze(1)
    denom = out_lengths.clamp(min=1).to(grad_feats.dtype).unsqueeze(-1)
    pooled = (grad_feats * mask.unsqueeze(-1)).sum(dim=1) / denom
    return uids, pooled.detach()


def _pooled_grad_vecs_for_ids(lit, ids: list[str], batch_size: int = 16,
                              device: str | None = None
                              ) -> dict[str, np.ndarray]:
    """TRAIN-side loop: {uid: pooled_grad_vec} — becomes the committed rows."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lit = lit.to(device).eval()
    recs = [r for r in load_records(load_paths().index_path, split="trainval")
            if r.utterance_id in set(ids)]
    collate = partial(collate_token_batch, token_pad_id=-1,
                      text_pad_id=lit.vocab.pad_id)
    dl = DataLoader(TokenDataset(recs, vocab=lit.vocab, include_text=True),
                    batch_size=batch_size, shuffle=False, num_workers=4,
                    collate_fn=collate)
    out: dict[str, np.ndarray] = {}
    handle, captured = _grad_feats_hook(lit)
    try:
        for batch in dl:
            uids, pooled = _batch_pooled_grads(lit, batch, captured, device)
            for uid, vec in zip(uids, pooled.cpu().numpy()):
                out[uid] = vec
    finally:
        handle.remove()
    return out


def _ref_vec_for_ids(lit, ids: list[str], batch_size: int = 16,
                     device: str | None = None) -> np.ndarray:
    """VAL-side loop: a bare mean vector, NEVER a keyed per-utterance
    structure — the structural §3.21 guarantee (nothing to leak downstream)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    lit = lit.to(device).eval()
    recs = [r for r in load_records(load_paths().index_path, split="trainval")
            if r.utterance_id in set(ids)]
    collate = partial(collate_token_batch, token_pad_id=-1,
                      text_pad_id=lit.vocab.pad_id)
    dl = DataLoader(TokenDataset(recs, vocab=lit.vocab, include_text=True),
                    batch_size=batch_size, shuffle=False, num_workers=4,
                    collate_fn=collate)
    d_model = lit.model.head.in_features
    sum_vec = np.zeros(d_model, dtype=np.float64)
    count = 0
    handle, captured = _grad_feats_hook(lit)
    try:
        for batch in dl:
            uids, pooled = _batch_pooled_grads(lit, batch, captured, device)
            if not uids:
                continue
            sum_vec += pooled.cpu().numpy().sum(axis=0)
            count += len(uids)
    finally:
        handle.remove()
    if count == 0:
        raise ValueError("no val utterances survived scoring — cannot build a reference")
    return (sum_vec / count).astype(np.float32)


def _grad_vecs_all_ckpts(lits, train_ids: list[str], val_ids: list[str],
                         batch_size: int = 16, on_ckpt_done=None
                         ) -> tuple[list[dict[str, np.ndarray]], list[np.ndarray]]:
    """The expensive, SEED-INDEPENDENT pass: one (train_vecs, ref_vec) pair
    per trajectory checkpoint. This is the part worth ~10s-of-minutes, so
    `on_ckpt_done(i, n_ckpts, ckpt_s, elapsed_s)` (if given) fires after EACH
    checkpoint -- callers use it for LIVE progress (W&B, print), not just a
    single report after everything is done. `ckpt_s` is THAT checkpoint's
    own duration (not just cumulative `elapsed_s`) -- PROTOCOL §3.24: an
    average hides a single slow or stuck unit, so per-unit timing is
    reported on its own, not folded into the running total."""
    train_vecs_per_ckpt = []
    ref_vec_per_ckpt = []
    t_start = time.monotonic()
    t_prev = t_start
    for i, lit in enumerate(lits):
        print(f"[less] checkpoint {i + 1}/{len(lits)}: scoring train pool...", flush=True)
        train_vecs_per_ckpt.append(_pooled_grad_vecs_for_ids(lit, train_ids, batch_size))
        print(f"[less] checkpoint {i + 1}/{len(lits)}: scoring val reference...", flush=True)
        ref_vec_per_ckpt.append(_ref_vec_for_ids(lit, val_ids, batch_size))
        now = time.monotonic()
        if on_ckpt_done is not None:
            on_ckpt_done(i + 1, len(lits), now - t_prev, now - t_start)
        t_prev = now
    return train_vecs_per_ckpt, ref_vec_per_ckpt


def _projection_matrix(seed: int, in_dim: int, proj_dim: int) -> np.ndarray:
    """Rademacher (+-1) projection, CPU-generator-seeded (reproducible
    independent of which GPU Ada happens to schedule the job onto)."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    bits = torch.randint(0, 2, (in_dim, proj_dim), generator=g)
    P = (bits.float() * 2 - 1) / (proj_dim ** 0.5)
    return P.numpy()


def _mean_across_ckpts(rows_per_ckpt: list[dict[str, float]]) -> dict[str, float]:
    acc: dict[str, list[float]] = {}
    for rows in rows_per_ckpt:
        for uid, v in rows.items():
            acc.setdefault(uid, []).append(float(v))
    return {uid: float(np.mean(vals)) for uid, vals in acc.items()}


def _influence_for_seed(train_vecs_per_ckpt: list[dict[str, np.ndarray]],
                        ref_vec_per_ckpt: list[np.ndarray], seed: int,
                        in_dim: int, proj_dim: int) -> dict[str, float]:
    """Cheap, seed-dependent part: one shared projection, one matmul per
    checkpoint (vectorized — no per-utterance Python loop)."""
    P = _projection_matrix(seed, in_dim, proj_dim)
    rows_per_ckpt = []
    for train_vecs, ref_vec in zip(train_vecs_per_ckpt, ref_vec_per_ckpt):
        uids = sorted(train_vecs)
        X = np.stack([train_vecs[u] for u in uids])
        scores = (X @ P) @ (ref_vec @ P)
        rows_per_ckpt.append(dict(zip(uids, scores.tolist())))
    return _mean_across_ckpts(rows_per_ckpt)


def _spearman(a: dict[str, float], b: dict[str, float]) -> float:
    """Rank correlation between two seeds' influence over their shared ids —
    the cross-seed sanity check: the raw projected values differ by
    construction (different Rademacher draw), but the RANKING they induce
    should broadly agree if the projection is preserving real signal rather
    than injecting noise that dominates it."""
    uids = sorted(set(a) & set(b))
    rho, _ = spearmanr([a[u] for u in uids], [b[u] for u in uids])
    return float(rho)


def _wandb_init(*, project: str, entity: str | None, mode: str, job_id: str,
                bundle: str, seeds: list[int], proj_dim: int, batch_size: int):
    """Opens the W&B run EARLY (before the expensive pass starts), not just
    at the end — SPELL-RQ2 policy (PROTOCOL §3.23): runs crucial to the
    research get logged, including partial progress if the job dies partway
    through. Scoped OUTSIDE the §3.17 formal-run contract: distinct
    job_type, less_scoring/*-prefixed series, never touching the frozen
    train/val series names."""
    import wandb

    run = wandb.init(project=project, entity=entity, mode=mode,
                     job_type="less_scoring",
                     name=f"less-scoring-{job_id}",
                     dir=str(Path(__file__).resolve().parents[1] / "outputs" / "wandb"),
                     tags=["less", "ctc-grad", "roster-5"])
    wandb.config.update({"bundle": str(bundle), "seeds": sorted(seeds),
                        "proj_dim": proj_dim, "batch_size": batch_size})
    return run


def _log_ckpt_progress(i: int, n_ckpts: int, ckpt_s: float, elapsed_s: float) -> None:
    """Fired after EACH trajectory checkpoint by _grad_vecs_all_ckpts's
    on_ckpt_done -- this is what makes the run show live progress instead of
    going dark until everything finishes. `ckpt_s` (this checkpoint's OWN
    duration) is reported alongside `elapsed_s` (cumulative) per PROTOCOL
    §3.24 -- a stuck/slow checkpoint must be visible on its own, not
    averaged away by the running total."""
    import wandb

    wandb.log({"less_scoring/checkpoints_done": i,
              "less_scoring/checkpoints_total": n_ckpts,
              "less_scoring/ckpt_duration_s": ckpt_s,
              "less_scoring/elapsed_s": elapsed_s})
    print(f"[less] checkpoint {i}/{n_ckpts} done in {ckpt_s:.0f}s "
          f"({elapsed_s:.0f}s total elapsed)", flush=True)


def log_seed_summary_to_wandb(seed_influences: dict[int, dict[str, float]]) -> None:
    """Per-seed influence stats + cross-seed Spearman agreement, logged onto
    the ALREADY-OPEN run from _wandb_init — this function does not init or
    finish the run itself, so it composes with the live progress logging
    above rather than replacing it."""
    import wandb

    for seed, influence in seed_influences.items():
        vals = np.array(list(influence.values()), dtype=float)
        wandb.log({
            f"less_scoring/influence_min_seed{seed}": float(vals.min()),
            f"less_scoring/influence_max_seed{seed}": float(vals.max()),
            f"less_scoring/influence_mean_seed{seed}": float(vals.mean()),
            f"less_scoring/influence_std_seed{seed}": float(vals.std()),
            f"less_scoring/n_rows_seed{seed}": int(vals.size),
        })
    seeds_sorted = sorted(seed_influences)
    for i in range(len(seeds_sorted)):
        for j in range(i + 1, len(seeds_sorted)):
            s1, s2 = seeds_sorted[i], seeds_sorted[j]
            rho = _spearman(seed_influences[s1], seed_influences[s2])
            wandb.run.summary[f"less_scoring/spearman_seed{s1}_vs_seed{s2}"] = rho
            print(f"[less] spearman(seed{s1}, seed{s2}) = {rho:.4f}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--bundle", required=True, type=Path,
                    help="proxy run bundle (runs/track_b/<proxy_id>)")
    ap.add_argument("--out-dir", type=Path, default=root / "scores")
    ap.add_argument("--seeds", type=int, nargs="+", default=[201, 202, 203],
                    help="§3.20 subset-identity seeds driving the Rademacher projection")
    ap.add_argument("--proj-dim", type=int, default=512,
                    help="PROVISIONAL — 256-dim input is already tiny; this "
                    "projection manufactures seed diversity, not compression")
    ap.add_argument("--batch-size", type=int, default=16,
                    help="lower than score_proxy's 32: grad-enabled forward "
                    "keeps more activation memory alive than no_grad inference")
    ap.add_argument("--wandb-project", default="spell-rq2")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"],
                    default="online")
    ap.add_argument("--job-id", default="local",
                    help="SLURM job id or other run tag for the W&B run name")
    args = ap.parse_args(argv)

    # opened BEFORE the expensive pass, not after -- so partial progress
    # (and a crash) are both visible in W&B, not just a final report
    run = _wandb_init(project=args.wandb_project, entity=args.wandb_entity,
                      mode=args.wandb_mode, job_id=args.job_id,
                      bundle=str(args.bundle), seeds=args.seeds,
                      proj_dim=args.proj_dim, batch_size=args.batch_size)
    try:
        enforce_gpu_policy()
        ckpts = _all_ckpts(args.bundle)
        print(f"[less] full trajectory: {[Path(c).name for c in ckpts]}", flush=True)

        train_ids = _collect_train_records()
        val_ids = _collect_val_records()
        train_recs = [r for r in load_records(load_paths().index_path, split="trainval")
                     if r.utterance_id in set(train_ids)]
        val_recs = [r for r in load_records(load_paths().index_path, split="trainval")
                   if r.utterance_id in set(val_ids)]
        _preflight(train_recs, k=50)
        _preflight(val_recs, k=50)

        lits = [_load_lit(c) for c in ckpts]
        in_dim = lits[0].model.head.in_features
        train_vecs_per_ckpt, ref_vec_per_ckpt = _grad_vecs_all_ckpts(
            lits, train_ids, val_ids, args.batch_size, on_ckpt_done=_log_ckpt_progress)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        val_set = set(val_ids)
        seed_influences: dict[int, dict[str, float]] = {}
        for seed in args.seeds:
            influence = _influence_for_seed(train_vecs_per_ckpt, ref_vec_per_ckpt,
                                            seed, in_dim, args.proj_dim)
            leaked = set(influence) & val_set
            if leaked:
                raise ValueError(
                    f"val-leak: {len(leaked)} val ids present in the about-to-be-"
                    f"written table (§3.21) — refusing to write: "
                    f"{sorted(leaked)[:5]}")
            seed_influences[int(seed)] = influence
            rows = [{"utterance_id": uid, "influence": v,
                    "n_ckpts": len(train_vecs_per_ckpt)}
                   for uid, v in influence.items()]
            df = pd.DataFrame(rows, columns=SCORE_COLUMNS).sort_values(
                "utterance_id", kind="mergesort").reset_index(drop=True)
            out = args.out_dir / f"less_influence_seed{int(seed)}.parquet"
            df.to_parquet(out, index=False)
            print(f"[less] seed={seed}: wrote {out} rows={len(df)} "
                  f"(influence min={df['influence'].min():.4f} "
                  f"max={df['influence'].max():.4f})")

        log_seed_summary_to_wandb(seed_influences)
        return 0
    finally:
        run.finish()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, FileNotFoundError, TypeError, RuntimeError) as e:
        print(f"FATAL[score_less] {e}", file=sys.stderr)
        raise SystemExit(2)
