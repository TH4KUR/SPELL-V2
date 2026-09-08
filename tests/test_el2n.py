"""EL2N per-utterance logging extension (proxy-model law, PROTOCOL §3.19 era).

The proxy run needs per-utterance error-L2-norm (EL2N; Paul et al. 2021, "Deep
Learning on a Data Diet") over late epochs — the disclosed CTC adaptation is
‖softmax(logp) − best-path one-hot‖₂ averaged over valid output frames. Laws:
  * the metric is pure CTC math, verified against a hand-computed case;
  * frames beyond out_length never contribute (§2.4 pad law);
  * accrual is GATED by cfg["logging"]["el2n_log"] — formal runs keep identical
    bookkeeping (no buffer churn, no metrics.parquet rows, no new W&B series);
  * rows land in metrics.parquet as per-(utt, epoch) metric="el2n" — schema
    unchanged, per-utterance data never reaches W&B (§3.17).
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import ctc as ctc_lib  # noqa: E402
from ckpt_bundle import BundleCallback  # noqa: E402
from dataset import TokenDataset, collate_token_batch  # noqa: E402
from lit_track_b import LitConformerCTC  # noqa: E402


def test_el2n_hand_computed_case():
    """Single frame, three classes: p = softmax([2,0,0]); best path = class 0;
    EL2N = ‖p − onehot(0)‖₂ = sqrt((1-p0)² + p1² + p2²)."""
    frame = torch.log_softmax(torch.tensor([2.0, 0.0, 0.0]), dim=-1)
    logp = frame.view(1, 1, 3)                       # [T=1, B=1, V=3]
    el2n = ctc_lib.el2n_per_utt(logp, torch.tensor([1]))
    z = torch.exp(torch.tensor(2.0)) + 2.0
    p0, p1, p2 = (float(torch.exp(torch.tensor(2.0)) / z), float(1.0 / z),
                  float(1.0 / z))
    expected = ((1 - p0) ** 2 + p1**2 + p2**2) ** 0.5
    assert el2n.shape == (1,)
    assert float(el2n[0]) == pytest.approx(expected, rel=1e-6)


def test_el2n_averages_valid_frames_only():
    """Frames beyond out_length are excluded (pad law); per-row values match an
    independent recomputation with different indexing."""
    torch.manual_seed(0)
    T, B, V = 5, 2, 7
    logits = torch.randn(T, B, V)
    logp = torch.log_softmax(logits, dim=-1)
    lengths = torch.tensor([5, 2])
    el2n = ctc_lib.el2n_per_utt(logp, lengths)
    assert el2n.shape == (2,)
    probs = logp.exp()                               # [T,B,V]
    best = logits.argmax(dim=-1)                     # [T,B]
    for i, L in enumerate((5, 2)):
        onehot = torch.zeros(L, V).scatter_(1, best[:L, i:i + 1], 1.0)
        expected = ((probs[:L, i] - onehot) ** 2).sum(-1).sqrt().mean()
        assert float(el2n[i]) == pytest.approx(float(expected), rel=1e-6)


def test_el2n_near_zero_when_confident():
    """A dominant logit drives the error vector toward zero — the sanity check
    that the norm is computed against the best path, not something else."""
    logits = torch.full((3, 1, 5), -50.0)
    logits[:, 0, 2] = 50.0                           # EVERY frame dominant on class 2
    logp = torch.log_softmax(logits, dim=-1)
    el2n = ctc_lib.el2n_per_utt(logp, torch.tensor([3]))
    assert float(el2n[0]) < 1e-5


def _tiny_cfg(el2n_log: bool) -> dict:
    return {
        "model": {"input_stream": 0, "codebook_size": 64, "d_model": 32,
                  "n_conformer_layers": 1, "n_conformer_heads": 4,
                  "conformer_ff_mult": 2, "conv_kernel_size": 15, "dropout": 0.0},
        "training": {"lr_peak": 5e-4, "warmup_steps": 10, "weight_decay": 0.0,
                     "grad_clip_norm": 1.0},
        "augment": {"freq_mask_n": 1, "freq_mask_width": 4, "time_mask_n": 1,
                    "time_mask_ratio_max": 0.1},
        "logging": {"grad_norm_log_every_batches": 50,
                    **({"el2n_log": True} if el2n_log else {})},
    }


def _fake_staged(tmp_path):
    """Four staged-layout utterances whose char targets fit inside T=12 frames
    (nothing dropped by the §3.12 input-length rule)."""
    from dataset import UtteranceRecord

    root = tmp_path / "data"
    texts = ["hello", "hi", "ok", "yes"]
    recs = []
    for i, text in enumerate(texts):
        vid, stem = f"v{i}", f"{50000 + i}"
        p = root / vid / f"{stem}.tokens.pt"
        p.parent.mkdir(parents=True)
        torch.save(torch.randint(0, 64, (3, 12)), p)
        recs.append(UtteranceRecord(
            utterance_id=f"{vid}/{stem}", split="trainval", video_id=vid,
            stem=stem, tokens_path=str(p), audio_path=None, audio_kind=None,
            txt_path=None, n_tokens=12, duration_s=12 / 50.0, conf=5,
            text_raw=text.upper(), text_norm=text, n_chars_norm=len(text),
        ))
    return recs


def _fit_one_epoch(tmp_path, el2n_log: bool):
    """One real epoch under BundleCallback; returns (lit, bundle_dir, spy)."""
    holder = LitConformerCTC(_tiny_cfg(el2n_log), use_augment=False)
    recs = _fake_staged(tmp_path)
    make_ds = lambda: TokenDataset(                  # noqa: E731
        recs, vocab=holder.vocab, include_text=True,
        path_resolver=lambda r: Path(r.tokens_path))
    collate = partial(collate_token_batch, token_pad_id=-1,
                      text_pad_id=holder.vocab.pad_id)
    train_dl = DataLoader(make_ds(), batch_size=4, collate_fn=collate)
    val_dl = DataLoader(make_ds(), batch_size=4, collate_fn=collate)

    from test_val_wandb_series import SpyLogger
    spy = SpyLogger()
    lit = LitConformerCTC(_tiny_cfg(el2n_log), use_augment=False)
    bundle = tmp_path / "bundle"
    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", devices=1, logger=spy,
        enable_checkpointing=False, enable_progress_bar=False,
        enable_model_summary=False, num_sanity_val_steps=2, log_every_n_steps=1,
        callbacks=[BundleCallback(bundle, ckpt_every_epochs=5)],
    )
    trainer.fit(lit, train_dl, val_dl)
    return lit, bundle, spy


def test_gated_fit_writes_el2n_rows_but_no_wandb_series(tmp_path):
    """el2n_log=true: per-utt el2n rows reach metrics.parquet; NOTHING el2n-like
    reaches the logger (§3.17: per-utterance data never streams to W&B)."""
    lit, bundle, spy = _fit_one_epoch(tmp_path, el2n_log=True)
    df = pd.read_parquet(bundle / "metrics.parquet")
    el2n_rows = df[df["metric"] == "el2n"]
    assert set(el2n_rows["split"]) == {"train"}
    assert el2n_rows["utterance_id"].notna().all()
    assert len(el2n_rows) == 4                       # one per utterance
    assert ((el2n_rows["value"] >= 0) & (el2n_rows["value"] <= 1.0)).all()
    for metrics in spy.series:
        assert not any("el2n" in k for k in metrics), f"el2n reached W&B: {metrics}"


def test_ungated_fit_writes_no_el2n_rows(tmp_path):
    """el2n_log absent (formal runs): metrics.parquet identical in kind to the
    pre-extension contract — zero el2n rows."""
    _, bundle, _ = _fit_one_epoch(tmp_path, el2n_log=False)
    df = pd.read_parquet(bundle / "metrics.parquet")
    assert not (df["metric"] == "el2n").any()


def test_rows_for_epoch_extension_is_backward_compatible():
    """Old-style drain dicts (no train_el2n key) still build rows unchanged."""
    data = {"train": [("a/1", 1.5)], "train_drops": 2, "val": None}
    rows = BundleCallback._rows_for_epoch(data, 7)
    assert all(r["metric"] != "el2n" for r in rows)
    data["train_el2n"] = [("a/1", 0.25)]
    rows = BundleCallback._rows_for_epoch(data, 7)
    el2n = [r for r in rows if r["metric"] == "el2n"]
    assert el2n == [{"epoch": 7, "split": "train", "utterance_id": "a/1",
                     "metric": "el2n", "value": 0.25}]