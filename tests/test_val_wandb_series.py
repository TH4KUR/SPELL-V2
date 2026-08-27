"""LIVE validation-series regression (2026-08-27 incident).

Pilot A trained fine and metrics.parquet carried full per-utterance val rows,
yet W&B showed NO ``val/*`` series: lightning fires the CALLBACK epoch-end
hook before the module's, BundleCallback drained the row buffers first, and
the module-side re-read was permanently empty — silently killing every
``self.log("val/…")``. The fix (ownership split: rows=callback-drained,
scalar aggregates=module-owned) is pinned here with a REAL one-epoch fit
against a spy logger — the exact boundary WandbLogger sits at — including
the sanity-suppression guarantee (partial sanity-pass data must NEVER emit).
"""

import sys
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import lightning.pytorch as pl
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import Logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import paths as paths_mod  # noqa: E402
from dataset import TokenDataset, collate_token_batch  # noqa: E402
from lit_track_b import LitConformerCTC  # noqa: E402


class SpyLogger(Logger):
    """Records every metrics/hparams dict reaching the logger boundary."""

    def __init__(self):
        super().__init__()
        self.series: list[dict] = []
        self.hparams: list[dict] = []

    @property
    def name(self) -> str:
        return "spy"

    @property
    def version(self) -> str:
        return "v0"

    def log_hyperparams(self, params, *args, **kwargs):
        self.hparams.append(dict(params))

    def log_metrics(self, metrics, step=None):
        self.series.append(dict(metrics))


def _tiny_cfg() -> dict:
    return {
        "model": {
            "input_stream": 0, "codebook_size": 64, "d_model": 32,
            "n_conformer_layers": 1, "n_conformer_heads": 4,
            "conformer_ff_mult": 2, "conv_kernel_size": 15, "dropout": 0.0,
        },
        "training": {"lr_peak": 5e-4, "warmup_steps": 10, "weight_decay": 0.0,
                     "grad_clip_norm": 1.0},
        "augment": {"freq_mask_n": 1, "freq_mask_width": 4, "time_mask_n": 1,
                    "time_mask_ratio_max": 0.1},
        "logging": {"grad_norm_log_every_batches": 50},
    }


def _fake_staged(tmp_path):
    """Four staged-layout utterances with texts whose char targets fit inside
    T=12 frames (so nothing is dropped by the §3.12 input-length rule)."""
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


def test_one_epoch_fit_emits_live_val_series(tmp_path):
    holder = LitConformerCTC(_tiny_cfg(), use_augment=False)   # vocab scaffold
    recs = _fake_staged(tmp_path)
    make_ds = lambda: TokenDataset(                            # noqa: E731
        recs, vocab=holder.vocab, include_text=True,
        path_resolver=lambda r: Path(r.tokens_path))
    collate = partial(collate_token_batch, token_pad_id=-1,
                      text_pad_id=holder.vocab.pad_id)
    train_dl = DataLoader(make_ds(), batch_size=4, collate_fn=collate)
    val_dl = DataLoader(make_ds(), batch_size=4, collate_fn=collate)

    spy = SpyLogger()
    lit = LitConformerCTC(_tiny_cfg(), use_augment=False)
    trainer = pl.Trainer(
        max_epochs=1, accelerator="cpu", devices=1,
        logger=spy, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=2,                       # exercise sanity suppression
        log_every_n_steps=1,
        # same assembly as scripts/train_track_b.py (§3.17 logging contract)
        callbacks=[LearningRateMonitor(logging_interval="step")],
    )
    trainer.fit(lit, train_dl, val_dl)

    # THE regression pin: live series reach the logger boundary under their
    # exact W&B names — once per fit (sanity pass suppressed), never zero.
    wer_points = [r for r in spy.series if "val/wer" in r]
    loss_points = [r for r in spy.series if "val/loss" in r]
    cer_points = [r for r in spy.series if "val/cer" in r]
    assert len(wer_points) == 1, [sorted(r) for r in spy.series]
    assert len(loss_points) == 1 and len(cer_points) == 1

    v = wer_points[0]["val/wer"]
    assert isinstance(v, float) and v >= 0.0          # a REAL point, not None/nan
    assert loss_points[0]["val/loss"] > 0.0

    # §3.17 contract surface: the rest of the mandatory series arrives at the
    # same boundary — step-level train loss and the realized LR schedule (no
    # LR series exists unless a LearningRateMonitor is actually attached).
    assert any("train/loss_step" in r for r in spy.series)
    lr_keys = {k for r in spy.series for k in r if k.lower().startswith("lr")}
    assert lr_keys, f"no LR series emitted: keys seen={sorted({k for r in spy.series for k in r})}"
    assert all(float(v) >= 0.0 for r in spy.series
               for k, v in r.items() if k.lower().startswith("lr"))

    # hyperparameters ride the same logger at fit start:
    assert len(spy.hparams) == 1 and "cfg" in spy.hparams[0]

    # and the trainer-visible metric the run manifest reads at finish:
    assert "val/wer" in trainer.callback_metrics
