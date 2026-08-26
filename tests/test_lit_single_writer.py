"""Single-writer regression: LitConformerCTC bookkeeping must be immune to
lightning's callback-before-module hook ordering (observed ≥2.x: a Callback's
``on_validation_epoch_end`` fires BEFORE the module's own).

The historical bug this pins: the module hook used to BUILD+STORE a snapshot and
the bundle callback drained it later — with module-after-callback ordering each
val event drained the PREVIOUS event's snapshot, so (a) sanity-pass rows were
recorded as 'epoch 1', (b) every recorded val epoch shifted forward, and (c) a
phantom final epoch leaked through the on_train_end leftover flush.

Contract enforced here:
  * step hooks APPEND only;
  * module ``on_validation_epoch_end`` touches NO row buffer (parquet feed) —
    it consumes ONLY its private ``_vallog_*`` aggregate scalars (live W&B
    path, restored 2026-08-27 after the empty-post-drain silent-log bug);
  * ``drain_epoch`` is the SOLE consumer of the row buffers;
  * sanity handling resets via ``reset_val_buffers``.
"""

import torch

from lit_track_b import LitConformerCTC


def _tiny_cfg() -> dict:
    return {
        "model": {
            "input_stream": 0, "codebook_size": 64, "d_model": 32,
            "n_conformer_layers": 2, "n_conformer_heads": 4,
            "conformer_ff_mult": 2, "conv_kernel_size": 15, "dropout": 0.0,
        },
        "training": {"lr_peak": 5e-4, "warmup_steps": 10, "weight_decay": 0.0,
                     "grad_clip_norm": 1.0},
        "augment": {"freq_mask_n": 1, "freq_mask_width": 4, "time_mask_n": 1,
                    "time_mask_ratio_max": 0.1},
        "logging": {"grad_norm_log_every_batches": 50},
    }


def _batch(vocab):
    t = torch.randint(0, 64, (2, 3, 12))               # [B, n_streams, T] REAL codes
    lengths = torch.tensor([12, 8])
    mask = torch.arange(12)[None] < lengths[:, None]
    r0, r1 = vocab.encode("hello"), vocab.encode("hi")   # 5 / 2 chars
    width = max(len(r0), len(r1))
    ids = torch.tensor([r0 + [vocab.pad_id] * (width - len(r0)),
                        r1 + [vocab.pad_id] * (width - len(r1))], dtype=torch.int64)
    return {
        "utterance_ids": ["u1", "u2"],
        "tokens": t,
        "lengths": lengths,
        "mask": mask,
        "text_ids": ids,
        "text_lengths": torch.tensor([5, 2]),
    }


def test_module_epoch_end_hook_is_read_only():
    """self.log is INSTANCE-OVERRIDDEN here (detached Module.log raises without
    a Trainer, lightning ≥2.x): this records exactly what the hook would emit,
    which no trainer-attached mock can do more faithfully."""
    lit = LitConformerCTC(_tiny_cfg(), use_augment=False)
    emitted: list[tuple[str, float]] = []
    lit.log = lambda name, value, **kw: emitted.append((name, float(value)))
    batch = _batch(lit.vocab)
    for _ in range(2):                                  # two val batches accumulate
        lit.validation_step(batch, 0)

    before_rows = len(lit._val_rows)
    drops_before = lit._val_drops
    assert lit._vallog_rows == 4                        # aggregate scalars fed live
    lit.on_validation_epoch_end()

    # live W&B series emitted under their exact final names
    assert {n for n, _ in emitted} == {"val/wer", "val/cer", "val/loss"}
    assert all(v >= 0.0 for _, v in emitted)

    assert len(lit._val_rows) == before_rows == 4       # parquet feed untouched
    assert lit._val_drops == drops_before
    assert lit._vallog_rows == 0                        # ...but log scalars consumed

    # ...and the same buffers then feed exactly ONE full record through drain:
    payload = lit.drain_epoch()
    assert len(payload["train"]) == 0                   # train side untouched here
    snap = payload["val"]
    assert snap is not None and snap["n_utterances"] == 4
    assert {r["utterance_id"] for r in snap["rows"]} == {"u1", "u2"}

    # drain emptied everything — a second call yields nothing (no phantom epochs)
    again = lit.drain_epoch()
    assert again["train"] == [] and again["val"] is None


def test_drain_then_callback_order_yields_exactly_one_record_per_event(tmp_path):
    """Adversarial replay of the real lightning sequence around the BundleCallback:
    callback fired BEFORE the module hook must still produce one correctly-labelled
    epoch record per validation pass — never a shifted or duplicated epoch."""
    from ckpt_bundle import METRICS_COLUMNS
    from types import SimpleNamespace

    import pandas as pd

    lit = LitConformerCTC(_tiny_cfg(), use_augment=False)
    live_series: list[str] = []                       # metric NAMES logged live
    lit.log = lambda name, value, **kw: (
        live_series.append(name) if name.startswith("val/") else None)
    batch = _batch(lit.vocab)
    cb = __import__("ckpt_bundle").BundleCallback(
        tmp_path / "run", ckpt_every_epochs=100)
    trainer_stub = SimpleNamespace(current_epoch=0, sanity_checking=False,
                                   state=SimpleNamespace(status="TrainerStatus.RUNNING"))
    trainer_stub.save_checkpoint = lambda filepath, **kw: torch.save({}, str(filepath))

    def one_val_pass(display_epoch_minus_one: int):
        trainer_stub.current_epoch = display_epoch_minus_one
        # lightning order: CALLBACK first, MODULE second (see docstring); steps
        # have already accumulated beforehand either way.
        vals_before = lit._build_val_snapshot()          # simulate completed steps
        assert vals_before is not None
        cb.on_validation_epoch_end(trainer_stub, lit)     # consumer fires first…
        lit.on_validation_epoch_end()                     # …logger-only hook second

    lit.validation_step(batch, 0)                         # epoch-0 validation data
    one_val_pass(0)                                       # → display epoch 1
    lit.validation_step(batch, 1)                         # epoch-1 validation data
    one_val_pass(1)                                       # → display epoch 2

    df = pd.read_parquet(cb.run_dir / "metrics.parquet")
    assert list(df.columns) == METRICS_COLUMNS
    val_wer = df[(df.split == "val") & (df.metric == "wer") & df.utterance_id.isna()]
    assert sorted(val_wer.epoch.tolist()) == [1, 2]       # no sanity bleed, no e3
    # live path fires too, ordering-immune (callback's row-drain ran first each
    # time; the scalar aggregates are module-owned): every epoch-end event emits
    # exactly ONE point of each val series (wer/cer/loss) — 3 names × 2 events.
    assert sorted(live_series) == sorted(
        ["val/wer", "val/cer", "val/loss"] * 2)


def test_reset_val_buffers_discards_sanity_rows():
    lit = LitConformerCTC(_tiny_cfg(), use_augment=False)
    batch = _batch(lit.vocab)
    lit.validation_step(batch, 0)
    assert lit._val_rows
    lit.reset_val_buffers()                              # what BundleCallback calls on sanity
    assert lit._val_rows == [] and lit._val_drops == 0
    assert lit.drain_epoch()["val"] is None              # nothing leaks into records
