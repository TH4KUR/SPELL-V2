"""Bundle contract (PROTOCOL §5.2): long-format metrics.parquet contents,
trajectory checkpoint naming cadence, atomic-file discipline, ≤20-file cap,
sanity-check suppression and the COMPLETED-marker semantics."""

from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as pl
import pandas as pd
import pytest
import torch

from ckpt_bundle import METRICS_COLUMNS, BundleCallback


# --------------------------------------------------------------- stub plumbing

class _StubTrainer:
    """Just enough Trainer surface for ckpt_bundle: epoch counter, sanity flag,
    status, save_checkpoint."""

    def __init__(self, epoch: int = 0, sanity: bool = False, status: str = "FINISHED"):
        self.current_epoch = epoch
        self.sanity_checking = sanity
        self.state = SimpleNamespace(status=status)

    def save_checkpoint(self, filepath, **kwargs):
        torch.save({"stub_weights_only": bool(kwargs.get("save_weights_only", False))},
                   str(filepath))


class DrainableModule(pl.LightningModule):
    """LightningModule exposing lit_track_b's drain_epoch()/reset_val_buffers()
    contract; payloads are CONSUMED on drain (sole consumer ⇒ order-immune)."""

    def __init__(self, payload):
        super().__init__()
        self._payload = payload
        self.sanity_resets = 0

    def stage_payload(self, payload):
        self._payload = payload

    def drain_epoch(self):
        p, self._payload = self._payload, {"train": [], "train_drops": 0, "val": None}
        return p

    def reset_val_buffers(self):
        self.sanity_resets += 1
        self._payload = {"train": [], "train_drops": 0, "val": None}


def _payload(uid_losses=(("utt_a", 0.5), ("utt_b", 0.75)), drops=2,
             val=None) -> dict:
    return {"train": list(uid_losses), "train_drops": drops, "val": val}


def _empty_payload():
    return {"train": [], "train_drops": 0, "val": None}


_VAL_SNAP = {
    "wer": 0.42,
    "cer": 0.30,
    "loss": 1.25,
    "n_utterances": 2,
    "dropped": 1,
    "rows": [
        {"utterance_id": "val_x", "wer": 0.5, "cer": 0.4, "loss": 1.1},
        {"utterance_id": "", "wer": None, "cer": None, "loss": 1.4},  # empty-ref row
    ],
}


# ------------------------------------------------------------ _rows_for_epoch

def test_rows_for_epoch_full_contents():
    data = _payload(val=_VAL_SNAP)
    rows = BundleCallback._rows_for_epoch(data, epoch=7)

    def key(r):
        return (r["split"], r["utterance_id"], r["metric"], r["value"])

    got = {key(r) for r in rows}
    expected = {
        ("train", "utt_a", "loss", 0.5),
        ("train", "utt_b", "loss", 0.75),
        ("train", None, "dropped_utts", 2.0),
        ("val", None, "dropped_utts", 1.0),
        ("val", None, "loss", 1.25),
        ("val", None, "wer", 0.42),
        ("val", None, "cer", 0.30),
        ("val", "val_x", "wer", 0.5),
        ("val", "val_x", "cer", 0.4),
        ("val", "val_x", "loss", 1.1),
        ("val", "", "loss", 1.4),                      # None-valued keys omitted
    }
    assert got == expected
    for r in rows:
        assert set(r.keys()) == set(METRICS_COLUMNS)
        assert isinstance(r["value"], float)


def test_rows_for_epoch_zero_train_drops_emit_no_row():
    rows = BundleCallback._rows_for_epoch(_payload(drops=0), epoch=1)
    assert all(not (r["metric"] == "dropped_utts" and r["split"] == "train")
               for r in rows)


def test_rows_for_epoch_without_validation_snapshot():
    rows = BundleCallback._rows_for_epoch(_payload(), epoch=3)
    assert all(r["split"] == "train" for r in rows)
    assert len(rows) == 3                              # 2 uid rows + drop bookkeeping


# ---------------------------------------------------------- callback lifecycle

def test_write_cycle_metrics_trajectory_and_cap(tmp_path):
    run_dir = tmp_path / "run"
    cb = BundleCallback(run_dir, ckpt_every_epochs=5)
    mod = DrainableModule(_empty_payload())
    trainer = _StubTrainer()

    # Lightning sanity pass writes NOTHING and wipes stale accumulators
    trainer.sanity_checking = True
    cb.on_validation_epoch_end(trainer, mod)
    assert not (run_dir / "metrics.parquet").exists()
    assert mod.sanity_resets == 1
    trainer.sanity_checking = False

    for e in range(1, 13):                             # display epochs 1..12
        trainer.current_epoch = e - 1
        mod.stage_payload(_payload(val=_VAL_SNAP))     # val aggregates every epoch
        cb.on_validation_epoch_end(trainer, mod)

    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert list(df.columns) == METRICS_COLUMNS
    # every epoch carries the PER-(utterance, epoch) mean train CTC loss rows…
    train_uid_rows = df[(df.split == "train") & (df.metric == "loss")
                        & (df.utterance_id == "utt_a")]
    assert sorted(train_uid_rows.epoch.tolist()) == list(range(1, 13))
    # …plus §3.12 drop accounting per epoch
    drops = df[(df.split == "train") & (df.metric == "dropped_utts")]
    assert len(drops) == 12 and (drops.value == 2.0).all()
    # display-space epochs start at 1 (current_epoch + 1)
    assert df.epoch.min() == 1 and df.epoch.max() == 12
    # val aggregates landed too
    val_wer = df[(df.split == "val") & (df.metric == "wer") & (df.utterance_id.isna())]
    assert len(val_wer) == 12

    # trajectory cadence: 5-boundary epochs only (12 → 0005, 0010)
    for present in ("ckpt_epoch0005.ckpt", "ckpt_epoch0010.ckpt", "last.ckpt"):
        assert (run_dir / present).exists(), present
    for absent in ("ckpt_epoch0004.ckpt", "ckpt_epoch0001.ckpt",
                   "ckpt_epoch0012.ckpt"):
        assert not (run_dir / absent).exists(), absent

    files = [p.name for p in run_dir.rglob("*") if p.is_file()]
    assert sorted(files) == ["ckpt_epoch0005.ckpt", "ckpt_epoch0010.ckpt",
                             "last.ckpt", "metrics.parquet"]     # ≤ 20-cap by far


def test_checkpoint_stub_records_weights_only_flag(tmp_path):
    """The LESS trajectories must be weights-ONLY while last.ckpt carries full
    training state — observable through what we ask save_checkpoint to do.
    (2026-08-27 incident: this used to bless the non-existent kwarg
    ``save_weights_only``, which a **kwargs stub cannot catch.)"""
    cb = BundleCallback(tmp_path / "r", ckpt_every_epochs=5)
    log: list[tuple[str, bool]] = []
    seen_kwargs: dict[str, tuple] = {}

    class LoggingTrainer(_StubTrainer):
        def save_checkpoint(self, filepath, **kwargs):
            # ckpt_bundle writes via `<dest>.tmp` + os.replace — strip the suffix
            p = str(filepath)
            if p.endswith(".tmp"):
                p = p[: -len(".tmp")]
            seen_kwargs[Path(p).name] = tuple(kwargs.keys())
            torch.save({}, str(filepath))

    tr = LoggingTrainer(epoch=4)                       # display epoch 5 hits trajectory
    mod = DrainableModule(_empty_payload())
    cb.on_validation_epoch_end(tr, mod)

    assert "weights_only" in seen_kwargs["ckpt_epoch0005.ckpt"]     # REAL param name
    assert "save_weights_only" not in seen_kwargs["ckpt_epoch0005.ckpt"]
    assert seen_kwargs["last.ckpt"] == ()                            # full state: bare call


def test_real_trainer_checkpoint_branches(tmp_path):
    """Integration pin for the 2026-08-27 pilot-A crash: the periodic
    trajectory branch had NEVER executed end-to-end anywhere before Ada
    (laptop smoke ran 2 epochs < every-5; unit stubs accepted any kwargs).
    This drives a REAL lightning.Trainer through BOTH save branches so any
    API drift fails here, on CPU, before any cluster submission."""
    class FitModule(DrainableModule):
        def __init__(self):
            super().__init__(_empty_payload())
            self.net = torch.nn.Linear(3, 1)

        def training_step(self, batch, _idx):
            loss = (self.net(batch[0].float()) ** 2).mean()
            self.stage_payload({"train": [("t0", float(loss))],
                                "train_drops": 0, "val": None})
            return loss

        def validation_step(self, batch, _idx):
            return None

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-2)

    from torch.utils.data import DataLoader, TensorDataset

    dl = DataLoader(TensorDataset(torch.randn(8, 3)), batch_size=4)

    class FitModule(DrainableModule):
        def __init__(self):
            super().__init__(_empty_payload())
            self.net = torch.nn.Linear(3, 1)

        def training_step(self, batch, _idx):
            loss = (self.net(batch[0].float()) ** 2).mean()
            self.stage_payload({"train": [("t0", float(loss))],
                                "train_drops": 0, "val": None})
            return loss

        def validation_step(self, batch, _idx):
            return None

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-2)

    run_dir = tmp_path / "fitrun"
    cb = BundleCallback(run_dir, ckpt_every_epochs=2)
    trainer = pl.Trainer(
        max_epochs=3, accelerator="cpu", devices=1,
        callbacks=[cb], logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(FitModule(), dl, dl)

    # cadence: periodic ONLY at display epoch 2; rolling everywhere
    assert (run_dir / "ckpt_epoch0002.ckpt").exists()
    assert not (run_dir / "ckpt_epoch0001.ckpt").exists()
    assert not (run_dir / "ckpt_epoch0003.ckpt").exists()
    assert (run_dir / "last.ckpt").exists()
    # periodic snapshot is WEIGHTS-ONLY; last.ckpt is the FULL resumable state
    traj = torch.load(run_dir / "ckpt_epoch0002.ckpt", map_location="cpu",
                      weights_only=False)
    last = torch.load(run_dir / "last.ckpt", map_location="cpu",
                      weights_only=False)
    assert "state_dict" in traj and "optimizer_states" not in traj
    assert "optimizer_states" in last
    # metrics flow + clean-finish marker
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert len(df[(df.split == "train") & (df.metric == "loss")]) >= 3
    assert (run_dir / "COMPLETED").exists()


def test_completed_marker_semantics(tmp_path):
    # The hook fires while status still reads RUNNING (flag flips after fit);
    # 'not STOPPED' ⇒ clean. Real statuses stringify as "TrainerStatus.X".
    d1 = tmp_path / "running"
    cb1 = BundleCallback(d1)
    cb1.on_train_end(_StubTrainer(status="TrainerStatus.RUNNING"),
                     DrainableModule(_payload()))       # leftover drain must not crash
    assert (d1 / "COMPLETED").exists()

    d2 = tmp_path / "finished"
    cb2 = BundleCallback(d2)
    cb2.on_train_end(_StubTrainer(status="TrainerStatus.FINISHED"),
                     DrainableModule(_empty_payload()))
    assert (d2 / "COMPLETED").exists()

    # interrupt ⇒ deliberately ABSENT so drainers skip the bundle
    d3 = tmp_path / "aborted"
    cb3 = BundleCallback(d3)
    cb3.on_train_end(_StubTrainer(status="TrainerStatus.STOPPED"),
                     DrainableModule(_empty_payload()))
    assert not (d3 / "COMPLETED").exists()


def test_file_cap_enforced_at_construction(tmp_path):
    for i in range(21):
        (tmp_path / f"junk_{i:02d}.bin").write_bytes(b"x")
    with pytest.raises(RuntimeError, match="cap"):
        BundleCallback(tmp_path)


def test_file_cap_enforced_mid_run(tmp_path):
    cb = BundleCallback(tmp_path / "grow", ckpt_every_epochs=100)
    mod = DrainableModule(_empty_payload())
    tr = _StubTrainer(epoch=0)
    cb.on_validation_epoch_end(tr, mod)                # healthy baseline epoch
    for i in range(25):                                # inode junk appears afterwards
        (tmp_path / "grow" / f"junk{i}.bin").write_bytes(b"x")
    tr.current_epoch = 1
    with pytest.raises(RuntimeError, match="cap"):
        cb.on_validation_epoch_end(tr, mod)


def test_constructor_rejects_nonpositive_cadence(tmp_path):
    with pytest.raises(ValueError, match="ckpt_every_epochs"):
        BundleCallback(tmp_path / "bad", ckpt_every_epochs=0)
