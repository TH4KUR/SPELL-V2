"""Bundle contract (PROTOCOL §5.2): long-format metrics.parquet contents,
trajectory checkpoint naming cadence, atomic-file discipline, ≤20-file cap,
sanity-check suppression, and the SINGLE-WRITER COMPLETED law (§10 item 9):
the callback never attests completion — only train_track_b.attest_completed
does, strictly after durable provenance."""

from pathlib import Path
from types import SimpleNamespace

import lightning.pytorch as pl
import pandas as pd
import pytest
import torch

from ckpt_bundle import METRICS_COLUMNS, BundleCallback
from scripts.train_track_b import attest_completed   # THE one sanctioned writer


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
    # metrics flow; single-writer law: a clean real fit does NOT attest by
    # itself — the callback wrote train rows, and no COMPLETED exists anywhere;
    df = pd.read_parquet(run_dir / "metrics.parquet")
    assert len(df[(df.split == "train") & (df.metric == "loss")]) >= 3
    assert not (run_dir / "COMPLETED").exists()
    # (attest_completed success-path pinning lives in
    # test_attestation_refuses_incomplete_provenance[healthy twin] and the
    # drain-validation subprocess suite — this fit has no val side.)


def test_callback_never_attests_completion(tmp_path):
    """Single-writer law (§10 item 9): BundleCallback.on_train_end writes NO
    COMPLETED under ANY status. Attestation belongs solely to the training
    entrypoint's last statement — the old status-sniffing here was also dead
    code (lightning has no STOPPED state; interrupts carry INTERRUPTED)."""
    for i, status in enumerate(("TrainerStatus.RUNNING",       # hook fires with this
                                "TrainerStatus.FINISHED",      # post-teardown value
                                "TrainerStatus.INTERRUPTED")): # real interrupt value
        d = tmp_path / f"status{i}"
        BundleCallback(d).on_train_end(
            _StubTrainer(status=status),
            DrainableModule(_payload()))         # leftover drain must not crash
        assert not (d / "COMPLETED").exists(), status


def _run_tiny_fit(run_dir, *, max_epochs: int = 1,
                  boom_after_batch: int | None = None):
    """Tiny REAL-Trainer CPU fit against the real bundle callback.
    ``boom_after_batch`` makes training_step RAISE on the n-th batch+1 call —
    a synthetic mid-training failure."""
    from torch.utils.data import DataLoader, TensorDataset

    class FitModule(DrainableModule):
        def __init__(self):
            super().__init__(_empty_payload())
            self.net = torch.nn.Linear(3, 1)
            self.calls = 0

        def training_step(self, batch, _idx):
            self.calls += 1
            if boom_after_batch is not None and self.calls >= boom_after_batch:
                raise RuntimeError("synthetic mid-run training failure")
            loss = (self.net(batch[0].float()) ** 2).mean()
            self.stage_payload({"train": [(f"t{self.calls}", float(loss))],
                                "train_drops": 0, "val": None})
            return loss

        def validation_step(self, batch, _idx):
            return None

        def configure_optimizers(self):
            return torch.optim.SGD(self.parameters(), lr=1e-2)

    dl = DataLoader(TensorDataset(torch.randn(8, 3)), batch_size=4)
    cb = BundleCallback(run_dir, ckpt_every_epochs=100)
    trainer = pl.Trainer(
        max_epochs=max_epochs, accelerator="cpu", devices=1,
        callbacks=[cb], logger=False, enable_checkpointing=False,
        enable_progress_bar=False, enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(FitModule(), dl, dl)
    return cb


def test_crashed_fit_leaves_no_marker(tmp_path):
    """Boundary pin: fit raising mid-training unwinds past on_train_end BY
    DESIGN of lightning's fit loops (no finally around on_run_end) — and under
    the single-writer law nothing may leave a COMPLETED behind regardless."""
    run_dir = tmp_path / "crashrun"
    with pytest.raises(RuntimeError, match="synthetic"):
        _run_tiny_fit(run_dir, max_epochs=1, boom_after_batch=2)

    assert list(run_dir.rglob("COMPLETED")) == []     # nowhere in the tree
    # whatever partial artifacts materialized stay INTACT but UNTRUSTED:
    for fname in ("metrics.parquet", "last.ckpt"):
        p = run_dir / fname
        if p.exists():
            assert p.is_file()
    # ...and attestation refuses loudly on that provenance — never marks.
    with pytest.raises(RuntimeError, match="REFUSING"):
        attest_completed(run_dir)
    assert not (run_dir / "COMPLETED").exists()


def test_attestation_refuses_incomplete_provenance(tmp_path):
    """attest_completed is the single writer AND a provenance gate: missing /
    empty metrics, or a missing last.ckpt each refuse WITHOUT creating the
    filename anywhere in the bundle tree."""
    base = tmp_path / "b"
    d_no_parquet = base / "no_parquet"
    d_empty = base / "empty_rows"
    d_train_only = base / "train_only"
    d_no_ckpt = base / "no_ckpt"
    healthy = base / "healthy"

    rows = [
        {"epoch": 1, "split": "train", "utterance_id": "u1", "metric": "loss", "value": 0.5},
        {"epoch": 1, "split": "val", "utterance_id": None, "metric": "wer", "value": 0.4},
        {"epoch": 1, "split": "val", "utterance_id": None, "metric": "loss", "value": 1.25},
    ]

    BundleCallback(d_no_parquet)                       # creates the dir only
    for d in (d_empty, d_train_only, healthy):
        d.mkdir(parents=True)
    pd.DataFrame(columns=METRICS_COLUMNS).to_parquet(d_empty / "metrics.parquet")
    pd.DataFrame([rows[0]]).to_parquet(d_train_only / "metrics.parquet")
    torch.save({"state_dict": {}}, d_train_only / "last.ckpt")
    d_no_ckpt.mkdir(parents=True)
    pd.DataFrame(rows).to_parquet(d_no_ckpt / "metrics.parquet")   # ckpt absent
    torch.save({"state_dict": {}}, healthy / "last.ckpt")
    pd.DataFrame(rows).to_parquet(healthy / "metrics.parquet")

    cases = [
        (d_no_parquet, r"metrics\.parquet \(missing\)"),
        (d_empty, "epoch row per split"),
        (d_train_only, "epoch row per split"),         # val side absent ⇒ both-splits rule
        (d_no_ckpt, r"last\.ckpt \(missing\)"),
    ]
    for d, fragment in cases:
        with pytest.raises(RuntimeError, match=fragment):
            attest_completed(d)
        assert list(d.rglob("COMPLETED")) == [], d.name

    # the healthy twin passes and marks — isolating the gate from the writer
    marker = attest_completed(healthy)
    assert marker == healthy / "COMPLETED" and marker.exists()


def test_clean_fit_returning_with_hollow_metrics_refuses(tmp_path):
    """The user's boundary case (fit RETURNS cleanly but the final metrics
    payload failed/was lost): the last-statement attestation inspects DURABLE
    state, not fit's return code — a schema-correct but row-less parquet gets
    REFUSED, and no COMPLETED can exist."""
    run_dir = tmp_path / "hollow-after-fit"
    cb = _run_tiny_fit(run_dir, max_epochs=1)
    assert cb.run_dir == run_dir                       # sanity on the helper seam
    # simulate lost final durability: replace with zero-row, correct-schema table
    pd.DataFrame(columns=METRICS_COLUMNS).to_parquet(run_dir / "metrics.parquet")
    torch.save({"state_dict": {"w": torch.zeros(1)}}, run_dir / "last.ckpt")
    assert not (run_dir / "COMPLETED").exists()
    with pytest.raises(RuntimeError, match="epoch row per split"):
        attest_completed(run_dir)
    assert list(run_dir.rglob("COMPLETED")) == []


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
