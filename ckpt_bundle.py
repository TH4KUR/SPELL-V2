"""Run-bundle callback implementing the Ada relay storage contract (PROTOCOL §5.2).

A run directory (bundle) contains EXACTLY:
  ``ckpt_epoch{NNNN}.ckpt``   every ``ckpt_every_epochs`` epochs, WEIGHTS ONLY —
                              trajectory snapshots feeding LESS (Phase 4);
  ``last.ckpt``               rolling full state (optimizer/schedulers), atomic;
  ``metrics.parquet``         ONE long-format table rewritten atomically each epoch:
                              columns [epoch, split, utterance_id, metric, value]
                              including PER-(utterance, epoch) mean train CTC loss
                              (Phase-4 loss ranking / Oracle-RHO feedstock) plus
                              aggregate + ``dropped_utts`` bookkeeping rows (§3.12);
  ``run_manifest.json``       written by the training script via ``manifest.py``;
  ``COMPLETED``               NEVER written here. The marker is TRAINER-ATTESTED
                              success (single-writer law, PROTOCOL §10 item 9): the
                              training entrypoint touches it IN-PROCESS as its last
                              statement after durable metrics — this callback and all
                              shell traps are cleanup-only. Crash/interrupt ⇒ bundle
                              reaches ``drain_runs.sh`` unmarked AND unvalidated.
A hard ≤ ``files_cap`` file assertion runs at construction and every epoch so inode
discipline violations surface immediately, not at drain time.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import pandas as pd

METRICS_COLUMNS = ["epoch", "split", "utterance_id", "metric", "value"]


class BundleCallback(pl.Callback):
    def __init__(
        self,
        run_dir: str | Path,
        *,
        ckpt_every_epochs: int = 5,
        metrics_filename: str = "metrics.parquet",
        files_cap: int = 20,
    ):
        super().__init__()
        if ckpt_every_epochs < 1:
            raise ValueError("ckpt_every_epochs must be >= 1")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_every_epochs = int(ckpt_every_epochs)
        self.metrics_path = self.run_dir / metrics_filename
        self.files_cap = int(files_cap)
        self.rows: list[dict[str, Any]] = []
        self._assert_file_cap()                       # fail fast on pre-existing junk

    # ------------------------------------------------------------------ files
    def _bundle_files(self) -> list[Path]:
        return [
            p for p in self.run_dir.rglob("*") if p.is_file() and p.name != "COMPLETED"
        ]

    def _assert_file_cap(self) -> None:
        files = self._bundle_files()
        if len(files) > self.files_cap:
            names = "\n  ".join(sorted(p.name for p in files)[:25])
            raise RuntimeError(
                f"bundle {self.run_dir} holds {len(files)} files > cap {self.files_cap} "
                f"(relay inode discipline, PROTOCOL §5.2). Contents:\n  {names}"
            )

    def _write_metrics(self) -> None:
        df = pd.DataFrame(self.rows, columns=METRICS_COLUMNS)
        tmp = self.metrics_path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, self.metrics_path)            # drainers never see partials

    @staticmethod
    def _save(trainer: pl.Trainer, dest: Path, **save_kwargs) -> None:
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        trainer.save_checkpoint(tmp, **save_kwargs)
        os.replace(tmp, dest)

    # ------------------------------------------------------------- row building
    @staticmethod
    def _rows_for_epoch(data: dict, epoch: int) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for uid, v in data["train"]:
            rows.append({"epoch": epoch, "split": "train", "utterance_id": uid,
                         "metric": "loss", "value": float(v)})
        if data["train_drops"]:
            rows.append({"epoch": epoch, "split": "train", "utterance_id": None,
                         "metric": "dropped_utts", "value": float(data["train_drops"])})
        snap = data["val"]
        if snap is not None:
            if snap.get("dropped"):
                rows.append({"epoch": epoch, "split": "val", "utterance_id": None,
                             "metric": "dropped_utts", "value": float(snap["dropped"])})
            for key in ("loss", "wer", "cer"):
                if snap.get(key) is not None:
                    rows.append({"epoch": epoch, "split": "val", "utterance_id": None,
                                 "metric": key, "value": float(snap[key])})
            for r in snap["rows"]:
                for key in ("loss", "wer", "cer"):
                    if r.get(key) is not None:
                        rows.append({"epoch": epoch, "split": "val",
                                     "utterance_id": r["utterance_id"],
                                     "metric": key, "value": float(r[key])})
        return rows

    # ------------------------------------------------------------------ hooks
    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if trainer.sanity_checking:
            reset = getattr(pl_module, "reset_val_buffers", None)
            if callable(reset):                       # drop sanity-pass rows entirely
                reset()
            return
        e = trainer.current_epoch + 1                 # display space: epochs start at 1
        new_rows = self._rows_for_epoch(pl_module.drain_epoch(), e)
        if new_rows:
            self.rows.extend(new_rows)
            self._write_metrics()

        if e % self.ckpt_every_epochs == 0:           # LESS trajectory snapshot
            self._save(trainer, self.run_dir / f"ckpt_epoch{e:04d}.ckpt",
                       weights_only=True)             # Trainer.save_checkpoint's real name
        self._save(trainer, self.run_dir / "last.ckpt")   # full state, resumable
        self._assert_file_cap()

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # final flush in case the last epoch ended without a validation pass
        leftover = self._rows_for_epoch(pl_module.drain_epoch(), trainer.current_epoch + 1)
        if leftover:
            self.rows.extend(leftover)
            self._write_metrics()
        self._assert_file_cap()
        # NOTE: this callback does NOT write COMPLETED, under any status. The
        # single-writer law (PROTOCOL §10 item 9) makes the training entrypoint's
        # last statement the only sanctioned attestation: fit-loop teardown fires
        # on_train_end unreliably on exceptions BY DESIGN (no finally around
        # on_run_end in lightning's _FitLoop.run), so a marker written anywhere on
        # the Lightning-hook path would re-create exactly the ambiguity that let
        # crashed bypass bundles pose as good ones.
