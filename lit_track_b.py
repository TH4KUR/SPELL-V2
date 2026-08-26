"""PyTorch-Lightning training module for Track B (Conformer-CTC, RVQ₁ stream).

Responsibilities here are deliberately narrow — everything storage-shaped lives in
``ckpt_bundle.BundleCallback``, everything identity-shaped in ``manifest.write_run_manifest``.

Protocol hooks:
- Reads ONLY stream ``model.input_stream`` (= dim-0 index 0 = RVQ₁, §2.3).
- **Input-length rule (§3.12)**: rows whose token length < char-target length are
  dropped before CTC; counts accumulate into the per-epoch record handed to the
  bundle callback (which writes them into metrics.parquet loudly).
- Per-(utterance, step) train losses accumulate so the callback can persist a
  PER-(utterance, EPOCH) mean train CTC loss — the Phase-4 loss-ranking / Oracle-RHO
  feedstock (approved amendment).
- Gradient L2 norm logged every ``logging.grad_norm_log_every_batches`` steps (§3.7).
- Schedule: AdamW + linear warmup then linear decay over ALL optimizer steps;
  constants come verbatim from the merged config and are ABSOLUTE (§3.15) — never
  scaled by subset size.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import lightning.pytorch as pl
import torch
from torch import Tensor

import ctc as ctc_lib
from augment import TokenSpecAugment
from model import ConformerCTC
from vocab import build_char_vocab


class LitConformerCTC(pl.LightningModule):
    """``cfg`` is the MERGED YAML mapping (see ``config.load_config``). Saved into
    checkpoints so evaluation can rebuild the exact model."""

    def __init__(self, cfg: dict[str, Any], *, use_augment: bool = True):
        super().__init__()
        m = cfg["model"]
        self.cfg = cfg
        self.use_augment = use_augment
        self.save_hyperparameters({"cfg": cfg, "use_augment": use_augment})

        self.vocab = build_char_vocab()
        self.blank_id = self.vocab.blank_id

        augment = TokenSpecAugment(**cfg["augment"]) if (use_augment and cfg.get("augment")) else None
        self.model = ConformerCTC(
            codebook_size=int(m.get("codebook_size", 1024)),
            d_model=int(m["d_model"]),
            n_layers=int(m["n_conformer_layers"]),
            n_heads=int(m["n_conformer_heads"]),
            ff_mult=int(m["conformer_ff_mult"]),
            conv_kernel=int(m["conv_kernel_size"]),
            dropout=float(m["dropout"]),
            spec_augment=augment,
        )
        self.input_stream = int(m.get("input_stream", 0))
        self._grad_log_every = int(cfg["logging"].get("grad_norm_log_every_batches", 50))

        # -- per-epoch bookkeeping drained by BundleCallback ------------------
        self._train_loss_acc: dict[str, list[float]] = defaultdict(list)
        self._train_drops = 0
        self._val_drops = 0
        self._val_rows: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ util
    def _select_valid(self, batch: dict) -> tuple[dict | None, list[str]]:
        """Apply the §3.12 rule: split batch into usable rows + dropped utterance ids."""
        keep = ctc_lib.input_length_keep_mask(batch["lengths"], batch["text_lengths"])
        idx = torch.nonzero(keep, as_tuple=False).flatten()
        dropped = [
            batch["utterance_ids"][i] for i in range(len(keep)) if not bool(keep[i])
        ]
        if idx.numel() == 0:
            return None, dropped
        sel = {
            "tokens": batch["tokens"][idx],
            "lengths": batch["lengths"][idx],
            "text_ids": batch["text_ids"][idx],
            "text_lengths": batch["text_lengths"][idx],
            "utterance_ids": [batch["utterance_ids"][i] for i in idx.tolist()],
        }
        return sel, dropped

    def _forward_ctc(self, sel: dict) -> tuple[Tensor, Tensor]:
        """One model pass over valid rows → (per-utterance losses [B], log_probs)."""
        stream = sel["tokens"][:, self.input_stream, :]              # §2.3 RVQ₁
        log_probs, out_lengths = self.model(stream, sel["lengths"])
        loss_vec = ctc_lib.ctc_loss_per_utt(
            log_probs.transpose(0, 1),
            sel["text_ids"],
            out_lengths,
            sel["text_lengths"],
            self.blank_id,
        )
        return loss_vec, log_probs

    # ------------------------------------------------------------------ hooks
    def training_step(self, batch: dict, batch_idx: int):
        sel, dropped = self._select_valid(batch)
        self._train_drops += len(dropped)
        if sel is None:
            print(
                f"[track-b] step {self.global_step}: ALL {len(dropped)} utterances "
                "dropped this batch (token length < target length, PROTOCOL §3.12)",
                flush=True,
            )
            return None                                              # skips optimizer step
        loss_vec, _ = self._forward_ctc(sel)
        for uid, v in zip(sel["utterance_ids"], loss_vec.detach().cpu().tolist()):
            self._train_loss_acc[uid].append(float(v))

        loss = loss_vec.mean()
        self.log("train/loss_step", loss, on_step=True, on_epoch=False, prog_bar=True)
        return loss

    # NOTE: module-level hook (Lightning calls it on the module with just the
    # optimizer; the trainer-arg variant belongs to Callbacks).
    def on_before_optimizer_step(self, optimizer) -> None:
        if self._grad_log_every <= 0 or self.global_step % self._grad_log_every != 0:
            return
        device = next(self.model.parameters()).device
        sq = torch.zeros((), device=device)
        found = False
        for p in self.parameters():
            if p.grad is not None:
                sq = sq + p.grad.detach().float().pow(2).sum()
                found = True
        if found:
            self.log("train/grad_l2", sq.sqrt().item())

    def validation_step(self, batch: dict, batch_idx: int):
        sel, dropped_v = self._select_valid(batch)
        self._val_drops += len(dropped_v)
        if sel is None:
            return
        loss_vec, log_probs = self._forward_ctc(sel)
        hyps = ctc_lib.greedy_decode(log_probs, sel["lengths"], self.vocab)
        for uid, lv, ref_ids, hyp in zip(
            sel["utterance_ids"],
            loss_vec.detach().cpu().tolist(),
            sel["text_ids"].tolist(),
            hyps,
        ):
            ref = self.vocab.decode(ref_ids, collapse_repeats=False)
            self._val_rows.append(
                {"utterance_id": uid, "loss": float(lv), "ref": ref, "hyp": hyp}
            )

    def _build_val_snapshot(self) -> dict[str, Any] | None:
        """Aggregate the live validation buffers into a snapshot WITHOUT consuming
        them. Single-writer discipline: only :meth:`drain_epoch` resets these."""
        if not self._val_rows:
            return None
        rows = []
        wers: list[float] = []
        cers: list[float] = []
        for r in self._val_rows:
            if r["ref"].strip():
                w = ctc_lib.wer_one(r["ref"], r["hyp"])
                c = ctc_lib.cer_one(r["ref"], r["hyp"])
                wers.append(w)
                cers.append(c)
            else:
                w = c = None                                        # unscoreable text
            rows.append(
                {"utterance_id": r["utterance_id"], "loss": r["loss"], "wer": w, "cer": c}
            )
        return {
            "rows": rows,
            "wer": sum(wers) / len(wers) if wers else None,
            "cer": sum(cers) / len(cers) if cers else None,
            "loss": sum(r["loss"] for r in rows) / len(rows),
            "n_utterances": len(rows),
            "dropped": self._val_drops,
        }

    def on_validation_epoch_end(self) -> None:
        """READ-ONLY pass over the buffers — progress-bar metrics only.

        This deliberately does NOT consume or rebuild any bookkeeping state:
        lightning (observed in-tree ≥2.x) fires CALLBACK ``on_validation_epoch_end``
        hooks BEFORE this module hook, so any snapshot built/consumed here raced
        BundleCallback.drain_epoch() and shifted every recorded val epoch by one
        (sanity-pass rows leaked in as 'epoch 1'; a phantom final row appeared via
        the leftover flush at on_train_end)."""
        snap = self._build_val_snapshot()
        if snap is None:
            return
        if snap["wer"] is not None:
            self.log("val/wer", snap["wer"], prog_bar=True)
            self.log("val/cer", snap["cer"], prog_bar=False)
        self.log("val/loss", snap["loss"], prog_bar=True)

    def reset_val_buffers(self) -> None:
        """Discard everything accumulated so far (lightning sanity-check passes,
        aborted segments) — never called mid-epoch with meaningful data."""
        self._val_rows.clear()
        self._val_drops = 0

    # -------------------------------------------------- bundle-callback API
    def drain_epoch(self) -> dict:
        """Hand this epoch's bookkeeping to BundleCallback; buffers reset HERE and
        nowhere else (sole consumer ⇒ immune to hook-invocation ordering).

        Returns {train:[(uid, mean_loss)], train_drops:int, val:snapshot|None}."""
        train = sorted((uid, sum(v) / len(v)) for uid, v in self._train_loss_acc.items())
        self._train_loss_acc.clear()
        drops = self._train_drops
        self._train_drops = 0
        val = self._build_val_snapshot()
        self._val_rows.clear()
        self._val_drops = 0
        return {"train": train, "train_drops": drops, "val": val}

    # ------------------------------------------------------------ optimizer
    def configure_optimizers(self):
        t = self.cfg["training"]
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=float(t["lr_peak"]),
            weight_decay=float(t["weight_decay"]),
        )
        warmup = max(int(t["warmup_steps"]), 0)
        try:
            total = int(self.trainer.estimated_stepping_batches)      # needs fit context
        except Exception:
            total = warmup + 1
        warmup = min(warmup, max(total - 1, 0))
        decay_span = max(total - warmup, 1)

        def scale(step: int) -> float:
            if step < warmup:
                return (step + 1) / max(warmup, 1)
            return max((total - step) / decay_span, 1e-8)

        sched = torch.optim.lr_scheduler.LambdaLR(opt, scale)
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "step"},
        }
