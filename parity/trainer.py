"""Plain-PyTorch training loop of the RAVEn-parity track (PROTOCOL §3.34).

Upstream RAVEn publishes no training code, so this loop follows the evidence recorded in
``parity/REVIEW_STAGE2.md``: the loss is the vendored ``E2E.forward`` loss (CTC weight 0.1 +
attention with label smoothing 0.1, per-utterance sums averaged over the batch); AdamW
(0.9, 0.98); per-update warm-up + cosine (``parity.optim``); gradient clipping at the update
level; fp32. An optimizer update is a set of micro-batches (``parity.train_data.plan_epoch``)
whose losses are weighted by ``n_utterances_in_micro_batch / n_utterances_in_update``, so every
utterance counts equally in the update's gradient however it was batched.

Everything an update or an epoch depends on is a function of ``(seed, epoch, step)`` or of the
checkpointed state, so ``--resume`` continues bit for bit (CPU) from the last epoch boundary.
The loop never touches the official test split: records must come from the index's
``trainval`` split.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader

from parity import checkpoint_avg, decode, metrics, optim, runinfo, wandb_run
from parity import text as parity_text
from parity import train_data
from parity.data import SAMPLE_RATE, ParityEvalDataset
from parity.progress import ProgressReporter
from parity.recipes import Recipe

CKPT_SCHEMA = "spell-rq2-parity-train-ckpt-v1"
SUMMARY_SCHEMA = "spell-rq2-parity-train-summary-v1"
LAST_NAME = "last.pt"
FINAL_NAME = "final.pt"
METRICS_NAME = "metrics.jsonl"
SUMMARY_NAME = "summary.json"
BEAM_PREDICTIONS_NAME = "val_beam_predictions.jsonl"
WEIGHTS_PREFIX = "weights_epoch"
FILES_CAP = 20                                  # relay bundle inode discipline (PROTOCOL §5.2)
PREFIX = "parity_train"


class TrainerError(RuntimeError):
    """Setup refused (exit 2 in the CLI)."""


# ------------------------------------------------------------------ determinism ------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2 ** 32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextlib.contextmanager
def fp32_deterministic_mode(deterministic_kernels: bool = True):
    """fp32 only (TF32 off for matmul and cuDNN, so an Ampere/Ada laptop and a Turing 2080 Ti
    compute in the same precision, no cuDNN autotuning) and, by default, deterministic kernels
    where they exist: PROTOCOL §10 item 1's ``warn_only`` policy, because CTC backward has no
    deterministic CUDA kernel. ``deterministic_kernels=False`` keeps fp32 but lets cuDNN pick its
    fastest convolution algorithms (measured on the development GPU: about 1.5x faster for this
    model; it is recorded in summary.json). Restores the previous global state on exit."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
              torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark,
              torch.are_deterministic_algorithms_enabled(),
              torch.is_deterministic_algorithms_warn_only_enabled())
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if deterministic_kernels:
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
         torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark) = before[:4]
        torch.use_deterministic_algorithms(before[4], warn_only=before[5])


# ------------------------------------------------------------------ one update --------

@dataclass
class UpdateStats:
    loss: float
    loss_ctc: float
    loss_att: float
    acc: float
    grad_norm: float
    n_utts: int
    n_micro: int
    audio_s: float
    padded_s: float


def forward_batch(model, batch: dict, device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """``E2E.forward`` on one collated micro-batch; returns (loss, loss_ctc, loss_att, acc)."""
    _, loss, loss_ctc, loss_att, acc = model(
        batch["x"].to(device), batch["padding_mask"].to(device), batch["lengths"].to(device),
        label=batch["labels"].to(device))
    return loss, loss_ctc, loss_att, float(acc)


def run_update(model, optimizer, batches: Sequence[dict], *, device, grad_clip: float) -> UpdateStats:
    """One optimizer update from ``batches`` (the micro-batches of the update).

    Each micro-batch's loss (the model's mean over its utterances of per-utterance sums) is
    weighted by ``n_b / N`` so the accumulated gradient is the gradient of the mean over ALL
    ``N`` utterances of the update. Clipping and the step happen once, after the last backward.
    """
    n_total = sum(len(b["utterance_ids"]) for b in batches)
    optimizer.zero_grad(set_to_none=True)
    sums = {"loss": 0.0, "ctc": 0.0, "att": 0.0, "acc": 0.0}
    audio = padded = 0.0
    for b in batches:
        n_b = len(b["utterance_ids"])
        weight = n_b / n_total
        loss, ctc, att, acc = forward_batch(model, b, device)
        loss_v = float(loss.detach())
        if not math.isfinite(loss_v):
            raise FloatingPointError(f"non-finite loss {loss_v} in utterances {b['utterance_ids'][:5]}")
        (loss * weight).backward()
        sums["loss"] += weight * loss_v
        sums["ctc"] += weight * float(ctc.detach())
        sums["att"] += weight * float(att.detach())
        sums["acc"] += weight * acc
        audio += float(b["lengths_samples"].sum()) / SAMPLE_RATE
        padded += b["x"].shape[0] * b["x"].shape[1] / SAMPLE_RATE
    params = [p for g in optimizer.param_groups for p in g["params"]]
    grad_norm = float(torch.nn.utils.clip_grad_norm_(params, grad_clip))
    if not math.isfinite(grad_norm):
        raise FloatingPointError(f"non-finite gradient norm {grad_norm}")
    optimizer.step()
    return UpdateStats(loss=sums["loss"], loss_ctc=sums["ctc"], loss_att=sums["att"], acc=sums["acc"],
                       grad_norm=grad_norm, n_utts=n_total, n_micro=len(batches),
                       audio_s=audio, padded_s=padded)


# ------------------------------------------------------------------ evaluation --------

def ctc_greedy_ids(frame_ids: Sequence[int], blank: int = parity_text.BLANK_ID) -> list[int]:
    """Best-path CTC decoding: collapse repeated frame labels, then drop blanks."""
    out, prev = [], None
    for t in frame_ids:
        if t != prev and t != blank:
            out.append(int(t))
        prev = t
    return out


@torch.no_grad()
def evaluate(model, dataset: train_data.ParityTrainDataset, tokens: list[str], *, device,
             micro_batch_seconds: float, num_workers: int = 0) -> dict:
    """Validation loss (teacher forcing: CTC, attention, total, token accuracy; per-utterance
    means) and CTC-greedy corpus WER, in eval mode (BatchNorm running statistics, no dropout).
    ``hyps``/``refs`` are the greedy hypotheses and references in evaluation-batch order."""
    if dataset.augment:
        raise ValueError("evaluation datasets must not augment")
    was_training = model.training
    model.eval()
    try:
        batches = train_data.plan_eval_batches(dataset.frames, micro_batch_seconds)
        loader = DataLoader(dataset, batch_sampler=batches, num_workers=num_workers,
                            collate_fn=train_data.collate_batch, worker_init_fn=train_data.worker_init_fn,
                            generator=torch.Generator().manual_seed(0))
        n = 0
        sums = {"loss": 0.0, "ctc": 0.0, "att": 0.0, "acc": 0.0}
        refs: list[str] = []
        hyps: list[str] = []
        for batch in loader:
            n_b = len(batch["utterance_ids"])
            enc, loss, ctc, att, acc = model(
                batch["x"].to(device), batch["padding_mask"].to(device), batch["lengths"].to(device),
                label=batch["labels"].to(device))
            sums["loss"] += n_b * float(loss)
            sums["ctc"] += n_b * float(ctc)
            sums["att"] += n_b * float(att)
            sums["acc"] += n_b * float(acc)
            n += n_b
            best = model.ctc.argmax(enc).cpu()
            for i, ref in enumerate(batch["refs"]):
                ids = ctc_greedy_ids(best[i, : int(batch["lengths"][i])].tolist())
                hyps.append(parity_text.pieces_to_text("".join(tokens[j] for j in ids)))
                refs.append(ref)
        scored = metrics.corpus_wer(refs, hyps)
        return {"n_utts": n, "loss": sums["loss"] / n, "loss_ctc": sums["ctc"] / n,
                "loss_att": sums["att"] / n, "acc": sums["acc"] / n,
                "ctc_greedy_wer": scored.corpus_wer, "ctc_greedy_edits": scored.total_edits,
                "ctc_greedy_ref_words": scored.total_ref_words, "hyps": hyps, "refs": refs}
    finally:
        model.train(was_training)


def beam_evaluate(model, records, tokens: list[str], *, device, beam_size: int, ctc_weight: float,
                  limit: int | None, pred_path: Path, path_resolver=None,
                  report_every_s: float = 30.0, on_report: Callable[[dict], None] | None = None) -> dict:
    """Corpus WER of the upstream joint CTC/attention beam search (parity.decode, exactly the
    Stage-1 recipe) over the first ``limit`` records. Slow: one utterance at a time."""
    chosen = list(records if limit is None else records[:limit])
    was_training = model.training
    model.eval()
    try:
        beam_search = decode.build_beam_search(model, tokens, beam_size=beam_size, ctc_weight=ctc_weight)
        dataset = ParityEvalDataset(chosen, path_resolver=path_resolver)
        loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=0)
        reporter = ProgressReporter(len(chosen), report_every_s=report_every_s,
                                    on_report=on_report or (lambda payload: None))
        per_utt: list[metrics.UttScore] = []
        decode_s = 0.0
        tmp = pred_path.with_name(pred_path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as pf:
            for uid, wave, ref in loader:
                t0 = time.perf_counter()
                hyp = decode.transcribe(model, beam_search, wave, tokens)
                dt = time.perf_counter() - t0
                decode_s += dt
                edits = metrics.pair_edits(ref, hyp)
                per_utt.append(metrics.UttScore(edits=edits, ref_words=len(ref.split())))
                pf.write(json.dumps({"utterance_id": uid, "ref": ref, "hyp": hyp, "edits": edits,
                                     "decode_s": dt}, ensure_ascii=False) + "\n")
                reporter.update(dt, running_corpus_wer=sum(u.edits for u in per_utt)
                                / sum(u.ref_words for u in per_utt))
            pf.flush()
            os.fsync(pf.fileno())
        os.replace(tmp, pred_path)
        reporter.finish()
        res = metrics.aggregate(per_utt)
        return {"corpus_wer": res.corpus_wer, "macro_wer": res.macro_wer, "n_utts": res.n_utts,
                "total_edits": res.total_edits, "total_ref_words": res.total_ref_words,
                "is_subset": len(chosen) < len(records), "beam_size": beam_size,
                "ctc_weight": ctc_weight, "decode_time_s": decode_s}
    finally:
        model.train(was_training)


# ------------------------------------------------------------------ checkpoints -------

def atomic_torch_save(obj, path: str | Path) -> None:
    """``torch.save`` to a temp file in the same directory, fsync, rename. A crash leaves the
    previous file intact and no temp file behind."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def atomic_write_text(path: str | Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def weights_path(out_dir: Path, epoch: int) -> Path:
    return out_dir / f"{WEIGHTS_PREFIX}{epoch:04d}.pt"


def cpu_state_dict(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def capture_rng() -> dict:
    return {"torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


# ------------------------------------------------------------------ the trainer -------

@dataclass
class TrainerOptions:
    out_dir: Path
    device: str = "cpu"
    num_workers: int = 0
    resume: bool = False
    stop_after_epochs: int | None = None
    beam_val_every: int = 0                         # 0: only when training finishes
    beam_val_limit: int | None = None
    keep_last_weights: int = 10
    report_every_s: float = 30.0
    micro_batch_seconds: float | None = None        # None: the recipe's
    path_resolver: Callable | None = None
    provenance: dict = field(default_factory=dict)  # hashes of every input; must match on resume
    environment: dict = field(default_factory=dict)  # paths, hardware: recorded, not compared


class Trainer:
    def __init__(self, *, recipe: Recipe, model, train_dataset: train_data.ParityTrainDataset,
                 val_dataset: train_data.ParityTrainDataset, tokens: list[str],
                 options: TrainerOptions, run=None):
        self.recipe = recipe
        self.model = model
        self.train_ds = train_dataset
        self.val_ds = val_dataset
        self.tokens = tokens
        self.opt = options
        self.run = run
        self.out_dir = Path(options.out_dir)
        self.device = torch.device(options.device)
        self.micro_s = recipe.micro_batch_seconds if options.micro_batch_seconds is None \
            else float(options.micro_batch_seconds)
        self._check_setup()
        self.optimizer = optim.build_optimizer(model, recipe)
        self.steps_per_epoch = self._plan(0).n_updates
        self.schedule = optim.UpdateSchedule(recipe, self.steps_per_epoch)
        self.epoch = 0                              # completed epochs
        self.global_step = 0
        self.history: list[dict] = []
        self.session_epoch_s: list[float] = []

    # ---- setup checks -----------------------------------------------------------------
    def _check_setup(self) -> None:
        r, o = self.recipe, self.opt
        for name, ds in (("train", self.train_ds), ("val", self.val_ds)):
            bad = [x.utterance_id for x in ds.records if x.split != "trainval"]
            if bad:
                raise TrainerError(f"{name} records outside the index's trainval split (the official test "
                                   f"split is never read by the trainer): {bad[:3]}")
        if not self.train_ds.augment or self.val_ds.augment:
            raise TrainerError("the train dataset must augment and the validation dataset must not")
        overlap = {x.utterance_id for x in self.train_ds.records} & {x.utterance_id for x in self.val_ds.records}
        if overlap:
            raise TrainerError(f"{len(overlap)} utterances are in both train and val, e.g. {sorted(overlap)[:3]}")
        if len(self.tokens) != self.model.odim or self.tokens[0] != "<blank>" or self.tokens[-1] != "<eos>":
            raise TrainerError(f"token list ({len(self.tokens)}) does not match the model's odim {self.model.odim}")
        if self.model.mtlalpha != r.ctc_weight or self.model.criterion.smoothing != r.label_smoothing:
            raise TrainerError("model loss weights differ from the recipe (build the model with recipe.arch_config)")
        dp = {getattr(layer.drop_path, "drop_prob", 0.0) for layer in self.model.encoder.encoders}
        if dp != {r.drop_path}:
            raise TrainerError(f"encoder drop-path {sorted(dp)} != recipe drop_path {r.drop_path}")
        if self.train_ds.window != int(round(r.time_mask_window_s * SAMPLE_RATE)) or \
                self.train_ds.stride != int(round(r.time_mask_stride_s * SAMPLE_RATE)):
            raise TrainerError("train dataset time-mask settings differ from the recipe")
        if self.train_ds.seed != r.seed:
            raise TrainerError("train dataset seed differs from the recipe seed")
        if o.stop_after_epochs is not None and not 1 <= o.stop_after_epochs <= r.epochs:
            raise TrainerError(f"stop_after_epochs must be in 1..{r.epochs}")
        if r.model_selection == "average_last_k" and o.keep_last_weights < r.average_last_k:
            raise TrainerError(f"model_selection average_last_k needs keep_last_weights >= {r.average_last_k}")
        if self.micro_s <= 0 or self.micro_s > r.effective_batch_seconds:
            raise TrainerError("micro_batch_seconds must be in (0, effective_batch_seconds]")

    def _plan(self, epoch: int) -> train_data.EpochPlan:
        return train_data.plan_epoch(
            self.train_ds.frames, seed=self.recipe.seed, epoch=epoch, micro_batch_seconds=self.micro_s,
            effective_batch_seconds=self.recipe.effective_batch_seconds)

    # ---- logging ----------------------------------------------------------------------
    def _log(self, sub: str, payload: dict) -> None:
        if self.run is None:
            return
        wandb_run.log(self.run, PREFIX + (f"/{sub}" if sub else ""), {"update": self.global_step, **payload})

    def _define_metrics(self) -> None:
        define = getattr(self.run, "define_metric", None)
        if define is None:
            return
        define(f"{PREFIX}/update")
        define(f"{PREFIX}/*", step_metric=f"{PREFIX}/update")

    # ---- checkpoint state ---------------------------------------------------------------
    def state(self) -> dict:
        return {
            "schema": CKPT_SCHEMA,
            "recipe": self.recipe.to_dict(),
            "epoch": self.epoch,
            "global_step": self.global_step,
            "steps_per_epoch": self.steps_per_epoch,
            "micro_batch_seconds": self.micro_s,
            "model": cpu_state_dict(self.model),
            "optimizer": self.optimizer.state_dict(),
            "rng": capture_rng(),
            "sampler": {"seed": self.recipe.seed, "next_epoch": self.epoch, "step_in_epoch": 0},
            "history": self.history,
            "provenance": self.opt.provenance,
        }

    def load_state(self, state: dict) -> None:
        if state.get("schema") != CKPT_SCHEMA:
            raise TrainerError(f"unexpected checkpoint schema {state.get('schema')!r}")
        if state["recipe"] != self.recipe.to_dict():
            raise TrainerError("checkpoint was written by a different recipe or recipe-file hash; refusing to resume")
        if state["steps_per_epoch"] != self.steps_per_epoch or state["micro_batch_seconds"] != self.micro_s:
            raise TrainerError("checkpoint batching (steps per epoch / micro-batch seconds) differs; refusing to resume")
        if state["provenance"] != self.opt.provenance:
            raise TrainerError("checkpoint data/tokenizer provenance differs from this run; refusing to resume")
        self.model.load_state_dict(state["model"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.epoch = int(state["epoch"])
        self.global_step = int(state["global_step"])
        self.history = list(state["history"])
        restore_rng(state["rng"])

    def _prepare_out_dir(self) -> None:
        last = self.out_dir / LAST_NAME
        if self.opt.resume:
            if not last.is_file():
                raise TrainerError(f"--resume but {last} does not exist")
            self.load_state(torch.load(last, map_location="cpu", weights_only=True))
            if self.epoch >= self.recipe.epochs:
                raise TrainerError(f"{last} is already at epoch {self.epoch} of {self.recipe.epochs}")
        else:
            if self.out_dir.exists() and any(self.out_dir.iterdir()):
                raise TrainerError(f"{self.out_dir} is not empty; use --resume to continue it or a new directory")
            self.out_dir.mkdir(parents=True, exist_ok=True)

    def _assert_file_cap(self) -> None:
        files = [p for p in self.out_dir.rglob("*") if p.is_file()]
        if len(files) > FILES_CAP:
            raise TrainerError(f"{self.out_dir} holds {len(files)} files > cap {FILES_CAP} (PROTOCOL §5.2)")

    # ---- the loop ---------------------------------------------------------------------
    def _train_loader(self, plan: train_data.EpochPlan) -> DataLoader:
        self.train_ds.set_epoch(plan.epoch)
        kwargs = {}
        if self.opt.num_workers > 0:
            kwargs.update(prefetch_factor=2, persistent_workers=False)
        return DataLoader(
            self.train_ds, batch_sampler=plan.flat_batches(), num_workers=self.opt.num_workers,
            collate_fn=train_data.collate_batch, worker_init_fn=train_data.worker_init_fn,
            generator=torch.Generator().manual_seed(train_data.derive_seed(self.recipe.seed, plan.epoch, 5)),
            pin_memory=self.device.type == "cuda", **kwargs)

    def _train_epoch(self, epoch: int, reporter: ProgressReporter) -> dict:
        plan = self._plan(epoch)
        if plan.n_updates != self.steps_per_epoch:
            raise AssertionError("updates per epoch changed between epochs; the LR schedule would be wrong")
        loader_iter = iter(self._train_loader(plan))
        self.model.train()
        cuda = self.device.type == "cuda"
        t_epoch = time.perf_counter()
        w_loss = n_seen = 0.0
        for u, micro in enumerate(plan.updates):
            t0 = time.perf_counter()
            batches = [next(loader_iter) for _ in micro]
            data_wait = time.perf_counter() - t0
            if cuda:
                torch.cuda.reset_peak_memory_stats(self.device)
            lrs = self.schedule.apply(self.optimizer, self.global_step + 1)
            stats = run_update(self.model, self.optimizer, batches, device=self.device,
                               grad_clip=self.recipe.grad_clip)
            if cuda:
                torch.cuda.synchronize(self.device)
            update_s = time.perf_counter() - t0
            self.global_step += 1
            w_loss += stats.loss * stats.n_utts
            n_seen += stats.n_utts
            payload = {
                "epoch": epoch + 1, "epoch_update": u + 1, "loss": stats.loss, "loss_ctc": stats.loss_ctc,
                "loss_att": stats.loss_att, "acc": stats.acc, "grad_norm": stats.grad_norm,
                "update_s": update_s, "data_wait_s": data_wait,
                "audio_s_per_s": stats.audio_s / update_s, "audio_s": stats.audio_s,
                "padded_audio_s": stats.padded_s, "n_utts": stats.n_utts, "n_micro_batches": stats.n_micro,
            }
            for g, lr in zip(self.optimizer.param_groups, lrs):
                payload[f"lr/{g['name']}"] = lr
            if cuda:
                payload["gpu_mem_peak_gb"] = torch.cuda.max_memory_allocated(self.device) / 2 ** 30
            self._log("", payload)
            reporter.update(update_s, epoch=epoch + 1, global_step=self.global_step, loss=stats.loss)
        return {"train_loss": w_loss / n_seen, "train_utts": int(n_seen),
                "epoch_train_s": time.perf_counter() - t_epoch, "n_updates": plan.n_updates,
                "lr_last": lrs[0]}

    def _prune_weights(self, epoch: int) -> None:
        keep = self.opt.keep_last_weights
        for p in self.out_dir.glob(f"{WEIGHTS_PREFIX}*.pt"):
            if int(p.stem[len(WEIGHTS_PREFIX):]) <= epoch - keep:
                p.unlink()

    def _write_metrics(self) -> None:
        atomic_write_text(self.out_dir / METRICS_NAME,
                          "".join(json.dumps(row, sort_keys=True) + "\n" for row in self.history))

    def _beam(self, epoch: int, final: bool) -> dict:
        def on_report(payload: dict) -> None:
            self._log("beam_val", {"epoch": epoch, "utts_done": payload["utts_done"],
                                   "utts_total": payload["utts_total"], "progress": payload["progress"],
                                   "elapsed_s": payload["elapsed_s"], "last_utt_s": payload["last_utt_s"],
                                   "eta_s": payload["eta_s"],
                                   "running_corpus_wer": payload["running_corpus_wer"]})

        res = beam_evaluate(
            self.model, self.val_ds.records, self.tokens, device=self.device,
            beam_size=self.recipe.beam_size, ctc_weight=self.recipe.decode_ctc_weight,
            limit=self.opt.beam_val_limit, pred_path=self.out_dir / BEAM_PREDICTIONS_NAME,
            path_resolver=self.opt.path_resolver, report_every_s=self.opt.report_every_s,
            on_report=on_report)
        res["epoch"] = epoch
        res["final"] = final
        return res

    def _end_of_epoch(self, epoch: int, train_row: dict, epoch_wall_s: float, last_epoch: bool) -> None:
        val = evaluate(self.model, self.val_ds, self.tokens, device=self.device,
                       micro_batch_seconds=self.micro_s, num_workers=self.opt.num_workers)
        row = {"epoch": epoch, "global_step": self.global_step, **train_row,
               **{f"val_{k}": v for k, v in val.items() if k not in ("hyps", "refs")},
               "val_example_hyp": val["hyps"][0], "val_example_ref": val["refs"][0]}
        periodic = self.opt.beam_val_every > 0 and epoch % self.opt.beam_val_every == 0
        if periodic and not last_epoch:
            beam = self._beam(epoch, final=False)
            row["val_beam_wer"] = beam["corpus_wer"]
            row["val_beam_decode_s"] = beam["decode_time_s"]
            row["val_beam_n_utts"] = beam["n_utts"]
        self.epoch = epoch
        if self.opt.keep_last_weights > 0:
            atomic_torch_save(cpu_state_dict(self.model), weights_path(self.out_dir, epoch))
            self._prune_weights(epoch)
        row["epoch_wall_s"] = epoch_wall_s
        self.history.append(row)
        atomic_torch_save(self.state(), self.out_dir / LAST_NAME)
        self._write_metrics()
        self._assert_file_cap()
        self.session_epoch_s.append(epoch_wall_s)
        remaining = self.recipe.epochs - epoch
        eta = remaining * (sum(self.session_epoch_s) / len(self.session_epoch_s))
        self._log("epoch", {
            "epoch": epoch, "epochs_total": self.recipe.epochs, "epoch_s": epoch_wall_s, "eta_s": eta,
            "train_loss": row["train_loss"], "val_loss": val["loss"], "val_loss_ctc": val["loss_ctc"],
            "val_loss_att": val["loss_att"], "val_acc": val["acc"], "val_ctc_greedy_wer": val["ctc_greedy_wer"],
            **({"val_beam_wer": row["val_beam_wer"]} if "val_beam_wer" in row else {})})
        print(f"[parity-train] epoch {epoch}/{self.recipe.epochs} train_loss {row['train_loss']:.3f} "
              f"val_loss {val['loss']:.3f} val_ctc_greedy_wer {100 * val['ctc_greedy_wer']:.2f}% "
              f"epoch {epoch_wall_s / 60:.1f} min, ETA {eta / 3600:.2f} h", flush=True)

    def _select_final_weights(self) -> tuple[dict, dict]:
        r = self.recipe
        if r.model_selection == "final_epoch":
            return cpu_state_dict(self.model), {"rule": "final_epoch", "epoch": r.epochs}
        epochs = list(range(r.epochs - r.average_last_k + 1, r.epochs + 1))
        paths = [weights_path(self.out_dir, e) for e in epochs]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise TrainerError(f"average_last_k needs {missing[:3]}")
        return checkpoint_avg.average_checkpoint_files(paths), {"rule": "average_last_k", "epochs": epochs}

    def _finalize(self) -> dict:
        r = self.recipe
        state, selection = self._select_final_weights()
        if selection["rule"] == "average_last_k":
            self.model.load_state_dict(state, strict=True)
        atomic_torch_save(state, self.out_dir / FINAL_NAME)
        if selection["rule"] == "final_epoch":
            final_val = {k: self.history[-1][f"val_{k}"] for k in
                         ("loss", "loss_ctc", "loss_att", "acc", "ctc_greedy_wer")}
        else:
            ev = evaluate(self.model, self.val_ds, self.tokens, device=self.device,
                          micro_batch_seconds=self.micro_s, num_workers=self.opt.num_workers)
            final_val = {k: ev[k] for k in ("loss", "loss_ctc", "loss_att", "acc", "ctc_greedy_wer")}
        beam = self._beam(r.epochs, final=True)
        self._log("final", {"val_beam_wer": beam["corpus_wer"], "val_beam_macro_wer": beam["macro_wer"],
                            "val_ctc_greedy_wer": final_val["ctc_greedy_wer"], "val_loss": final_val["loss"]})
        return {"selection": selection, "val": final_val, "val_beam": beam,
                "final_sha256": runinfo.sha256_file(self.out_dir / FINAL_NAME)}

    def fit(self) -> dict:
        """Train to the recipe's last epoch (or ``stop_after_epochs``). Starts from the recipe seed
        whatever consumed the global RNG before; a resume then overrides it with the checkpointed
        RNG state."""
        r = self.recipe
        seed_everything(r.seed)
        self._prepare_out_dir()
        if self.run is not None:
            self._define_metrics()
        start_epoch = self.epoch
        stop = r.epochs if self.opt.stop_after_epochs is None else self.opt.stop_after_epochs
        if start_epoch >= stop:
            raise TrainerError(f"nothing to do: {start_epoch} epochs are already done (stop at {stop})")
        total_updates = (stop - start_epoch) * self.steps_per_epoch

        def on_progress(p: dict) -> None:
            self._log("progress", {
                "updates_done": p["utts_done"], "updates_total": p["utts_total"], "progress": p["progress"],
                "global_progress": self.global_step / (r.epochs * self.steps_per_epoch),
                "elapsed_s": p["elapsed_s"], "last_update_s": p["last_utt_s"],
                "window_max_update_s": p["window_max_utt_s"], "updates_per_s": p["utts_per_s"],
                "eta_s": p["eta_s"], "loss": p["loss"]})

        reporter = ProgressReporter(total_updates, report_every_s=self.opt.report_every_s, on_report=on_progress)
        t_fit = time.perf_counter()
        finished = None
        for epoch in range(start_epoch + 1, stop + 1):
            t_epoch = time.perf_counter()
            train_row = self._train_epoch(epoch - 1, reporter)
            self._end_of_epoch(epoch, train_row, time.perf_counter() - t_epoch, last_epoch=epoch == r.epochs)
        reporter.finish()
        if self.epoch == r.epochs:
            finished = self._finalize()
        summary = self._summary(finished, time.perf_counter() - t_fit, start_epoch)
        atomic_write_text(self.out_dir / SUMMARY_NAME, json.dumps(summary, indent=2, sort_keys=True) + "\n")
        return summary

    def _summary(self, finished: dict | None, wall_s: float, start_epoch: int) -> dict:
        r = self.recipe
        doc = {
            "schema": SUMMARY_SCHEMA,
            "complete": finished is not None,
            "recipe": r.to_dict(),
            "epochs_done": self.epoch,
            "epochs_total": r.epochs,
            "resumed_from_epoch": start_epoch if self.opt.resume else None,
            "global_step": self.global_step,
            "steps_per_epoch": self.steps_per_epoch,
            "micro_batch_seconds": self.micro_s,
            "micro_batch_seconds_override": None if self.micro_s == r.micro_batch_seconds else self.micro_s,
            "param_groups": optim.group_table(self.optimizer),
            "n_parameters": sum(p.numel() for p in self.model.parameters()),
            "n_train_utts": len(self.train_ds), "n_val_utts": len(self.val_ds),
            "session_wall_s": wall_s,
            "history": self.history,
            "device": str(self.device),
            "torch_version": torch.__version__,
            "provenance": self.opt.provenance,
            "environment": self.opt.environment,
            "finished_utc": runinfo.utc_now_iso(),
            **runinfo.git_state(),
        }
        if finished is not None:
            doc["final"] = finished
        return doc
