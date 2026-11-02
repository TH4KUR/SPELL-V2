"""Optimizer, per-group learning rates and the per-update schedule of the parity trainer.

* Parameter groups: uniform (one group) or, when the recipe has a layer-wise decay, upstream's
  own ``get_param_groups`` (vendored ``raven_utils``): after_norm and the last encoder block at
  the encoder LR, block ``i`` at ``decay ** (n_blocks - 1 - i)`` times it, ``embed`` and
  ``frontend`` at the block-0 scale, CTC head and decoder at the decoder LR, every group floored
  at ``min_lr``. AdamW applies ``weight_decay`` to EVERY parameter (biases, norms, LayerScale
  gammas included), as upstream's group list carries no per-group decay and as Auto-AVSR does.
* Schedule: Auto-AVSR's ``WarmupCosineScheduler`` (linear warm-up from 0 to the group's peak,
  then cosine) stepped per UPDATE, generalised to end at ``min_lr`` instead of 0:
  ``lr(s) = peak * s / W`` for ``s < W`` and
  ``lr(s) = floor + (peak - floor) * (1 + cos(pi * (s - W) / (T - W))) / 2`` afterwards, with
  ``s`` the 1-based update count (the first update runs at ``peak / W``, as Auto-AVSR's
  ``_step_count`` does) and ``floor = min(min_lr, peak)``. With ``min_lr = 0`` this is
  Auto-AVSR's formula exactly.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch

from parity import upstream
from parity.recipes import Recipe

upstream.activate()
from raven_utils import get_param_groups  # noqa: E402


def lr_at(step_count: int, *, base_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    """Learning rate of the update whose 1-based index is ``step_count``."""
    if not 1 <= step_count <= total_steps:
        raise ValueError(f"step_count {step_count} outside 1..{total_steps}")
    if step_count < warmup_steps:
        return base_lr * step_count / warmup_steps
    decay = total_steps - warmup_steps
    floor = min(min_lr, base_lr)
    cos_val = math.cos(math.pi * (step_count - warmup_steps) / decay)
    return floor + 0.5 * (base_lr - floor) * (1.0 + cos_val)


def _encoder_blocks(model: torch.nn.Module) -> int:
    return len(model.encoder.encoders)


def build_param_groups(model: torch.nn.Module, recipe: Recipe) -> list[dict]:
    """Groups with ``name``, ``lr`` (= the group's peak LR) and ``params``."""
    if recipe.layer_decay is None:
        params = [p for p in model.parameters() if p.requires_grad]
        return [{"name": "uniform", "lr": recipe.lr_encoder, "params": params}]
    return get_param_groups(
        model, _encoder_blocks(model), recipe.lr_encoder, recipe.lr_decoder, recipe.layer_decay,
        ctc_equals_other=True, min_lr=recipe.min_lr)


def build_optimizer(model: torch.nn.Module, recipe: Recipe) -> torch.optim.AdamW:
    groups = build_param_groups(model, recipe)
    for g in groups:
        g["base_lr"] = g["lr"]
    return torch.optim.AdamW(groups, lr=recipe.lr_encoder, betas=recipe.betas,
                             weight_decay=recipe.weight_decay)


def group_table(optimizer: torch.optim.Optimizer) -> list[dict]:
    return [{"name": g["name"], "peak_lr": g["base_lr"], "n_params": sum(p.numel() for p in g["params"])}
            for g in optimizer.param_groups]


class UpdateSchedule:
    """Sets every group's LR for a given update; holds no state beyond its constants."""

    def __init__(self, recipe: Recipe, steps_per_epoch: int):
        if steps_per_epoch < 1:
            raise ValueError("steps_per_epoch must be >= 1")
        self.steps_per_epoch = steps_per_epoch
        self.warmup_steps = recipe.warmup_epochs * steps_per_epoch
        self.total_steps = recipe.epochs * steps_per_epoch
        self.min_lr = recipe.min_lr

    def lrs(self, step_count: int, base_lrs: Iterable[float]) -> list[float]:
        return [lr_at(step_count, base_lr=b, min_lr=self.min_lr, warmup_steps=self.warmup_steps,
                      total_steps=self.total_steps) for b in base_lrs]

    def apply(self, optimizer: torch.optim.Optimizer, step_count: int) -> list[float]:
        values = self.lrs(step_count, [g["base_lr"] for g in optimizer.param_groups])
        for g, lr in zip(optimizer.param_groups, values):
            g["lr"] = lr
        return values
