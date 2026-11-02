"""Checkpoint averaging with the arithmetic of the vendored ``raven_utils.average_checkpoints``.

Upstream sums the state dicts in order, in each tensor's own dtype, then divides: floating
point tensors by ``/= n`` and integer tensors by ``//= n`` (floor of the mean; this is what
happens to BatchNorm ``num_batches_tracked``, which only matters for a BatchNorm with
``momentum=None`` - ours use the default - so the value is carried, not relied on). BatchNorm
running means and variances are averaged like any float tensor; they are not re-estimated.

Upstream reads Lightning checkpoints (``["state_dict"]`` with a ``model.`` prefix); the parity
trainer writes plain weights-only state dicts, so this module averages those and a test proves
the two give identical tensors on the same inputs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch


def average_state_dicts(states: Sequence[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("nothing to average")
    keys = list(states[0].keys())
    for i, s in enumerate(states[1:], start=1):
        if list(s.keys()) != keys:
            raise ValueError(f"state dict {i} has different keys than state dict 0")
    avg = {k: v.detach().clone() for k, v in states[0].items()}
    for s in states[1:]:
        for k in keys:
            if s[k].shape != avg[k].shape or s[k].dtype != avg[k].dtype:
                raise ValueError(f"{k}: shape/dtype mismatch between the checkpoints")
            avg[k] += s[k]
    n = len(states)
    for k in keys:
        if avg[k].is_floating_point():
            avg[k] /= n
        else:
            avg[k] //= n
    return avg


def average_checkpoint_files(paths: Sequence[str | Path]) -> dict[str, torch.Tensor]:
    """Average weights-only checkpoint files (plain ``state_dict``s, loaded with ``weights_only``)."""
    states = []
    for p in paths:
        state = torch.load(p, map_location="cpu", weights_only=True)
        if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
            raise ValueError(f"{p}: expected a plain state_dict of tensors")
        states.append(state)
    return average_state_dicts(states)
