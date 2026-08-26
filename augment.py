"""SpecAugment-style masking over embedded token features (Track B, fixed config).

Applied post-embedding (PROTOCOL §3.6: fixed augmentation for all Track B runs):
the "frequency" axis of classic SpecAugment maps onto the d_model CHANNEL axis,
"time" onto RVQ frames. Masks are zero-fills, drawn per training step from the
process-global RNG — deterministic because every entrypoint seeds globally
(``pl.seed_everything(train_seed)``) before dataloaders/training begin; any test
re-seeds explicitly around both calls it compares.

Only REAL frames are ever touched (pad frames are exact zeros already and would be
re-zeroed by the block containment anyway); time-mask bounds are computed against
TRUE utterance lengths, never padded Tmax. Training-only: ``eval()`` ⇒ identity.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TokenSpecAugment(nn.Module):
    """Time masks (n per sample, ≤ ``time_mask_ratio_max`` of TRUE length) +
    feature/channel masks (``freq_mask_n`` bands of width ≤ ``freq_mask_width``).
    Identity outside ``train()`` mode."""

    def __init__(
        self,
        *,
        freq_mask_n: int = 2,
        freq_mask_width: int = 12,
        time_mask_n: int = 2,
        time_mask_ratio_max: float = 0.15,
    ):
        super().__init__()
        if min(freq_mask_n, time_mask_n) < 0:
            raise ValueError("mask counts must be >= 0")
        if not 0 <= time_mask_ratio_max <= 1:
            raise ValueError("time_mask_ratio_max must lie in [0, 1]")
        self.freq_mask_n = freq_mask_n
        self.freq_mask_width = freq_mask_width
        self.time_mask_n = time_mask_n
        self.time_mask_ratio_max = time_mask_ratio_max

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """x [B,T,D], mask [B,T] True=real → augmented copy."""
        if not self.training:
            return x
        bsz, seq_len, dim = x.shape
        lengths = mask.sum(dim=-1)                       # true lengths [B]
        out = x.clone()

        # --- channel ("frequency") masks — same bands across the batch like SpecAugment
        for _ in range(self.freq_mask_n):
            width = int(torch.randint(0, self.freq_mask_width + 1, (1,)).item())
            if width == 0:
                continue
            start = int(torch.randint(0, max(dim - width, 0) + 1, (1,)).item())
            out[:, :, start : start + width] = 0.0

        # --- time masks — per-sample bounds against true length
        for i in range(bsz):
            true_len = int(lengths[i].item())
            max_width = int(true_len * self.time_mask_ratio_max)
            if max_width < 1:
                continue
            for _ in range(self.time_mask_n):
                width = int(torch.randint(0, max_width + 1, (1,)).item())
                if width == 0:
                    continue
                start = int(torch.randint(0, true_len - width + 1, (1,)).item())
                out[i, start : start + width, :] = 0.0
        return out
