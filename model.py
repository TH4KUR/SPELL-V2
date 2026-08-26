"""Conformer-CTC head model for Track B (PROTOCOL §2.3: RVQ₁ stream only).

Thin wrapper: token ids → guarded embedding (sentinel-safe, see conformer.py) →
frame-level Conformer encoder (optional SpecAugment on embedded features while
training) → linear character head → log-softmax. ``out_lengths == lengths`` — no
temporal downsampling anywhere, so CTC input lengths equal token frame lengths and
Phase-4 loss ranking can align losses to utterances exactly.

Pad handling contract (PROTOCOL §2.6 / §3.13):
- input tokens may contain the collate sentinel −1;
- sentinel frames are replaced by a legal id BEFORE the guarded lookup and their
  features re-zeroed immediately after it (pads contribute exact zeros to every
  mixing op except the documented conv halo);
- every block re-zeros pads after each sublayer;
- output log-probs at pad frames are overwritten with 0.0 — never consumed anyway
  (CTC slices to ``in_lengths``, greedy decode slices likewise).
"""

from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor, nn

from conformer import ConformerEncoder
from vocab import build_char_vocab


class ConformerCTC(nn.Module):
    def __init__(
        self,
        *,
        codebook_size: int = 1024,
        vocab_size: int | None = None,          # defaults to the frozen char vocab
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 4,
        conv_kernel: int = 15,
        dropout: float = 0.1,
        spec_augment: nn.Module | None = None,   # TokenSpecAugment (training-time only)
    ):
        super().__init__()
        if vocab_size is None:
            vocab_size = len(build_char_vocab())
        self.encoder = ConformerEncoder(
            codebook_size=codebook_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            ff_mult=ff_mult,
            conv_kernel=conv_kernel,
            dropout=dropout,
            spec_augment=spec_augment,
        )
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        """tokens [B,T] int64 (may contain −1 sentinels), lengths [B]
        → (log_probs [B,T,V], out_lengths == lengths)."""
        feats, mask = self.encoder(tokens, lengths)
        logits = self.head(feats)                        # [B,T,V]
        log_probs = F.log_softmax(logits.float(), dim=-1)
        # inert values on pad frames (consumers slice to true lengths regardless)
        log_probs = log_probs.masked_fill(~mask.unsqueeze(-1), 0.0)
        return log_probs, lengths.clone()
