"""Conformer encoder stack for Track B (ASR over the RVQ₁ token stream).

Frame-level (no subsampling): token frames stay 1:1 with RVQ time steps, keeping
the crop law (PROTOCOL §2.2) and Phase-4 per-(utterance, epoch) loss ranking exact.

Protocol hooks implemented here (see PROTOCOL.md §2.6 / §3.13 / §3.14):

- **GuardedEmbedding**: the padding sentinel is ``-1``, which PyTorch would silently
  wrap to the LAST vocabulary row. This embedding RAISES on any negative index — it
  is defense in depth behind the caller-side ``masked_fill`` that replaces sentinel
  frames before lookup.
- **Pad containment (§3.13)**: the encoder accepts a ``mask`` (True = real frame).
  Every ``ConformerBlock`` sublayer re-zeroes pad frames afterwards, bounding pad
  contamination to a ``conv_kernel // 2`` halo per block (only the depthwise conv
  mixes neighbouring positions; attention keys are column-masked, FFNs are
  position-wise).
- **GroupNorm, not BatchNorm (§3.14)**: BN running stats would depend on each run's
  batch-length distribution (100% vs 25% cells differ structurally). GroupNorm has
  no running state, keeping eval statistics comparable across cells.

Design notes:
- Attention is relative-position (Transformer-XL style, ESPnet formulation): content
  terms plus a ``rel_shift``-skewed positional term, with learnable ``u``/``v`` biases.
  Positional embeddings are FIXED sinusoids (no learnable PE — keeps the param count
  pinned and results stable across subsets).
- Macaron structure: FFN×(mult) half-step residuals flank MHA + conv per block,
  pre-LN wrapped, per the approved Phase-1 pin (d_model=256, 4 blocks, PROVISIONAL).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class GuardedEmbedding(nn.Embedding):
    """``nn.Embedding`` that refuses negative indices outright.

    The last-resort guard against sentinel leakage (PROTOCOL §2.6): without this,
    a ``-1`` quietly embeds the FINAL vocabulary row and corrupts the batch.
    Callers must still mask/pad before lookup — this class exists so that a
    forgotten mask fails loudly instead of silently.
    """

    def forward(self, x: Tensor) -> Tensor:
        if bool(torch.any(x < 0)):
            mn = int(x.min().item())
            raise ValueError(
                f"GuardedEmbedding received negative index {mn} — the pad sentinel "
                "leaked past masking (PyTorch would wrap it onto the LAST row). "
                "Fix the consumer: mask/replace pad frames BEFORE the lookup."
            )
        return super().forward(x)


def sinusoid_pe(length: int, dim: int) -> Tensor:
    """Fixed sinusoidal positional encoding, [length, dim] (Transformer-XL recipe)."""
    pos = torch.arange(length, dtype=torch.float32).unsqueeze(1)          # [L, 1]
    div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.empty(length, dim, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe


class RelPosMHA(nn.Module):
    """Relative-position multi-head attention (bidirectional; full-sequence).

    Scores follow the ESPnet Conformer formulation:
        (q + u)·k               content term
        (q + v)·r[j - i]  (rel-shifted)   positional term
    with fixed sinusoidal ``r`` inputs through a learned projection. Padded keys are
    additively masked to -inf; padded QUERY rows may emit garbage (they carry zeros),
    which the enclosing block re-zeroes immediately after this sublayer.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.dropout = dropout

        self.linear_q = nn.Linear(d_model, d_model, bias=False)
        self.linear_k = nn.Linear(d_model, d_model, bias=False)
        self.linear_v = nn.Linear(d_model, d_model, bias=False)
        self.linear_out = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        # content-bias u / position-bias v (the "u"/"v" of Dai et al.)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.d_head))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.d_head))

        self._pe_cache: Tensor | None = None   # CPU buffer; moved/sliced lazily
        self._shift_idx: dict[tuple[int, torch.device], Tensor] = {}

    # -- positional supply ---------------------------------------------------
    def _pos_emb(self, seq_len: int, ref: Tensor) -> Tensor:
        need = 2 * seq_len - 1                 # relative offsets -(T-1)..(T-1)
        if self._pe_cache is None or self._pe_cache.numel() == 0 or \
                self._pe_cache.size(0) < need or self._pe_cache.device != ref.device:
            self._pe_cache = sinusoid_pe(max(need, 128), self.d_model).to(ref.device)
        return self._pe_cache[:need]

    def _rel_shift(self, x: Tensor) -> Tensor:
        """Re-index skewed positional scores [B,H,T,2T-1] so entry [...,i,j]
        holds the relative distance j-i. The classical 'flatten & re-view' trick
        only matches widths for specific sequence parities — replaced by the
        directly-equivalent gather over a per-length cached index grid (identical
        outputs, correct for every T)."""
        t = x.size(-2)
        key = (t, x.device)
        idx = self._shift_idx.get(key)
        if idx is None:
            jj = torch.arange(t)
            ii = torch.arange(t).unsqueeze(1)
            idx = (jj - ii + (t - 1)).to(x.device)         # [query i, key j]
            self._shift_idx[key] = idx
        return x.gather(-1, idx.expand(*x.shape[:-1], t))

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """x [B,T,D], mask [B,T] True=real → contextualised [B,T,D]."""
        bsz, seq_len, _ = x.shape
        q = self.linear_q(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = self.linear_k(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = self.linear_v(x).view(bsz, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        pe = self._pos_emb(seq_len, x)                    # [2T-1, D]
        r = self.linear_pos(pe).view(2 * seq_len - 1, self.n_heads, self.d_head)

        # content term
        mat_ac = torch.matmul((q + self.pos_bias_u.view(1, -1, 1, self.d_head)), k.transpose(-2, -1))
        # positional term
        mat_bd = torch.einsum("bhtd,shd->bhts", q + self.pos_bias_v.view(1, -1, 1, self.d_head), r)
        mat_bd = self._rel_shift(mat_bd)

        scores = (mat_ac + mat_bd) * (self.d_head ** -0.5)
        pad_keys = ~mask                                   # [B, T]
        scores = scores.masked_fill(
            pad_keys[:, None, None, :], torch.finfo(scores.dtype).min
        )
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)                        # [B,H,T,dh]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.linear_out(out)


class MacaronFFN(nn.Module):
    """Position-wise feed-forward (hidden = mult × d_model), pre-LN applied by the block."""

    def __init__(self, d_model: int, mult: int, dropout: float):
        super().__init__()
        self.w_1 = nn.Linear(d_model, mult * d_model)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)
        self.w_2 = nn.Linear(mult * d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.w_2(self.drop(self.act(self.w_1(x))))


class ConvModule(nn.Module):
    """Canonical Conformer convolution module with GroupNorm (PROTOCOL §3.14).

    pointwise expand → GLU → depthwise k (symmetric 'same' padding) →
    GroupNorm(1, C) → SiLU → pointwise back. Input/output [B,T,D].
    """

    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError(f"conv_kernel_size must be odd, got {kernel_size}")
        self.kernel_size = kernel_size
        self.pw_1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size, groups=d_model, padding=kernel_size // 2)
        self.norm = nn.GroupNorm(1, d_model)          # §3.14: no running stats
        self.pw_2 = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        xt = x.transpose(1, 2)                        # [B,D,T]
        xt = self.pw_1(xt)
        xt = self.glu(xt)
        xt = self.dw(xt)
        xt = self.norm(xt)
        xt = self.pw_2(F.silu(xt))
        return self.drop(xt).transpose(1, 2)


class ConformerBlock(nn.Module):
    """Pre-LN macaron block: FFN½ → MHA → Conv → FFN½, re-masking pads after EVERY
    sublayer (PROTOCOL §3.13). Residual factors: 0.5 for each macaron half."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int, conv_kernel: int, dropout: float):
        super().__init__()
        self.ln_ff1 = nn.LayerNorm(d_model)
        self.ff1 = MacaronFFN(d_model, ff_mult, dropout)
        self.ln_mha = nn.LayerNorm(d_model)
        self.mha = RelPosMHA(d_model, n_heads, dropout)
        self.ln_conv = nn.LayerNorm(d_model)
        self.conv = ConvModule(d_model, conv_kernel, dropout)
        self.ln_ff2 = nn.LayerNorm(d_model)
        self.ff2 = MacaronFFN(d_model, ff_mult, dropout)

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        real = mask.unsqueeze(-1)                     # [B,T,1] True=real
        pad = ~real

        x = x + 0.5 * self.ff1(self.ln_ff1(x))
        x = x.masked_fill(pad, 0.0)

        x = x + self.mha(self.ln_mha(x), mask)
        x = x.masked_fill(pad, 0.0)

        x = x + self.conv(self.ln_conv(x))
        x = x.masked_fill(pad, 0.0)

        x = x + 0.5 * self.ff2(self.ln_ff2(x))
        x = x.masked_fill(pad, 0.0)
        return x


class ConformerEncoder(nn.Module):
    """GuardedEmbedding → zero-fill pads → dropout → N × ConformerBlock → final LN.

    Frame-level: input [B,T] int64 (sentinels at pad frames replaced with a legal id
    BEFORE lookup, whose embedded value is then overwritten with exact zeros — pad
    features therefore never mix anything into neighbours except through the known
    conv halo). Returns features [B,T,D] (pads zeroed) and the boolean mask.
    """

    def __init__(
        self,
        *,
        codebook_size: int = 1024,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 4,
        ff_mult: int = 4,
        conv_kernel: int = 15,
        dropout: float = 0.1,
        spec_augment: nn.Module | None = None,
    ):
        super().__init__()
        self.embed = GuardedEmbedding(codebook_size, d_model)
        self.spec_augment = spec_augment       # TokenSpecAugment; identity outside train()
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            ConformerBlock(d_model, n_heads, ff_mult, conv_kernel, dropout) for _ in range(n_layers)
        )
        self.final_ln = nn.LayerNorm(d_model)

    def forward(self, tokens: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        bsz, seq_len = tokens.shape
        mask = torch.arange(seq_len, device=tokens.device).unsqueeze(0) < lengths.unsqueeze(1)
        safe = tokens.masked_fill(~mask, 0)           # legal id at pad frames; guarded
        x = self.embed(safe)
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)   # exact zeros BEFORE any mixing
        if self.spec_augment is not None:
            x = self.spec_augment(x, mask)
        x = self.embed_dropout(x)
        for block in self.blocks:
            x = block(x, mask)
        x = self.final_ln(x)
        x = x.masked_fill(~mask.unsqueeze(-1), 0.0)
        return x, mask


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
