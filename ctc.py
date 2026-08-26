"""CTC loss, greedy decoding and WER/CER evaluation for Track B.

Protocol hooks:
- CTC loss uses ``zero_infinity=True`` (PROTOCOL §2.4: pads never become classes);
  reduction 'none' so every call yields PER-UTTERANCE losses — the lit module
  averages per batch for backprop but records per-utterance means per epoch into
  ``metrics.parquet`` (amendment: required by Phase-4 loss ranking / Oracle-RHO).
- Greedy decoding = argmax → slice to true input length →
  ``vocab.decode(collapse_repeats=True)`` (the frozen vocabulary already implements
  exactly the CTC semantics: repeated ids collapse, blanks break runs).
- **Input-length rule (PROTOCOL §3.12)**: a training/eval example whose token length
  is shorter than its target id length has no valid CTC alignment space; such rows
  are DROPPED by :func:`input_length_keep_mask` and the drop count logged loudly.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from vocab import CharVocab


def ctc_loss_per_utt(
    log_probs: Tensor,        # [T_in, B, V] log-softmax over char vocab
    targets: Tensor,          # [B, S] padded char ids (pad value ignored beyond tgt lens)
    in_lengths: Tensor,       # [B]
    tgt_lengths: Tensor,      # [B]
    blank_id: int,
) -> Tensor:
    """Per-utterance CTC losses [B]. Callers must pre-filter with
    :func:`input_length_keep_mask` so every row satisfies in_len >= tgt_len."""
    return F.ctc_loss(
        log_probs,
        targets,
        in_lengths,
        tgt_lengths,
        blank=blank_id,
        zero_infinity=True,
        reduction="none",
    )


def input_length_keep_mask(tok_lengths: Tensor, tgt_lengths: Tensor) -> Tensor:
    """True where the utterance is usable for CTC (tok_len >= tgt_len)."""
    return tok_lengths >= tgt_lengths


def greedy_decode(log_probs: Tensor, lengths: Tensor, vocab: CharVocab) -> list[str]:
    """Greedy CTC decoding → list of strings, one per batch row."""
    best_ids = log_probs.argmax(dim=-1)              # [B,T]
    hyps: list[str] = []
    for i in range(best_ids.size(0)):
        seq = best_ids[i, : int(lengths[i].item())].tolist()
        hyps.append(vocab.decode(seq, collapse_repeats=True))
    return hyps


def _per_utt_scores(metric_fn, refs: list[str], hyps: list[str]) -> tuple[float, list[float]]:
    """jiwer metric per pair + macro mean; empty-reference rows are skipped from
    the mean (logged upstream via metrics). jiwer normalizes whitespace itself."""
    scores = [float(metric_fn(r, h)) for r, h in zip(refs, hyps)]
    if not scores:
        return 0.0, []
    mean = sum(scores) / len(scores)
    return mean, scores


def compute_wer(refs: list[str], hyps: list[str]) -> tuple[float, list[float]]:
    """(batch_mean_wer, per-utterance wer) via jiwer."""
    from jiwer import wer as jiwer_wer

    return _per_utt_scores(jiwer_wer, refs, hyps)


def compute_cer(refs: list[str], hyps: list[str]) -> tuple[float, list[float]]:
    """(batch_mean_cer, per-utterance cer) via jiwer."""
    from jiwer import cer as jiwer_cer

    return _per_utt_scores(jiwer_cer, refs, hyps)


def wer_one(ref: str, hyp: str) -> float:
    """Single-pair WER (validation/eval loops that need strict row alignment)."""
    from jiwer import wer as jiwer_wer

    return float(jiwer_wer(ref, hyp))


def cer_one(ref: str, hyp: str) -> float:
    """Single-pair CER."""
    from jiwer import cer as jiwer_cer

    return float(jiwer_cer(ref, hyp))


def char_accuracy(ref_ids: list[list[int]], hyps: list[str], vocab: CharVocab) -> float:
    """1 − batch-mean CER computed against DECODED reference id sequences.
    The overfit gate uses this (char-level ⇒ sensitive even for one-word refs)."""
    refs = [vocab.decode(ids) if ids else "" for ids in ref_ids]
    cer, _ = compute_cer(refs, hyps)
    return max(0.0, 1.0 - cer)


def ids_from_text(text_norm: str, vocab: CharVocab) -> list[int]:
    """Normalized text → char ids (encode drops unknown chars with warning)."""
    return vocab.encode(text_norm, warn_unknown=False)
