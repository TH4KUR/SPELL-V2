"""Joint CTC/attention beam search exactly as upstream RAVEn evaluates
(``finetune_learner.Learner.get_beam_search`` + ``calculate_wer``), minus Lightning.

* waveform: float32 [-1, 1], 16 kHz mono, NO normalisation (paper App. A.2);
* the encoder takes ``[1, T, 1]`` (batch, samples, channel) and truncates to a
  multiple of 640 samples itself (25 frames/s);
* ``BatchBeamSearch`` with scorers {decoder, ctc: CTCPrefixScorer, lm: None,
  length_bonus: LengthBonus}, weights decoder = 1 - ctc_weight, ctc = ctc_weight,
  lm = 0, length_bonus = 0; sos = eos = odim - 1; pre-beam on the decoder score
  (disabled iff ctc_weight == 1.0); ``maxlenratio = minlenratio = 0.0``; 1-best.
"""

from __future__ import annotations

import torch

from parity import upstream
from parity.text import pieces_to_text

upstream.activate()
from espnet.asr.asr_utils import add_results_to_json  # noqa: E402
from espnet.nets.batch_beam_search import BatchBeamSearch  # noqa: E402
from espnet.nets.scorers.length_bonus import LengthBonus  # noqa: E402

SAMPLES_PER_FRAME = 640


def build_beam_search(model, token_list: list[str], *, beam_size: int = 40,
                      ctc_weight: float = 0.1) -> BatchBeamSearch:
    """Move ``model`` to its final device BEFORE calling: the decoder and CTC
    scorers are the model's own modules and share its device."""
    odim = len(token_list)
    scorers = model.scorers()
    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(odim)
    weights = dict(decoder=1.0 - ctc_weight, ctc=ctc_weight, lm=0.0, length_bonus=0.0)
    return BatchBeamSearch(
        beam_size=beam_size,
        vocab_size=odim,
        weights=weights,
        scorers=scorers,
        sos=odim - 1,
        eos=odim - 1,
        token_list=token_list,
        pre_beam_score_key=None if ctc_weight == 1.0 else "decoder",
    )


def transcribe(model, beam_search: BatchBeamSearch, waveform: torch.Tensor,
               token_list: list[str]) -> str:
    """1-best transcription of one utterance. ``waveform``: 1-D float tensor."""
    if model.training:
        raise RuntimeError("model must be in eval() mode (BatchNorm running statistics)")
    if waveform.ndim != 1:
        raise ValueError(f"waveform must be 1-D (samples,), got shape {tuple(waveform.shape)}")
    if waveform.numel() < SAMPLES_PER_FRAME:
        raise ValueError(
            f"{waveform.numel()} samples is shorter than one encoder frame "
            f"({SAMPLES_PER_FRAME} samples); the encoder would output nothing")
    device = next(model.parameters()).device
    with torch.no_grad():
        x = waveform.to(device=device, dtype=torch.float32).reshape(1, -1, 1)
        feat, _ = model.encoder(x, None)
        nbest = beam_search(x=feat.squeeze(0), maxlenratio=0.0, minlenratio=0.0)
        if not nbest:
            raise RuntimeError("beam search returned no hypothesis")
        nbest = [h.asdict() for h in nbest[: min(len(nbest), 1)]]
        return pieces_to_text(add_results_to_json(nbest, token_list))
