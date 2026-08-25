"""Crop sampling over discrete-token streams and their waveforms.

THE alignment law (verified during the Phase-0 audit): RVQ frames run at
``token_hz`` over ``sample_rate`` audio, so frame *f* covers exactly
``samples_per_frame = sample_rate // token_hz`` samples:

    frame f  <->  samples [spf * f, spf * (f+1))

A canonical crop of ``crop_frames`` frames therefore spans EXACTLY
``crop_frames * spf`` samples (= 16000 samples <-> 50 frames with the locked
protocol config). Tokens are full-utterance streams (audit verdict), so no
re-extraction was needed — this sampler is the corrected alignment path.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CropWindow:
    """A half-open frame range with its exact sample counterpart."""

    frame_start: int          # inclusive
    frame_end: int            # exclusive
    samples_per_frame: int

    def __post_init__(self) -> None:
        if self.frame_end <= self.frame_start:
            raise ValueError(f"empty crop [{self.frame_start}, {self.frame_end})")

    @property
    def n_frames(self) -> int:
        return self.frame_end - self.frame_start

    @property
    def sample_start(self) -> int:
        return self.frame_start * self.samples_per_frame

    @property
    def sample_end(self) -> int:
        return self.frame_end * self.samples_per_frame

    @property
    def n_samples(self) -> int:
        return self.sample_end - self.sample_start


def n_frames_for_samples(n_samples: int, samples_per_frame: int) -> int:
    """Frames produced by a full-utterance tokenization of ``n_samples`` audio."""
    return -(-n_samples // samples_per_frame)  # ceil


def random_crop_window(
    n_tokens: int,
    crop_frames: int,
    rng: random.Random,
    samples_per_frame: int,
) -> CropWindow:
    """Uniformly sample a crop window. Deterministic given ``rng``'s state."""
    if n_tokens < crop_frames:
        raise ValueError(
            f"utterance has {n_tokens} frames < crop_frames={crop_frames}; "
            "exclude short utterances from crop-based training"
        )
    start = rng.randint(0, n_tokens - crop_frames)
    return CropWindow(start, start + crop_frames, samples_per_frame)


def apply_crop(tokens: torch.Tensor, window: CropWindow) -> torch.Tensor:
    """Slice a [n_streams, T] token tensor to the window."""
    if tokens.ndim != 2:
        raise ValueError(f"expected [n_streams, T] tokens, got shape {tuple(tokens.shape)}")
    return tokens[:, window.frame_start : window.frame_end].contiguous()


def slice_waveform(wav: torch.Tensor, window: CropWindow) -> torch.Tensor:
    """Slice a [N] or [1, N] waveform to the window's exact sample range."""
    return wav[..., window.sample_start : window.sample_end]


def pad_waveform_to_frames(wav: torch.Tensor, n_frames: int, samples_per_frame: int) -> torch.Tensor:
    """Right-pad (zeros) a waveform to exactly ``n_frames * samples_per_frame``.

    The last tokenized frame may cover a partial final frame worth of audio
    (T = ceil(N / spf)); crops ending at the final frame need the tail present.
    Raises if the waveform is LONGER than the frame grid — that would indicate a
    tokens↔audio misalignment and must never be silently truncated.
    """
    target = n_frames * samples_per_frame
    n = wav.shape[-1]
    if n > target:
        raise ValueError(
            f"waveform has {n} samples > {target} expected for {n_frames} frames; "
            "tokens/audio mismatch — audit before training"
        )
    if n == target:
        return wav
    pad_shape = list(wav.shape)
    pad_shape[-1] = target - n
    return torch.cat([wav, wav.new_zeros(pad_shape)], dim=-1)
