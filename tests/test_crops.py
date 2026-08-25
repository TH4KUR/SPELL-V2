"""Tests for the frame<->sample alignment law — the heart of Phase 0."""

import random

import pytest
import torch

from crops import (
    CropWindow,
    apply_crop,
    n_frames_for_samples,
    pad_waveform_to_frames,
    random_crop_window,
    slice_waveform,
)

SAMPLES_PER_FRAME = 320  # 16000 Hz / 50 Hz
CROP_FRAMES = 50


def test_protocol_law_samples_per_frame():
    # The locked protocol must satisfy: crop_frames * spf == crop_samples == 16000
    import config

    proto = config.load_protocol()
    assert proto.sample_rate // proto.token_hz == SAMPLES_PER_FRAME
    assert proto.crop_frames * SAMPLES_PER_FRAME == proto.crop_samples


def test_window_sample_ranges():
    w = CropWindow(7, 7 + CROP_FRAMES, SAMPLES_PER_FRAME)
    assert (w.sample_start, w.sample_end, w.n_samples) == (7 * 320, 57 * 320, CROP_FRAMES * 320)


def test_full_utterance_frame_count_is_ceil():
    assert n_frames_for_samples(24576, SAMPLES_PER_FRAME) == 77   # real test clip
    assert n_frames_for_samples(16000, SAMPLES_PER_FRAME) == 50
    assert n_frames_for_samples(16001, SAMPLES_PER_FRAME) == 51


def _marked_wave_and_tokens(n_frames):
    """Wave whose every 320-sample block is a distinct constant; tokens carry the
    same marker so cropped audio chunks and cropped tokens can be compared."""
    wav = torch.arange(n_frames, dtype=torch.float32).repeat_interleave(SAMPLES_PER_FRAME)
    tokens = torch.arange(n_frames, dtype=torch.int64).unsqueeze(0).repeat(8, 1)
    return wav, tokens


def test_alignment_proof_crop_audio_matches_tokens_exactly():
    """THE proof: a 50-frame crop of tokens corresponds to EXACTLY the same
    16000 samples of waveform, frame for frame, at every offset."""
    n_frames = 300
    for k in [0, 1, 149, n_frames - CROP_FRAMES]:  # boundaries included
        window = CropWindow(k, k + CROP_FRAMES, SAMPLES_PER_FRAME)
        _, tokens = _marked_wave_and_tokens(n_frames)
        wav, _ = _marked_wave_and_tokens(n_frames)
        tok_crop = apply_crop(tokens, window)          # [8, 50]
        wav_crop = slice_waveform(wav, window)         # [16000]
        assert tok_crop.shape == (8, CROP_FRAMES)
        assert wav_crop.shape == (CROP_FRAMES * SAMPLES_PER_FRAME,)
        # chunk the crop into per-frame blocks; each block's marker must equal
        # the corresponding token id on every stream
        blocks = wav_crop.view(CROP_FRAMES, SAMPLES_PER_FRAME)
        for f in range(CROP_FRAMES):
            assert torch.all(blocks[f] == float(k + f))
            assert torch.all(tok_crop[:, f] == k + f)


def test_random_window_deterministic_per_seed():
    r1, r2 = random.Random(42), random.Random(42)
    w1 = random_crop_window(500, CROP_FRAMES, r1, SAMPLES_PER_FRAME)
    w2 = random_crop_window(500, CROP_FRAMES, r2, SAMPLES_PER_FRAME)
    assert (w1.frame_start, w1.frame_end) == (w2.frame_start, w2.frame_end)
    r3 = random.Random(43)
    w3 = random_crop_window(500, CROP_FRAMES, r3, SAMPLES_PER_FRAME)
    assert (w1.frame_start, w1.frame_end) != (w3.frame_start, w3.frame_end)


def test_too_short_raises():
    with pytest.raises(ValueError):
        random_crop_window(49, CROP_FRAMES, random.Random(0), SAMPLES_PER_FRAME)


def test_shortest_valid_crop_boundaries():
    rng = random.Random(0)
    w = random_crop_window(CROP_FRAMES, CROP_FRAMES, rng, SAMPLES_PER_FRAME)
    assert (w.frame_start, w.n_frames) == (0, CROP_FRAMES)


def test_pad_waveform_zero_fills_partial_last_frame():
    n_frames = 77                      # like the real clip: 24576 samples -> ceil
    wav = torch.ones(n_frames * SAMPLES_PER_FRAME - 100)  # 100-sample short tail
    padded = pad_waveform_to_frames(wav, n_frames, SAMPLES_PER_FRAME)
    assert padded.shape[-1] == n_frames * SAMPLES_PER_FRAME
    assert torch.all(padded[-100:] == 0)
    assert torch.all(padded[:-100] == 1)


def test_pad_waveform_rejects_longer_than_grid():
    with pytest.raises(ValueError):
        pad_waveform_to_frames(torch.zeros(101), 0, SAMPLES_PER_FRAME)
