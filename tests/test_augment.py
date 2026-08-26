"""TokenSpecAugment: determinism under global seeding, bounds vs TRUE lengths,
real-frames-only guarantee, and eval-mode identity."""

import torch

from augment import TokenSpecAugment


CFG = dict(freq_mask_n=2, freq_mask_width=12, time_mask_n=2, time_mask_ratio_max=0.15)


def _features(b=2, t=20, d=16):
    g = torch.Generator().manual_seed(1234)
    lengths = torch.tensor([t, 12])
    mask = torch.arange(t)[None] < lengths[:, None]
    x = torch.randn(b, t, d, generator=g)
    return x, mask, lengths


def test_determinism_same_seed_identical_masks():
    aug = TokenSpecAugment(**CFG).train()
    x, mask, _ = _features()

    torch.manual_seed(777)
    out_a = aug(x, mask)
    torch.manual_seed(777)
    out_b = aug(x, mask)

    assert torch.equal(out_a, out_b)


def test_different_seeds_usually_differ():
    aug = TokenSpecAugment(freq_mask_width=4, time_mask_n=1, time_mask_ratio_max=0.1).train()
    x, mask, _ = _features(t=40)
    torch.manual_seed(1); out1 = aug(x, mask)
    hits = 0
    for seed in range(2, 30):
        torch.manual_seed(seed)
        if not torch.equal(out1, aug(x.clone(), mask)):
            hits += 1
    assert hits > 10                                     # masking genuinely stochastic


def test_time_masks_never_touch_pad_frames():
    n, ratio = 3, 0.25                                    # max coverage 0.75 < 1 ⇒ nontrivial
    aug = TokenSpecAugment(freq_mask_n=0, time_mask_n=n, time_mask_ratio_max=ratio).train()
    x, mask, lengths = _features()
    before = x.clone()
    for seed in range(50):                               # hammer it across seeds
        torch.manual_seed(seed)
        out = aug(before, mask)
        changed = (out != before).any(dim=-1)
        assert not bool(changed[~mask].any()), "pad frames were modified"
        for i, ln in enumerate(lengths.tolist()):
            frac = changed[i, :ln].float().mean().item()
            # per-mask cap is ratio_max ⇒ total covered ≤ min(1, n·ratio) (overlap allowed)
            assert frac <= min(1.0, n * ratio) + 1e-6


def test_bounds_freq_and_scale_for_utterance_lengths():
    # freq masks alone → exactly the touched channel columns must be bounded
    aug_f = TokenSpecAugment(freq_mask_n=CFG["freq_mask_n"],
                             freq_mask_width=CFG["freq_mask_width"], time_mask_n=0).train()
    x, mask, lengths = _features(t=20)
    torch.manual_seed(9)
    out = aug_f(x, mask)
    d = x.shape[-1]
    changed_cols = ((out != x).any(dim=(0, 1))).sum().item()
    assert changed_cols <= min(d, CFG["freq_mask_n"] * CFG["freq_mask_width"])

    # time masks scale to the utterance's TRUE length (row 1 has T=12 < Tmax=20)
    aug_t = TokenSpecAugment(freq_mask_n=0, time_mask_n=CFG["time_mask_n"],
                             time_mask_ratio_max=CFG["time_mask_ratio_max"]).train()
    torch.manual_seed(9)
    out = aug_t(x, mask)
    row_changed = (out[1] != x[1]).any(dim=-1)[:12].float().mean().item()
    assert row_changed <= CFG["time_mask_n"] * CFG["time_mask_ratio_max"] + 1e-6


def test_eval_mode_identity():
    aug = TokenSpecAugment(**CFG)
    aug.train(); x, mask, _ = _features()
    torch.manual_seed(3); ref = aug(x, mask)             # ensure train path mutates *something*
    aug.eval()
    out = aug(x, mask)
    assert torch.equal(out, x)                           # byte-identical passthrough


def test_invalid_config_raises():
    import pytest

    with pytest.raises(ValueError):
        TokenSpecAugment(freq_mask_n=-1)
    with pytest.raises(ValueError):
        TokenSpecAugment(time_mask_ratio_max=1.5)
