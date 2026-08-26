"""Conformer encoder block tests: shapes, mask semantics, containment guarantees,
and the §3.14 GroupNorm (not BatchNorm) requirement."""

import pytest
import torch
from torch import nn

from conformer import ConformerBlock, ConformerEncoder, GuardedEmbedding, sinusoid_pe


TINY = dict(codebook_size=64, d_model=32, n_layers=2, n_heads=4, ff_mult=2,
            conv_kernel=15, dropout=0.0)


def _encoder() -> ConformerEncoder:
    enc = ConformerEncoder(**TINY).eval()
    return enc


def _batch(bsz=3, tmax=11):
    lens = torch.tensor([tmax, 7, 1])
    tokens = torch.randint(0, TINY["codebook_size"], (bsz, tmax))
    return tokens, lens


def test_output_shape_is_frame_level():
    enc = _encoder()
    tokens, lens = _batch()
    feats, mask = enc(tokens, lens)
    assert feats.shape == (3, 11, TINY["d_model"])       # NO subsampling anywhere
    assert mask.dtype == torch.bool and bool(mask[0].all()) and int(mask[1].sum()) == 7
    assert int(mask[2].sum()) == 1


def test_out_lengths_equal_input_lengths_through_model():
    from model import ConformerCTC

    m = ConformerCTC(codebook_size=TINY["codebook_size"], d_model=32, n_layers=2,
                     n_heads=4, ff_mult=2, conv_kernel=15)
    tokens, lens = _batch()
    _, out_lens = m(tokens, lens)
    assert torch.equal(out_lens, lens)                   # Phase-4 alignment guarantee


def test_no_attention_leakage_beyond_containment_halo():
    """Real-frame outputs must not depend on pad-region content. Per §3.13 the conv
    halo may COUPLE pads to neighbours, but pads are zero-filled INSIDE the network,
    so even halo frames are content-invariant here (stronger than the bound)."""
    torch.manual_seed(0)
    enc = _encoder()
    tokens, lens = _batch()
    mask = torch.arange(11)[None] < lens[:, None]
    variant = tokens.clone()
    variant[~mask] = 63                                  # different junk in pads
    with torch.no_grad():
        f1, _ = enc(tokens, lens)
        f2, _ = enc(variant, lens)
    assert torch.equal(f1[mask], f2[mask])
    # ...and pad features stay exact zeros at the ENCODER boundary:
    assert bool((f1[~mask] == 0).all())


def test_mask_all_true_equals_unmasked_semantics():
    """Single full-length utterance runs the plain forward path cleanly."""
    enc = _encoder()
    tokens = torch.randint(0, TINY["codebook_size"], (1, 9))
    feats, mask = enc(tokens, torch.tensor([9]))
    assert feats.shape == (1, 9, TINY["d_model"]) and bool(mask.all())
    assert torch.isfinite(feats).all()


@pytest.mark.parametrize("t", [1, 2, 7, 8])              # odd AND even parities
def test_rel_shift_correct_across_parities(t):
    """Rel-pos attention must run for every length (the classical flatten-trick
    broke on odd T; our gather formulation must not regress that)."""
    enc = _encoder()
    tokens = torch.randint(0, TINY["codebook_size"], (2, t))
    feats, _ = enc(tokens, torch.tensor([t, max(t - 1, 1)]))
    assert feats.shape == (2, t, TINY["d_model"])
    assert torch.isfinite(feats).all()


def test_block_zeroes_pads_after_every_sublayer():
    """Drive a single block manually and confirm final pad rows are exact zeros."""
    blk = ConformerBlock(TINY["d_model"], TINY["n_heads"], TINY["ff_mult"],
                         TINY["conv_kernel"], dropout=0.0).eval()
    x = torch.randn(2, 6, TINY["d_model"])
    x[1, 4:] = torch.full((2, TINY["d_model"]), 99.0)     # garbage seeded into pads
    lens = torch.tensor([6, 4])
    mask = torch.arange(6)[None] < lens[:, None]
    out = blk(x, mask)
    assert bool((out[~mask] == 0).all())
    assert torch.isfinite(out[mask]).all()


def test_conv_module_uses_groupnorm_not_batchnorm():
    """§3.14 regression guard: running-stat norms are forbidden in the conv module."""
    blk = ConformerBlock(TINY["d_model"], TINY["n_heads"], 2, TINY["conv_kernel"], 0.1)
    norm = blk.conv.norm
    assert isinstance(norm, nn.GroupNorm)
    assert norm.num_groups == 1 and norm.num_channels == TINY["d_model"]
    for mod in blk.modules():
        assert not isinstance(mod, nn.BatchNorm1d), \
            "BatchNorm crept back in — violates PROTOCOL §3.14"


def test_guarded_embedding_inherits_state_dict_compat():
    enc = _encoder()
    assert "embed.weight" in dict(enc.named_parameters())
    clone = _encoder()
    missing, unexpected = clone.load_state_dict(enc.state_dict(), strict=True), None
    torch.testing.assert_close(clone.embed.weight, enc.embed.weight)


def test_sinusoid_pe_shapes_and_finiteness():
    pe = sinusoid_pe(13, 32)
    assert pe.shape == (13, 32)
    assert torch.isfinite(pe).all()
    # distinct positions give distinct encodings
    assert not torch.equal(pe[0], pe[5])
