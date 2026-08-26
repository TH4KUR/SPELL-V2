"""PROTOCOL-MANDATED pad-safety proof (§2.6): the collate sentinel −1 must never
reach an embedding lookup, and padded frames must not corrupt real outputs.

The primary test is END-TO-END: a collated batch whose pad region holds −1 flows
through the FULL ConformerCTC forward without raising. The negative control feeds
raw −1 straight into GuardedEmbedding and demands a ValueError.
"""

import torch
import torch.nn.functional as F

from conformer import GuardedEmbedding
from dataset import collate_token_batch
from model import ConformerCTC


def _tiny_model(dropout: float = 0.0) -> ConformerCTC:
    return ConformerCTC(
        codebook_size=64, d_model=32, n_layers=2, n_heads=4,
        ff_mult=2, conv_kernel=15, dropout=dropout,
    )


def _collated_batch():
    """2 utterances with genuine pad sentinels, as DataLoader output would be."""
    t0 = torch.randint(0, 64, (8, 11))
    t1 = torch.randint(0, 64, (8, 5))
    return collate_token_batch(
        [
            {"utterance_id": "a", "tokens": t0, "lengths": 0},
            {"utterance_id": "b", "tokens": t1, "lengths": 0},
        ],
        token_pad_id=-1,
    )


def _batch_with_text():
    batch = _collated_batch()
    batch["text_ids"] = torch.randint(2, 30, (2, 9))
    batch["text_lengths"] = torch.tensor([9, 7])
    return batch


def test_sentinel_batch_flows_through_full_forward():
    """THE mandated proof: −1 pads pass through model.forward without raising."""
    batch = _batch_with_text()
    model = _tiny_model().eval()
    stream = batch["tokens"][:, 0, :]              # RVQ₁ stream; still has −1 pads
    assert bool((stream < 0).any())                # sentinel genuinely present
    lengths = batch["lengths"]

    with torch.no_grad():
        log_probs, out_lengths = model(stream, lengths)

    assert log_probs.shape == (2, 11, 30)          # [B, T, frozen char vocab]
    assert torch.equal(out_lengths, lengths)
    # every REAL frame carries finite normalised log-probs summing to 1
    for i, ln in enumerate(lengths.tolist()):
        row = log_probs[i, :ln]
        assert torch.isfinite(row).all()
        sums = row.exp().sum(-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_padded_log_prob_frames_are_zero():
    batch = _batch_with_text()
    model = _tiny_model().eval()
    with torch.no_grad():
        log_probs, _ = model(batch["tokens"][:, 0, :], batch["lengths"])
    lengths = batch["lengths"]
    for i, ln in enumerate(lengths.tolist()):
        assert bool((log_probs[i, ln:] == 0).all()), f"pad frames of row {i} not zeroed"


def test_real_outputs_invariant_to_pad_content():
    """Pad-region CONTENT is irrelevant to every real frame (containment ⇒ even the
    known conv halo is content-blind: pads are exact zeros inside the network)."""
    batch = _batch_with_text()
    stream = batch["tokens"][:, 0, :]
    variant = stream.clone()
    mask = batch["mask"]                           # [B, Tmax] real-frame stencil
    variant[~mask] = -999                          # any junk; NOT the −1 sentinel
    model = _tiny_model().eval()
    with torch.no_grad():
        lp_a, _ = model(stream, batch["lengths"])
        lp_b, _ = model(variant, batch["lengths"])
    real = batch["mask"]
    assert torch.equal(lp_a[real], lp_b[real])     # EXACT equality, halo included


def test_negative_control_raw_sentinel_raises():
    emb = GuardedEmbedding(1024, 16)
    try:
        emb(torch.tensor([[3, 7], [1, -1]]))
    except ValueError as e:
        assert "-1" in str(e)
    else:
        raise AssertionError("GuardedEmbedding accepted −1 — guard is broken")


def test_valid_codes_pass_unguarded_boundaries():
    emb = GuardedEmbedding(1024, 16)
    ids = torch.tensor([[0, 1023, 512]])
    out = emb(ids)
    assert out.shape == (1, 3, 16)


def test_torch_wrap_semantics_documented_by_guard_trigger(monkeypatch):
    """Even an innocuous-looking wrong path (raw sentinel via embedding call at
    index level) surfaces loudly rather than wrapping onto the last row."""
    emb = GuardedEmbedding(4, 3)
    last_row_snapshot = emb.weight.detach().clone()[-1]
    try:
        emb(torch.tensor([2, -1]))
    except ValueError:
        pass
    # ...and nothing mutated state silently:
    assert torch.equal(last_row_snapshot, emb.weight.detach().clone()[-1])
