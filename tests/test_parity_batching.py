"""Does a padded batch mean what the vendored E2E.forward thinks it means? (PROTOCOL §3.34)

The most error-prone part of the trainer is the contract between the collate and
``E2E.forward(x, padding_mask, lengths, label=...)``: waveform padding, frame lengths, the
(B, 1, T') bool padding mask, label padding and the way the loss is reduced. These tests pin it.

What is and is not batch-invariant (measured here, quoted in parity/REVIEW_STAGE2.md):

* Everything after the ResNet front-end is exactly batch-invariant in eval mode: attention keys
  beyond an utterance's length are masked in the encoder and in the decoder's source attention,
  CTC gets per-utterance frame lengths, the decoder is causal, BatchNorm1d uses running
  statistics, LayerNorm and the feed-forward layers act per frame.
* The ResNet front-end mixes neighbouring samples (convolutions) and its BatchNorm + Swish map a
  zero input to a non-zero constant, so the front-end features of the last valid frame(s) of a
  utterance shorter than its batch-mates differ slightly from the utterance processed alone
  (alone: the next layer's zero padding; in a batch: that constant). Upstream's own pad-with-
  zeros collate has the same property. With BatchNorm statistics that map zero to zero the
  difference vanishes exactly, which is how the exact tests below isolate it.
* In TRAIN mode BatchNorm normalises with the statistics of the whole micro-batch, padded frames
  included, so an utterance's loss then depends on its batch-mates; no collate can remove that.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from parity import model as parity_model  # noqa: E402
from parity import train_data as D  # noqa: E402
from parity import trainer as T  # noqa: E402
from parity_train_helpers import tiny_arch  # noqa: E402

ODIM = 30


def _items(waves, labels):
    return [{"utterance_id": f"u{i}", "wave": w, "ids": torch.tensor(l), "ref": "X"}
            for i, (w, l) in enumerate(zip(waves, labels))]


def _batch(waves, labels):
    return D.collate_batch(_items(waves, labels))


def _make_model(seed=0, *, identity_bn=False, **arch):
    torch.manual_seed(seed)
    model = parity_model.build_model(tiny_arch(**arch), odim=ODIM)
    model.train()
    for _ in range(4):                                         # non-trivial BatchNorm running statistics
        b = _batch([torch.randn(16000) * 0.1 for _ in range(3)], [[3, 4, 5]] * 3)
        model(b["x"], b["padding_mask"], b["lengths"], label=b["labels"])
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm1d):
            m.running_mean += 0.05 * torch.randn_like(m.running_mean)
            m.running_var *= 1.0 + 0.2 * torch.rand_like(m.running_var)
            m.bias.data += 0.1 * torch.randn_like(m.bias)
    if identity_bn:
        for m in model.encoder.frontend.modules():
            if isinstance(m, torch.nn.BatchNorm1d):
                m.running_mean.zero_()
                m.bias.data.zero_()
    return model.eval()


@pytest.fixture(scope="module")
def clips():
    g = torch.Generator().manual_seed(1)
    return {
        "a": torch.randn(32000 + 100, generator=g) * 0.1,      # 50 frames + remainder
        "b": torch.randn(48000 + 300, generator=g) * 0.1,      # 75 frames + remainder
        "c": torch.randn(40000, generator=g) * 0.1,            # 62 frames + remainder
        "a2": torch.randn(32000, generator=g) * 0.1,           # same length as a
    }


LABELS = {"a": [3, 4, 5, 6], "b": [7, 8, 9, 10, 11, 12], "c": [13, 14, 15], "a2": [16, 17, 18, 19]}


def _losses(model, names, clips):
    b = _batch([clips[n] for n in names], [LABELS[n] for n in names])
    return T.forward_batch(model, b, "cpu")


def _grads(model, names, clips, weights=None):
    model.zero_grad(set_to_none=True)
    loss, *_ = _losses(model, names, clips)
    loss.backward()
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])


def _f(t):
    return float(t.detach())


def _cos(a, b):
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0))


def _rel(a, b):
    """Relative L2 difference of two flattened gradients (element-wise tolerances are meaningless
    for the many entries that are numerically zero)."""
    return float((a - b).norm() / b.norm())


# ---------------------------------------------------------------- padding / masks ------

def test_padding_mask_marks_valid_frames_and_attention_ignores_padded_frames(clips):
    """Encoder and decoder see only valid frames: running the post-front-end encoder on a padded
    feature sequence with the mask gives the same valid-frame output as on the unpadded one."""
    model = _make_model()
    enc = model.encoder
    a, b = clips["a"], clips["b"]
    batch = _batch([a, b], [LABELS["a"], LABELS["b"]])
    assert batch["padding_mask"].shape == (2, 1, 75) and batch["padding_mask"].dtype == torch.bool
    assert batch["padding_mask"][0, 0].tolist() == [True] * 50 + [False] * 25
    assert batch["padding_mask"][1, 0].tolist() == [True] * 75

    def post_frontend(feats, mask):
        xs = enc.embed(feats)
        xs, _ = enc.encoders(xs, mask)
        xs = xs[0] if isinstance(xs, tuple) else xs
        return enc.after_norm(xs)

    with torch.no_grad():
        feats_a = enc.frontend(a.reshape(1, -1, 1))                      # (1, 50, 512)
        garbage = torch.randn(1, 25, feats_a.shape[-1]) * 5
        padded = torch.cat([feats_a, garbage], dim=1)
        mask = batch["padding_mask"][:1]
        alone = post_frontend(feats_a, None)
        masked = post_frontend(padded, mask)
        unmasked = post_frontend(padded, None)
        assert torch.allclose(masked[:, :50], alone, atol=1e-5)
        assert not torch.allclose(unmasked[:, :50], alone, atol=1e-3)    # the mask is what makes it work

        # decoder cross-attention uses the same mask
        ys_in = torch.tensor([[ODIM - 1, 3, 4, 5, 6]])
        ys_mask = D.torch.ones(1, 5, 5, dtype=torch.bool).tril()[None][0]
        out_alone, _ = model.decoder(ys_in, ys_mask, alone, None)
        out_masked, _ = model.decoder(ys_in, ys_mask, masked, mask)
        assert torch.allclose(out_alone, out_masked, atol=1e-5)


def test_ctc_gets_per_utterance_frame_lengths_and_labels_are_unpadded(clips):
    """CTC and attention losses of a padded batch equal the mean of the per-utterance losses
    (exact front-end), i.e. neither the padded frames nor the -1 label padding leak in."""
    model = _make_model(identity_bn=True)
    names = ["a", "b", "c"]
    with torch.no_grad():
        _, ctc_b, att_b, _ = _losses(model, names, clips)
        singles = [_losses(model, [n], clips) for n in names]
    assert float(ctc_b) == pytest.approx(sum(float(s[1]) for s in singles) / 3, rel=1e-4)
    assert float(att_b) == pytest.approx(sum(float(s[2]) for s in singles) / 3, rel=1e-4)


# ---------------------------------------------------------------- batch invariance -----

def test_exact_batch_invariance_of_loss_and_gradients_when_the_frontend_maps_zero_to_zero(clips):
    """BatchNorm eval + a front-end whose BN maps 0 -> 0: the padded batch's loss is the mean of
    the utterances' own losses and its gradient the mean of their gradients (per-utterance sums
    averaged over the batch, as E2E.forward reduces)."""
    model = _make_model(identity_bn=True)
    names = ["a", "b", "c"]
    batch_loss = _f(_losses(model, names, clips)[0])
    g_batch = _grads(model, names, clips)
    singles = [_f(_losses(model, [n], clips)[0]) for n in names]
    g_mean = sum(_grads(model, [n], clips) for n in names) / 3
    assert batch_loss == pytest.approx(sum(singles) / 3, rel=1e-5)
    assert _rel(g_batch, g_mean) < 2e-4 and _cos(g_batch, g_mean) > 0.9999999
    print(f"exact case: loss rel diff {abs(batch_loss - sum(singles) / 3) / batch_loss:.2e}, "
          f"grad rel L2 diff {_rel(g_batch, g_mean):.2e}, cos {_cos(g_batch, g_mean):.9f}")


def test_realistic_batchnorm_statistics_break_invariance_only_through_the_last_frames(clips):
    """With ordinary BN running statistics only the front-end's last valid frame(s) of the
    shorter utterances move; the loss changes by well under a per-mille and the gradient
    direction is unchanged to four digits."""
    model = _make_model()
    names = ["a", "b", "c"]
    batch_loss = _f(_losses(model, names, clips)[0])
    singles = [_f(_losses(model, [n], clips)[0]) for n in names]
    rel = abs(batch_loss - sum(singles) / 3) / batch_loss
    g_batch = _grads(model, names, clips)
    g_mean = sum(_grads(model, [n], clips) for n in names) / 3
    cos = _cos(g_batch, g_mean)
    rel_grad = float((g_batch - g_mean).norm() / g_mean.norm())
    print(f"realistic BN: loss rel diff {rel:.2e}, grad rel diff {rel_grad:.2e}, cos {cos:.6f}")
    assert 0 < rel < 1e-3 and cos > 0.9999 and rel_grad < 1e-2

    enc = model.encoder
    with torch.no_grad():
        alone = enc.frontend(clips["a"].reshape(1, -1, 1))[0]                       # 50 frames
        b = _batch([clips["a"], clips["b"]], [LABELS["a"], LABELS["b"]])
        in_batch = enc.frontend(b["x"])[0, :50]
    diff = (alone - in_batch).abs().amax(-1)
    assert float(diff[:-1].max()) < 1e-5                                           # frames 0..48 identical
    assert float(diff[-1]) > 1e-4                                                  # only the last frame moves


def test_utterances_of_equal_length_are_batch_invariant_with_realistic_statistics(clips):
    model = _make_model()
    names = ["a2", "a2"]
    loss_b = _f(_losses(model, names, clips)[0])
    loss_1 = _f(_losses(model, ["a2"], clips)[0])
    assert loss_b == pytest.approx(loss_1, rel=1e-5)


def test_in_train_mode_batchnorm_couples_the_utterances_of_a_batch(clips):
    """Train mode normalises with the micro-batch's statistics (padded frames included), so an
    utterance's loss depends on its batch-mates. No collate can remove this; it is why the
    recipe's micro_batch_seconds is a documented assumption."""
    model = _make_model().train()
    for m in model.modules():                                    # isolate BatchNorm: no stochastic layers
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
    state = copy.deepcopy(model.state_dict())
    alone = _f(_losses(model, ["a"], clips)[0])
    model.load_state_dict(state)
    pair = _losses(model, ["a", "b"], clips)
    model.load_state_dict(state)
    only_b = _f(_losses(model, ["b"], clips)[0])
    model.load_state_dict(state)
    assert abs((alone + only_b) / 2 - _f(pair[0])) / _f(pair[0]) > 1e-3


# ---------------------------------------------------------------- accumulation ---------

def _capture_update_grads(model, micro_batches, clip=1e9):
    opt = torch.optim.AdamW(model.parameters(), lr=0.0)
    stats = T.run_update(model, opt, micro_batches, device="cpu", grad_clip=clip)
    grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
    return stats, grads


def test_accumulated_micro_batches_weigh_every_utterance_equally(clips):
    model = _make_model(identity_bn=True)
    mb1 = _batch([clips["a"], clips["b"]], [LABELS["a"], LABELS["b"]])        # 2 utterances
    mb2 = _batch([clips["c"]], [LABELS["c"]])                                 # 1 utterance
    stats, g_acc = _capture_update_grads(model, [mb1, mb2])
    full = _batch([clips["a"], clips["b"], clips["c"]], [LABELS["a"], LABELS["b"], LABELS["c"]])
    stats_full, g_full = _capture_update_grads(model, [full])
    assert stats.n_utts == 3 and stats.n_micro == 2
    assert stats.loss == pytest.approx(stats_full.loss, rel=1e-5)
    assert _rel(g_acc, g_full) < 2e-4
    # the tempting alternative (mean of the micro-batch means) would weigh c twice as much
    l1 = _f(T.forward_batch(model, mb1, "cpu")[0])
    l2 = _f(T.forward_batch(model, mb2, "cpu")[0])
    assert abs((l1 + l2) / 2 - stats_full.loss) > 1e-3 * stats_full.loss
    assert stats.loss == pytest.approx((2 * l1 + 1 * l2) / 3, rel=1e-5)


def test_run_update_clips_at_the_update_level_and_reports_the_pre_clip_norm(clips):
    model = _make_model(identity_bn=True)
    mb1 = _batch([clips["a"], clips["b"]], [LABELS["a"], LABELS["b"]])
    mb2 = _batch([clips["c"]], [LABELS["c"]])
    _, g_free = _capture_update_grads(model, [mb1, mb2], clip=1e9)
    norm = float(g_free.norm())
    assert norm > 0.5
    clip = norm / 4
    stats, g_clipped = _capture_update_grads(model, [mb1, mb2], clip=clip)
    assert stats.grad_norm == pytest.approx(norm, rel=1e-4)                   # pre-clip total norm
    assert float(g_clipped.norm()) == pytest.approx(clip, rel=1e-3)           # clipped once, over both micro-batches
    assert _rel(g_clipped, g_free * (clip / norm)) < 1e-4
    stats, g_loose = _capture_update_grads(model, [mb1, mb2], clip=norm * 10)
    assert _rel(g_loose, g_free) < 1e-6                                       # below the threshold: untouched


def test_non_finite_loss_stops_training_loudly(clips):
    model = _make_model()
    mb = _batch([clips["a"]], [LABELS["a"]])
    mb["x"][0, 100, 0] = float("nan")
    opt = torch.optim.AdamW(model.parameters(), lr=0.0)
    with pytest.raises(FloatingPointError):
        T.run_update(model, opt, [mb], device="cpu", grad_clip=1.0)
