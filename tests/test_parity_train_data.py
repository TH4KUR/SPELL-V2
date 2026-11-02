"""parity/train_data.py: zero-mask augmentation, the length-bucketed deterministic sampler and the
collate that builds E2E.forward's inputs (PROTOCOL §3.34)."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import paths as data_paths  # noqa: E402
from parity import train_data as D  # noqa: E402
from parity import upstream  # noqa: E402
from parity_train_helpers import make_record, make_world  # noqa: E402

upstream.activate()
from raven_data_transforms import AdaptiveLengthTimeMask  # noqa: E402

WINDOW, STRIDE = 6400, 16000


def _mask(wave, seed_t, seed_r, window=WINDOW, stride=STRIDE):
    return D.time_mask(wave, window=window, stride=stride,
                       generator=torch.Generator().manual_seed(seed_t), rng=random.Random(seed_r))


# ------------------------------------------------------------------- time mask ------

@pytest.mark.parametrize("n_samples", [1, 700, 16000, 16001, 40000, 54720, 99999])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_time_mask_is_bit_identical_to_the_vendored_class_for_the_same_seeds(n_samples, seed):
    wave = torch.randn(n_samples) + 5.0                  # no zeros in the input
    torch.manual_seed(seed)
    random.seed(seed + 100)
    upstream_out = AdaptiveLengthTimeMask(WINDOW, STRIDE, replace_with_zero=True)(wave.unsqueeze(0))[0]
    ours = _mask(wave, seed, seed + 100)
    assert torch.equal(ours, upstream_out)


def test_time_mask_only_zeroes_and_never_touches_the_input():
    wave = torch.randn(48000) + 5.0
    before = wave.clone()
    out = _mask(wave, 3, 4)
    assert torch.equal(wave, before)                     # input untouched
    changed = out != wave
    assert changed.any()
    assert torch.all(out[changed] == 0.0)                # only zeros are written
    assert torch.equal(out[~changed], wave[~changed])


def test_time_mask_is_a_pure_function_of_its_two_seeds():
    wave = torch.randn(48000) + 5.0
    assert torch.equal(_mask(wave, 7, 8), _mask(wave, 7, 8))
    assert not torch.equal(_mask(wave, 7, 8), _mask(wave, 9, 8))
    assert not torch.equal(_mask(wave, 7, 8), _mask(wave, 7, 9))


def _independent_masked_fraction(n, rng):
    """The documented semantics re-derived with numpy only: n_mask masks per started second,
    second integer = length in [0, 6400), start uniform in [0, n - t)."""
    covered = np.zeros(n, dtype=bool)
    for _ in range(int((n + STRIDE - 0.1) // STRIDE)):
        t, w = rng.integers(0, WINDOW, size=2)
        if n - t <= 0 or t == 0:
            continue
        s = rng.integers(0, n - t)
        covered[s:s + w] = True
    return covered.mean()


def test_masked_fraction_matches_the_expected_value_and_an_independent_numpy_model():
    n = 160000                                           # 10 s -> 10 masks of mean 3199.5 samples
    wave = torch.ones(n)
    ours = np.array([float((_mask(wave, 1000 + i, 2000 + i) == 0).float().mean()) for i in range(300)])
    rng = np.random.default_rng(0)
    ref = np.array([_independent_masked_fraction(n, rng) for _ in range(300)])
    expected = 3199.5 / STRIDE                           # one mask of mean 3199.5 samples per second
    assert abs(ours.mean() - ref.mean()) < 4 * np.sqrt(ours.var() / 300 + ref.var() / 300)
    assert 0.8 * expected < ours.mean() < 1.0 * expected  # overlaps and end clipping only reduce it


def test_n_masks_is_one_per_started_stride():
    for n, want in [(1, 1), (16000, 1), (16001, 2), (32000, 2), (54720, 4)]:
        assert int((n + STRIDE - 0.1) // STRIDE) == want
    # a wave shorter than the stride still gets one mask attempt
    out = _mask(torch.ones(500), 0, 0)
    assert out.numel() == 500


def test_time_mask_rejects_non_1d_input():
    with pytest.raises(ValueError):
        _mask(torch.ones(2, 100), 0, 0)


# ------------------------------------------------------------------- sampler ---------

def _frames(n=600, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(rng.gamma(4.0, 21.0, size=n).astype(int), 13, 155).tolist()


def _plan(frames, epoch=0, seed=42, micro=60.0, eff=198.0):
    return D.plan_epoch(frames, seed=seed, epoch=epoch, micro_batch_seconds=micro,
                        effective_batch_seconds=eff)


def test_every_utterance_appears_exactly_once_per_epoch():
    frames = _frames()
    for epoch in range(3):
        seen = sorted(i for mb in _plan(frames, epoch).flat_batches() for i in mb)
        assert seen == list(range(len(frames)))


def test_micro_batches_respect_the_padded_budget_and_hold_similar_lengths():
    frames = _frames()
    budget = 60 * D.FRAMES_PER_SECOND
    plan = _plan(frames)
    for mb in plan.flat_batches():
        assert len(mb) * max(frames[i] for i in mb) <= budget
    # greedy longest-first packing: padding waste stays small
    padded = sum(len(mb) * max(frames[i] for i in mb) for mb in plan.flat_batches())
    assert padded / sum(frames) < 1.08


def test_plan_is_a_pure_function_of_seed_and_epoch_and_batches_are_reconstructible():
    frames = _frames()
    a, b = _plan(frames, epoch=3), _plan(frames, epoch=3)
    assert a == b
    assert _plan(frames, epoch=3) != _plan(frames, epoch=4)
    assert _plan(frames, epoch=3, seed=42) != _plan(frames, epoch=3, seed=43)
    # a resume at epoch 3 needs nothing but (seed, epoch): update 5 is the same micro-batches
    assert _plan(frames, epoch=3).updates[5] == a.updates[5]


def test_update_count_is_the_same_every_epoch_and_matches_the_effective_batch():
    frames = _frames(3000)
    plans = [_plan(frames, epoch=e) for e in range(5)]
    assert len({p.n_updates for p in plans}) == 1
    total_s = sum(frames) / D.FRAMES_PER_SECOND
    assert abs(plans[0].n_updates - total_s / 198.0) <= 0.1 * total_s / 198.0 + 1
    padded = np.array(plans[0].padded_seconds)
    assert abs(padded.mean() - 198.0) < 0.1 * 198.0
    assert padded.min() > 198.0 - 61 and padded.max() < 198.0 + 61      # within one micro-batch
    assert plans[0].n_utterances == len(frames)


def test_micro_batch_membership_changes_between_epochs_but_batch_sizes_do_not():
    frames = _frames(3000)
    p0, p1 = _plan(frames, 0), _plan(frames, 1)
    key = lambda p: sorted((len(mb), max(frames[i] for i in mb)) for mb in p.flat_batches())
    assert key(p0) == key(p1)                                           # same partition shape
    assert sorted(map(tuple, p0.flat_batches())) != sorted(map(tuple, p1.flat_batches()))   # ties re-drawn


def test_ties_are_broken_by_the_seeded_permutation_only():
    frames = [50] * 40 + [20] * 40
    mbs0 = _plan(frames, 0, micro=10.0, eff=20.0).flat_batches()
    assert mbs0 == _plan(frames, 0, micro=10.0, eff=20.0).flat_batches()
    for mb in mbs0:
        assert len({frames[i] for i in mb}) == 1                        # sorted by length first


def test_too_long_or_too_short_utterances_are_refused():
    with pytest.raises(ValueError, match="longer than the micro-batch budget"):
        _plan([10, 200], micro=6.0, eff=12.0)
    with pytest.raises(ValueError, match="shorter than one frame"):
        _plan([10, 0])
    with pytest.raises(ValueError):
        _plan([])
    with pytest.raises(ValueError):
        D.plan_epoch([10], seed=0, epoch=0, micro_batch_seconds=60, effective_batch_seconds=10)


def test_tiny_dataset_gives_one_update():
    plan = _plan([30, 25, 20], micro=60.0, eff=198.0)
    assert plan.n_updates == 1 and plan.n_utterances == 3


def test_eval_batches_are_deterministic_cover_everything_and_respect_the_budget():
    frames = _frames(200)
    b1, b2 = D.plan_eval_batches(frames, 60.0), D.plan_eval_batches(frames, 60.0)
    assert b1 == b2
    assert sorted(i for b in b1 for i in b) == list(range(200))
    assert all(len(b) * max(frames[i] for i in b) <= 60 * 25 for b in b1)


@pytest.mark.skipif("LRS3_VALID_ID_FILE" not in os.environ,
                    reason="needs the licensed AV-HuBERT id file (LRS3_VALID_ID_FILE)")
def test_real_train_split_update_count_and_padding():
    import pandas as pd
    from parity import splits as S
    index = pd.read_parquet(PROJECT_ROOT / "data_index.parquet")
    res = S.build_splits(index, S.read_valid_id_file(os.environ["LRS3_VALID_ID_FILE"]))
    dur = dict(zip(index["utterance_id"], index["duration_s"]))
    frames = [D.frames_of(int(round(dur[u] * 16000))) for u in res.train_ids]
    plan = _plan(frames)
    assert max(frames) == 155 and plan.n_utterances == 30782
    padded = sum(len(mb) * max(frames[i] for i in mb) for mb in plan.flat_batches())
    print(f"real train split: {plan.n_updates} updates/epoch, {plan.n_micro_batches} micro-batches, "
          f"padding overhead {padded / sum(frames):.4f}")
    assert 500 < plan.n_updates < 600 and padded / sum(frames) < 1.05
    assert len({_plan(frames, e).n_updates for e in range(4)}) == 1


# ------------------------------------------------------------------- collate ---------

def _items(lengths, label_lengths):
    out = []
    for k, (n, l) in enumerate(zip(lengths, label_lengths)):
        out.append({"utterance_id": f"u{k}", "wave": torch.full((n,), float(k + 1)),
                    "ids": torch.arange(1, l + 1), "ref": "X"})
    return out


def test_collate_builds_what_e2e_forward_needs():
    items = _items([1280 + 100, 640 * 4, 640 * 2 + 639], [3, 5, 2])
    b = D.collate_batch(items)
    assert b["x"].shape == (3, 2560, 1) and b["x"].dtype == torch.float32
    assert b["lengths"].tolist() == [2, 4, 2]                     # samples // 640 (CTC input lengths)
    assert b["lengths_samples"].tolist() == [1380, 2560, 1919]
    assert b["padding_mask"].shape == (3, 1, 4) and b["padding_mask"].dtype == torch.bool
    assert b["padding_mask"][:, 0].tolist() == [[True, True, False, False],
                                                [True, True, True, True],
                                                [True, True, False, False]]
    assert b["labels"].shape == (3, 5) and b["labels"].dtype == torch.long
    assert b["labels"].tolist() == [[1, 2, 3, -1, -1], [1, 2, 3, 4, 5], [1, 2, -1, -1, -1]]
    # right zero padding, values preserved
    assert torch.all(b["x"][0, :1380, 0] == 1.0) and torch.all(b["x"][0, 1380:, 0] == 0.0)
    assert torch.all(b["x"][2, 1919:, 0] == 0.0)
    assert b["padding_mask"].shape[-1] == int(b["lengths"].max()) == b["x"].shape[1] // 640
    assert b["utterance_ids"] == ["u0", "u1", "u2"]


def test_collate_refuses_a_sub_frame_utterance():
    with pytest.raises(ValueError, match="shorter than one frame"):
        D.collate_batch(_items([639, 1280], [1, 1]))
    with pytest.raises(ValueError):
        D.collate_batch([])


# ------------------------------------------------------------------- dataset ---------

@pytest.fixture
def world(tmp_path):
    return make_world(tmp_path / "data", [1.0, 1.5, 2.0, 2.5, 1.2, 1.8])


def _ds(world, **kw):
    args = dict(seed=42, time_mask_window_s=0.4, time_mask_stride_s=1.0, augment=True,
                path_resolver=world["resolver"])
    args.update(kw)
    return D.ParityTrainDataset(world["records"], world["token_ids"], **args)


def test_dataset_items_and_frames_from_the_index(world):
    ds = _ds(world, augment=False)
    item = ds[2]
    assert set(item) == {"utterance_id", "wave", "ids", "ref"}
    assert item["wave"].dtype == torch.float32 and item["wave"].ndim == 1 and item["wave"].numel() == 32000
    assert item["ids"].dtype == torch.long and item["ids"].tolist() == list(world["token_ids"]["vid/50002"])
    assert ds.frames == [25, 37, 50, 62, 30, 45]                   # int(round(d * 16000)) // 640


def test_augmentation_is_off_for_eval_and_deterministic_per_seed_epoch_index(world):
    off = _ds(world, augment=False)
    on = _ds(world)
    assert torch.equal(off[1]["wave"], off[1]["wave"])
    a = on[1]["wave"]
    assert not torch.equal(a, off[1]["wave"]) and (a == 0).sum() > (off[1]["wave"] == 0).sum()
    assert torch.equal(a, on[1]["wave"])                           # same (seed, epoch, index)
    on.set_epoch(1)
    assert not torch.equal(a, on[1]["wave"])                       # new epoch, new mask
    on.set_epoch(0)
    assert torch.equal(a, on[1]["wave"])
    assert not torch.equal(on[1]["wave"], _ds(world, seed=43)[1]["wave"])


def test_augmentation_does_not_depend_on_the_number_of_workers(world):
    ds = _ds(world)
    ds.set_epoch(2)
    batches = D.plan_eval_batches(ds.frames, 60.0)
    out = {}
    for workers in (0, 2):
        loader = DataLoader(ds, batch_sampler=batches, num_workers=workers, collate_fn=D.collate_batch,
                            worker_init_fn=D.worker_init_fn)
        out[workers] = [b["x"].clone() for b in loader]
    assert len(out[0]) == len(out[2])
    assert all(torch.equal(a, b) for a, b in zip(out[0], out[2]))


def test_missing_token_ids_are_refused(world):
    ids = dict(world["token_ids"])
    ids.pop("vid/50003")
    with pytest.raises(ValueError, match="no token ids"):
        D.ParityTrainDataset(world["records"], ids, seed=0, time_mask_window_s=0.4,
                             time_mask_stride_s=1.0, augment=False, path_resolver=world["resolver"])


def test_dataset_reads_through_the_layout_authority(tmp_path, monkeypatch):
    w = make_world(tmp_path / "root", [1.0, 1.4])
    monkeypatch.setenv("SPELL_DATA_ROOT", str(tmp_path / "root"))
    data_paths.reset_cache()
    try:
        ds = D.ParityTrainDataset(w["records"], w["token_ids"], seed=0, time_mask_window_s=0.4,
                                  time_mask_stride_s=1.0, augment=False)
        assert ds[1]["wave"].numel() == 22400
    finally:
        data_paths.reset_cache()


def test_worker_init_fn_seeds_deterministically(world):
    ds = _ds(world)
    seen = {}

    class Probe(torch.utils.data.Dataset):
        seed, epoch = 5, 3

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return torch.tensor([random.random(), float(np.random.rand()), float(torch.rand(1))])

    for run in range(2):
        loader = DataLoader(Probe(), batch_size=1, num_workers=2, worker_init_fn=D.worker_init_fn)
        seen[run] = torch.cat(list(loader)).tolist()
    assert seen[0] == seen[1]
    assert ds.epoch == 0 and make_record("a/1", 1.0).duration_s == 1.0
