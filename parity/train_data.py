"""Training data of the RAVEn-parity track (PROTOCOL §3.34): dataset, length-bucketed
sampler, time-mask augmentation and the collate that builds exactly what the vendored
``E2E.forward(x, padding_mask, lengths, mask=None, label=None)`` needs.

What ``E2E.forward`` needs (read from the vendored espnet sources):

* ``x``: ``(B, T, 1)`` float32 waveform, zero-padded on the right to the longest utterance.
  The front-end cuts the batch to ``T // 640 * 640`` samples (640 samples = one 25 frames/s
  frame) and returns ``T // 640`` frames per utterance row.
* ``lengths``: ``(B,)`` valid FRAMES per utterance, ``len_samples // 640`` (CTC input lengths).
* ``padding_mask``: ``(B, 1, T // 640)`` bool, True = valid frame. The attention does
  ``mask.unsqueeze(1).eq(0)`` and the decoder's source attention uses the same tensor.
* ``label``: ``(B, Lmax)`` int64 token ids padded with -1 (``ignore_id``); ids are 1..n_units
  (0 is <blank>; the model adds <sos>/<eos> = odim - 1 itself).

Batches are built by padded-seconds budget in the units RAVEn's sampler used (25 frames/s
video-frame counts, fairseq ``batch_by_size`` semantics: ``n_utterances * longest <= budget``).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from parity.data import SAMPLE_RATE, load_waveform
from parity.text import reference_text

SAMPLES_PER_FRAME = 640
FRAMES_PER_SECOND = SAMPLE_RATE // SAMPLES_PER_FRAME      # 25
IGNORE_ID = -1


def derive_seed(*parts: int) -> int:
    """A 64-bit seed that is a pure function of ``parts`` (not of any process or worker state)."""
    return int(np.random.SeedSequence(list(parts)).generate_state(1, dtype=np.uint64)[0])


def frames_of(n_samples: int) -> int:
    return int(n_samples) // SAMPLES_PER_FRAME


def time_mask(wave: torch.Tensor, *, window: int, stride: int,
              generator: torch.Generator, rng: random.Random) -> torch.Tensor:
    """Zero-mask augmentation with the semantics of the vendored upstream
    ``AdaptiveLengthTimeMask(window, stride, replace_with_zero=True)`` (and of Auto-AVSR's
    ``AdaptiveTimeMask``), on a 1-D waveform and with explicit RNGs.

    ``n_mask = int((len + stride - 0.1) // stride)`` masks, i.e. one per started ``stride``
    samples. Each draws two integers in ``[0, window)``: ``t`` only bounds where the mask may
    start (``start ~ U[0, len - t)``, skipped when ``len - t <= 0`` or ``t == 0``) and the
    second integer is the mask length. Draws are consumed in upstream's order, so with the
    same two seeds the output is identical to the vendored class's.
    """
    if wave.ndim != 1:
        raise ValueError(f"expected a 1-D waveform, got shape {tuple(wave.shape)}")
    out = wave.clone()
    n = out.numel()
    n_mask = int((n + stride - 0.1) // stride)
    for t, width in torch.randint(0, window, size=(n_mask, 2), generator=generator).tolist():
        if n - t <= 0:
            continue
        start = rng.randrange(0, n - t)
        if t == 0:
            continue
        out[start:start + width] = 0.0
    return out


# ---------------------------------------------------------------------------- sampler --

@dataclass(frozen=True)
class EpochPlan:
    """One epoch's batches: ``updates[u][m]`` is the tuple of dataset indices of micro-batch
    ``m`` of optimizer update ``u``. A pure function of the inputs of :func:`plan_epoch`."""

    epoch: int
    updates: tuple[tuple[tuple[int, ...], ...], ...]
    padded_seconds: tuple[float, ...]            # per update

    @property
    def n_updates(self) -> int:
        return len(self.updates)

    @property
    def n_micro_batches(self) -> int:
        return sum(len(u) for u in self.updates)

    @property
    def n_utterances(self) -> int:
        return sum(len(mb) for u in self.updates for mb in u)

    def flat_batches(self) -> list[list[int]]:
        return [list(mb) for u in self.updates for mb in u]


def _budget_frames(seconds: float) -> int:
    return int(round(seconds * FRAMES_PER_SECOND))


def _pack(order: Sequence[int], frames: Sequence[int], budget: int) -> list[list[int]]:
    """Greedy packing of a longest-first ordering: a batch is full when
    ``(n + 1) * longest > budget`` (fairseq ``batch_by_size``)."""
    batches: list[list[int]] = []
    cur: list[int] = []
    cur_max = 0
    for idx in order:
        f = frames[idx]
        if f < 1:
            raise ValueError(f"utterance index {idx} is shorter than one frame ({SAMPLES_PER_FRAME} samples)")
        if f > budget:
            raise ValueError(f"utterance index {idx} is {f / FRAMES_PER_SECOND:.2f} s, longer than the "
                             f"micro-batch budget of {budget / FRAMES_PER_SECOND:.2f} s")
        if cur and max(cur_max, f) * (len(cur) + 1) > budget:
            batches.append(cur)
            cur, cur_max = [], 0
        cur.append(idx)
        cur_max = max(cur_max, f)
    if cur:
        batches.append(cur)
    return batches


def _batch_frames(batch: Sequence[int], frames: Sequence[int]) -> int:
    return len(batch) * max(frames[i] for i in batch)


def plan_epoch(frames: Sequence[int], *, seed: int, epoch: int, micro_batch_seconds: float,
               effective_batch_seconds: float) -> EpochPlan:
    """Deterministic batches for ``epoch`` from per-utterance frame counts.

    1. Utterances are ordered longest-first by frame count with a (seed, epoch)-seeded random
       tie-break, and packed greedily into micro-batches of at most ``micro_batch_seconds``
       PADDED seconds (``n * longest``). Frame counts have few distinct values, so the
       tie-break is what changes batch membership between epochs, as in RAVEn's sampler.
    2. The micro-batch order is shuffled with a (seed, epoch)-seeded permutation.
    3. Consecutive micro-batches form an update; update boundaries are the first micro-batch
       whose cumulative padded seconds reach ``k * total / U`` for ``k = 1 .. U - 1``, where
       ``U = round(total / effective_batch_seconds)`` (reduced while one micro-batch would
       cross two boundaries). ``U`` depends only on the multiset of frame counts, so every
       epoch has the same number of updates, which is what a per-update LR schedule needs.
    """
    n = len(frames)
    if n == 0:
        raise ValueError("no utterances")
    if micro_batch_seconds <= 0 or effective_batch_seconds < micro_batch_seconds:
        raise ValueError("need 0 < micro_batch_seconds <= effective_batch_seconds")
    budget = _budget_frames(micro_batch_seconds)
    tiebreak = np.random.default_rng(derive_seed(seed, epoch, 0)).permutation(n)
    order = np.lexsort((tiebreak, np.asarray(frames)))[::-1].tolist()
    micro = _pack(order, frames, budget)

    sizes = np.array([_batch_frames(mb, frames) for mb in micro], dtype=np.float64) / FRAMES_PER_SECOND
    total = float(sizes.sum())
    n_updates = max(1, int(round(total / effective_batch_seconds)))
    while n_updates > 1 and float(sizes.max()) >= total / n_updates:
        n_updates -= 1

    perm = np.random.default_rng(derive_seed(seed, epoch, 1)).permutation(len(micro))
    shuffled = [micro[i] for i in perm]
    shuffled_sizes = sizes[perm]
    cumulative = np.cumsum(shuffled_sizes)
    cuts = [int(np.searchsorted(cumulative, k * total / n_updates, side="left")) + 1
            for k in range(1, n_updates)]
    bounds = [0] + cuts + [len(shuffled)]
    updates, padded = [], []
    for a, b in zip(bounds[:-1], bounds[1:]):
        updates.append(tuple(tuple(mb) for mb in shuffled[a:b]))
        padded.append(float(shuffled_sizes[a:b].sum()))
    if any(len(u) == 0 for u in updates):
        raise AssertionError("empty update: micro-batch sizes are inconsistent with the boundaries")
    return EpochPlan(epoch=epoch, updates=tuple(updates), padded_seconds=tuple(padded))


def plan_eval_batches(frames: Sequence[int], micro_batch_seconds: float) -> list[list[int]]:
    """Deterministic evaluation batches: longest-first, ties by index, same padded budget."""
    order = np.lexsort((np.arange(len(frames)), np.asarray(frames)))[::-1].tolist()
    return _pack(order, frames, _budget_frames(micro_batch_seconds))


# ---------------------------------------------------------------------------- dataset --

class ParityTrainDataset(Dataset):
    """``dict(utterance_id, wave, ids, ref)`` per utterance.

    ``wave`` is the float32 [-1, 1] waveform with NO normalisation (RAVEn App. A.2), zero-masked
    when ``augment`` is on. The mask RNGs are seeded from ``(seed, epoch, index)`` only, so the
    augmentation of an utterance in an epoch is the same whatever the worker count, batch order
    or resume point.
    """

    def __init__(self, records, token_ids: Mapping[str, Sequence[int]], *, seed: int,
                 time_mask_window_s: float, time_mask_stride_s: float, augment: bool,
                 path_resolver=None, loader=load_waveform):
        if not records:
            raise ValueError("empty record list")
        missing = [r.utterance_id for r in records if r.utterance_id not in token_ids]
        if missing:
            raise ValueError(f"{len(missing)} utterances have no token ids, e.g. {missing[:3]}")
        self.records = list(records)
        self.token_ids = token_ids
        self.seed = int(seed)
        self.augment = bool(augment)
        self.window = int(round(time_mask_window_s * SAMPLE_RATE))
        self.stride = int(round(time_mask_stride_s * SAMPLE_RATE))
        if self.augment and (self.window < 1 or self.stride < 1):
            raise ValueError("time-mask window and stride must be at least one sample")
        self.epoch = 0
        self._loader = loader
        if path_resolver is None:
            import paths as _paths

            path_resolver = _paths.resolve_audio_path
        self._resolve = path_resolver

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def frames(self) -> list[int]:
        """Frame counts used for bucketing, from the index's ``duration_s`` (no audio is read)."""
        return [frames_of(int(round(r.duration_s * SAMPLE_RATE))) for r in self.records]

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]
        wave = self._loader(self._resolve(rec))
        if self.augment:
            wave = time_mask(
                wave, window=self.window, stride=self.stride,
                generator=torch.Generator().manual_seed(derive_seed(self.seed, self.epoch, idx, 2)),
                rng=random.Random(derive_seed(self.seed, self.epoch, idx, 3)))
        return {"utterance_id": rec.utterance_id, "wave": wave,
                "ids": torch.tensor(list(self.token_ids[rec.utterance_id]), dtype=torch.long),
                "ref": reference_text(rec.text_raw)}


def worker_init_fn(worker_id: int) -> None:
    """Seed python/numpy/torch inside a DataLoader worker from (seed, epoch, worker). Nothing in
    the pipeline draws from them (the mask uses explicit RNGs); this keeps any future use
    deterministic."""
    info = get_worker_info()
    ds = info.dataset
    s = derive_seed(getattr(ds, "seed", 0), getattr(ds, "epoch", 0), worker_id, 4)
    random.seed(s)
    np.random.seed(s % (2 ** 32))
    torch.manual_seed(s)


def collate_batch(items: Sequence[dict]) -> dict:
    """Pad a list of dataset items into the tensors ``E2E.forward`` consumes (see module doc)."""
    if not items:
        raise ValueError("empty batch")
    samples = torch.tensor([it["wave"].numel() for it in items], dtype=torch.long)
    lengths = samples // SAMPLES_PER_FRAME
    if int(lengths.min()) < 1:
        bad = [it["utterance_id"] for it, n in zip(items, lengths.tolist()) if n < 1]
        raise ValueError(f"utterances shorter than one frame: {bad}")
    t_max = int(samples.max())
    x = torch.zeros(len(items), t_max, 1, dtype=torch.float32)
    for i, it in enumerate(items):
        x[i, : it["wave"].numel(), 0] = it["wave"]
    n_frames = t_max // SAMPLES_PER_FRAME
    padding_mask = torch.arange(n_frames).unsqueeze(0) < lengths.unsqueeze(1)          # (B, T')
    l_max = max(it["ids"].numel() for it in items)
    labels = torch.full((len(items), l_max), IGNORE_ID, dtype=torch.long)
    for i, it in enumerate(items):
        labels[i, : it["ids"].numel()] = it["ids"]
    return {
        "utterance_ids": [it["utterance_id"] for it in items],
        "refs": [it["ref"] for it in items],
        "x": x,
        "lengths": lengths,
        "lengths_samples": samples,
        "padding_mask": padding_mask.unsqueeze(1),
        "labels": labels,
    }
