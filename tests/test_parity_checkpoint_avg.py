"""parity/checkpoint_avg.py: arithmetic of the vendored ``raven_utils.average_checkpoints``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from parity import checkpoint_avg as A  # noqa: E402
from parity import upstream  # noqa: E402

upstream.activate()
from raven_utils import average_checkpoints  # noqa: E402


def _states(n=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for k in range(n):
        out.append({
            "w": torch.randn(4, 3, generator=g),
            "bn.running_mean": torch.randn(5, generator=g),
            "bn.running_var": torch.rand(5, generator=g) + 0.5,
            "bn.num_batches_tracked": torch.tensor(10 * (k + 1) + k, dtype=torch.long),
        })
    return out


def test_floats_are_the_arithmetic_mean():
    states = _states(3)
    avg = A.average_state_dicts(states)
    for key in ("w", "bn.running_mean", "bn.running_var"):
        assert torch.allclose(avg[key], sum(s[key] for s in states) / 3, atol=1e-7)
        assert avg[key].dtype == torch.float32


def test_integer_buffers_use_the_floor_of_the_mean_like_upstream():
    states = _states(3)                                    # num_batches_tracked 10, 21, 32 -> sum 63
    avg = A.average_state_dicts(states)
    assert avg["bn.num_batches_tracked"].dtype == torch.long
    assert int(avg["bn.num_batches_tracked"]) == 63 // 3
    two = A.average_state_dicts([{"n": torch.tensor(1)}, {"n": torch.tensor(2)}])
    assert int(two["n"]) == 1                              # (1 + 2) // 2: floor, not round
    neg = A.average_state_dicts([{"n": torch.tensor(-1)}, {"n": torch.tensor(-2)}])
    assert int(neg["n"]) == -2                             # floor division of the sum, as `//=`


def test_inputs_are_not_modified_and_one_checkpoint_is_the_identity():
    states = _states(3)
    before = [{k: v.clone() for k, v in s.items()} for s in states]
    A.average_state_dicts(states)
    for s, b in zip(states, before):
        assert all(torch.equal(s[k], b[k]) for k in s)
    single = A.average_state_dicts(states[:1])
    assert all(torch.equal(single[k], states[0][k]) for k in states[0])


def test_identical_to_the_vendored_function_on_lightning_layout_files(tmp_path):
    states = _states(4, seed=3)
    paths = []
    for i, s in enumerate(states):
        p = tmp_path / f"ckpt{i}.ckpt"
        torch.save({"state_dict": {f"model.{k}": v for k, v in s.items()}}, p)
        paths.append(p)
    upstream_avg = average_checkpoints(paths)
    ours = A.average_state_dicts(states)
    assert list(upstream_avg) == list(ours)
    for k in ours:
        assert torch.equal(upstream_avg[k], ours[k]), k       # bit-identical, floats and integers


def test_average_of_weight_files_round_trips_through_disk(tmp_path):
    states = _states(3)
    paths = []
    for i, s in enumerate(states):
        p = tmp_path / f"w{i}.pt"
        torch.save(s, p)
        paths.append(p)
    via_files = A.average_checkpoint_files(paths)
    direct = A.average_state_dicts(states)
    assert all(torch.equal(via_files[k], direct[k]) for k in direct)
    bad = tmp_path / "bad.pt"
    torch.save({"state_dict": {}}, bad)
    with pytest.raises(ValueError, match="plain state_dict"):
        A.average_checkpoint_files([bad])


def test_mismatching_checkpoints_are_refused():
    a = _states(2)
    b = dict(a[1])
    b.pop("w")
    with pytest.raises(ValueError, match="different keys"):
        A.average_state_dicts([a[0], b])
    c = dict(a[1])
    c["w"] = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="shape/dtype"):
        A.average_state_dicts([a[0], c])
    with pytest.raises(ValueError, match="nothing to average"):
        A.average_state_dicts([])
