"""parity/optim.py: AdamW parameter groups (uniform / upstream layer-wise decay) and the per-update
warm-up + cosine schedule, against independent references (PROTOCOL §3.34)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from parity import model as parity_model  # noqa: E402
from parity import optim  # noqa: E402
from parity import recipes as R  # noqa: E402
from parity_train_helpers import tiny_arch  # noqa: E402


# Auto-AVSR's scheduler, copied from github.com/mpc001/auto_avsr (Apache-2.0), file cosine.py,
# with the deprecated `verbose` argument dropped. It is the independent reference of the formula.
class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, steps_per_epoch, last_epoch=-1):
        self.warmup_steps = warmup_epochs * steps_per_epoch
        self.total_steps = total_epochs * steps_per_epoch
        super().__init__(optimizer, last_epoch=last_epoch)

    def get_lr(self):
        if self._step_count < self.warmup_steps:
            return [self._step_count / self.warmup_steps * base_lr for base_lr in self.base_lrs]
        decay_steps = self.total_steps - self.warmup_steps
        cos_val = math.cos(math.pi * (self._step_count - self.warmup_steps) / decay_steps)
        return [0.5 * base_lr * (1 + cos_val) for base_lr in self.base_lrs]


def _recipe(name="R-A", **over):
    import dataclasses
    return dataclasses.replace(R.load_recipe(name), **over)


# ------------------------------------------------------------------ schedule ----------

@pytest.mark.parametrize("warmup_epochs,epochs,steps", [(3, 10, 7), (20, 50, 5), (1, 2, 13), (0, 4, 3)])
def test_schedule_equals_autoavsrs_scheduler_for_min_lr_zero(warmup_epochs, epochs, steps):
    peaks = [1e-3, 5e-3]
    params = [torch.nn.Parameter(torch.zeros(1)) for _ in peaks]
    opt = torch.optim.AdamW([{"params": [p], "lr": lr} for p, lr in zip(params, peaks)], lr=1.0)
    ref = WarmupCosineScheduler(opt, warmup_epochs, epochs, steps)
    total, warm = epochs * steps, warmup_epochs * steps
    for u in range(total):
        want = [g["lr"] for g in opt.param_groups]                 # the LR update u would run at
        got = [optim.lr_at(u + 1, base_lr=b, min_lr=0.0, warmup_steps=warm, total_steps=total) for b in peaks]
        assert got == pytest.approx(want, rel=1e-12, abs=1e-15), u
        opt.step()
        ref.step()


def test_warmup_is_linear_from_zero_and_the_first_update_runs_at_peak_over_warmup():
    kw = dict(base_lr=2e-3, min_lr=1e-5, warmup_steps=100, total_steps=1000)
    assert optim.lr_at(1, **kw) == pytest.approx(2e-3 / 100)
    steps = [optim.lr_at(s, **kw) for s in range(1, 100)]
    diffs = [b - a for a, b in zip(steps, steps[1:])]
    assert max(diffs) - min(diffs) < 1e-15 and diffs[0] == pytest.approx(2e-3 / 100)
    assert optim.lr_at(100, **kw) == pytest.approx(2e-3)           # end of warm-up = peak


def test_cosine_decays_monotonically_to_min_lr_and_hits_the_midpoint():
    kw = dict(base_lr=3e-3, min_lr=1e-5, warmup_steps=100, total_steps=500)
    vals = [optim.lr_at(s, **kw) for s in range(100, 501)]
    assert all(b <= a + 1e-18 for a, b in zip(vals, vals[1:]))
    assert vals[0] == pytest.approx(3e-3) and vals[-1] == pytest.approx(1e-5, abs=1e-15)
    assert optim.lr_at(300, **kw) == pytest.approx((3e-3 + 1e-5) / 2)       # halfway through the decay
    ref = lambda s: 1e-5 + 0.5 * (3e-3 - 1e-5) * (1 + math.cos(math.pi * (s - 100) / 400))
    assert [optim.lr_at(s, **kw) for s in (150, 250, 400)] == pytest.approx([ref(s) for s in (150, 250, 400)])


def test_a_group_whose_peak_is_at_or_below_min_lr_stays_constant_after_warmup():
    kw = dict(min_lr=1e-5, warmup_steps=10, total_steps=100)
    assert [optim.lr_at(s, base_lr=1e-5, **kw) for s in (10, 50, 100)] == pytest.approx([1e-5] * 3)


def test_the_schedule_steps_every_update_not_every_epoch():
    recipe = _recipe("R-B")
    sched = optim.UpdateSchedule(recipe, steps_per_epoch=530)
    assert sched.warmup_steps == 20 * 530 and sched.total_steps == 75 * 530
    first_epoch = [sched.lrs(s, [3e-3])[0] for s in range(1, 531)]
    assert len(set(first_epoch)) == 530                              # a new LR at every update
    with pytest.raises(ValueError):
        sched.lrs(0, [1.0])
    with pytest.raises(ValueError):
        sched.lrs(sched.total_steps + 1, [1.0])


def test_apply_sets_every_groups_lr_from_its_own_peak():
    model = parity_model.build_model(tiny_arch(), odim=30)
    recipe = _recipe("R-A", epochs=4, warmup_epochs=1)
    opt = optim.build_optimizer(model, recipe)
    sched = optim.UpdateSchedule(recipe, steps_per_epoch=10)
    lrs = sched.apply(opt, 5)
    assert [g["lr"] for g in opt.param_groups] == lrs
    assert lrs == [optim.lr_at(5, base_lr=g["base_lr"], min_lr=1e-5, warmup_steps=10, total_steps=40)
                   for g in opt.param_groups]


# ------------------------------------------------------------------ param groups -------

def _expected_groups(model, enc_lr, dec_lr, decay, min_lr):
    """Group membership and LR re-derived from the documented rule (names -> scale)."""
    n = len(model.encoder.encoders)
    scale = lambda i: decay ** (n - 1 - i)
    floor = lambda v: max(v, min_lr)
    groups: dict[str, tuple[float, set[int]]] = {}
    for name, p in model.named_parameters():
        if name.startswith("encoder.after_norm"):
            key, lr = "after_norm", floor(enc_lr)
        elif name.startswith("encoder.embed") or name.startswith("encoder.frontend"):
            key, lr = name.split(".")[1], floor(scale(0) * enc_lr)
        elif name.startswith("encoder.encoders."):
            i = int(name.split(".")[2])
            key, lr = f"block_{i}", floor(scale(i) * enc_lr)
        elif name.startswith("ctc"):
            key, lr = "ctc", floor(dec_lr)
        else:
            assert name.startswith("decoder")
            key, lr = "other", floor(dec_lr)
        groups.setdefault(key, (lr, set()))[1].add(id(p))
    return groups


def test_layer_decay_groups_match_the_rule_for_the_full_depth():
    model = parity_model.build_model(tiny_arch(elayers=12), odim=30)
    recipe = _recipe("R-A")
    opt = optim.build_optimizer(model, recipe)
    want = _expected_groups(model, 2e-3, 6e-3, 0.5, 1e-5)
    got = {g["name"]: (g["base_lr"], {id(p) for p in g["params"]}) for g in opt.param_groups}
    assert set(got) == set(want)
    for name in want:
        assert got[name][0] == pytest.approx(want[name][0], rel=1e-12), name
        assert got[name][1] == want[name][1], name
    # the literal numbers a reviewer wants to see for R-A
    lr = {k: v[0] for k, v in got.items()}
    assert lr["block_11"] == pytest.approx(2e-3) and lr["after_norm"] == pytest.approx(2e-3)
    assert lr["block_10"] == pytest.approx(1e-3) and lr["block_9"] == pytest.approx(5e-4)
    assert lr["block_4"] == pytest.approx(2e-3 * 0.5 ** 7) and lr["block_4"] > 1e-5
    for floored in ("block_3", "block_2", "block_1", "block_0", "embed", "frontend"):
        assert lr[floored] == pytest.approx(1e-5), floored                 # decayed below min_lr: floored
    assert lr["ctc"] == pytest.approx(6e-3) and lr["other"] == pytest.approx(6e-3)


def test_every_trainable_parameter_is_in_exactly_one_group_for_both_modes():
    model = parity_model.build_model(tiny_arch(), odim=30)
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    for name in R.RECIPE_NAMES:
        opt = optim.build_optimizer(model, R.load_recipe(name))
        ids = [id(p) for g in opt.param_groups for p in g["params"]]
        assert len(ids) == len(set(ids)) == len([p for p in model.parameters() if p.requires_grad])
        assert sum(r["n_params"] for r in optim.group_table(opt)) == total


def test_without_layer_decay_there_is_one_uniform_group():
    model = parity_model.build_model(tiny_arch(), odim=30)
    opt = optim.build_optimizer(model, R.load_recipe("R-B"))
    assert [g["name"] for g in opt.param_groups] == ["uniform"]
    assert opt.param_groups[0]["base_lr"] == 3e-3


def test_adamw_settings_and_decay_on_every_parameter():
    model = parity_model.build_model(tiny_arch(), odim=30)
    for name, wd in (("R-A", 0.1), ("R-B", 0.04)):
        opt = optim.build_optimizer(model, R.load_recipe(name))
        assert isinstance(opt, torch.optim.AdamW)
        for g in opt.param_groups:
            assert g["betas"] == (0.9, 0.98) and g["weight_decay"] == wd
        assert sum(len(g["params"]) for g in opt.param_groups) == len(list(model.parameters()))


def test_the_vendored_get_param_groups_is_what_builds_the_layer_decay_groups():
    model = parity_model.build_model(tiny_arch(), odim=30)
    direct = optim.get_param_groups(model, len(model.encoder.encoders), 2e-3, 6e-3, 0.5,
                                    ctc_equals_other=True, min_lr=1e-5)
    built = optim.build_param_groups(model, R.load_recipe("R-A"))
    assert [(g["name"], g["lr"]) for g in direct] == [(g["name"], g["lr"]) for g in built]
    assert optim.get_param_groups.__module__ == "raven_utils"
