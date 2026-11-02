"""parity/trainer.py: the plain-PyTorch loop (PROTOCOL §3.34), run on tiny synthetic data.

Real FLAC files in a staged-layout tree, the real vendored model with small transformer widths,
the real data pipeline and checkpoint code; nothing is mocked except the W&B run object.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from parity import checkpoint_avg  # noqa: E402
from parity import model as parity_model  # noqa: E402
from parity import optim  # noqa: E402
from parity import recipes as R  # noqa: E402
from parity import train_data as D  # noqa: E402
from parity import trainer as T  # noqa: E402
from parity_train_helpers import FakeRun, make_world, tiny_arch  # noqa: E402

N_UNITS = 20
TOKENS = ["<blank>"] + [f"U{i}" for i in range(1, N_UNITS + 1)] + ["<eos>"]
TRAIN_DURS = [1.0, 1.5, 2.0, 1.2, 1.8, 1.4, 1.1, 1.6]
VAL_DURS = [1.3, 1.7, 1.0]


@pytest.fixture(autouse=True)
def _remove_run_outputs(tmp_path):
    """Every training run writes 50-130 MB of checkpoints even with the small model; pytest keeps tmp dirs of
    the whole session, which would not fit on a nearly full disk. Remove them as soon as each test is done."""
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


def small_recipe(name="R-B", **over):
    base = dict(epochs=3, warmup_epochs=1, micro_batch_seconds=4.0, effective_batch_seconds=8.0,
                beam_size=2)
    base.update(over)
    return dataclasses.replace(R.load_recipe(name), **base)


@pytest.fixture(scope="module")
def worlds(tmp_path_factory):
    root = tmp_path_factory.mktemp("staged")
    train = make_world(root, TRAIN_DURS, n_units=N_UNITS, seed=0, video="tr")
    val = make_world(root, VAL_DURS, n_units=N_UNITS, seed=100, video="va")
    token_ids = {**train["token_ids"], **val["token_ids"]}
    resolver = train["resolver"]
    return {"train": train["records"], "val": val["records"], "ids": token_ids, "resolver": resolver}


def build(worlds, tmp_path, recipe=None, *, run=None, out="run", seed_model=0, train_records=None,
          val_records=None, **opts):
    recipe = recipe or small_recipe()
    torch.manual_seed(seed_model)
    model = parity_model.build_model(recipe.arch_config(tiny_arch()), odim=len(TOKENS))
    common = dict(seed=recipe.seed, time_mask_window_s=recipe.time_mask_window_s,
                  time_mask_stride_s=recipe.time_mask_stride_s, path_resolver=worlds["resolver"])
    train_ds = D.ParityTrainDataset(train_records or worlds["train"], worlds["ids"], augment=True, **common)
    val_ds = D.ParityTrainDataset(val_records or worlds["val"], worlds["ids"], augment=False, **common)
    options = T.TrainerOptions(out_dir=tmp_path / out, path_resolver=worlds["resolver"], report_every_s=0.0,
                               provenance={"tokens": "x"}, **opts)
    return T.Trainer(recipe=recipe, model=model, train_dataset=train_ds, val_dataset=val_ds,
                     tokens=TOKENS, options=options, run=run)


def comparable(history):
    drop = {"epoch_wall_s", "epoch_train_s"}
    return [{k: v for k, v in row.items() if k not in drop} for row in history]


# ------------------------------------------------------------------ refusals -----------

def test_the_official_test_split_is_refused(worlds, tmp_path):
    leaked = [dataclasses.replace(worlds["val"][0], split="test")] + worlds["val"][1:]
    with pytest.raises(T.TrainerError, match="official test split"):
        build(worlds, tmp_path, val_records=leaked)
    leaked_train = [dataclasses.replace(worlds["train"][0], split="test")] + worlds["train"][1:]
    with pytest.raises(T.TrainerError, match="official test split"):
        build(worlds, tmp_path, train_records=leaked_train)


def test_train_val_overlap_is_refused(worlds, tmp_path):
    with pytest.raises(T.TrainerError, match="both train and val"):
        build(worlds, tmp_path, val_records=worlds["train"][:2])


def test_model_and_recipe_must_agree(worlds, tmp_path):
    recipe = small_recipe()
    model = parity_model.build_model(recipe.arch_config(tiny_arch()) | {"drop_path": 0.3}, odim=len(TOKENS))
    train_ds = D.ParityTrainDataset(worlds["train"], worlds["ids"], seed=recipe.seed, augment=True,
                                    time_mask_window_s=0.4, time_mask_stride_s=1.0,
                                    path_resolver=worlds["resolver"])
    val_ds = D.ParityTrainDataset(worlds["val"], worlds["ids"], seed=recipe.seed, augment=False,
                                  time_mask_window_s=0.4, time_mask_stride_s=1.0,
                                  path_resolver=worlds["resolver"])
    with pytest.raises(T.TrainerError, match="drop-path"):
        T.Trainer(recipe=recipe, model=model, train_dataset=train_ds, val_dataset=val_ds, tokens=TOKENS,
                  options=T.TrainerOptions(out_dir=tmp_path / "x"))
    with pytest.raises(T.TrainerError, match="odim"):
        T.Trainer(recipe=recipe, model=parity_model.build_model(recipe.arch_config(tiny_arch()), odim=30),
                  train_dataset=train_ds, val_dataset=val_ds, tokens=TOKENS,
                  options=T.TrainerOptions(out_dir=tmp_path / "x"))


def test_selection_rule_and_stop_epoch_are_validated(worlds, tmp_path):
    with pytest.raises(T.TrainerError, match="keep_last_weights"):
        build(worlds, tmp_path, small_recipe(model_selection="average_last_k", average_last_k=3),
              keep_last_weights=2)
    with pytest.raises(T.TrainerError, match="stop_after_epochs"):
        build(worlds, tmp_path, stop_after_epochs=4)
    with pytest.raises(T.TrainerError, match="micro_batch_seconds"):
        build(worlds, tmp_path, micro_batch_seconds=9.0)


def test_augmentation_flags_are_checked(worlds, tmp_path):
    recipe = small_recipe()
    model = parity_model.build_model(recipe.arch_config(tiny_arch()), odim=len(TOKENS))
    mk = lambda recs, aug: D.ParityTrainDataset(recs, worlds["ids"], seed=recipe.seed, augment=aug,
                                                time_mask_window_s=0.4, time_mask_stride_s=1.0,
                                                path_resolver=worlds["resolver"])
    with pytest.raises(T.TrainerError, match="must augment"):
        T.Trainer(recipe=recipe, model=model, train_dataset=mk(worlds["train"], False),
                  val_dataset=mk(worlds["val"], False), tokens=TOKENS, options=T.TrainerOptions(out_dir=tmp_path))


# ------------------------------------------------------------------ a full run ---------

def test_a_complete_run_writes_the_bundle_and_logs_live(worlds, tmp_path):
    run = FakeRun()
    tr = build(worlds, tmp_path, run=run, keep_last_weights=10)
    summary = tr.fit()
    out = tmp_path / "run"
    names = sorted(p.name for p in out.iterdir())
    assert names == sorted(["last.pt", "weights_epoch0001.pt", "weights_epoch0002.pt", "weights_epoch0003.pt",
                            "final.pt", "metrics.jsonl", "summary.json", "val_beam_predictions.jsonl"])
    assert summary["complete"] is True and summary["epochs_done"] == 3 and summary["global_step"] == 3 * tr.steps_per_epoch
    assert [json.loads(l)["epoch"] for l in (out / "metrics.jsonl").read_text().splitlines()] == [1, 2, 3]
    on_disk = json.loads((out / "summary.json").read_text())
    assert on_disk["schema"] == T.SUMMARY_SCHEMA and on_disk["recipe"]["sha256"] == R.RECIPES_SHA256
    assert on_disk["final"]["selection"] == {"rule": "final_epoch", "epoch": 3}
    assert 0.0 <= on_disk["final"]["val_beam"]["corpus_wer"] and on_disk["final"]["val_beam"]["n_utts"] == 3
    # final.pt is a plain state_dict that parity_eval / load_released_checkpoint can read
    state = torch.load(out / "final.pt", map_location="cpu", weights_only=True)
    assert set(state) == set(tr.model.state_dict())
    assert all(torch.equal(state[k], v.cpu()) for k, v in tr.model.state_dict().items())

    # W&B: live, per-update, parity_train/ series only, never train/* or val/*
    keys = {k for payload in run.logged for k in payload}
    assert keys and all(k.startswith("parity_train/") for k in keys)
    assert not any(k.split("/")[0] in ("train", "val") for k in keys)
    per_update = [p for p in run.logged if "parity_train/loss" in p]
    assert len(per_update) == summary["global_step"]
    for needed in ("loss", "loss_ctc", "loss_att", "acc", "grad_norm", "update_s", "audio_s_per_s",
                   "lr/uniform", "update", "epoch", "data_wait_s", "n_utts"):
        assert f"parity_train/{needed}" in per_update[0], needed
    assert [p["parity_train/update"] for p in per_update] == list(range(1, summary["global_step"] + 1))
    epochs = [p for p in run.logged if "parity_train/epoch/val_loss" in p]
    assert [p["parity_train/epoch/epoch"] for p in epochs] == [1, 2, 3]
    assert all("parity_train/epoch/eta_s" in p and "parity_train/epoch/val_ctc_greedy_wer" in p for p in epochs)
    assert any(k.startswith("parity_train/progress/") for k in keys)       # wall-clock progress + ETA
    assert any(k.startswith("parity_train/beam_val/") for k in keys)
    assert any(k.startswith("parity_train/final/") for k in keys)
    assert ("parity_train/*",) in [a for a, _ in run.defined] or any(a == ("parity_train/*",) for a, _ in run.defined)


def test_lr_follows_the_per_update_schedule_across_epochs(worlds, tmp_path):
    run = FakeRun()
    tr = build(worlds, tmp_path, run=run, stop_after_epochs=3)
    tr.fit()
    logged = [p["parity_train/lr/uniform"] for p in run.logged if "parity_train/lr/uniform" in p]
    total = tr.schedule.total_steps
    want = [optim.lr_at(s, base_lr=3e-3, min_lr=1e-5, warmup_steps=tr.schedule.warmup_steps, total_steps=total)
            for s in range(1, total + 1)]
    assert logged == pytest.approx(want, rel=1e-12)
    assert logged[0] < logged[tr.schedule.warmup_steps - 1] <= 3e-3         # warming up
    assert logged[-1] == pytest.approx(1e-5, abs=1e-12)                     # cosine reached min_lr


def test_validation_runs_in_eval_mode_and_the_model_is_back_in_train_mode(worlds, tmp_path):
    tr = build(worlds, tmp_path, stop_after_epochs=1)
    seen = []
    tr.model.register_forward_pre_hook(lambda module, args, kwargs: seen.append((module.training, torch.is_grad_enabled())),
                                       with_kwargs=True)
    tr.fit()
    train_calls = [s for s in seen if s == (True, True)]
    eval_calls = [s for s in seen if s == (False, False)]
    assert len(train_calls) + len(eval_calls) == len(seen)               # nothing ran in a mixed state
    assert train_calls and eval_calls                                      # both phases happened
    assert seen[:len(train_calls)] == train_calls                          # all training first, then validation
    assert tr.model.training is True                                       # back in train mode afterwards
    row = tr.history[0]
    assert {"val_loss", "val_loss_ctc", "val_loss_att", "val_acc", "val_ctc_greedy_wer"} <= set(row)


# ------------------------------------------------------------------ resume -------------

@pytest.mark.parametrize("recipe_name", ["R-B", "R-A"])
def test_resume_is_bit_for_bit_equivalent_to_an_uninterrupted_run(worlds, tmp_path, recipe_name):
    """N epochs straight vs 1 epoch, stop, resume in a fresh process-like trainer, N-1 more."""
    recipe = small_recipe(recipe_name, epochs=3, warmup_epochs=1, drop_path=0.2)
    straight = build(worlds, tmp_path, recipe, out="straight", keep_last_weights=3)
    sum_a = straight.fit()

    first = build(worlds, tmp_path, recipe, out="split", keep_last_weights=3, stop_after_epochs=1)
    first.fit()
    assert (tmp_path / "split" / "last.pt").is_file()
    second = build(worlds, tmp_path, recipe, out="split", keep_last_weights=3, resume=True, seed_model=999)
    sum_b = second.fit()

    sd_a, sd_b = straight.model.state_dict(), second.model.state_dict()
    assert set(sd_a) == set(sd_b)
    for k in sd_a:
        assert torch.equal(sd_a[k], sd_b[k]), k                          # exact, incl. BatchNorm statistics
    oa, ob = straight.optimizer.state_dict(), second.optimizer.state_dict()
    assert oa["param_groups"] == ob["param_groups"]
    for pid in oa["state"]:
        for name, va in oa["state"][pid].items():
            assert torch.equal(va, ob["state"][pid][name]), (pid, name)
    assert comparable(sum_a["history"]) == comparable(sum_b["history"])  # same losses and WERs every epoch
    assert sum_b["resumed_from_epoch"] == 1 and sum_b["global_step"] == sum_a["global_step"]
    fa = torch.load(tmp_path / "straight" / "final.pt", weights_only=True)
    fb = torch.load(tmp_path / "split" / "final.pt", weights_only=True)
    assert all(torch.equal(fa[k], fb[k]) for k in fa)


def test_loader_worker_count_does_not_change_the_training_result(worlds, tmp_path):
    a = build(worlds, tmp_path, out="w0", stop_after_epochs=1, num_workers=0)
    b = build(worlds, tmp_path, out="w2", stop_after_epochs=1, num_workers=2)
    sa = a.fit()
    torch.rand(1000)                                       # fit() starts from the recipe seed, not from whatever ran before
    sb = b.fit()
    assert comparable(sa["history"]) == comparable(sb["history"])
    for k, v in a.model.state_dict().items():
        assert torch.equal(v, b.model.state_dict()[k]), k


def test_micro_batch_override_is_recorded_as_a_deviation(worlds, tmp_path):
    base = build(worlds, tmp_path, out="a", stop_after_epochs=1).fit()
    over = build(worlds, tmp_path, out="b", stop_after_epochs=1, micro_batch_seconds=2.5).fit()
    assert base["micro_batch_seconds_override"] is None and base["micro_batch_seconds"] == 4.0
    assert over["micro_batch_seconds_override"] == 2.5 and over["micro_batch_seconds"] == 2.5


def test_batches_after_a_resume_are_the_same_micro_batches(worlds, tmp_path):
    tr = build(worlds, tmp_path)
    assert tr._plan(2) == tr._plan(2) and tr._plan(2) != tr._plan(3)
    state = tr.state()
    assert state["sampler"] == {"seed": 42, "next_epoch": 0, "step_in_epoch": 0}


def test_resume_refusals(worlds, tmp_path):
    first = build(worlds, tmp_path, stop_after_epochs=1)
    first.fit()
    with pytest.raises(T.TrainerError, match="not empty"):
        build(worlds, tmp_path).fit()                                       # never overwrite a run
    with pytest.raises(T.TrainerError, match="does not exist"):
        build(worlds, tmp_path, out="nothing", resume=True).fit()
    other = small_recipe(weight_decay=0.5)
    with pytest.raises(T.TrainerError, match="different recipe"):
        build(worlds, tmp_path, other, resume=True).fit()
    tr = build(worlds, tmp_path, resume=True)
    tr.opt.provenance = {"tokens": "changed"}
    with pytest.raises(T.TrainerError, match="provenance"):
        tr.fit()
    tr = build(worlds, tmp_path, resume=True, micro_batch_seconds=2.0)
    with pytest.raises(T.TrainerError, match="batching"):
        tr.fit()


def test_resume_of_a_finished_run_is_refused(worlds, tmp_path):
    build(worlds, tmp_path).fit()
    with pytest.raises(T.TrainerError, match="already at epoch"):
        build(worlds, tmp_path, resume=True).fit()


def test_stopped_timing_run_has_no_final_evaluation_and_can_be_continued(worlds, tmp_path):
    summary = build(worlds, tmp_path, stop_after_epochs=1).fit()
    assert summary["complete"] is False and "final" not in summary and summary["epochs_done"] == 1
    assert not (tmp_path / "run" / "final.pt").exists()
    done = build(worlds, tmp_path, resume=True).fit()
    assert done["complete"] is True and done["epochs_done"] == 3


# ------------------------------------------------------------------ checkpoints --------

def test_atomic_save_leaves_the_previous_file_when_the_write_fails(tmp_path, monkeypatch):
    path = tmp_path / "last.pt"
    T.atomic_torch_save({"v": 1}, path)
    assert torch.load(path, weights_only=True) == {"v": 1} and not list(tmp_path.glob("*.tmp"))

    real_save = torch.save

    def failing_save(obj, f, *a, **kw):
        f.write(b"half a checkpoint")
        raise OSError("disk full")

    monkeypatch.setattr(torch, "save", failing_save)
    with pytest.raises(OSError):
        T.atomic_torch_save({"v": 2}, path)
    monkeypatch.setattr(torch, "save", real_save)
    assert torch.load(path, weights_only=True) == {"v": 1}                  # untouched
    assert not list(tmp_path.glob("*.tmp"))                                  # no debris


def test_only_the_last_k_weight_files_are_kept_and_the_bundle_stays_small(worlds, tmp_path):
    tr = build(worlds, tmp_path, small_recipe(epochs=4), keep_last_weights=2, stop_after_epochs=4)
    tr.fit()
    assert sorted(p.name for p in (tmp_path / "run").glob("weights_epoch*.pt")) == \
        ["weights_epoch0003.pt", "weights_epoch0004.pt"]
    assert len(list((tmp_path / "run").iterdir())) <= T.FILES_CAP


def test_file_cap_is_enforced(worlds, tmp_path, monkeypatch):
    monkeypatch.setattr(T, "FILES_CAP", 3)
    with pytest.raises(T.TrainerError, match="cap"):
        build(worlds, tmp_path, stop_after_epochs=2).fit()


def test_final_epoch_selection_keeps_the_last_epoch_weights(worlds, tmp_path):
    build(worlds, tmp_path, small_recipe(epochs=3)).fit()
    out = tmp_path / "run"
    last = torch.load(out / "weights_epoch0003.pt", weights_only=True)
    final = torch.load(out / "final.pt", weights_only=True)
    assert all(torch.equal(last[k], final[k]) for k in last)


def test_average_last_k_selection_averages_the_weight_files(worlds, tmp_path):
    recipe = small_recipe(epochs=3, model_selection="average_last_k", average_last_k=2)
    summary = build(worlds, tmp_path, recipe, keep_last_weights=2).fit()
    out = tmp_path / "run"
    assert summary["final"]["selection"] == {"rule": "average_last_k", "epochs": [2, 3]}
    avg = checkpoint_avg.average_checkpoint_files([out / "weights_epoch0002.pt", out / "weights_epoch0003.pt"])
    final = torch.load(out / "final.pt", weights_only=True)
    assert all(torch.equal(avg[k], final[k]) for k in avg)
    last = torch.load(out / "weights_epoch0003.pt", weights_only=True)
    assert not all(torch.equal(last[k], final[k]) for k in last)
    assert "val" in summary["final"] and summary["final"]["val"]["loss"] > 0


# ------------------------------------------------------------------ evaluation ---------

def test_ctc_greedy_collapses_repeats_then_drops_blanks():
    assert T.ctc_greedy_ids([0, 3, 3, 0, 3, 4, 4, 0, 0, 5]) == [3, 3, 4, 5]
    assert T.ctc_greedy_ids([0, 0, 0]) == []
    assert T.ctc_greedy_ids([7, 7, 7]) == [7]
    assert T.ctc_greedy_ids([1, 0, 1]) == [1, 1]                     # a blank separates genuine repeats


def test_evaluate_is_consistent_across_batch_sizes_and_refuses_augmentation(worlds, tmp_path):
    tr = build(worlds, tmp_path)
    tr.model.eval()
    kw = dict(device=torch.device("cpu"), num_workers=0)
    big = T.evaluate(tr.model, tr.val_ds, TOKENS, micro_batch_seconds=8.0, **kw)
    small = T.evaluate(tr.model, tr.val_ds, TOKENS, micro_batch_seconds=2.0, **kw)
    assert big["n_utts"] == small["n_utts"] == 3
    assert big["loss"] == pytest.approx(small["loss"], rel=2e-3)     # only the front-end's last frame differs
    assert tr.model.training is False                                # restored to what it was
    with pytest.raises(ValueError, match="must not augment"):
        T.evaluate(tr.model, tr.train_ds, TOKENS, micro_batch_seconds=8.0, **kw)


def test_periodic_beam_pass_is_logged_and_timed_for_the_timing_run(worlds, tmp_path):
    summary = build(worlds, tmp_path, stop_after_epochs=1, beam_val_every=1, beam_val_limit=2).fit()
    row = summary["history"][0]
    assert 0.0 <= row["val_beam_wer"] and row["val_beam_n_utts"] == 2 and row["val_beam_decode_s"] > 0
    assert (tmp_path / "run" / "val_beam_predictions.jsonl").is_file() and summary["complete"] is False


def test_beam_evaluation_limit_is_flagged_as_a_subset(worlds, tmp_path):
    tr = build(worlds, tmp_path)
    tr.model.eval()
    res = T.beam_evaluate(tr.model, tr.val_ds.records, TOKENS, device=torch.device("cpu"), beam_size=2,
                          ctc_weight=0.1, limit=2, pred_path=tmp_path / "p.jsonl",
                          path_resolver=worlds["resolver"])
    assert res["n_utts"] == 2 and res["is_subset"] is True and res["decode_time_s"] > 0
    assert len((tmp_path / "p.jsonl").read_text().splitlines()) == 2


# ------------------------------------------------------------------ misc ---------------

def test_fp32_deterministic_mode_sets_and_restores_global_flags():
    before = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
              torch.are_deterministic_algorithms_enabled())
    with T.fp32_deterministic_mode():
        assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.deterministic is True and torch.backends.cudnn.benchmark is False
        assert torch.are_deterministic_algorithms_enabled() and torch.is_deterministic_algorithms_warn_only_enabled()
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
            torch.are_deterministic_algorithms_enabled()) == before


def test_fp32_mode_without_deterministic_kernels_still_pins_fp32_and_restores():
    before = (torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled(),
              torch.backends.cuda.matmul.allow_tf32)
    with T.fp32_deterministic_mode(deterministic_kernels=False):
        assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.benchmark is False
        assert torch.are_deterministic_algorithms_enabled() == before[1]       # untouched
    assert (torch.backends.cudnn.deterministic, torch.are_deterministic_algorithms_enabled(),
            torch.backends.cuda.matmul.allow_tf32) == before


def test_seed_everything_seeds_torch_numpy_and_python():
    import random
    import numpy as np
    T.seed_everything(7)
    a = (torch.rand(2).tolist(), float(np.random.rand()), random.random())
    T.seed_everything(7)
    assert a == (torch.rand(2).tolist(), float(np.random.rand()), random.random())


def test_production_modules_do_not_pull_in_sentencepiece_or_lightning():
    code = ("import sys; sys.path.insert(0, 'scripts'); import parity.trainer, parity.train_data, parity.optim, "
            "parity.checkpoint_avg, parity.recipes, parity.spm_tokens, parity_train; "
            "bad = [m for m in ('sentencepiece', 'lightning', 'pytorch_lightning', 'torchaudio') if m in sys.modules]; "
            "assert not bad, bad")
    subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, check=True)


def test_param_groups_and_group_table_are_in_the_summary_for_the_layer_decay_recipe(worlds, tmp_path):
    summary = build(worlds, tmp_path, small_recipe("R-A", epochs=2, warmup_epochs=1),
                    stop_after_epochs=1).fit()
    names = [g["name"] for g in summary["param_groups"]]
    assert "block_0" in names and "ctc" in names and "other" in names and "frontend" in names
    assert summary["recipe"]["layer_decay"] == 0.5 and copy.deepcopy(summary["steps_per_epoch"]) >= 1
