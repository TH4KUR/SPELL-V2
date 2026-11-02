"""scripts/parity_train.py end to end on a tiny staged tree: real paths authority, real FLAC,
real tokenizer artifacts on disk, real recipe file (pinned R-A / R-B), real W&B module (disabled
mode) - a randomly initialised small model, so this proves the plumbing and the refusals, not a WER."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import parity_train as P  # noqa: E402
import paths as data_paths  # noqa: E402
from parity import spm_tokens as K  # noqa: E402
from parity import trainer as T  # noqa: E402
from parity_train_helpers import FakeRun, make_world, tiny_arch  # noqa: E402

INDEX_COLUMNS = ["utterance_id", "split", "video_id", "stem", "tokens_path", "audio_path",
                 "audio_kind", "txt_path", "n_tokens", "duration_s", "selectable", "conf",
                 "text_raw", "text_norm", "n_chars_norm"]
N_UNITS = 20


@pytest.fixture(autouse=True)
def _remove_run_outputs(tmp_path):
    """Every training run writes 50-130 MB of checkpoints even with the small model; pytest keeps tmp dirs of
    the whole session, which would not fit on a nearly full disk. Remove them as soon as each test is done."""
    yield
    shutil.rmtree(tmp_path, ignore_errors=True)


def _row(rec, split):
    return {"utterance_id": rec.utterance_id, "split": split, "video_id": rec.video_id, "stem": rec.stem,
            "tokens_path": "", "audio_path": "", "audio_kind": "flac", "txt_path": "", "n_tokens": 5,
            "duration_s": rec.duration_s, "selectable": split == "trainval", "conf": 5.0,
            "text_raw": rec.text_raw, "text_norm": rec.text_raw.lower(), "n_chars_norm": len(rec.text_raw)}


@pytest.fixture
def cli(tmp_path, monkeypatch):
    root = tmp_path / "data"
    train = make_world(root, [1.0, 1.5, 2.0, 1.2, 1.8, 1.4], n_units=N_UNITS, seed=0, video="tr")
    val = make_world(root, [1.3, 1.7], n_units=N_UNITS, seed=100, video="va")
    leak = make_world(root, [1.1], n_units=N_UNITS, seed=200, video="te")
    rows = [_row(r, "trainval") for r in train["records"] + val["records"]]
    test_rec = leak["records"][0]
    rows.append({**_row(test_rec, "test"), "utterance_id": "9999", "video_id": "", "stem": "9999"})
    index = tmp_path / "index.parquet"
    pd.DataFrame(rows, columns=INDEX_COLUMNS).to_parquet(index)
    ids = tmp_path / "ids"
    ids.mkdir()
    (ids / "train.txt").write_text("".join(f"{r.utterance_id}\n" for r in train["records"]))
    (ids / "val.txt").write_text("".join(f"{r.utterance_id}\n" for r in val["records"]))
    (ids / "test.txt").write_text("9999\n")
    tokens = tmp_path / "spm"
    tokens.mkdir()
    units = [f"U{i}" for i in range(1, N_UNITS + 1)]
    K.write_units_file(units, tokens / K.UNITS_FILENAME)
    K.write_token_ids({**train["token_ids"], **val["token_ids"]}, tokens / K.TOKEN_IDS_FILENAME)
    (tokens / K.MODEL_FILENAME).write_bytes(b"not a real model; the trainer never opens it")
    files = {n: {"sha256": K.sha256_path(tokens / n)}
             for n in (K.UNITS_FILENAME, K.TOKEN_IDS_FILENAME, K.MODEL_FILENAME)}
    K.write_json_atomic(K.provenance_document(sources={}, units=units, files=files, fidelity={},
                                              roundtrip={}, spm_version="x"), tokens / K.PROVENANCE_FILENAME)
    arch = tmp_path / "arch_tiny.yaml"
    arch.write_text(yaml.safe_dump(tiny_arch()))
    monkeypatch.setenv("SPELL_DATA_ROOT", str(root))
    monkeypatch.delenv("SPELL_FINAL_EVAL", raising=False)
    data_paths.reset_cache()
    yield {"tmp": tmp_path, "index": index, "ids": ids, "tokens": tokens, "arch": arch}
    data_paths.reset_cache()


def argv(c, *extra, recipe="R-B", out="out"):
    return ["--recipe", recipe, "--index", str(c["index"]), "--train-ids", str(c["ids"] / "train.txt"),
            "--val-ids", str(c["ids"] / "val.txt"), "--tokens-dir", str(c["tokens"]),
            "--out-dir", str(c["tmp"] / out), "--arch-config", str(c["arch"]), "--device", "cpu",
            "--num-workers", "0", "--wandb-mode", "disabled", "--job-id", "t1", *extra]


def test_timing_run_end_to_end(cli, capsys):
    assert P.main(argv(cli, "--stop-after-epochs", "1")) == 0
    out = cli["tmp"] / "out"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete"] is False and summary["epochs_done"] == 1
    assert summary["recipe"]["name"] == "R-B" and summary["steps_per_epoch"] >= 1
    assert summary["environment"]["ids_verified_against_provenance"] is False
    assert summary["environment"]["duration_check"]["n_checked"] > 0
    assert summary["provenance"]["n_units"] == N_UNITS and summary["provenance"]["n_train_utts"] == 6
    assert (out / "last.pt").is_file() and (out / "weights_epoch0001.pt").is_file()
    printed = capsys.readouterr().out
    assert "group uniform" in printed and "recipe R-B" in printed and "stopped after 1/75" in printed


def test_deterministic_kernels_are_the_default_and_the_choice_is_recorded(cli):
    seen = []
    real = P.trainer.fp32_deterministic_mode

    def spy(flag=True):
        seen.append(flag)
        return real(flag)

    P.trainer.fp32_deterministic_mode = spy
    try:
        assert P.main(argv(cli, "--stop-after-epochs", "1")) == 0
        assert P.main(argv(cli, "--stop-after-epochs", "1", "--no-deterministic-kernels", out="out2")) == 0
    finally:
        P.trainer.fp32_deterministic_mode = real
    assert seen == [True, False]
    for out, want in (("out", True), ("out2", False)):
        env = json.loads((cli["tmp"] / out / "summary.json").read_text())["environment"]
        assert env["deterministic_kernels"] is want


def test_two_cli_runs_with_the_same_recipe_give_identical_weights(cli):
    """Seeded initial weights, seeded sampler / mask / dropout: reruns are reproducible (CPU)."""
    import torch
    assert P.main(argv(cli, "--stop-after-epochs", "1", out="r1")) == 0
    torch.manual_seed(12345)                                    # whatever ran in between must not matter
    assert P.main(argv(cli, "--stop-after-epochs", "1", out="r2")) == 0
    a = torch.load(cli["tmp"] / "r1" / "weights_epoch0001.pt", weights_only=True)
    b = torch.load(cli["tmp"] / "r2" / "weights_epoch0001.pt", weights_only=True)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_layer_decay_recipe_prints_its_group_table(cli, capsys):
    assert P.main(argv(cli, "--stop-after-epochs", "1", recipe="R-A")) == 0
    printed = capsys.readouterr().out
    for name in ("frontend", "embed", "block_0", "block_1", "ctc", "other", "after_norm"):
        assert f"group {name}" in printed


def test_resume_through_the_cli_continues_the_same_directory(cli):
    assert P.main(argv(cli, "--stop-after-epochs", "1")) == 0
    assert P.main(argv(cli, "--stop-after-epochs", "2", "--resume")) == 0
    summary = json.loads((cli["tmp"] / "out" / "summary.json").read_text())
    assert summary["epochs_done"] == 2 and summary["resumed_from_epoch"] == 1
    assert [r["epoch"] for r in summary["history"]] == [1, 2]


def test_refuses_the_test_split_ids(cli, capsys):
    rc = P.main(argv(cli)[:6] + ["--val-ids", str(cli["ids"] / "test.txt")] + argv(cli)[8:])
    assert rc == 2
    assert "trainval" in capsys.readouterr().err
    assert not (cli["tmp"] / "out" / "last.pt").exists()
    rc = P.main(["--recipe", "R-B", "--index", str(cli["index"]), "--train-ids", str(cli["ids"] / "test.txt"),
                 "--val-ids", str(cli["ids"] / "val.txt"), "--tokens-dir", str(cli["tokens"]),
                 "--out-dir", str(cli["tmp"] / "o2"), "--arch-config", str(cli["arch"]), "--device", "cpu",
                 "--wandb-mode", "disabled"])
    assert rc == 2


def test_the_script_has_no_test_unlock_and_never_names_the_test_ids_file():
    src = (PROJECT_ROOT / "scripts" / "parity_train.py").read_text()
    assert "SPELL_FINAL_EVAL" not in src
    assert 'SPLIT_FILES["test"]' not in src and "parity_test_ids" not in src
    assert '"test"' not in src.replace('"""', "").split("def parse_args")[1].split("def main")[0].replace(
        "the official test split is never read", "")


def test_setup_refusals_exit_2(cli, capsys):
    assert P.main(argv(cli, "--resume", out="missing")) == 2                    # no last.pt
    assert "does not exist" in capsys.readouterr().err
    (cli["tokens"] / K.UNITS_FILENAME).write_text("<unk> 1\n")
    assert P.main(argv(cli, out="o3")) == 2                                      # tokenizer artifacts edited
    assert "tokenizer artifacts unusable" in capsys.readouterr().err
    assert P.main(["--recipe", "R-B", "--tokens-dir", str(cli["tokens"] / "nope"), "--index", str(cli["index"]),
                   "--train-ids", str(cli["ids"] / "train.txt"), "--val-ids", str(cli["ids"] / "val.txt"),
                   "--device", "cpu", "--wandb-mode", "disabled"]) == 2
    assert P.main(["--recipe", "R-B", "--train-ids", str(cli["ids"] / "train.txt"), "--device", "cpu",
                   "--wandb-mode", "disabled"]) == 2                              # only one of the two id lists


def test_a_tampered_recipe_file_is_refused_before_anything_runs(cli, tmp_path, monkeypatch, capsys):
    from parity import recipes as R
    edited = tmp_path / "recipes.yaml"
    edited.write_text(R.RECIPES_PATH.read_text().replace("seed: 42", "seed: 1"))
    monkeypatch.setattr(R.load_recipe, "__defaults__", (edited,))             # default path -> the edited copy
    opened = []
    monkeypatch.setattr(P.wandb_run, "init_run", lambda **kw: opened.append(kw))
    assert P.main(argv(cli)) == 2
    assert "RECIPES_SHA256" in capsys.readouterr().err
    assert opened == []                                                          # no W&B run for a refused recipe


def test_crash_marks_the_wandb_run_failed_and_run_name_obeys_the_naming_law(cli, monkeypatch):
    run = FakeRun()
    captured = {}

    def fake_init(**kw):
        captured.update(kw)
        return run

    def boom(self):
        raise RuntimeError("simulated crash")

    monkeypatch.setattr(P.wandb_run, "init_run", fake_init)
    monkeypatch.setattr(T.Trainer, "fit", boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        P.main(argv(cli))
    assert run.finished == [1] and "simulated crash" in run.summary["error"]
    assert captured["job_type"] == "parity_train" and captured["name"] == "parity-train-R-B-t1"
    P.wandb_run.check_run_name(captured["name"])                                  # flat, not a formal run id
    assert not captured["name"].startswith("track_")
    assert captured["config"]["recipe_config"]["sha256"]


def test_successful_run_finishes_the_wandb_run_cleanly_and_opens_it_first(cli, monkeypatch):
    order = []
    run = FakeRun()
    monkeypatch.setattr(P.wandb_run, "init_run", lambda **kw: order.append("init") or run)
    original = P._setup
    monkeypatch.setattr(P, "_setup", lambda *a, **k: order.append("setup") or original(*a, **k))
    assert P.main(argv(cli, "--stop-after-epochs", "1")) == 0
    assert order[:2] == ["init", "setup"] and run.finished == [0]
    assert run.summary["parity_train/epochs_done"] == 1 and "parity_train/steps_per_epoch" in run.summary


def test_final_checkpoint_evaluates_with_parity_eval_to_the_same_wer_as_the_trainer(cli, monkeypatch):
    """The trainer's end-of-run beam WER and scripts/parity_eval.py on the produced final.pt are
    the same decoding recipe on the same weights and data, so they must agree exactly."""
    import dataclasses

    import parity_eval as E
    short = dataclasses.replace(P.recipes.load_recipe("R-B"), epochs=2, warmup_epochs=1, beam_size=2)
    monkeypatch.setattr(P.recipes, "load_recipe", lambda name: short)
    assert P.main(argv(cli)) == 0
    out = cli["tmp"] / "out"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete"] is True and summary["final"]["selection"]["rule"] == "final_epoch"
    rc = E.main(["--checkpoint", str(out / "final.pt"), "--split", "val", "--index", str(cli["index"]),
                 "--ids", str(cli["ids"] / "val.txt"), "--units", str(cli["tokens"] / K.UNITS_FILENAME),
                 "--expect-sha256", "none", "--arch-config", str(cli["arch"]), "--device", "cpu",
                 "--beam-size", "2", "--num-workers", "0", "--wandb-mode", "disabled",
                 "--out-dir", str(cli["tmp"] / "eval")])
    assert rc == 0
    ev = json.loads((cli["tmp"] / "eval" / "summary.json").read_text())
    beam = summary["final"]["val_beam"]
    assert ev["corpus_wer"] == beam["corpus_wer"] and ev["total_edits"] == beam["total_edits"]
    assert ev["n_utts"] == beam["n_utts"] == 2 and ev["checkpoint"]["sha256"] == summary["final"]["final_sha256"]
    hyps_trainer = [json.loads(l)["hyp"] for l in (out / "val_beam_predictions.jsonl").read_text().splitlines()]
    hyps_eval = [json.loads(l)["hyp"] for l in (cli["tmp"] / "eval" / "predictions.jsonl").read_text().splitlines()]
    assert hyps_trainer == hyps_eval
