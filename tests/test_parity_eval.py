"""scripts/parity_eval.py - the Phase-1 evaluation-harness gate (PROTOCOL §3.33).

Decodes the released RAVEn checkpoint over a parity split exactly as upstream does
and reports CORPUS WER (headline) next to the macro mean. End-to-end tests run the
real script against a tiny staged tree (real ``paths`` layout authority, real FLAC
files, real beam search) with a randomly initialised model; they establish the
plumbing, not any WER.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import parity_eval as S  # noqa: E402
import paths as data_paths  # noqa: E402
from parity import model as parity_model  # noqa: E402

INDEX_COLUMNS = ["utterance_id", "split", "video_id", "stem", "tokens_path", "audio_path",
                 "audio_kind", "txt_path", "n_tokens", "duration_s", "selectable", "conf",
                 "text_raw", "text_norm", "n_chars_norm"]

TRAINVAL = [("vidA/50001", "vidA", "50001", "HELLO WORLD"),
            ("vidA/50002", "vidA", "50002", "IT'S 5 O'CLOCK"),
            ("vidB/50001", "vidB", "50001", "ONE TWO THREE")]
TEST = [("0", "", "0", "the first lesson"), ("1", "", "1", "we change the economics")]


def _row(uid, split, video, stem, text):
    return {"utterance_id": uid, "split": split, "video_id": video, "stem": stem,
            "tokens_path": "", "audio_path": "", "audio_kind": "flac", "txt_path": "",
            "n_tokens": 50, "duration_s": 1.0, "selectable": split == "trainval", "conf": 5.0,
            "text_raw": text, "text_norm": text.lower(), "n_chars_norm": len(text)}


def _tone(seconds=1.0, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(16000 * seconds)) / 16000
    x = 0.2 * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(t.shape)
    return (x * 32767).astype(np.int16)


@pytest.fixture(scope="module")
def shared_model():
    torch.manual_seed(0)
    return parity_model.build_model().eval()


@pytest.fixture(scope="module")
def ckpt_file(tmp_path_factory, shared_model):
    p = tmp_path_factory.mktemp("ckpt") / "random_init.pth"
    torch.save(shared_model.state_dict(), p)
    return p


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Tiny staged tree + parquet index + id lists; the real layout authority."""
    root = tmp_path / "data"
    rows = []
    for k, (uid, vid, stem, text) in enumerate(TRAINVAL):
        rows.append(_row(uid, "trainval", vid, stem, text))
        (root / vid).mkdir(parents=True, exist_ok=True)
        sf.write(str(root / vid / f"{stem}.flac"), _tone(1.0, k), 16000, subtype="PCM_16", format="FLAC")
    for k, (uid, vid, stem, text) in enumerate(TEST):
        rows.append(_row(uid, "test", vid, stem, text))
        sf.write(str(root / f"{stem}.flac"), _tone(1.0, 10 + k), 16000, subtype="PCM_16", format="FLAC")
    index = tmp_path / "index.parquet"
    pd.DataFrame(rows, columns=INDEX_COLUMNS).to_parquet(index)
    ids = tmp_path / "ids"
    ids.mkdir()
    (ids / "val.txt").write_text("".join(f"{u}\n" for u, *_ in TRAINVAL[:2]))
    (ids / "test.txt").write_text("".join(f"{u}\n" for u, *_ in TEST))
    monkeypatch.setenv("SPELL_DATA_ROOT", str(root))
    monkeypatch.delenv("SPELL_FINAL_EVAL", raising=False)
    data_paths.reset_cache()
    yield {"root": root, "index": index, "ids": ids, "tmp": tmp_path}
    data_paths.reset_cache()


def _argv(world, ckpt, split="val", out="out", *extra):
    return ["--checkpoint", str(ckpt), "--split", split, "--index", str(world["index"]),
            "--ids", str(world["ids"] / f"{split}.txt"), "--out-dir", str(world["tmp"] / out),
            "--device", "cpu", "--skip-sha-check", "--beam-size", "2", "--num-workers", "0",
            "--wandb-mode", "disabled", *extra]


def _read(out_dir: Path):
    rows = [json.loads(l) for l in (out_dir / "predictions.jsonl").read_text().splitlines()]
    return rows, json.loads((out_dir / "summary.json").read_text())


@pytest.fixture
def fast_model(monkeypatch, shared_model):
    """Skip the 223 MB state_dict round-trip: hand the shared random model to the script."""
    monkeypatch.setattr(S.parity_model, "load_released_checkpoint",
                        lambda path, **kw: shared_model)


class FakeRun:
    def __init__(self, events=None):
        self.logged, self.summary, self.finished = [], {}, []
        self.events = events if events is not None else []

    def log(self, payload, step=None):
        self.logged.append(dict(payload))

    def finish(self, exit_code=0):
        self.finished.append(exit_code)
        self.events.append("finish")


# ------------------------------------------------------------- quarantine gate --

def test_test_split_is_refused_without_final_eval_env_and_nothing_is_created(world, ckpt_file, monkeypatch):
    opened = []
    monkeypatch.setattr(S.wandb_run, "init_run", lambda **kw: opened.append(kw))
    rc = S.main(_argv(world, ckpt_file, split="test"))
    assert rc == 3
    assert not (world["tmp"] / "out").exists()
    assert opened == []                              # refused before any W&B run or model work


def test_test_split_runs_with_final_eval_env(world, ckpt_file, fast_model, monkeypatch):
    monkeypatch.setenv("SPELL_FINAL_EVAL", "1")
    assert S.main(_argv(world, ckpt_file, split="test")) == 0
    rows, summary = _read(world["tmp"] / "out")
    assert summary["split"] == "test" and summary["n_utts"] == 2 == len(rows)
    assert summary["final_eval_unlocked"] is True


def test_val_split_needs_no_unlock(world, ckpt_file, fast_model):
    assert S.main(_argv(world, ckpt_file, split="val")) == 0


# ----------------------------------------------------------------- end to end --

def test_end_to_end_real_decode_schema_and_wer_arithmetic(world, ckpt_file):
    """Real BatchBeamSearch on the real tree, strict load of a (random-init) state_dict."""
    rc = S.main(_argv(world, ckpt_file, "val", "e2e"))
    assert rc == 0
    rows, summary = _read(world["tmp"] / "e2e")
    assert [r["utterance_id"] for r in rows] == ["vidA/50001", "vidA/50002"]
    assert [r["ref"] for r in rows] == ["HELLO WORLD", "IT'S 5 O'CLOCK"]
    for r in rows:
        assert set(r) == {"utterance_id", "ref", "hyp", "edits", "ref_words", "audio_s", "decode_s"}
        assert isinstance(r["hyp"], str) and r["ref_words"] == len(r["ref"].split())
        assert r["edits"] >= 0 and r["audio_s"] == pytest.approx(1.0) and r["decode_s"] > 0
    total_edits = sum(r["edits"] for r in rows)
    total_words = sum(r["ref_words"] for r in rows)
    assert total_words == 2 + 3
    assert summary["total_edits"] == total_edits and summary["total_ref_words"] == total_words
    assert summary["corpus_wer"] == pytest.approx(total_edits / total_words)
    macro = sum(r["edits"] / r["ref_words"] for r in rows) / 2
    assert summary["macro_wer"] == pytest.approx(macro)


def test_summary_records_config_provenance_and_the_smoke_flag(world, ckpt_file, fast_model):
    assert S.main(_argv(world, ckpt_file, "val", "o", "--limit", "1")) == 0
    rows, s = _read(world["tmp"] / "o")
    assert len(rows) == 1 and s["n_utts"] == 1
    assert s["limit"] == 1 and s["is_subset_smoke"] is True
    assert s["config"] == {"beam_size": 2, "ctc_weight": 0.1, "lm_weight": 0.0, "length_bonus": 0.0,
                           "maxlenratio": 0.0, "minlenratio": 0.0, "device": "cpu"}
    assert s["checkpoint"]["sha256_verified"] is False
    assert s["checkpoint"]["state_dict_fingerprint"] == parity_model.RELEASED_STATE_DICT_FINGERPRINT
    assert s["checkpoint"]["pinned_sha256"] == parity_model.RELEASED_CKPT_SHA256
    for key in ("git_rev", "git_dirty", "wall_time_s", "decode_time_s", "finished_utc",
                "ids", "units_sha256", "arch_config", "audio_subtype_sample", "data_root",
                "predictions_sha256", "complete"):
        assert key in s, key
    assert s["audio_subtype_sample"] == "PCM_16" and s["complete"] is True
    assert s["split"] == "val" and s["final_eval_unlocked"] is False


def test_full_split_is_not_flagged_as_a_smoke_subset(world, ckpt_file, fast_model):
    assert S.main(_argv(world, ckpt_file, "val", "o")) == 0
    _, s = _read(world["tmp"] / "o")
    assert s["limit"] is None and s["is_subset_smoke"] is False


def test_decoding_runs_in_eval_mode_on_float32_waveforms(world, ckpt_file, fast_model, monkeypatch):
    seen = []

    def spy(model, beam_search, wave, tokens):
        seen.append((model.training, wave.dtype, tuple(wave.shape), len(tokens)))
        return "HELLO"

    monkeypatch.setattr(S.decode, "transcribe", spy)
    S.parity_model.load_released_checkpoint(None).train()      # handed over in train mode
    assert S.main(_argv(world, ckpt_file)) == 0
    assert seen == [(False, torch.float32, (16000,), 1049)] * 2


def test_wer_arithmetic_with_a_scripted_decoder(world, ckpt_file, fast_model, monkeypatch):
    # ref 1 "HELLO WORLD": 0 edits.  ref 2 "IT'S 5 O'CLOCK": 2 substitutions + 1 insertion = 3 edits.
    hyps = iter(["HELLO WORLD", "ITS 5 OCLOCK EXTRA"])
    monkeypatch.setattr(S.decode, "transcribe", lambda *a: next(hyps))
    assert S.main(_argv(world, ckpt_file)) == 0
    rows, s = _read(world["tmp"] / "out")
    assert [r["edits"] for r in rows] == [0, 3]
    assert s["total_edits"] == 3 and s["total_ref_words"] == 5
    assert s["corpus_wer"] == pytest.approx(3 / 5)
    assert s["macro_wer"] == pytest.approx((0 + 3 / 3) / 2)


def test_predictions_are_written_incrementally(world, ckpt_file, fast_model, monkeypatch):
    out = world["tmp"] / "out"
    sizes = []

    def spy(model, bs, wave, tokens):
        pred = out / "predictions.jsonl"
        sizes.append(len(pred.read_text().splitlines()) if pred.exists() else 0)
        return "X"

    monkeypatch.setattr(S.decode, "transcribe", spy)
    assert S.main(_argv(world, ckpt_file)) == 0
    assert sizes == [0, 1]                           # row 0 already on disk when utterance 1 is decoded


# ---------------------------------------------------------------- setup refusals --

def test_checkpoint_sha256_mismatch_is_refused_unless_skipped(world, ckpt_file, fast_model, capsys):
    argv = [a for a in _argv(world, ckpt_file) if a != "--skip-sha-check"]
    assert S.main(argv) == 2
    assert "sha256" in capsys.readouterr().err
    assert not (world["tmp"] / "out" / "summary.json").exists()


def test_explicit_expect_sha256_is_honoured(world, ckpt_file, fast_model):
    real = parity_model.sha256_file(ckpt_file)
    argv = [a for a in _argv(world, ckpt_file) if a != "--skip-sha-check"] + ["--expect-sha256", real]
    assert S.main(argv) == 0
    _, s = _read(world["tmp"] / "out")
    assert s["checkpoint"]["sha256"] == real and s["checkpoint"]["sha256_verified"] is True


def test_missing_checkpoint_is_fatal(world, capsys):
    assert S.main(_argv(world, world["tmp"] / "nope.pth")) == 2
    assert "checkpoint not found" in capsys.readouterr().err


def test_strict_load_rejects_a_checkpoint_with_a_missing_tensor(world, shared_model, tmp_path, capsys):
    sd = {k: v for k, v in shared_model.state_dict().items() if k != "ctc.ctc_lo.bias"}
    bad = tmp_path / "bad.pth"
    torch.save(sd, bad)
    assert S.main(_argv(world, bad)) == 2
    assert "ctc.ctc_lo.bias" in capsys.readouterr().err


def test_ids_outside_the_requested_split_are_refused(world, ckpt_file, fast_model, capsys):
    argv = _argv(world, ckpt_file, "val")
    argv[argv.index("--ids") + 1] = str(world["ids"] / "test.txt")     # test ids with --split val
    assert S.main(argv) == 2
    assert "absent" in capsys.readouterr().err


def test_an_empty_id_list_is_refused(world, ckpt_file, fast_model, capsys):
    empty = world["tmp"] / "empty_ids.txt"
    empty.write_text("# nothing\n")
    argv = _argv(world, ckpt_file)
    argv[argv.index("--ids") + 1] = str(empty)
    assert S.main(argv) == 2
    assert "selects no utterances" in capsys.readouterr().err


def test_overwrite_protection_for_a_finished_result(world, ckpt_file, fast_model):
    assert S.main(_argv(world, ckpt_file)) == 0
    assert S.main(_argv(world, ckpt_file)) == 2
    assert S.main(_argv(world, ckpt_file, "val", "out", "--overwrite")) == 0


def test_default_ids_are_the_verified_parity_split_files(world, ckpt_file, fast_model, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(S.splits, "SPLITS_DIR", tmp_path / "no_such_dir")
    argv = _argv(world, ckpt_file)
    i = argv.index("--ids")
    del argv[i:i + 2]
    assert S.main(argv) == 2
    assert "provenance" in capsys.readouterr().err


def test_limit_must_be_positive(world, ckpt_file):
    with pytest.raises(SystemExit):
        S.main(_argv(world, ckpt_file, "val", "o", "--limit", "0"))


# ------------------------------------------------------------------ hardware guard --

def test_cuda_device_goes_through_the_hardware_guard(world, ckpt_file, fast_model, monkeypatch):
    calls = []

    def refuse():
        calls.append(1)
        raise RuntimeError("HARDWARE DRIFT GUARD")

    monkeypatch.setattr(S, "enforce_gpu_policy", refuse)
    argv = _argv(world, ckpt_file)
    argv[argv.index("--device") + 1] = "cuda"
    with pytest.raises(RuntimeError, match="DRIFT"):
        S.main(argv)
    assert calls == [1]


def test_cpu_device_never_touches_the_guard(world, ckpt_file, fast_model, monkeypatch):
    monkeypatch.setattr(S, "enforce_gpu_policy", lambda: pytest.fail("guard must not run on cpu"))
    assert S.main(_argv(world, ckpt_file)) == 0


# ------------------------------------------------------------------------- W&B --

def test_wandb_run_opens_before_the_model_loads_and_logs_live(world, ckpt_file, monkeypatch, shared_model):
    events, opened = [], {}
    run = FakeRun(events)

    def fake_init(**kw):
        events.append("init")
        opened.update(kw)
        return run

    def fake_load(path, **kw):
        events.append("load")
        return shared_model

    monkeypatch.setattr(S.wandb_run, "init_run", fake_init)
    monkeypatch.setattr(S.parity_model, "load_released_checkpoint", fake_load)
    argv = [a for a in _argv(world, ckpt_file) if a not in ("--wandb-mode", "disabled")]
    assert S.main(argv + ["--report-every-s", "0", "--job-id", "777"]) == 0
    assert events[:2] == ["init", "load"] and events[-1] == "finish"
    assert opened["job_type"] == "parity_eval" and opened["name"] == "parity-eval-val-777"
    assert "/" not in opened["name"]
    keys = {k for p in run.logged for k in p}
    assert keys and all(k.startswith("parity_eval/") for k in keys)
    assert {"parity_eval/progress", "parity_eval/last_utt_s", "parity_eval/eta_s",
            "parity_eval/running_corpus_wer", "parity_eval/utts_done"} <= keys
    assert run.finished == [0]
    assert run.summary["parity_eval/corpus_wer"] >= 0 and run.summary["parity_eval/n_utts"] == 2


def test_wandb_run_is_failed_with_the_error_when_decoding_crashes(world, ckpt_file, fast_model, monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(S.wandb_run, "init_run", lambda **kw: run)

    def boom(*a):
        raise RuntimeError("decoder exploded")

    monkeypatch.setattr(S.decode, "transcribe", boom)
    argv = [a for a in _argv(world, ckpt_file) if a not in ("--wandb-mode", "disabled")]
    with pytest.raises(RuntimeError, match="decoder exploded"):
        S.main(argv)
    assert run.finished == [1] and "decoder exploded" in run.summary["error"]
    assert not (world["tmp"] / "out" / "summary.json").exists()       # an incomplete run never looks complete


def test_wandb_run_is_failed_on_a_setup_refusal_too(world, ckpt_file, fast_model, monkeypatch):
    run = FakeRun()
    monkeypatch.setattr(S.wandb_run, "init_run", lambda **kw: run)
    argv = [a for a in _argv(world, ckpt_file) if a not in ("--wandb-mode", "disabled", "--skip-sha-check")]
    assert S.main(argv) == 2
    assert run.finished == [1] and "sha256" in run.summary["error"]


# ---------------------------------------- evaluating a checkpoint written by the parity trainer --

CUSTOM_UNITS = ["<unk>", "▁HELLO", "▁WORLD", "S", "T", "▁IT"]


@pytest.fixture
def custom(world, tmp_path):
    """A small-architecture model with a 6-unit tokenizer (odim 8) saved as a plain state_dict,
    the shape of file scripts/parity_train.py writes as final.pt."""
    import yaml
    from parity import spm_tokens
    arch = parity_model.load_arch_config()
    arch.update(adim=32, aheads=2, eunits=64, elayers=2, ddim=32, dheads=2, dunits=64, dlayers=1)
    arch_path = tmp_path / "arch_small.yaml"
    arch_path.write_text(yaml.safe_dump(arch))
    torch.manual_seed(0)
    model = parity_model.build_model(arch, odim=len(CUSTOM_UNITS) + 2).eval()
    ckpt = tmp_path / "final.pt"
    torch.save(model.state_dict(), ckpt)
    units = tmp_path / "units.txt"
    spm_tokens.write_units_file(CUSTOM_UNITS, units)
    return {"ckpt": ckpt, "units": units, "arch": arch_path}


def _custom_argv(world, c, *extra):
    argv = [a for a in _argv(world, c["ckpt"]) if a != "--skip-sha-check"]
    return argv + ["--units", str(c["units"]), "--arch-config", str(c["arch"]), *extra]


def test_trainer_checkpoint_with_its_own_units_evaluates_with_the_same_decoding_recipe(world, custom):
    assert S.main(_custom_argv(world, custom, "--expect-sha256", "none")) == 0
    rows, s = _read(world["tmp"] / "out")
    assert [r["ref"] for r in rows] == ["HELLO WORLD", "IT'S 5 O'CLOCK"]
    assert s["units"] == {"path": str(custom["units"]), "n_units": 6, "custom": True}
    assert s["units_sha256"] == parity_model.sha256_file(custom["units"])
    assert s["checkpoint"]["sha256_verified"] is False
    assert s["checkpoint"]["released_architecture_asserted"] is False
    assert s["config"]["beam_size"] == 2 and s["config"]["lm_weight"] == 0.0 and s["config"]["maxlenratio"] == 0.0
    assert s["arch_config"]["path"] == str(custom["arch"])


def test_trainer_checkpoint_hypotheses_are_spelled_with_the_custom_pieces(world, custom, monkeypatch):
    seen = []
    real = S.decode.build_beam_search

    def spy(model, tokens, **kw):
        seen.append((len(tokens), tokens[:3], tokens[-1], model.odim))
        return real(model, tokens, **kw)

    monkeypatch.setattr(S.decode, "build_beam_search", spy)
    assert S.main(_custom_argv(world, custom, "--expect-sha256", "none")) == 0
    assert seen == [(8, ["<blank>", "<unk>", "▁HELLO"], "<eos>", 8)]


def test_a_custom_units_file_needs_the_sha256_pin_switched_off_explicitly(world, custom, capsys):
    assert S.main(_custom_argv(world, custom)) == 2                     # released-file pin still applies
    err = capsys.readouterr().err
    assert "sha256" in err and "--expect-sha256 none" in err


def test_units_that_do_not_match_the_checkpoints_output_size_are_refused(world, custom, tmp_path, capsys):
    from parity import spm_tokens
    wrong = tmp_path / "units7.txt"
    spm_tokens.write_units_file(CUSTOM_UNITS + ["EXTRA"], wrong)
    argv = _custom_argv(world, custom, "--expect-sha256", "none")
    argv[argv.index("--units") + 1] = str(wrong)
    assert S.main(argv) == 2
    assert "does not load strictly" in capsys.readouterr().err


def test_missing_or_malformed_units_files_are_refused(world, custom, tmp_path, capsys):
    argv = _custom_argv(world, custom, "--expect-sha256", "none")
    argv[argv.index("--units") + 1] = str(tmp_path / "nope.txt")
    assert S.main(argv) == 2 and "units file not found" in capsys.readouterr().err
    bad = tmp_path / "bad_units.txt"
    bad.write_text("<unk> 1\n▁A 5\n")
    argv[argv.index("--units") + 1] = str(bad)
    assert S.main(argv) == 2 and "units file unusable" in capsys.readouterr().err


def test_expect_sha256_none_also_works_for_the_default_units_and_is_recorded(world, ckpt_file, fast_model):
    argv = [a for a in _argv(world, ckpt_file) if a != "--skip-sha-check"] + ["--expect-sha256", "none"]
    assert S.main(argv) == 0
    _, s = _read(world["tmp"] / "out")
    assert s["checkpoint"]["sha256_verified"] is False
    assert s["units"]["custom"] is False and s["units_sha256"] == S.parity_text.UNITS_SHA256
    assert s["checkpoint"]["released_architecture_asserted"] is True      # default path still asserts it


def test_default_behaviour_still_asserts_the_released_architecture(world, tmp_path, capsys):
    arch = parity_model.load_arch_config()
    arch["elayers"] = 11
    import yaml
    p = tmp_path / "arch11.yaml"
    p.write_text(yaml.safe_dump(arch))
    torch.manual_seed(0)
    smaller = tmp_path / "small.pth"
    torch.save(parity_model.build_model(arch).state_dict(), smaller)
    argv = [a for a in _argv(world, smaller)] + ["--arch-config", str(p)]
    assert S.main(argv) == 2
    assert "released architecture" in capsys.readouterr().err


def test_the_test_quarantine_is_untouched_by_the_generalisation(world, custom, monkeypatch):
    argv = _custom_argv(world, custom, "--expect-sha256", "none")
    argv[argv.index("--split") + 1] = "test"
    assert S.main(argv) == 3
    assert not (world["tmp"] / "out").exists()
