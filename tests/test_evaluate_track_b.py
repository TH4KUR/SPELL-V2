"""scripts/evaluate_track_b.py contract — the OFFICIAL post-hoc WER/CER
evaluator (§7 gate-signal policy: source of official final numbers).

Laws pinned here:
  * a REAL end-to-end run through main() -- a real LitConformerCTC
    checkpoint, real TokenDataset/DataLoader, real greedy decode -- proves
    the per-batch loop actually EXECUTES, not just that its pieces exist in
    isolation. 2026-09-19 incident: `text_lengths_dev = text_lengths.to(device)`
    referenced a bare name that was never assigned anywhere in the
    function -- a NameError on the very first batch of the very first real
    invocation. This script had ZERO test coverage before that, on a
    script PROTOCOL's own §7 policy calls "the source of official final
    numbers" -- not a script to leave unexercised.
  * official-test quarantine (§3.5): --split test is refused outright
    without SPELL_FINAL_EVAL=1, with no decode attempted at all.
  * predictions.jsonl / summary.json carry the documented schema.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import evaluate_track_b as S  # noqa: E402
from dataset import TokenDataset, UtteranceRecord  # noqa: E402
from lit_track_b import LitConformerCTC  # noqa: E402


def _tiny_cfg() -> dict:
    return {
        "model": {"input_stream": 0, "codebook_size": 64, "d_model": 16,
                  "n_conformer_layers": 1, "n_conformer_heads": 4,
                  "conformer_ff_mult": 2, "conv_kernel_size": 15, "dropout": 0.0},
        "training": {"lr_peak": 5e-4, "warmup_steps": 10, "weight_decay": 0.0,
                     "grad_clip_norm": 1.0},
        "augment": {},
        "logging": {"grad_norm_log_every_batches": 50},
    }


def _toy_lit() -> LitConformerCTC:
    return LitConformerCTC(_tiny_cfg(), use_augment=False).to("cpu").eval()


def _fake_records(tmp_path) -> list[UtteranceRecord]:
    specs = [("hi", 12), ("ok", 8), ("yes", 10)]
    root = tmp_path / "data"
    recs = []
    for i, (text, n_frames) in enumerate(specs):
        vid, stem = f"v{i}", f"{50000 + i}"
        p = root / vid / f"{stem}.tokens.pt"
        p.parent.mkdir(parents=True)
        torch.save(torch.randint(0, 64, (3, n_frames)), p)
        recs.append(UtteranceRecord(
            utterance_id=f"{vid}/{stem}", split="trainval", video_id=vid,
            stem=stem, tokens_path=str(p), audio_path=None, audio_kind=None,
            txt_path=None, n_tokens=n_frames, duration_s=n_frames / 50.0, conf=5,
            text_raw=text.upper(), text_norm=text, n_chars_norm=len(text)))
    return recs


def _write_fake_ckpt(lit: LitConformerCTC, path: Path) -> None:
    import lightning.pytorch as pl

    torch.save({
        "state_dict": lit.state_dict(),
        "hyper_parameters": dict(lit.hparams),
        "pytorch-lightning_version": pl.__version__,
        "epoch": 0, "global_step": 0,
    }, path)


def test_main_val_split_runs_end_to_end_and_writes_schema(tmp_path, monkeypatch):
    """The anchor regression test for the 2026-09-19 NameError: runs the
    REAL per-batch decode loop (not mocked away) against a real checkpoint
    and real records."""
    lit = _toy_lit()
    ckpt_path = tmp_path / "fake.ckpt"
    _write_fake_ckpt(lit, ckpt_path)

    recs = _fake_records(tmp_path)
    val_ids = {recs[0].utterance_id, recs[1].utterance_id}

    monkeypatch.setattr(S, "enforce_gpu_policy", lambda: {"gpu_name": "fake"})
    monkeypatch.setattr(S, "load_records", lambda index_path, split: recs)
    monkeypatch.setattr(S, "load_id_list", lambda path: val_ids)
    monkeypatch.setattr(
        S, "TokenDataset",
        lambda records, **kw: TokenDataset(
            records, path_resolver=lambda r: Path(r.tokens_path), **kw))
    monkeypatch.setattr(S.data_paths, "preflight_resolve", lambda recs, k=50: None)
    monkeypatch.setattr(
        S.data_paths, "current",
        lambda: type("P", (), {"root": str(tmp_path), "layout": "staged"})())

    out_dir = tmp_path / "eval_out"
    rc = S.main(["--ckpt", str(ckpt_path), "--split", "val",
                "--output-dir", str(out_dir), "--num-workers", "0"])
    assert rc == 0

    summary = json.loads((out_dir / "summary.json").read_text())
    assert summary["n_utterances"] == 2                 # only val_ids survive filter_records
    assert summary["mean_wer"] is not None
    assert summary["mean_cer"] is not None
    assert summary["split"] == "val"
    assert summary["checkpoint"] == str(ckpt_path.resolve())

    rows = [json.loads(line) for line in
            (out_dir / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    for r in rows:
        assert set(r) == {"utterance_id", "ref", "hyp", "loss", "wer", "cer"}
        assert isinstance(r["hyp"], str)
        assert r["wer"] is not None and r["cer"] is not None


def test_official_test_split_refused_without_final_eval_env(tmp_path, monkeypatch):
    monkeypatch.delenv("SPELL_FINAL_EVAL", raising=False)
    rc = S.main(["--ckpt", str(tmp_path / "nonexistent.ckpt"), "--split", "test",
                "--output-dir", str(tmp_path / "out")])
    assert rc == 3
    assert not (tmp_path / "out").exists()               # refused before any decode


def test_official_test_split_allowed_with_final_eval_env_but_missing_ckpt(
        tmp_path, monkeypatch):
    """SPELL_FINAL_EVAL=1 lifts the quarantine refusal specifically -- it
    does not bypass ordinary error handling (missing checkpoint still
    fails loudly, just past the quarantine gate rather than because of it)."""
    monkeypatch.setenv("SPELL_FINAL_EVAL", "1")
    monkeypatch.setattr(S, "enforce_gpu_policy", lambda: {"gpu_name": "fake"})
    rc = S.main(["--ckpt", str(tmp_path / "nonexistent.ckpt"), "--split", "test",
                "--output-dir", str(tmp_path / "out")])
    assert rc == 2                                       # FATAL: checkpoint not found
