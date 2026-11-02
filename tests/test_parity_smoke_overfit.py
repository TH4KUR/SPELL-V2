"""End-to-end CPU smoke of the whole training machinery on real speech (PROTOCOL §3.34): four short
LibriSpeech clips (real 16 kHz FLAC through the real loader, sampler, collate, time-mask, vendored
model, AdamW + per-update schedule, gradient accumulation code path, validation and greedy CTC
decoding) are overfit for 140 updates with hand-made character tokens. It shows the loop learns
something real - it is NOT a WER and says nothing about LRS3.

Opt-in: set LIBRISPEECH_SMOKE_PARQUET to a parquet with columns ``audio`` ({bytes}) and ``text``
(73 LibriSpeech validation clips are what the Stage-2 author used). About 100 s with 4 CPU threads.
"""

from __future__ import annotations

import dataclasses
import io
import os
import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

PARQUET = os.environ.get("LIBRISPEECH_SMOKE_PARQUET")
pytestmark = pytest.mark.skipif(not PARQUET or not Path(PARQUET).is_file(),
                                reason="needs LIBRISPEECH_SMOKE_PARQUET (real LibriSpeech clips)")

from parity import metrics  # noqa: E402
from parity import model as parity_model  # noqa: E402
from parity import recipes as R  # noqa: E402
from parity import text as parity_text  # noqa: E402
from parity import train_data as D  # noqa: E402
from parity import trainer as T  # noqa: E402
from parity_train_helpers import make_record, tiny_arch, write_flac  # noqa: E402

EPOCHS = 140
MAX_SECONDS = 2.0


def _cer(refs, hyps):
    edits = sum(metrics._edit_distance(list(r), list(h)) for r, h in zip(refs, hyps))
    return edits / sum(len(r) for r in refs)


@pytest.fixture(autouse=True)
def four_cpu_threads():
    """The cost is dominated by small convolutions: on a busy or many-core machine the default thread
    count oversubscribes (measured: 94 s with 4 threads, over 10 min with 20 threads under load)."""
    before = torch.get_num_threads()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    yield
    torch.set_num_threads(before)


def test_overfit_four_real_clips_loss_falls_and_greedy_output_moves_toward_the_reference(tmp_path, capsys, monkeypatch):
    import pandas as pd
    import soundfile as sf

    df = pd.read_parquet(PARQUET)
    clips = []
    for _, row in df.iterrows():
        wave, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if sr == 16000 and wave.size / sr <= MAX_SECONDS:
            clips.append((wave, row["text"].upper()))
    clips = sorted(clips, key=lambda c: c[0].size)[:4]
    assert len(clips) == 4, "need four clips of at most two seconds"

    chars = sorted({c for _, text in clips for c in text.replace(" ", "")})
    units = ["<unk>", "▁"] + chars                               # one piece per character, ▁ = word boundary
    tokens = parity_text.token_list(units)
    token_ids, records, val_records = {}, [], []
    for k, (wave, text) in enumerate(clips):
        write_flac(tmp_path / "tr" / f"{k}.flac", wave)
        pieces = [p for w in text.split() for p in ["▁", *w]]
        ids = tuple(units.index(p) + 1 for p in pieces)
        assert wave.size // 640 >= len(ids), "CTC needs at least one frame per token"
        for video, bucket in (("tr", records), ("va", val_records)):      # same audio under a disjoint val id
            rec = make_record(f"{video}/{k}", wave.size / 16000, text)
            token_ids[rec.utterance_id] = ids
            bucket.append(rec)
    resolver = lambda rec: tmp_path / "tr" / f"{rec.stem}.flac"
    total_s = sum(r.duration_s for r in records)

    recipe = dataclasses.replace(
        R.load_recipe("R-B"), epochs=EPOCHS, warmup_epochs=8, lr_encoder=6e-3, lr_decoder=6e-3,
        micro_batch_seconds=total_s + 1, effective_batch_seconds=total_s + 1, beam_size=2,
        time_mask_window_s=0.05)                                 # light masking so 140 updates can memorise
    torch.manual_seed(0)
    model = parity_model.build_model(recipe.arch_config(tiny_arch()), odim=len(tokens))
    common = dict(seed=recipe.seed, time_mask_window_s=recipe.time_mask_window_s,
                  time_mask_stride_s=recipe.time_mask_stride_s, path_resolver=resolver)
    train_ds = D.ParityTrainDataset(records, token_ids, augment=True, **common)
    val_ds = D.ParityTrainDataset(val_records, token_ids, augment=False, **common)
    before = T.evaluate(model, val_ds, tokens, device=torch.device("cpu"), micro_batch_seconds=100.0)

    # Checkpoint I/O (a 54 MB fsynced last.pt per epoch) dominates the wall time on a nearly full disk and
    # is covered by tests/test_parity_trainer.py; only final.pt is written here.
    real_save = T.atomic_torch_save
    monkeypatch.setattr(T, "atomic_torch_save",
                        lambda obj, path: real_save(obj, path) if Path(path).name == T.FINAL_NAME else None)

    trainer = T.Trainer(recipe=recipe, model=model, train_dataset=train_ds, val_dataset=val_ds, tokens=tokens,
                        options=T.TrainerOptions(out_dir=tmp_path / "run", keep_last_weights=0,
                                                 path_resolver=resolver, report_every_s=1e9))
    assert trainer.steps_per_epoch == 1
    summary = trainer.fit()
    after = T.evaluate(model, val_ds, tokens, device=torch.device("cpu"), micro_batch_seconds=100.0)

    first, last = summary["history"][0], summary["history"][-1]
    cer_before, cer_after = _cer(before["refs"], before["hyps"]), _cer(after["refs"], after["hyps"])
    with capsys.disabled():
        print(f"\n[smoke-overfit] {len(clips)} real clips, {total_s:.1f} s, {EPOCHS} updates, tiny arch, char tokens")
        print(f"[smoke-overfit] train loss {first['train_loss']:.2f} -> {last['train_loss']:.2f}; "
              f"val loss {before['loss']:.2f} -> {after['loss']:.2f} (ctc {before['loss_ctc']:.1f} -> "
              f"{after['loss_ctc']:.1f}, att {before['loss_att']:.1f} -> {after['loss_att']:.1f}); "
              f"token acc {before['acc']:.2f} -> {after['acc']:.2f}")
        print(f"[smoke-overfit] greedy CTC char error rate {cer_before:.3f} -> {cer_after:.3f}")
        for ref, b, a in zip(before["refs"], before["hyps"], after["hyps"]):
            print(f"[smoke-overfit]   ref {ref!r}\n[smoke-overfit]     before {b!r}\n[smoke-overfit]     after  {a!r}")

    assert last["train_loss"] < 0.25 * first["train_loss"]
    assert after["loss"] < 0.25 * before["loss"]
    assert after["loss_ctc"] < 0.4 * before["loss_ctc"] and after["loss_att"] < 0.2 * before["loss_att"]
    assert after["acc"] > 0.8 > before["acc"]
    assert cer_after < 0.8 and cer_after < 0.85 * cer_before          # measured: ~0.98 -> ~0.59 on 4 threads
    assert any(h.strip() for h in after["hyps"])                      # no longer the all-blank or constant output
    assert summary["complete"] is True and summary["global_step"] == EPOCHS
