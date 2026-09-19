"""scripts/score_less.py contract (roster #5 scorer, LESS CTC-grad variant).

Laws pinned here:
  * FULL trajectory (every ckpt_epoch*.ckpt), not score_proxy's late-only window;
  * the batch-independence gradient trick is PROVEN numerically, not assumed:
    a batched grad-w.r.t.-feats call must match naive per-example processing
    to floating-point tolerance (test_batched_grad_feats_matches_naive_...);
  * pad frames carry exactly-zero gradient (justifies length-normalized,
    not Tmax-normalized, pooling);
  * val is reference-only: _ref_vec_for_ids returns a bare vector, never a
    keyed structure — a type-level proof of §3.21, not just a filter;
  * the Rademacher projection is seed-deterministic and seed-divergent
    (§3.20 ×3), and main() actually produces DIFFERENT influence values per
    seed end-to-end (the projection is not an accidental no-op);
  * main() refuses to write anything if a val id would appear in the table;
  * W&B logging (SPELL-RQ2 policy: runs crucial to the research get logged)
    runs structurally (mode="disabled" needs no network/credentials) and
    reports per-seed stats plus cross-seed spearman agreement.
"""

from __future__ import annotations

import sys
import time
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import score_less as S  # noqa: E402
from dataset import TokenDataset, UtteranceRecord, collate_token_batch  # noqa: E402
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
    # forced onto CPU regardless of local CUDA availability: PROTOCOL §10
    # item 1 already documents CTC backward as non-deterministic on GPU, and
    # these tests isolate the batch-independence MATH, not device/kernel
    # reproducibility -- CPU keeps the comparison to genuine floating-point
    # non-associativity between padded-batch and singleton execution shapes.
    return LitConformerCTC(_tiny_cfg(), use_augment=False).to("cpu").eval()


def _fake_records(tmp_path) -> list[UtteranceRecord]:
    """Four utterances with DIFFERENT lengths (so padding is genuinely
    exercised) and char targets that fit inside their own length (nothing
    dropped by the §3.12 input-length rule)."""
    specs = [("hi", 12), ("ok", 8), ("yes", 10), ("a", 6)]
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


def _loader(lit, recs, batch_size):
    collate = partial(collate_token_batch, token_pad_id=-1, text_pad_id=lit.vocab.pad_id)
    return DataLoader(
        TokenDataset(recs, vocab=lit.vocab, include_text=True,
                    path_resolver=lambda r: Path(r.tokens_path)),
        batch_size=batch_size, shuffle=False, collate_fn=collate)


# --------------------------------------------------------------------- ckpts

def test_all_ckpts_uses_full_trajectory_not_late_only(tmp_path):
    d = tmp_path / "bundle"
    d.mkdir()
    for e in (5, 10, 15, 20):
        (d / f"ckpt_epoch{e:04d}.ckpt").touch()
    got = S._all_ckpts(d)
    assert got == [d / f"ckpt_epoch{e:04d}.ckpt" for e in (5, 10, 15, 20)]

    from score_proxy import _late_ckpts
    late = _late_ckpts(d, late_frac=0.75)
    assert len(late) < len(got)                     # contrast: late-only window is smaller


def test_all_ckpts_empty_trajectory_is_loud(tmp_path):
    d = tmp_path / "bundle"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="trajectory"):
        S._all_ckpts(d)


# ----------------------------------------------------------------------- hook

def test_grad_feats_hook_captures_head_input(tmp_path):
    lit = _toy_lit()
    recs = _fake_records(tmp_path)
    handle, captured = S._grad_feats_hook(lit)
    try:
        batch = next(iter(_loader(lit, recs, batch_size=4)))
        stream = batch["tokens"][:, 0, :]
        lit.model(stream, batch["lengths"])
        assert captured["feats"].shape[0] == 4
        assert captured["feats"].shape[-1] == lit.model.head.in_features
    finally:
        handle.remove()


def test_grad_feats_hook_rejects_wrong_head_type():
    lit = _toy_lit()
    lit.model.head = torch.nn.Identity()
    with pytest.raises(TypeError):
        S._grad_feats_hook(lit)


# --------------------------------------------------- the anchor correctness test

def _slice_batch(batch: dict, i: int) -> dict:
    """A length-1 'batch' sliced directly out of an already-padded batch —
    keeps that row's padding pattern IDENTICAL to how it appeared in the
    full batch. This matters: re-collating a singleton from scratch would
    pad it to ITS OWN (shorter) length instead, changing which frames sit
    at the sequence boundary -- and PROTOCOL §3.13 already documents that
    boundary-adjacent frames legitimately see a small halo of contamination
    from the depthwise conv's padding treatment. Comparing against a
    re-collated singleton would conflate that already-known, already-
    accepted effect with the actual claim under test here (cross-BATCH
    independence, not cross-TIME padding behavior) -- slicing controls for it."""
    return {"tokens": batch["tokens"][i:i + 1], "lengths": batch["lengths"][i:i + 1],
            "text_ids": batch["text_ids"][i:i + 1], "text_lengths": batch["text_lengths"][i:i + 1],
            "utterance_ids": batch["utterance_ids"][i:i + 1]}


def test_batched_grad_feats_matches_naive_per_example_processing(tmp_path):
    """The whole design rests on: torch.autograd.grad(loss_per_utt.sum(), feats)
    gives EVERY row's true individual gradient in one backward call, with zero
    cross-batch contamination. Prove it: process 4 differently-lengthed
    utterances as one real batch, then again one at a time — each sliced
    OUT OF that same padded batch (see _slice_batch) so only "are other rows
    sharing the tensor" varies, nothing about padding/boundary handling.
    If batch elements leaked gradient into each other, batched vs. sliced-
    singleton processing would differ and this assertion would fail.
    """
    lit = _toy_lit()
    recs = _fake_records(tmp_path)

    handle, captured = S._grad_feats_hook(lit)
    try:
        batch = next(iter(_loader(lit, recs, batch_size=4)))
        uids_batched, pooled_batched = S._batch_pooled_grads(lit, batch, captured, "cpu")

        singleton_vecs = {}
        for i in range(len(recs)):
            uids_one, pooled_one = S._batch_pooled_grads(lit, _slice_batch(batch, i), captured, "cpu")
            assert uids_one == [recs[i].utterance_id]
            singleton_vecs[recs[i].utterance_id] = pooled_one[0]
    finally:
        handle.remove()

    for uid, vec_batched in zip(uids_batched, pooled_batched):
        assert torch.allclose(vec_batched, singleton_vecs[uid], atol=1e-5), (
            f"{uid}: batched vs singleton gradient mismatch -- batch "
            "elements are leaking gradient into each other")


def test_pooled_grad_zero_at_pad_frames(tmp_path):
    lit = _toy_lit()
    recs = _fake_records(tmp_path)
    handle, captured = S._grad_feats_hook(lit)
    try:
        batch = next(iter(_loader(lit, recs, batch_size=4)))
        keep = torch.ones(4, dtype=torch.bool)   # nothing dropped for this fixture
        tokens, lengths = batch["tokens"], batch["lengths"]
        stream = tokens[:, 0, :]
        log_probs, out_lengths = lit.model(stream, lengths)
        import ctc as ctc_lib
        loss_vec = ctc_lib.ctc_loss_per_utt(
            log_probs.transpose(0, 1), batch["text_ids"], out_lengths,
            batch["text_lengths"], lit.blank_id)
        feats = captured["feats"]
        grad_feats = torch.autograd.grad(loss_vec.sum(), feats)[0]
        for b, length in enumerate(out_lengths.tolist()):
            if length < grad_feats.shape[1]:
                assert grad_feats[b, length:].abs().max().item() < 1e-7
    finally:
        handle.remove()


def test_pooled_grad_vecs_for_ids_matches_independent_masked_mean(tmp_path, monkeypatch):
    lit = _toy_lit()
    recs = _fake_records(tmp_path)
    monkeypatch.setattr(S, "load_records", lambda index_path, split: recs)
    # redirect path resolution to the fixture's real on-disk files, bypassing
    # the global SPELL_DATA_ROOT/paths.current() singleton entirely (no env
    # mutation needed) -- production score_less.py passes no path_resolver.
    monkeypatch.setattr(
        S, "TokenDataset",
        lambda records, **kw: TokenDataset(
            records, path_resolver=lambda r: Path(r.tokens_path), **kw))

    ids = [r.utterance_id for r in recs]
    got = S._pooled_grad_vecs_for_ids(lit, ids, batch_size=4, device="cpu")
    assert set(got) == set(ids)
    for uid, vec in got.items():
        assert vec.shape == (lit.model.head.in_features,)
        assert np.isfinite(vec).all()

    # independent recomputation: rebuild the same padded batch and slice each
    # row out of it individually (see _slice_batch's docstring for why a
    # re-collated singleton would confound this with §3.13's padding halo),
    # mirroring test_el2n.py's "different indexing" cross-check style
    handle, captured = S._grad_feats_hook(lit)
    try:
        batch = next(iter(_loader(lit, recs, batch_size=4)))
        for i, rec in enumerate(recs):
            uids_one, pooled_one = S._batch_pooled_grads(lit, _slice_batch(batch, i), captured, "cpu")
            assert uids_one == [rec.utterance_id]
            assert np.allclose(got[rec.utterance_id], pooled_one[0].numpy(), atol=1e-5)
    finally:
        handle.remove()


def test_ref_vec_for_ids_returns_bare_array_not_a_dict(monkeypatch):
    d_model = 4
    vecs = {"v0/1": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
           "v1/2": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)}

    class FakeHead:
        in_features = d_model

    class FakeModel:
        head = FakeHead()

    class FakeVocab:
        pad_id = 0

    class FakeLit:
        model = FakeModel()
        vocab = FakeVocab()

        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(S, "_grad_feats_hook", lambda lit: (
        type("H", (), {"remove": lambda self: None})(), {}))
    monkeypatch.setattr(S, "load_records", lambda index_path, split: [])
    monkeypatch.setattr(
        S, "_batch_pooled_grads",
        lambda lit, batch, captured, device: (
            list(vecs.keys()), torch.from_numpy(np.stack(list(vecs.values())))))
    monkeypatch.setattr(
        S, "DataLoader", lambda dataset, **kw: [{"dummy": True}])
    monkeypatch.setattr(
        S, "TokenDataset", lambda recs, **kw: recs)

    ref = S._ref_vec_for_ids(FakeLit(), list(vecs), batch_size=8, device="cpu")
    assert isinstance(ref, np.ndarray)
    assert ref.shape == (d_model,)
    expected = np.mean(list(vecs.values()), axis=0)
    assert np.allclose(ref, expected)


# ------------------------------------------------------------------ projection

def test_projection_matrix_is_rademacher_seed_deterministic_and_divergent():
    P1 = S._projection_matrix(201, in_dim=16, proj_dim=8)
    P2 = S._projection_matrix(201, in_dim=16, proj_dim=8)
    P3 = S._projection_matrix(202, in_dim=16, proj_dim=8)
    scaled = np.round(P1 * (8 ** 0.5))
    assert set(np.unique(scaled).tolist()) <= {-1.0, 1.0}
    assert np.array_equal(P1, P2)
    assert not np.array_equal(P1, P3)


def test_influence_for_seed_mirrors_mean_across_ckpts():
    """Verify against a hand-mirrored computation using the SAME projection
    matrix -- NOT against an assumed ordering: a random 2x2 Rademacher
    projection has no obligation to preserve raw-dot-product ordering at
    such tiny dimensionality (JL guarantees are asymptotic, not exact at
    in_dim=proj_dim=2), so this only checks the mean-across-checkpoints
    aggregation logic itself, matching score_proxy._mean_across's contract."""
    train_vecs_per_ckpt = [
        {"a/1": np.array([1.0, 0.0]), "a/2": np.array([0.0, 1.0])},
        {"a/1": np.array([2.0, 0.0]), "a/2": np.array([0.0, 2.0])},
    ]
    ref_vec_per_ckpt = [np.array([1.0, 0.0]), np.array([1.0, 0.0])]
    got = S._influence_for_seed(train_vecs_per_ckpt, ref_vec_per_ckpt,
                                seed=201, in_dim=2, proj_dim=2)
    assert set(got) == {"a/1", "a/2"}

    P = S._projection_matrix(201, in_dim=2, proj_dim=2)
    expected = {}
    for uid in ("a/1", "a/2"):
        per_ckpt = [float((tv[uid] @ P) @ (rv @ P))
                   for tv, rv in zip(train_vecs_per_ckpt, ref_vec_per_ckpt)]
        expected[uid] = float(np.mean(per_ckpt))
    assert got == pytest.approx(expected)


def test_grad_vecs_all_ckpts_reports_per_checkpoint_not_cumulative_duration(monkeypatch):
    """PROTOCOL §3.24: on_ckpt_done's ckpt_s must be THAT checkpoint's own
    duration, not the running total -- an average/cumulative-only view
    would hide a single slow or stuck checkpoint. Three fake checkpoints
    with deliberately uneven, distinguishable sleep times prove ckpt_s
    tracks the delta, not elapsed_s."""
    sleep_schedule = [0.05, 0.02, 0.08]
    call_index = {"i": 0}

    def fake_pooled(lit, ids, batch_size):
        time.sleep(sleep_schedule[call_index["i"]])
        return {"u/1": np.zeros(2)}

    monkeypatch.setattr(S, "_pooled_grad_vecs_for_ids", fake_pooled)
    monkeypatch.setattr(S, "_ref_vec_for_ids", lambda lit, ids, batch_size: np.zeros(2))

    seen = []

    def on_done(i, n, ckpt_s, elapsed_s):
        seen.append((i, n, ckpt_s, elapsed_s))
        call_index["i"] += 1

    S._grad_vecs_all_ckpts(["ck1", "ck2", "ck3"], ["u/1"], ["u/1"],
                          batch_size=4, on_ckpt_done=on_done)

    assert [i for i, *_ in seen] == [1, 2, 3]
    assert all(n == 3 for _, n, _, _ in seen)
    # each checkpoint's own duration roughly matches ITS sleep, not the sum
    for idx, (_, _, ckpt_s, _) in enumerate(seen):
        assert ckpt_s == pytest.approx(sleep_schedule[idx], abs=0.05)
    # cumulative elapsed strictly increases across checkpoints; the per-idx
    # loop above is what actually proves ckpt_s tracks the DELTA rather than
    # the running total -- if it reported cumulative elapsed instead, the
    # 3rd checkpoint's ckpt_s would be ~0.15 (sum of all sleeps) and fail
    # that approx(0.08) check
    assert seen[0][3] < seen[1][3] < seen[2][3]


# --------------------------------------------------------------------- wandb

def test_spearman_perfect_agreement_and_disagreement():
    a = {"u1": 1.0, "u2": 2.0, "u3": 3.0}
    b_agree = {"u1": 10.0, "u2": 20.0, "u3": 30.0}
    b_disagree = {"u1": 30.0, "u2": 20.0, "u3": 10.0}
    assert S._spearman(a, b_agree) == pytest.approx(1.0)
    assert S._spearman(a, b_disagree) == pytest.approx(-1.0)


def test_spearman_uses_only_shared_ids():
    a = {"u1": 1.0, "u2": 2.0, "u3": 3.0, "only_in_a": 99.0}
    b = {"u1": 10.0, "u2": 20.0, "u3": 30.0, "only_in_b": -5.0}
    assert S._spearman(a, b) == pytest.approx(1.0)


def test_main_marks_wandb_run_failed_on_exception(monkeypatch, tmp_path):
    """Caught live (2026-09-18) on score_proxy.py's identical finally:
    pattern -- job 2701229 crashed but still showed as a completed run in
    the W&B UI, because `run.finish()` with no args defaults to success
    regardless of whether an exception is propagating. A fake run object
    (not real wandb) proves main() now passes exit_code=1 on a crash."""
    calls = []

    class FakeRun:
        def __init__(self):
            self.summary = {}

        def finish(self, exit_code=0):
            calls.append(exit_code)

    fake_run = FakeRun()
    monkeypatch.setattr(S, "_wandb_init", lambda **kw: fake_run)
    monkeypatch.setattr(S, "enforce_gpu_policy", lambda: {"gpu_name": "fake"})

    def boom(bundle):
        raise RuntimeError("boom")

    monkeypatch.setattr(S, "_all_ckpts", boom)

    with pytest.raises(RuntimeError, match="boom"):
        S.main(["--bundle", str(tmp_path / "b")])
    assert calls == [1]
    assert "RuntimeError: boom" in fake_run.summary["error"]
    assert "boom" in fake_run.summary["traceback"]


def test_wandb_init_disabled_mode_runs_without_network():
    """mode='disabled' no-ops all wandb network/credential activity -- proves
    the run opens (and can be finished) without needing live credentials."""
    run = S._wandb_init(project="spell-rq2", entity=None, mode="disabled",
                        job_id="test", bundle="runs/track_b/fake",
                        seeds=[201, 202], proj_dim=512, batch_size=16)
    run.finish()


def test_log_ckpt_progress_disabled_mode_runs_without_network():
    run = S._wandb_init(project="spell-rq2", entity=None, mode="disabled",
                        job_id="test", bundle="runs/track_b/fake",
                        seeds=[201], proj_dim=512, batch_size=16)
    S._log_ckpt_progress(1, 4, 12.5, 12.5)
    run.finish()


def test_log_seed_summary_to_wandb_disabled_mode_runs_without_network():
    """Proves the logging code path (including the cross-seed spearman
    computation) is structurally correct without needing a live run. Must
    run against an already-open run, matching how main() actually calls it
    (after _wandb_init, before run.finish())."""
    run = S._wandb_init(project="spell-rq2", entity=None, mode="disabled",
                        job_id="test", bundle="runs/track_b/fake",
                        seeds=[201, 202], proj_dim=512, batch_size=16)
    seed_influences = {
        201: {"v0/1": 1.0, "v0/2": 2.0, "v0/3": 3.0},
        202: {"v0/1": 1.5, "v0/2": 2.5, "v0/3": 2.9},
    }
    S.log_seed_summary_to_wandb(seed_influences)
    run.finish()


# ------------------------------------------------------------------------ main

def _fake_main_setup(monkeypatch, train_vecs_per_ckpt, ref_vec_per_ckpt, val_ids=()):
    monkeypatch.setattr(S, "enforce_gpu_policy", lambda: {"gpu_name": "fake"})
    monkeypatch.setattr(S, "_all_ckpts", lambda bundle: ["ck1", "ck2"])
    monkeypatch.setattr(S, "_load_lit", lambda ckpt: object())
    monkeypatch.setattr(S, "_collect_train_records",
                        lambda: sorted({u for d in train_vecs_per_ckpt for u in d}))
    monkeypatch.setattr(S, "_collect_val_records", lambda: sorted(val_ids))
    monkeypatch.setattr(S, "load_records", lambda index_path, split: [])
    monkeypatch.setattr(S, "_preflight", lambda recs, k=50: None)
    monkeypatch.setattr(
        S, "_grad_vecs_all_ckpts",
        lambda lits, train_ids, vids, batch_size, on_ckpt_done=None:
            (train_vecs_per_ckpt, ref_vec_per_ckpt))

    class FakeHead:
        in_features = 2

    class FakeModel:
        head = FakeHead()

    class FakeLit:
        model = FakeModel()
    monkeypatch.setattr(S, "_load_lit", lambda ckpt: FakeLit())


def test_main_writes_one_parquet_per_seed_and_seeds_actually_differ(tmp_path, monkeypatch):
    train_vecs_per_ckpt = [{"v0/1": np.array([1.0, 0.0]), "v0/2": np.array([0.0, 1.0])}]
    ref_vec_per_ckpt = [np.array([1.0, 0.5])]
    _fake_main_setup(monkeypatch, train_vecs_per_ckpt, ref_vec_per_ckpt)

    out = tmp_path / "scores"
    rc = S.main(["--bundle", str(tmp_path / "b"), "--out-dir", str(out),
                "--seeds", "201", "202", "--proj-dim", "8",
                "--wandb-mode", "disabled"])
    assert rc == 0

    df201 = pd.read_parquet(out / "less_influence_seed201.parquet")
    df202 = pd.read_parquet(out / "less_influence_seed202.parquet")
    assert list(df201.columns) == ["utterance_id", "influence", "n_ckpts"]
    assert list(df201["utterance_id"]) == ["v0/1", "v0/2"]
    assert (df201["n_ckpts"] == 1).all()
    assert not df201["influence"].tolist() == df202["influence"].tolist()


def test_main_refuses_to_write_val_leaked_rows(tmp_path, monkeypatch):
    train_vecs_per_ckpt = [{"v0/1": np.array([1.0, 0.0]), "v1/9": np.array([0.0, 1.0])}]
    ref_vec_per_ckpt = [np.array([1.0, 0.0])]
    _fake_main_setup(monkeypatch, train_vecs_per_ckpt, ref_vec_per_ckpt, val_ids=["v1/9"])

    out = tmp_path / "scores"
    with pytest.raises(ValueError, match="val-leak"):
        S.main(["--bundle", str(tmp_path / "b"), "--out-dir", str(out), "--seeds", "201",
               "--wandb-mode", "disabled"])
    assert not out.exists() or not list(out.glob("*.parquet"))
