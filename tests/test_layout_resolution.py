"""Layout-resolution contract suite (PROTOCOL §10 items 4–6).

Pins THE failure class that killed Pilot A twice: loaders assuming a
``trainval/`` segment the staged tree does not carry. Every test here proves
one face of the same law — locations come ONLY from paths.py keyed on
(index.video_id, index.stem) against the canonical STAGED flat tree declared
in docs/layout.md; split membership is index metadata, never a path segment.

Named regression case throughout: ``0D9QIG36J9Q/50001`` (the exact UID Pilot A
failed to resolve as ``$HOME/spell/data/trainval/0D9QIG36J9Q/50001.tokens.pt``
when the real file lives at ``$HOME/spell/data/0D9QIG36J9Q/50001.tokens.pt``).
"""

from __future__ import annotations

import dataclasses
import os
import random
import sys
from pathlib import Path, PurePosixPath

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import paths as paths_mod  # noqa: E402
from dataset import UtteranceRecord, load_id_list  # noqa: E402
from paths import DataPaths, LayoutError  # noqa: E402

REGRESSION_VIDEO = "0D9QIG36J9Q"
REGRESSION_STEM = "50001"
CONTRACT_SAMPLE_N = 100
SAMPLE_SEED = 20260826


# --------------------------------------------------------------- helpers --

def _index_df() -> pd.DataFrame:
    return pd.read_parquet(PROJECT_ROOT / "data_index.parquet")


def _row(df: pd.DataFrame, uid: str) -> pd.Series:
    return df.loc[df["utterance_id"] == uid].iloc[0]


def _rec_from_row(row) -> UtteranceRecord:
    return UtteranceRecord(
        utterance_id=row.utterance_id,
        split=row.split,
        video_id=str(row.video_id),
        stem=str(row.stem),
        tokens_path=str(row.tokens_path),      # legacy provenance string, unused
        audio_path=str(row.audio_path),
        audio_kind=str(row.audio_kind),
        txt_path=None if pd.isna(row.txt_path) else str(row.txt_path),
        n_tokens=int(row.n_tokens),
        duration_s=float(row.duration_s),
        conf=None if pd.isna(row.conf) else int(row.conf),
        text_raw=str(row.text_raw),
        text_norm=str(row.text_norm),
        n_chars_norm=int(row.n_chars_norm),
    )


def _oracle_rel(video_id: str, stem: str, suffix: str = ".tokens.pt") -> tuple[str, ...]:
    """Independent restatement of the documented law — the test-side twin.
    Written FROM docs/layout.md, deliberately NOT imported from paths.py."""
    return ((video_id, f"{stem}{suffix}") if video_id else (f"{stem}{suffix}",))


def _fake_staged_tree(tmp_path: Path, uids: list[str], df: pd.DataFrame,
                      suffix: str = ".tokens.pt") -> dict[str, Path]:
    """Materialize one dummy marker per uid where the CANONICAL law says it sits.
    Resolvers only stat files, so content is irrelevant."""
    made = {}
    for uid in uids:
        row = _row(df, uid)
        rel = _oracle_rel(str(row.video_id), str(row.stem), suffix)
        p = tmp_path.joinpath(*rel)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
        made[uid] = p
    return made


# ------------------------------------------- (a) manifest-vs-fake-tree contract

def test_contract_manifest_uids_resolve_one_to_one(tmp_path):
    """~100 UIDs drawn deterministically from the COMMITTED pilot manifests all
    resolve through the authority to exactly the docs/layout.md location built
    from index metadata — 1:1, with no split segment anywhere."""
    df = _index_df()
    ids = sorted(
        load_id_list(PROJECT_ROOT / "subsets/random_25pct_seed101.txt")
        | load_id_list(PROJECT_ROOT / "subsets/random_25pct_seed102.txt")
    )
    sample = random.Random(SAMPLE_SEED).sample(ids, CONTRACT_SAMPLE_N)

    made = _fake_staged_tree(tmp_path, sample, df)
    dp = DataPaths(root=tmp_path, layout="staged")

    resolved = {}
    for uid in sample:
        row = _row(df, uid)
        got = dp.resolve_tokens(_rec_from_row(row))
        assert Path(got) == tmp_path.joinpath(
            *_oracle_rel(str(row.video_id), str(row.stem)))
        assert "trainval" not in Path(got).parts          # THE regression class
        resolved[uid] = got
    assert set(resolved) == set(sample)                   # 1:1, none skipped
    assert set(made) == set(sample)

    pairs = dp.preflight([_rec_from_row(_row(df, u)) for u in sample],
                         k=len(sample))
    assert dict(pairs) == resolved


# --------------------------------------------- (b) live-layout probe (guarded)

@pytest.mark.skipif(not os.environ.get("SPELL_DATA_ROOT"),
                    reason="SPELL_DATA_ROOT unset — no live staged tree to probe")
def test_live_layout_probe_on_real_tree():
    """Hits the REAL staged tree at $SPELL_DATA_ROOT when present (e.g. Ada with
    SPELL_DATA_ROOT=$HOME/spell/data). Skips cleanly everywhere else."""
    try:
        paths_mod.reset_cache()
        df = _index_df()
        rec = _rec_from_row(_row(df, f"{REGRESSION_VIDEO}/{REGRESSION_STEM}"))
        got = paths_mod.resolve_token_path(rec)
        assert got.is_file(), f"live tree missing {got}"
        assert got.parts[-2:] == (REGRESSION_VIDEO, f"{REGRESSION_STEM}.tokens.pt")
    finally:
        paths_mod.reset_cache()


# --------------------------------------------- (c) NAMED regression case

def test_regression_pilot_a_uid_resolves_split_free(tmp_path):
    """THE UID from the incident report. Endswith the exact on-Ada location;
    the string 'trainval' may never appear."""
    df = _index_df()
    uid = f"{REGRESSION_VIDEO}/{REGRESSION_STEM}"
    made = _fake_staged_tree(tmp_path, [uid], df)
    got = DataPaths(root=tmp_path, layout="staged").resolve_tokens(
        _rec_from_row(_row(df, uid)))
    assert str(got).endswith(f"/{REGRESSION_VIDEO}/{REGRESSION_STEM}.tokens.pt")
    assert got == made[uid]
    assert "trainval" not in got.parts


def test_regression_missing_file_aborts_with_full_diff(tmp_path):
    """No silent fallback: an absent file is a LOUD abort naming root, resolved
    path, and the declaring document — the diff the operator needs."""
    df = _index_df()
    rec = _rec_from_row(_row(df, f"{REGRESSION_VIDEO}/{REGRESSION_STEM}"))
    dp = DataPaths(root=tmp_path, layout="staged")
    with pytest.raises(LayoutError) as ei:
        dp.resolve_tokens(rec)
    msg = str(ei.value)
    assert "[spell] LAYOUT MISMATCH" in msg
    assert str(dp.root) in msg                  # data_root visible...
    assert REGRESSION_VIDEO in msg              # ...resolved path...
    assert paths_mod.LAYOUT_DOC in msg          # ...and the declaring doc


# ------------------------------------------------------------ resolver faces

def test_audio_prefers_flac_twin_on_staged(tmp_path):
    df = _index_df()
    uid = f"{REGRESSION_VIDEO}/{REGRESSION_STEM}"
    flac = tmp_path / REGRESSION_VIDEO / f"{REGRESSION_STEM}.flac"
    flac.parent.mkdir(parents=True)
    flac.write_bytes(b"f")
    decoy = tmp_path / REGRESSION_VIDEO / f"{REGRESSION_STEM}.mp4"
    decoy.write_bytes(b"m")                     # legacy container must NOT win
    got = DataPaths(root=tmp_path, layout="staged").resolve_audio(
        _rec_from_row(_row(df, uid)))
    assert got == flac


def test_legacy_mode_strips_prefix_and_finds_container(tmp_path):
    row = _row(_index_df(), f"{REGRESSION_VIDEO}/{REGRESSION_STEM}")
    rec = dataclasses.replace(
        _rec_from_row(row),
        audio_path=(f"datasets/LRS3/trainval/{REGRESSION_VIDEO}/"
                    f"{REGRESSION_STEM}.mp4"))
    mp4 = tmp_path / "trainval" / REGRESSION_VIDEO / f"{REGRESSION_STEM}.mp4"
    mp4.parent.mkdir(parents=True)
    mp4.write_bytes(b"m")
    got = DataPaths(root=tmp_path, layout="legacy").resolve_audio(rec)
    assert got == mp4
    assert got.parts[-3:-2] == ("trainval",)   # raw audit-era tree DOES carry it


def test_unknown_layout_value_rejected(tmp_path):
    with pytest.raises(LayoutError, match="not one of"):
        DataPaths(root=tmp_path, layout="splitdirs")


def test_preflight_empty_record_list_is_loud(tmp_path):
    with pytest.raises(LayoutError, match="empty record list"):
        DataPaths(root=tmp_path, layout="staged").preflight([])


# ------------------------------ stager ↔ resolver twin equality (divergence kill)

def test_stage_jobs_relpaths_are_the_resolver_twins():
    """stage_to_ada builds DESTINATION relpaths via the SAME paths.*_relpath
    twins the runtime resolver uses — verified here over the ENTIRE committed
    index. If these ever diverge, staging and training disagree about the
    tree; this test makes that state unreachable without a red suite."""
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    import stage_to_ada as stager  # noqa: E402

    df = _index_df()
    dp = DataPaths(root=Path("/twins-never-resolve"), layout="staged")

    want_tokens: set[str] = set()
    want_txt: set[str] = set()
    for r in df.itertuples():
        vid, stem = str(r.video_id), str(r.stem)
        want_tokens.add(dp.tokens_relpath(vid, stem).as_posix())
        if r.txt_path and not pd.isna(r.txt_path):
            want_txt.add(dp.transcript_relpath(vid, stem).as_posix())

    jobs = stager.plan_jobs(source=PROJECT_ROOT, dest=Path("/tmp"), limit=None)
    got_tokens = {j["rel"] for j in jobs if j["kind"] == stager.KIND_TOKENS}
    got_txt = {j["rel"] for j in jobs if j["kind"] == stager.KIND_TRANSCRIPT}
    n_flac = sum(j["kind"] == stager.KIND_AUDIO for j in jobs)

    assert got_tokens == want_tokens                    # tokens: exact twin set
    assert got_txt == want_txt                          # transcripts: exact twin set
    assert n_flac == len(df)                            # one .flac job per utterance
    assert len({j["rel"] for j in jobs}) == len(jobs)   # and zero collisions overall
    # NO emitted relpath carries a split segment, anywhere in the staging plan
    assert not any("trainval" in PurePosixPath(j["rel"]).parts or
                   "test" in PurePosixPath(j["rel"]).parts for j in jobs)


# -------------------------------------------------- declaration-document pins

def test_docs_layout_md_declares_the_canonical_patterns():
    text = (PROJECT_ROOT / paths_mod.LAYOUT_DOC).read_text(encoding="utf-8")
    assert "<root>/<VIDEO_ID>/<stem>.tokens.pt" in text      # trainval pattern
    assert "<root>/<stem>.tokens.pt" in text                 # test bare-file pattern
    assert f"{REGRESSION_VIDEO}/{REGRESSION_STEM}.tokens.pt" in text   # worked example
    assert "METADATA ONLY" in text                            # split-membership rule
    assert "INDEX IS TRUTH" in text                           # §10 item 5 pointer
    assert "SPELL_DATA_LAYOUT=legacy" in text                 # quarantine knob
