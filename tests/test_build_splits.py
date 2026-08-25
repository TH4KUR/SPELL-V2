import json

import pandas as pd

from config import ProtocolConfig
from scripts.build_splits import build_split_frames, write_split_files


def _proto(seed=7, val_size=6):
    return ProtocolConfig(
        sample_rate=16000, token_hz=50, codebook_size=1024, n_rvq_streams=8,
        crop_frames=50, crop_samples=16000, min_crop_frames=50, token_pad_id=-1,
        budget_fraction=0.25, val_size_utts=val_size, split_seed=seed,
    )


def _fake_index(n_videos=10, utts_per_video=3):
    rows = []
    for v in range(n_videos):
        vid = f"video{v:02d}"
        for u in range(utts_per_video):
            rows.append({
                "utterance_id": f"{vid}/{50001 + u}", "split": "trainval", "video_id": vid,
                "stem": str(50001 + u), "tokens_path": f"x/{vid}/{u}.pt",
                "audio_path": f"x/{vid}/{u}.mp4", "audio_kind": "mp4",
                "txt_path": f"x/{vid}/{u}.txt", "n_tokens": 100, "duration_s": 2.0,
                "conf": 4, "text_raw": "HELLO", "text_norm": "hello", "n_chars_norm": 5,
            })
    for i in range(4):
        rows.append({
            "utterance_id": str(i), "split": "test", "video_id": "", "stem": str(i),
            "tokens_path": f"t/{i}.tokens.pt", "audio_path": f"t/{i}.wav",
            "audio_kind": "wav", "txt_path": None, "n_tokens": 60, "duration_s": 1.2,
            "conf": None, "text_raw": "", "text_norm": "", "n_chars_norm": 0,
        })
    return pd.DataFrame(rows)


def test_video_disjoint_and_size_reached():
    df = _fake_index()
    train_df, val_df = build_split_frames(df, seed=7, val_size_utts=6)
    assert set(train_df["video_id"]).isdisjoint(set(val_df["video_id"]))
    assert len(val_df) >= 6                      # whole videos only -> >= target
    assert len(val_df) % 3 == 0                  # videos are atomic units of 3 utts
    assert len(train_df) + len(val_df) == len(df[df["split"] == "trainval"])


def test_determinism_same_seed():
    df = _fake_index()
    t1, v1 = build_split_frames(df, seed=123, val_size_utts=5)
    t2, v2 = build_split_frames(df, seed=123, val_size_utts=5)
    assert sorted(t1["utterance_id"]) == sorted(t2["utterance_id"])
    assert sorted(v1["utterance_id"]) == sorted(v2["utterance_id"])


def test_seed_changes_split():
    df = _fake_index(n_videos=30)
    _, v_a = build_split_frames(df, seed=1, val_size_utts=4)
    _, v_b = build_split_frames(df, seed=2, val_size_utts=4)
    assert set(v_a["video_id"]) != set(v_b["video_id"])


def test_written_files_are_byte_identical_on_rerun(tmp_path):
    df = _fake_index()
    proto = _proto()
    for run_dir in (tmp_path / "a", tmp_path / "b"):
        train_df, val_df = build_split_frames(df, proto.split_seed, proto.val_size_utts)
        write_split_files(train_df, val_df, df[df["split"] == "test"], run_dir, proto)
    for name in ("train_ids.txt", "val_ids.txt", "test_pointers.tsv"):
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_freeze_manifest_records_hashes(tmp_path):
    df = _fake_index()
    proto = _proto()
    train_df, val_df = build_split_frames(df, proto.split_seed, proto.val_size_utts)
    manifest = write_split_files(train_df, val_df, df[df["split"] == "test"], tmp_path, proto)
    on_disk = json.loads((tmp_path / "freeze_manifest.json").read_text())
    assert on_disk == manifest
    import hashlib
    for fname, rec in manifest["files"].items():
        h = hashlib.sha256((tmp_path / fname).read_bytes()).hexdigest()
        assert h == rec["sha256"]
