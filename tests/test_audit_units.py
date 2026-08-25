"""Unit tests for audit_data internals that don't need a dataset walk."""

import torch

from config import load_protocol
from scripts.audit_data import Anomalies, _token_row


def _proto():
    return load_protocol()


def test_token_row_selectable_flag(tmp_path):
    proto = _proto()
    anomalies = Anomalies()
    common = dict(
        split="trainval", video_id="v", stem="50001",
        audio_path="a.mp4", audio_kind="mp4", txt_path="t.txt",
        conf=4, text_raw="HELLO WORLD",
    )

    long_p = tmp_path / "long.tokens.pt"
    torch.save(torch.randint(0, 1024, (8, 120)), long_p)
    row = _token_row("v/50001", tokens_path=str(long_p), proto=proto,
                     anomalies=anomalies, **common)
    assert row["selectable"] is True
    assert row["n_tokens"] == 120

    short_p = tmp_path / "short.tokens.pt"
    torch.save(torch.randint(0, 1024, (8, 30)), short_p)
    row = _token_row("v/60001", tokens_path=str(short_p), proto=proto,
                     anomalies=anomalies, **common)
    assert row["selectable"] is False          # below the frame floor
    assert anomalies.counters["trainval_too_short_for_crops"] == 1

    # test-split rows are never selectable regardless of length (eval-only pool)
    test_p = tmp_path / "test.tokens.pt"
    torch.save(torch.randint(0, 1024, (8, 200)), test_p)
    row = _token_row("42", tokens_path=str(test_p), proto=proto, anomalies=anomalies,
                     split="test", video_id="", stem="42",
                     audio_path="t.wav", audio_kind="wav", txt_path=None,
                     conf=None, text_raw="")
    assert row["selectable"] is False


def test_token_row_unreadable_is_not_selectable(tmp_path):
    proto = _proto()
    anomalies = Anomalies()
    bad = tmp_path / "bad.tokens.pt"
    bad.write_bytes(b"not a torch file")
    row = _token_row("v/x", tokens_path=str(bad), proto=proto, anomalies=anomalies,
                     split="trainval", video_id="v", stem="x",
                     audio_path="a.mp4", audio_kind="mp4", txt_path="t.txt",
                     conf=4, text_raw="HI")
    assert row["selectable"] is False and row["n_tokens"] == 0
    assert anomalies.counters["trainval_unreadable_tokens"] == 1
