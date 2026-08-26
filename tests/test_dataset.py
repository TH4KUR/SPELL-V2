import pytest
import torch

from dataset import TokenDataset, collate_token_batch


CODEBOOK_SIZE = 1024
PAD_SENTINEL = -1  # configs/protocol.yaml token_pad_id


def _make_item(uid, t, seed=0, text=None):
    g = torch.Generator().manual_seed(seed)
    return {
        "utterance_id": uid,
        "tokens": torch.randint(0, CODEBOOK_SIZE, (8, t), generator=g),
        "n_tokens": t,
        **({"text_ids": text} if text is not None else {}),
    }


def test_collate_pads_with_negative_sentinel():
    batch = [_make_item("a/1", 10, 0), _make_item("a/2", 4, 1), _make_item("b/9", 7, 2)]
    out = collate_token_batch(batch, token_pad_id=PAD_SENTINEL)
    assert out["tokens"].shape == (3, 8, 10)
    assert out["tokens"].dtype == torch.int64
    assert out["lengths"].tolist() == [10, 4, 7]
    # padding region uses the sentinel and never collides with real codes
    assert torch.all(out["tokens"][1, :, 4:] == PAD_SENTINEL)
    assert torch.all(out["tokens"][2, :, 7:] == PAD_SENTINEL)
    assert int(out["tokens"].min()) >= PAD_SENTINEL
    # every real (unmasked) frame holds an in-range code on all streams
    real = out["tokens"].transpose(1, 2)[out["mask"]]   # [n_real_frames, n_streams]
    assert int(real.min()) >= 0 and int(real.max()) < CODEBOOK_SIZE


def test_collate_mask_matches_lengths():
    batch = [_make_item("x", 6, 3), _make_item("y", 2, 4)]
    out = collate_token_batch(batch, token_pad_id=PAD_SENTINEL)
    assert out["mask"].shape == (2, 6)
    assert out["mask"][0].all()
    assert out["mask"][1].tolist() == [True, True, False, False, False, False]


def test_collate_rejects_positive_pad():
    with pytest.raises(ValueError, match="negative"):
        collate_token_batch([_make_item("x", 5)], token_pad_id=0)


def test_collate_text_ids():
    texts = [[2, 3, 4], [5]]  # pretend vocab ids (pad would be 0)
    batch = [
        _make_item("a", 3, 5, text=texts[0]),
        _make_item("b", 2, 6, text=texts[1]),
    ]
    out = collate_token_batch(batch, token_pad_id=PAD_SENTINEL, text_pad_id=0)
    assert out["text_ids"].tolist() == [[2, 3, 4], [5, 0, 0]]
    assert out["text_lengths"].tolist() == [3, 1]


def test_tokendataset_loads_via_injected_resolver(tmp_path):
    """Unit-test fixtures pin their own files through the path_resolver seam
    (production wires no resolver -> paths.resolve_token_path, DATA LAYOUT LAW)."""
    from pathlib import Path

    from dataset import UtteranceRecord

    tokens = torch.randint(0, CODEBOOK_SIZE, (8, 13))
    p = tmp_path / "tok.pt"
    torch.save(tokens, p)
    rec = UtteranceRecord(
        utterance_id="vid/50001",
        split="trainval",
        video_id="vid",
        stem="50001",
        tokens_path=str(p),
        audio_path=None,
        audio_kind="mp4",
        txt_path=None,
        n_tokens=13,
        duration_s=0.26,
        conf=4,
        text_raw="HELLO",
        text_norm="hello",
        n_chars_norm=5,
    )
    ds = TokenDataset([rec], path_resolver=lambda r: Path(r.tokens_path))
    item = ds[0]
    assert item["tokens"].shape == (8, 13)
    assert item["n_tokens"] == 13
    assert item["utterance_id"] == "vid/50001"
