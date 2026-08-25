import pytest

import config


def test_protocol_loads_and_validates():
    proto = config.load_protocol()
    assert proto.crop_samples == 16000 and proto.crop_frames == 50
    assert proto.token_pad_id < 0
    h = config.config_hash(proto)
    assert len(h) == 64 and h == config.config_hash(proto)  # stable


def test_universe_budget_rounding_rule():
    # The real Phase-0 numbers: 31,071 selectable utts at 25% -> 7,768 (round-half-up)
    assert config.universe_budget(31071, 0.25) == 7768
    assert config.universe_budget(31071, 1.0) == 31071
    assert config.universe_budget(1000, 0.25) == 250
    assert config.universe_budget(3, 0.5) == 2      # 1.5 -> 2
    assert config.universe_budget(5, 0.5) == 3      # 2.5 -> 3 (half UP, unlike round())


def test_universe_budget_rejects_bad_input():
    with pytest.raises(ValueError):
        config.universe_budget(0, 0.25)
    with pytest.raises(ValueError):
        config.universe_budget(100, 0.0)
    with pytest.raises(ValueError):
        config.universe_budget(100, 1.5)


def test_protocol_rejects_inconsistent_crop_law():
    d = {
        "sample_rate": 16000, "token_hz": 50, "codebook_size": 1024, "n_rvq_streams": 8,
        "crop_frames": 50, "crop_samples": 8000,  # WRONG: must be 16000
        "min_crop_frames": 50, "token_pad_id": -1, "budget_fraction": 0.25,
        "val_size_utts": 2000, "split_seed": 20260825,
    }
    with pytest.raises(ValueError, match="inconsistent"):
        config.ProtocolConfig.from_dict(d)


def test_protocol_rejects_in_range_pad():
    d = {
        "sample_rate": 16000, "token_hz": 50, "codebook_size": 1024, "n_rvq_streams": 8,
        "crop_frames": 50, "crop_samples": 16000,
        "min_crop_frames": 50, "token_pad_id": 1023,  # in-range pad would corrupt targets
        "budget_fraction": 0.25, "val_size_utts": 2000, "split_seed": 20260825,
    }
    with pytest.raises(ValueError, match="token_pad_id"):
        config.ProtocolConfig.from_dict(d)
