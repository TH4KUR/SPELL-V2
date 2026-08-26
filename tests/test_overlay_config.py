"""Overlay config inheritance: deep_merge precedence, `base:` chain resolution,
cycle detection, and the real pilot configs inheriting the pinned track_b.yaml."""

import pytest

import yaml

from config import PROJECT_ROOT, deep_merge, load_config


# ------------------------------------------------------------------- deep_merge

def test_deep_merge_override_wins_leafwise():
    base = {"model": {"d_model": 256, "dropout": 0.1}, "training": {"lr": 5.0e-4}}
    override = {"model": {"dropout": 0.2}}
    merged = deep_merge(base, override)
    assert merged["model"]["d_model"] == 256          # untouched base key survives
    assert merged["model"]["dropout"] == 0.2          # leaf replaced
    assert merged["training"]["lr"] == 5.0e-4         # sibling section untouched
    assert base["model"]["dropout"] == 0.1            # inputs not mutated


def test_deep_merge_lists_and_scalars_replace_wholesale():
    base = {"seeds": [1, 2, 3], "opts": {"flags": ["a"], "x": 1}}
    override = {"seeds": [9], "opts": {"flags": ["b", "c"]}}
    merged = deep_merge(base, override)
    assert merged["seeds"] == [9]
    assert merged["opts"]["flags"] == ["b", "c"]      # list NOT recursed into
    assert merged["opts"]["x"] == 1                   # dict siblings still merge


def test_deep_merge_new_keys_added():
    merged = deep_merge({"a": 1}, {"b": {"c": 2}})
    assert merged == {"a": 1, "b": {"c": 2}}


# ------------------------------------------------------------------ load_config

def _write_yaml(path, payload):
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_base_chain_resolution_and_precedence(tmp_path):
    _write_yaml(tmp_path / "base.yaml", {
        "model": {"d_model": 256, "dropout": 0.1},
        "training": {"lr_peak": 5.0e-4, "n_epochs": None},
    })
    _write_yaml(tmp_path / "pilot.yaml", {
        "base": "base.yaml",
        "model": {"dropout": 0.15},
        "training": {"n_epochs": 20},
    })
    cfg = load_config(tmp_path / "pilot.yaml", configs_dir=tmp_path)
    assert cfg["model"] == {"d_model": 256, "dropout": 0.15}
    assert cfg["training"] == {"lr_peak": 5.0e-4, "n_epochs": 20}
    assert "base" not in cfg                          # consumed by the loader


def test_multi_level_chain_goes_deeper(tmp_path):
    _write_yaml(tmp_path / "l1.yaml", {"model": {"k": 15}, "train_seed": 20260826})
    _write_yaml(tmp_path / "l2.yaml", {"base": "l1.yaml", "model": {"layers": 4}})
    _write_yaml(tmp_path / "l3.yaml", {"base": "l2.yaml", "model": {"k": 32}})
    cfg = load_config(tmp_path / "l3.yaml", configs_dir=tmp_path)
    assert cfg["model"] == {"k": 32, "layers": 4}
    assert cfg["train_seed"] == 20260826              # threaded through every level


def test_base_cycle_raises(tmp_path):
    _write_yaml(tmp_path / "a.yaml", {"base": "b.yaml"})
    _write_yaml(tmp_path / "b.yaml", {"base": "a.yaml"})
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "a.yaml", configs_dir=tmp_path)


def test_self_base_cycle_raises(tmp_path):
    _write_yaml(tmp_path / "selfish.yaml", {"base": "selfish.yaml"})
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "selfish.yaml", configs_dir=tmp_path)


def test_missing_base_file_raises():
    p = tmp_other = None                              # keep flake-lints quiet
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        from pathlib import Path
        td_path = Path(td)
        p = td_path / "orphan.yaml"
        _write_yaml(p, {"base": "does_not_exist.yaml"})
        with pytest.raises(FileNotFoundError):
            load_config(p, configs_dir=td_path)
    del tmp_other


def test_returned_dict_is_fresh_per_call(tmp_path):
    _write_yaml(tmp_path / "c.yaml", {"x": 1})
    c1 = load_config(tmp_path / "c.yaml", configs_dir=tmp_path)
    c1["x"] = 999                                     # callers may mutate freely
    c2 = load_config(tmp_path / "c.yaml", configs_dir=tmp_path)
    assert c2["x"] == 1


# ------------------------------------------- real pilot configs inherit track_b

@pytest.mark.parametrize("yaml_name", ["pilot_100pct.yaml", "pilot_25pct_random_x2seeds.yaml"])
def test_pilot_configs_resolve_through_track_b_base(yaml_name):
    cfg = load_config(PROJECT_ROOT / "configs" / yaml_name)
    # the PROVISIONAL pilot budget rides on top of the pinned base — equal epochs
    assert cfg["training"]["n_epochs"] == 20
    # the ENTIRE frozen track_b surface arrives via base:
    assert "base" not in cfg
    assert cfg["track"] == "asr_ctc"
    model = cfg["model"]
    assert model["d_model"] == 256
    assert model["n_conformer_layers"] == 4
    assert model["n_conformer_heads"] == 4
    assert model["conformer_ff_mult"] == 4
    assert model["conv_kernel_size"] == 15            # recorded k deviation
    assert model["input_stream"] == 0                 # RVQ₁ only
    train = cfg["training"]
    assert train["train_seed"] == 20260826            # FIXED constant (§3.11)
    assert train["lr_peak"] == 5.0e-4
    assert train["batch_size"] == 32
    assert cfg["decode"]["greedy"] is True            # no external LM, ever
    assert cfg["logging"]["wandb"]["mode"] == "online"  # ONLINE default policy
    assert cfg["augment"]["time_mask_ratio_max"] == 0.15


def test_track_b_baseline_has_no_builtin_epoch_budget():
    cfg = load_config(PROJECT_ROOT / "configs" / "track_b.yaml")
    assert cfg["training"]["n_epochs"] is None        # per-run driver sets it; FATAL if unset
