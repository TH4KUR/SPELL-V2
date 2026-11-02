"""configs/parity/recipes.yaml: the two pre-registered recipes (PROTOCOL §3.34), pinned by
sha256, every value tagged with its provenance, values equal to what the primary sources
state (RAVEN_TABLE_7 / USR_TABLE_8 below are transcribed from the papers, independently of
the yaml)."""

from __future__ import annotations

import re
import shutil

import pytest
import yaml

from parity import model as parity_model
from parity import recipes as R

# RAVEn (arXiv 2212.06246) Table 7, low-resource fine-tuning column; USR (arXiv 2411.02256)
# Table 8 / App. C.5. Transcribed from the papers as read by the Stage-2 brief.
RAVEN_TABLE_7 = dict(epochs=50, warmup_epochs=20, lr_encoder=2e-3, lr_decoder=6e-3, weight_decay=0.1,
                     layer_decay=0.5, min_lr=1e-5, grad_clip=5.0, drop_path=0.2)
USR_TABLE_8 = dict(epochs=75, warmup_epochs=20, lr_encoder=3e-3, lr_decoder=3e-3, weight_decay=0.04,
                   layer_decay=None, min_lr=1e-5, grad_clip=3.0, drop_path=0.1)


def test_pinned_hash_matches_the_file():
    assert R.recipes_sha256() == R.RECIPES_SHA256


def test_a_changed_byte_is_refused(tmp_path):
    edited = tmp_path / "recipes.yaml"
    edited.write_text(R.RECIPES_PATH.read_text().replace("seed: 42", "seed: 43"))
    with pytest.raises(R.RecipeError, match="RECIPES_SHA256"):
        R.load_recipe("R-A", edited)


def test_identical_copy_at_another_path_loads(tmp_path):
    copy = tmp_path / "r.yaml"
    shutil.copy(R.RECIPES_PATH, copy)
    assert R.load_recipe("R-B", copy).sha256 == R.RECIPES_SHA256


@pytest.mark.parametrize("name,table", [("R-A", RAVEN_TABLE_7), ("R-B", USR_TABLE_8)])
def test_recipe_values_equal_the_papers(name, table):
    r = R.load_recipe(name)
    for key, want in table.items():
        assert getattr(r, key) == want, key


def test_common_values():
    for name in R.RECIPE_NAMES:
        r = R.load_recipe(name)
        assert r.seed == 42 and r.betas == (0.9, 0.98)
        assert r.ctc_weight == 0.1 and r.label_smoothing == 0.1 and r.decode_ctc_weight == 0.1
        assert r.time_mask_window_s == 0.4 and r.time_mask_stride_s == 1.0
        assert r.waveform_normalisation == "none" and r.precision == "fp32"
        assert r.effective_batch_seconds == 198 and r.beam_size == 40
        assert r.model_selection == "final_epoch"
        assert r.sha256 == R.RECIPES_SHA256


def test_exactly_two_recipes_and_unknown_names_are_refused():
    assert R.RECIPE_NAMES == ("R-A", "R-B")
    assert sorted(yaml.safe_load(R.RECIPES_PATH.read_text())["recipes"]) == ["R-A", "R-B"]
    with pytest.raises(R.RecipeError, match="unknown recipe"):
        R.load_recipe("R-C")


def test_every_value_line_carries_a_provenance_tag():
    lines = R.RECIPES_PATH.read_text().splitlines()
    value_lines = [ln for ln in lines
                   if re.match(r"^\s+[a-z_]+:\s+\S", ln) and not ln.strip().startswith(("#", "description:"))]
    assert len(value_lines) == 14 + 2 * 9
    tag_re = re.compile(r"#\s*(" + "|".join(R.PROVENANCE_TAGS) + r")\b")
    untagged = [ln for ln in value_lines if not tag_re.search(ln)]
    assert untagged == []


def test_arch_overrides_touch_only_the_three_recipe_owned_keys():
    base = parity_model.load_arch_config()
    for name in R.RECIPE_NAMES:
        r = R.load_recipe(name)
        cfg = r.arch_config(base)
        changed = {k for k in cfg if cfg[k] != base[k]}
        assert changed <= {"drop_path", "mtlalpha", "lsm_weight"}
        assert cfg["drop_path"] == r.drop_path and cfg["mtlalpha"] == 0.1 and cfg["lsm_weight"] == 0.1
        parity_model._check_keys(cfg)


def _write(tmp_path, mutate):
    doc = yaml.safe_load(R.RECIPES_PATH.read_text())
    mutate(doc)
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d["common"].pop("seed"), "missing keys"),
    (lambda d: d["common"].update(extra=1), "unknown keys"),
    (lambda d: d["recipes"]["R-A"].update(lr_decoder=1e-3, layer_decay=None), "lr_encoder must equal lr_decoder"),
    (lambda d: d["common"].update(waveform_normalisation="peak"), "waveform_normalisation"),
    (lambda d: d["common"].update(precision="bf16"), "fp32"),
    (lambda d: d["common"].update(micro_batch_seconds=500), "micro_batch_seconds"),
    (lambda d: d["recipes"]["R-A"].update(warmup_epochs=50), "warmup_epochs"),
    (lambda d: d["common"].update(model_selection="best_val"), "model_selection"),
    (lambda d: d["recipes"].pop("R-B"), "exactly"),
])
def test_malformed_recipes_are_refused(tmp_path, mutate, match):
    with pytest.raises(R.RecipeError, match=match):
        R.load_recipe("R-A", _write(tmp_path, mutate), expect_sha256=None)
