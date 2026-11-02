"""The parity documents cannot drift from what they describe: REVIEW.md embeds the
manifest's provenance table verbatim, the claims ledger keeps its shape (measured
results stay PENDING until measured), and PROTOCOL §3.33 carries the binding rules."""

from __future__ import annotations

import re

from config import PROJECT_ROOT
from parity import vendor_provenance as vp

REVIEW = (PROJECT_ROOT / "parity" / "REVIEW.md").read_text(encoding="utf-8")
CLAIMS = (PROJECT_ROOT / "parity" / "CLAIMS.md").read_text(encoding="utf-8")
PROTOCOL = (PROJECT_ROOT / "PROTOCOL.md").read_text(encoding="utf-8")

STATUSES = ("VERIFIED", "CARRIED", "ASSUMED", "PENDING", "FAILED")


def _claim_rows() -> list[list[str]]:
    lines = CLAIMS.splitlines()
    start = lines.index("| claim | value | evidence | status |")
    assert lines[start + 1].startswith("|---")
    rows = []
    for ln in lines[start + 2:]:
        if not ln.startswith("|"):
            break
        rows.append([c.strip() for c in ln.strip().strip("|").split("|")])
    return rows


def test_review_embeds_the_current_manifest_table_verbatim():
    m = re.search(r"<!-- provenance-table:start -->\n(.*?)\n<!-- provenance-table:end -->", REVIEW, re.S)
    assert m, "provenance table markers missing"
    assert m.group(1) == vp.render_provenance_table(vp.load_manifest())


def test_review_lists_the_two_phase1_commands_in_order_and_they_exist():
    val = REVIEW.index("PARITY_SPLIT=val")
    test = REVIEW.index("SPELL_FINAL_EVAL=1 sbatch --export=ALL,PARITY_SPLIT=test")
    assert val < test
    for rel in ("slurm/parity_eval.sbatch", "scripts/parity_eval.py", "scripts/parity_rescore_track_b.py",
                "scripts/parity_build_splits.py", "scripts/parity_vendor_manifest.py"):
        assert (PROJECT_ROOT / rel).is_file(), rel
        assert rel in REVIEW


def test_review_states_the_pass_band_and_that_nothing_is_claimed():
    assert "0.045 and" in REVIEW and "0.049" in REVIEW
    assert "Nothing here claims parity" in REVIEW


def test_claims_ledger_has_four_columns_and_known_statuses():
    rows = _claim_rows()
    assert len(rows) >= 30
    for r in rows:
        assert len(r) == 4, r
        assert r[3].split(" ")[0] in STATUSES, r
        assert r[0] and r[1] and r[2], r


def test_measured_results_are_pending_until_measured():
    measured = [r for r in _claim_rows() if r[0].startswith("MEASURED")]
    assert len(measured) >= 5
    assert all(r[3] == "PENDING" for r in measured)


def test_every_verified_claim_names_its_evidence():
    for r in _claim_rows():
        if r[3].startswith("VERIFIED"):
            assert len(r[2]) > 15, r


def test_protocol_item_33_exists_once_and_carries_the_binding_rules():
    assert len(re.findall(r"^33\. \*\*RAVEn-parity track", PROTOCOL, re.M)) == 1
    start = PROTOCOL.index("33. **RAVEn-parity track")
    item = PROTOCOL[start:PROTOCOL.index("## 4. Known caveats", start)]
    for needle in ("corpus WER", "text_raw", "30,782", "1,200", "1,321", "SPELL_FINAL_EVAL=1",
                   "4.7 ± 0.2", "HARD STOP", "parity/CLAIMS.md", "§3.23", "byte-for-byte"):
        assert needle in item, needle
    assert re.search(r"^32\. \*\*", PROTOCOL, re.M) and PROTOCOL.index("32. **") < start


# ------------------------------------------------------------------------ Stage 2 ------

import math  # noqa: E402

import yaml  # noqa: E402

from parity import recipes as R  # noqa: E402

REVIEW2 = (PROJECT_ROOT / "parity" / "REVIEW_STAGE2.md").read_text(encoding="utf-8")


def _recipe_values_and_tags() -> dict[str, dict[str, tuple[object, str]]]:
    """key -> {recipe name -> (value, leading provenance tag)}, read from recipes.yaml itself."""
    doc = yaml.safe_load(R.RECIPES_PATH.read_text(encoding="utf-8"))
    tag_re = re.compile(r"#\s*(" + "|".join(R.PROVENANCE_TAGS) + r")\b")
    section, tags = None, {}
    for ln in R.RECIPES_PATH.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^(common):|^  (R-[AB]):", ln)
        if m:
            section = m.group(1) or m.group(2)
            continue
        m = re.match(r"^\s+([a-z_]+):\s+\S", ln)
        t = tag_re.search(ln)
        if m and t and section:
            tags[(section, m.group(1))] = t.group(1)
    out: dict[str, dict[str, tuple[object, str]]] = {}
    for name in R.RECIPE_NAMES:
        for key, value in {**doc["common"], **doc["recipes"][name]}.items():
            if key == "description":
                continue
            tag = tags[(name, key)] if (name, key) in tags else tags[("common", key)]
            out.setdefault(key, {})[name] = (value, tag)
    return out


def _same(cell: str, value) -> bool:
    if value is None:
        return cell == "none"
    if isinstance(value, list):
        return [float(x) for x in cell.split(",")] == value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return math.isclose(float(cell), value, rel_tol=1e-12)
    return cell == str(value)


def test_review_stage2_recipe_table_equals_the_pinned_yaml_values_and_tags():
    m = re.search(r"<!-- recipe-table:start -->\n(.*?)\n<!-- recipe-table:end -->", REVIEW2, re.S)
    assert m, "recipe table markers missing"
    lines = m.group(1).splitlines()
    assert lines[0].startswith("| key | R-A | R-A tag | R-B | R-B tag |") and lines[1].startswith("|---")
    rows = {cells[0]: cells for cells in ([c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines[2:])}
    expected = _recipe_values_and_tags()
    assert set(rows) == set(expected) and len(rows) == 23
    for key, cells in rows.items():
        for name, vcol, tcol in (("R-A", 1, 2), ("R-B", 3, 4)):
            value, tag = expected[key][name]
            assert _same(cells[vcol], value), (key, name, cells[vcol], value)
            assert cells[tcol] == tag, (key, name, cells[tcol], tag)


def test_review_stage2_names_the_pin_and_lists_the_commands_in_order_and_they_exist():
    assert R.RECIPES_SHA256 in PROTOCOL
    for rel in ("slurm/parity_train.sbatch", "scripts/parity_train.py", "scripts/parity_train_spm.py",
                "parity/trainer.py", "parity/train_data.py", "parity/optim.py", "parity/recipes.py",
                "parity/spm_tokens.py", "parity/checkpoint_avg.py", "configs/parity/recipes.yaml"):
        assert (PROJECT_ROOT / rel).is_file(), rel
        assert rel in REVIEW2, rel
    order = [REVIEW2.index(s) for s in (
        "parity_train_spm.py --valid-id-file", "PARITY_STOP_AFTER_EPOCHS=1", "PARITY_RECIPE=R-A slurm/parity_train.sbatch",
        "PARITY_RESUME=1,PARITY_RUN_ID=<RUN_ID>", "PARITY_SPLIT=val,PARITY_CKPT=", "SPELL_FINAL_EVAL=1 sbatch --export=ALL,PARITY_SPLIT=test,PARITY_CKPT=")]
    assert order == sorted(order)
    assert "Nothing here claims parity or any WER" in REVIEW2 and "Independent audit of Stage 1" in REVIEW2


def test_review_stage2_ranks_the_unknowns_effective_batch_first():
    sec = REVIEW2[REVIEW2.index("## 5. Assumptions and known unknowns"):REVIEW2.index("## 6. Numbers obtained")]
    items = re.findall(r"^\d\. \*\*([^*]+)\*\*", sec, re.M)
    assert items[0].startswith("Effective batch size") and items[1].startswith("Layer-wise decay")
    assert items[2].startswith("BatchNorm") and items[3].startswith("SentencePiece") and items[4].startswith("Checkpoint averaging")
    assert items[5].startswith("Weight decay")


def test_protocol_item_34_exists_once_after_item_33_and_carries_the_binding_rules():
    assert len(re.findall(r"^34\. \*\*RAVEn-parity track", PROTOCOL, re.M)) == 1
    start = PROTOCOL.index("34. **RAVEn-parity track")
    assert PROTOCOL.index("33. **RAVEn-parity track") < start < PROTOCOL.index("## 4. Known caveats")
    assert not re.search(r"^35\. \*\*", PROTOCOL, re.M)
    item = PROTOCOL[start:PROTOCOL.index("## 4. Known caveats", start)]
    for needle in ("R-A", "R-B", "RECIPES_SHA256", "VALIDATION WER only", "once", "SPELL_FINAL_EVAL=1", "16.5-20.5",
                   "professor", "Deviation log", "third recipe", "198 s", "BatchNorm", "§3.23-§3.25"):
        assert needle in item, needle


def test_stage2_ledger_rows_keep_every_measured_number_pending():
    rows = _claim_rows()
    names = [r[0] for r in rows]
    for needle in ("MEASURED: Ada 1-epoch timing", "MEASURED: R-A from scratch", "MEASURED: R-B from scratch",
                   "MEASURED: chosen recipe, official test"):
        row = next(r for r in rows if r[0].startswith(needle))
        assert row[3] == "PENDING", row
    assert sum(n.startswith("STAGE 2") for n in names) == 1
    assert any("Effective batch of 198 s" in n and r[3] == "ASSUMED" for n, r in zip(names, rows))
