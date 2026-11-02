"""parity/spm_tokens.py and scripts/parity_train_spm.py: the offline SentencePiece step.

The artifact formats and the cross-checks the trainer relies on are tested without
sentencepiece (it is not in Ada's frozen environment); the tests that train or encode
with a real SentencePiece model skip cleanly when it is not installed.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from parity import spm_tokens as K  # noqa: E402
from parity import text as T  # noqa: E402

UNITS = ["<unk>", "▁THE", "▁CAT", "S", "AT", "'"]


# ------------------------------------------------------------- formats (no spm) --


def test_units_file_uses_upstreams_piece_id_format_and_round_trips(tmp_path):
    p = tmp_path / "units.txt"
    K.write_units_file(UNITS, p)
    assert p.read_text(encoding="utf-8").splitlines()[:3] == ["<unk> 1", "▁THE 2", "▁CAT 3"]
    assert T.load_units(p) == UNITS
    assert T.token_list(UNITS) == ["<blank>"] + UNITS + ["<eos>"]


def test_vendored_upstream_units_file_has_the_same_line_format():
    raw = T.VENDORED_UNITS_PATH.read_text(encoding="utf-8").splitlines()
    assert raw[0] == "<unk> 1"
    assert all(len(line.split()) == 2 for line in raw)
    # the writer reproduces upstream's own file byte for byte
    assert K.units_file_text(T.load_units()).encode("utf-8") == T.VENDORED_UNITS_PATH.read_bytes()


def test_token_id_file_round_trip_and_range_checks(tmp_path):
    rows = {"vidA/50001": [2, 3, 4], "vidB/50002": [5]}
    p = tmp_path / "ids.tsv"
    K.write_token_ids(rows, p)
    assert p.read_text(encoding="utf-8") == "vidA/50001\t2 3 4\nvidB/50002\t5\n"
    assert K.read_token_ids(p, n_units=6) == {"vidA/50001": (2, 3, 4), "vidB/50002": (5,)}
    with pytest.raises(ValueError, match="1\\.\\.4"):
        K.read_token_ids(p, n_units=4)                   # id 5 is out of range
    for bad, match in [("u\t0 1\n", "1..6"),             # 0 is <blank>
                       ("u\t7 1\n", "1..6"),             # 7 = n_units + 1 is <eos>
                       ("u\t1 x\n", "non-integer"),
                       ("u\t\n", "expected"),
                       ("u 1 2\n", "expected"),
                       ("u\t1\nu\t2\n", "duplicate")]:
        p.write_text(bad)
        with pytest.raises(ValueError, match=match):
            K.read_token_ids(p, n_units=6)
    p.write_text("")
    with pytest.raises(ValueError, match="empty"):
        K.read_token_ids(p, n_units=6)


def test_token_id_writer_refuses_bad_rows(tmp_path):
    with pytest.raises(ValueError):
        K.write_token_ids({"a\tb": [1]}, tmp_path / "x")
    with pytest.raises(ValueError):
        K.write_token_ids({"a": []}, tmp_path / "x")


def _write_artifacts(directory: Path, units=UNITS, rows=None):
    directory.mkdir(parents=True, exist_ok=True)
    rows = rows or {"v/1": [2, 3], "v/2": [4, 5, 6]}
    K.write_units_file(units, directory / K.UNITS_FILENAME)
    K.write_token_ids(rows, directory / K.TOKEN_IDS_FILENAME)
    (directory / K.MODEL_FILENAME).write_bytes(b"model-bytes")
    files = {n: {"sha256": K.sha256_path(directory / n)}
             for n in (K.UNITS_FILENAME, K.TOKEN_IDS_FILENAME, K.MODEL_FILENAME)}
    doc = K.provenance_document(sources={}, units=units, files=files, fidelity={}, roundtrip={},
                                spm_version="x")
    K.write_json_atomic(doc, directory / K.PROVENANCE_FILENAME)


def test_load_artifacts_returns_units_tokens_and_ids(tmp_path):
    _write_artifacts(tmp_path)
    art = K.load_artifacts(tmp_path)
    assert art["units"] == UNITS
    assert art["tokens"][0] == "<blank>" and art["tokens"][-1] == "<eos>" and len(art["tokens"]) == len(UNITS) + 2
    assert art["token_ids"]["v/2"] == (4, 5, 6)


@pytest.mark.parametrize("victim", [K.UNITS_FILENAME, K.TOKEN_IDS_FILENAME, K.MODEL_FILENAME])
def test_load_artifacts_refuses_files_edited_after_the_provenance_was_written(tmp_path, victim):
    _write_artifacts(tmp_path)
    with open(tmp_path / victim, "ab") as f:
        f.write(b"\n" if victim != K.MODEL_FILENAME else b"x")
    with pytest.raises(ValueError, match="sha256"):
        K.load_artifacts(tmp_path)


def test_load_artifacts_refuses_a_missing_provenance_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="parity_train_spm"):
        K.load_artifacts(tmp_path)


class FakeProcessor:
    """encode(text, out_type=str) over a fixed vocabulary (greedy longest match per word)."""

    def __init__(self, pieces):
        self.pieces = set(pieces)

    def encode(self, text, out_type=str):
        out = []
        for w in text.split():
            rest = "▁" + w
            while rest:
                for n in range(len(rest), 0, -1):
                    if rest[:n] in self.pieces or n == 1:
                        out.append(rest[:n])
                        rest = rest[n:]
                        break
        return out


def test_fidelity_report_counts_overlap_and_identical_rows():
    raven_units = ["▁THE", "▁CAT", "S", "AT", "▁DOG", "ZZ"]
    tokens = T.token_list(raven_units)
    rows = [
        T.RavenTestRow("t", "a", 1, (1, 2), "THE CAT"),            # identical under mine
        T.RavenTestRow("t", "b", 1, (1, 2, 3), "THE CATS"),        # mine: ▁CATS unseen -> splits differently
        T.RavenTestRow("t", "c", 1, (5, 6), "DOG ZZ"),             # RAVEn has ZZ, mine lacks it
    ]
    mine_units = ["<unk>", "▁THE", "▁CAT", "S", "▁DOG"]
    rep = K.fidelity_report(FakeProcessor(mine_units), mine_units, raven_units, rows)
    assert (rep["n_mine"], rep["n_raven"], rep["n_common"]) == (5, 6, 4)
    assert rep["common_fraction_of_mine"] == pytest.approx(4 / 5)
    assert rep["common_fraction_of_raven"] == pytest.approx(4 / 6)
    assert rep["jaccard"] == pytest.approx(4 / 7)
    assert rep["raven_test_rows"] == 3
    assert rep["test_rows_with_identical_piece_sequence"] == 2     # rows a and b
    assert rep["raven_test_distinct_pieces_not_in_my_vocab"] == 1
    assert rep["raven_test_piece_occurrences_not_in_my_vocab"] == 1
    assert tokens[6] == "ZZ"


def test_trainer_side_modules_never_import_sentencepiece():
    code = ("import sys; import parity.spm_tokens, parity.text; "
            "assert 'sentencepiece' not in sys.modules, 'imported'")
    subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, check=True)


def test_script_refuses_a_missing_and_a_wrong_valid_id_file(tmp_path, capsys):
    import parity_train_spm as P
    assert P.main(["--valid-id-file", str(tmp_path / "nope"), "--out-dir", str(tmp_path / "o")]) == 2
    wrong = tmp_path / "wrong.id"
    wrong.write_text("trainval/a/b\n")
    assert P.main(["--valid-id-file", str(wrong), "--out-dir", str(tmp_path / "o")]) == 3
    assert not (tmp_path / "o").exists()


# --------------------------------------------------------------- real sentencepiece --

@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory):
    pytest.importorskip("sentencepiece")
    rng = random.Random(0)
    words = ["THE", "CAT", "SAT", "ON", "A", "MAT", "DOG", "RAN", "FAST", "HOME", "WE", "DON'T",
             "KNOW", "WHY", "BUT", "IT'S", "TRUE", "5", "O'CLOCK", "{LG}"]
    texts = [" ".join(rng.choice(words) for _ in range(rng.randint(3, 9))) for _ in range(400)]
    prefix = tmp_path_factory.mktemp("spm") / "tiny"
    model = K.train_unigram(texts, prefix, vocab_size=48)
    return texts, prefix, model


def test_trained_model_ids_are_spm_ids_plus_one_with_unk_first(tiny_model):
    texts, _, model = tiny_model
    sp = K.load_processor(model)
    units = K.units_from_processor(sp)
    assert len(units) == 48 and units[0] == "<unk>"
    tokens = T.token_list(units)
    ids = K.encode_token_ids(sp, texts[0])
    assert min(ids) >= 1 and max(ids) <= len(units)
    assert [tokens[i] for i in ids] == sp.encode(texts[0], out_type=str)
    assert [sp.id_to_piece(i - 1) for i in ids] == sp.encode(texts[0], out_type=str)


def test_no_bos_eos_pad_unit_and_every_training_character_is_kept(tiny_model):
    texts, _, model = tiny_model
    sp = K.load_processor(model)
    units = K.units_from_processor(sp)
    assert "<s>" not in units and "</s>" not in units
    chars = set("".join(texts).replace(" ", ""))
    assert chars <= set("".join(units).replace("▁", ""))      # character_coverage = 1.0
    assert K.roundtrip_mismatches(sp, units, {f"u{i}": t for i, t in enumerate(texts)}) == []


def test_training_is_byte_reproducible_independent_of_the_output_location(tiny_model, tmp_path):
    texts, prefix, model = tiny_model
    again = K.train_unigram(texts, tmp_path / "elsewhere" / "other_name", vocab_size=48)
    assert again.read_bytes() == model.read_bytes()


def test_units_file_written_from_a_model_matches_its_piece_order(tiny_model, tmp_path):
    _, _, model = tiny_model
    sp = K.load_processor(model)
    units = K.units_from_processor(sp)
    K.write_units_file(units, tmp_path / "u.txt")
    assert T.load_units(tmp_path / "u.txt") == units


def test_training_refuses_empty_input_and_newlines(tmp_path):
    pytest.importorskip("sentencepiece")
    with pytest.raises(ValueError):
        K.train_unigram([], tmp_path / "m")
    with pytest.raises(ValueError, match="newline"):
        K.train_unigram(["A\nB"], tmp_path / "m")


@pytest.mark.skipif("LRS3_VALID_ID_FILE" not in os.environ,
                    reason="needs the licensed AV-HuBERT id file (LRS3_VALID_ID_FILE)")
def test_real_pipeline_on_the_real_index(tmp_path):
    pytest.importorskip("sentencepiece")
    import parity_train_spm as P
    assert P.main(["--valid-id-file", os.environ["LRS3_VALID_ID_FILE"], "--out-dir", str(tmp_path)]) == 0
    art = K.load_artifacts(tmp_path)
    doc = art["provenance"]
    assert doc["n_units"] == 1000 and len(art["token_ids"]) == 31982
    assert all(not v["n"] for v in doc["roundtrip_mismatches"].values())
    assert json.loads((tmp_path / K.PROVENANCE_FILENAME).read_text())["fidelity_vs_raven"]["raven_test_rows"] == 1321
