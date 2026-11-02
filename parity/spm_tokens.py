"""SentencePiece unigram-1000 vocabulary of the parity track (PROTOCOL §3.34).

Offline preparation only. ``sentencepiece`` is imported lazily inside the functions that
need it (training, encoding); the production trainer reads the plain-text artifacts
written here (units file, token-id file, ``tokenizer.json``) through the functions that
do not, so Ada's frozen environment, which has no sentencepiece, never needs it.

Token ids follow the model's token list (``parity.text.token_list``): ``<blank>`` is 0,
the SentencePiece pieces occupy ids 1..N in SentencePiece id order (SentencePiece id
``i`` is token id ``i + 1``; the ``<unk>`` piece is SentencePiece id 0, hence token 1),
and ``<eos>`` is N + 1. Targets only ever use 1..N.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence

from parity import text as parity_text

VOCAB_SIZE = 1000
UNITS_FILENAME = "unigram1000_units.txt"
MODEL_FILENAME = "unigram1000.model"
VOCAB_FILENAME = "unigram1000.vocab"
TOKEN_IDS_FILENAME = "token_ids_trainval.tsv"
PROVENANCE_FILENAME = "tokenizer.json"
SCHEMA = "spell-rq2-parity-tokenizer-v1"

# unk_id=0 keeps <unk>; bos/eos/pad = -1 means none of them is a unit (the model adds its own
# <blank> and <eos>). character_coverage=1.0 keeps every character seen in training.
# num_threads=1 so the EM accumulation order, hence the model, does not depend on scheduling.
SPM_OPTIONS = dict(
    model_type="unigram",
    vocab_size=VOCAB_SIZE,
    character_coverage=1.0,
    unk_id=0,
    bos_id=-1,
    eos_id=-1,
    pad_id=-1,
    num_threads=1,
)


def _sentencepiece():
    try:
        import sentencepiece
    except ImportError as e:
        raise RuntimeError(
            "sentencepiece is not installed. It is needed only for the offline tokenizer step "
            "(scripts/parity_train_spm.py), which runs on the laptop in a throwaway venv; the "
            "trainer reads the artifacts it writes and never imports it") from e
    return sentencepiece


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: str | Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def train_unigram(texts: Iterable[str], model_prefix: str | Path, *, vocab_size: int = VOCAB_SIZE) -> Path:
    """Train on ``texts`` (one sentence each) and write ``<model_prefix>.model/.vocab``.

    SentencePiece stores the corpus and model-prefix strings it was given inside the model
    file, so training runs inside a scratch directory with fixed relative names: the model
    bytes then depend on the texts and options only, not on where or when it was trained.
    """
    spm = _sentencepiece()
    texts = list(texts)
    if not texts:
        raise ValueError("no training texts")
    for n, t in enumerate(texts):
        if "\n" in t or not t.strip():
            raise ValueError(f"training text {n} is empty or contains a newline")
    model_prefix = Path(model_prefix).resolve()
    model_prefix.parent.mkdir(parents=True, exist_ok=True)
    previous = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            Path("corpus.txt").write_text("".join(t + "\n" for t in texts), encoding="utf-8")
            spm.SentencePieceTrainer.train(
                input="corpus.txt", model_prefix="unigram", **{**SPM_OPTIONS, "vocab_size": vocab_size})
        finally:
            os.chdir(previous)
        shutil.copyfile(Path(tmp) / "unigram.model", model_prefix.with_suffix(".model"))
        shutil.copyfile(Path(tmp) / "unigram.vocab", model_prefix.with_suffix(".vocab"))
    return model_prefix.with_suffix(".model")


def load_processor(model_path: str | Path):
    spm = _sentencepiece()
    return spm.SentencePieceProcessor(model_file=str(model_path))


def units_from_processor(sp) -> list[str]:
    """Pieces in SentencePiece id order; refuses a model that has bos/eos/pad pieces."""
    units = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    if sp.unk_id() != 0 or units[0] != "<unk>":
        raise ValueError("expected <unk> as SentencePiece id 0")
    if sp.bos_id() != -1 or sp.eos_id() != -1 or sp.pad_id() != -1:
        raise ValueError("bos/eos/pad must be disabled: they are not units")
    if any(not u or any(c.isspace() for c in u) for u in units):
        raise ValueError("a piece is empty or contains whitespace: the units file would not round-trip")
    if len(set(units)) != len(units):
        raise ValueError("duplicate pieces")
    return units


def units_file_text(units: Sequence[str]) -> str:
    """Upstream's ``piece id`` format; the id is the piece's index in the token list (first = 1)."""
    return "".join(f"{u} {i}\n" for i, u in enumerate(units, start=1))


def write_units_file(units: Sequence[str], path: str | Path) -> None:
    Path(path).write_text(units_file_text(units), encoding="utf-8")


def encode_token_ids(sp, text: str) -> list[int]:
    """Token ids (SentencePiece id + 1) of one reference string."""
    return [i + 1 for i in sp.encode(text, out_type=int)]


def token_ids_text(rows: dict[str, Sequence[int]]) -> str:
    out = []
    for uid, ids in rows.items():
        if "\t" in uid or "\n" in uid or not uid:
            raise ValueError(f"bad utterance id {uid!r}")
        if not ids:
            raise ValueError(f"{uid}: empty token sequence")
        out.append(f"{uid}\t{' '.join(str(int(i)) for i in ids)}\n")
    return "".join(out)


def write_token_ids(rows: dict[str, Sequence[int]], path: str | Path) -> None:
    Path(path).write_text(token_ids_text(rows), encoding="utf-8")


def read_token_ids(path: str | Path, *, n_units: int) -> dict[str, tuple[int, ...]]:
    """``utterance_id -> ids``; every id must be a unit (1..n_units): never blank, never eos."""
    path = Path(path)
    rows: dict[str, tuple[int, ...]] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        fields = line.split("\t")
        if len(fields) != 2 or not fields[0] or not fields[1].strip():
            raise ValueError(f"{path}:{n}: expected `utterance_id<TAB>ids`, got {line!r}")
        uid, raw_ids = fields
        try:
            ids = tuple(int(i) for i in raw_ids.split(" "))
        except ValueError as e:
            raise ValueError(f"{path}:{n}: non-integer id in {raw_ids!r}") from e
        if min(ids) < 1 or max(ids) > n_units:
            raise ValueError(f"{path}:{n}: ids must lie in 1..{n_units} (0 is <blank>, {n_units + 1} is <eos>)")
        if uid in rows:
            raise ValueError(f"{path}:{n}: duplicate utterance id {uid!r}")
        rows[uid] = ids
    if not rows:
        raise ValueError(f"{path}: empty token-id file")
    return rows


def load_artifacts(directory: str | Path) -> dict:
    """Read and cross-check the tokenizer artifacts the trainer needs, without sentencepiece.

    Returns ``{"units", "tokens", "token_ids", "provenance"}``; the units file and the token-id
    file must hash to what ``tokenizer.json`` recorded when they were written.
    """
    directory = Path(directory)
    prov_path = directory / PROVENANCE_FILENAME
    if not prov_path.is_file():
        raise FileNotFoundError(f"{prov_path} missing: run scripts/parity_train_spm.py first")
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    if prov.get("schema") != SCHEMA:
        raise ValueError(f"{prov_path}: unexpected schema {prov.get('schema')!r}")
    for name in (UNITS_FILENAME, TOKEN_IDS_FILENAME, MODEL_FILENAME):
        want = prov["files"][name]["sha256"]
        got = sha256_path(directory / name)
        if got != want:
            raise ValueError(f"{directory / name}: sha256 {got} != {want} recorded in {PROVENANCE_FILENAME} "
                             "(edited after it was written)")
    units = parity_text.load_units(directory / UNITS_FILENAME)
    if len(units) != prov["n_units"]:
        raise ValueError(f"{directory / UNITS_FILENAME}: {len(units)} units != {prov['n_units']} recorded")
    return {
        "units": units,
        "tokens": parity_text.token_list(units),
        "token_ids": read_token_ids(directory / TOKEN_IDS_FILENAME, n_units=len(units)),
        "provenance": prov,
    }


def roundtrip_mismatches(sp, units: Sequence[str], refs: dict[str, str]) -> list[str]:
    """Utterance ids whose reference does not survive encode -> decode unchanged."""
    tokens = parity_text.token_list(list(units))
    return [uid for uid, ref in refs.items()
            if parity_text.ids_to_text(encode_token_ids(sp, ref), tokens) != ref]


def fidelity_report(sp, units: Sequence[str], raven_units: Sequence[str],
                    raven_rows: Sequence[parity_text.RavenTestRow]) -> dict:
    """How close this vocabulary is to RAVEn's released one.

    * piece-set overlap with RAVEn's units list (1,047 entries; not the same size);
    * on RAVEn's 1,321 official test rows: the fraction whose piece sequence under this model
      equals the piece sequence RAVEn's own ids spell (ids mapped to pieces through RAVEn's
      units list), plus how many of RAVEn's pieces this vocabulary lacks.
    """
    mine, theirs = set(units), set(raven_units)
    raven_tokens = parity_text.token_list(list(raven_units))
    equal = 0
    differing_examples = []
    mine_tokens_total = theirs_tokens_total = 0
    theirs_pieces_missing = Counter()
    for row in raven_rows:
        raven_pieces = [raven_tokens[i] for i in row.token_ids]
        my_pieces = sp.encode(row.text, out_type=str)
        mine_tokens_total += len(my_pieces)
        theirs_tokens_total += len(raven_pieces)
        for p in raven_pieces:
            if p not in mine:
                theirs_pieces_missing[p] += 1
        if my_pieces == raven_pieces:
            equal += 1
        elif len(differing_examples) < 5:
            differing_examples.append({"text": row.text, "raven": raven_pieces, "mine": my_pieces})
    n = len(raven_rows)
    inter = mine & theirs
    return {
        "n_mine": len(mine),
        "n_raven": len(theirs),
        "n_common": len(inter),
        "common_fraction_of_mine": len(inter) / len(mine),
        "common_fraction_of_raven": len(inter) / len(theirs),
        "jaccard": len(inter) / len(mine | theirs),
        "raven_test_rows": n,
        "test_rows_with_identical_piece_sequence": equal,
        "test_rows_identical_fraction": equal / n,
        "mean_tokens_per_test_row_mine": mine_tokens_total / n,
        "mean_tokens_per_test_row_raven": theirs_tokens_total / n,
        "raven_test_piece_occurrences_not_in_my_vocab": sum(theirs_pieces_missing.values()),
        "raven_test_distinct_pieces_not_in_my_vocab": len(theirs_pieces_missing),
        "differing_examples": differing_examples,
    }


def provenance_document(*, sources: dict, units: Sequence[str], files: dict, fidelity: dict,
                        roundtrip: dict, spm_version: str) -> dict:
    return {
        "schema": SCHEMA,
        "definition": ("SentencePiece unigram, trained on the upper-cased reference text of the parity TRAIN "
                       "split only (never val, never test); token id = SentencePiece id + 1; "
                       "<blank> = 0, <eos> = n_units + 1"),
        "spm_options": SPM_OPTIONS,
        "sentencepiece_version": spm_version,
        "n_units": len(units),
        "sources": sources,
        "files": files,
        "fidelity_vs_raven": fidelity,
        "roundtrip_mismatches": roundtrip,
    }


def write_json_atomic(doc: dict, path: str | Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(doc, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)
