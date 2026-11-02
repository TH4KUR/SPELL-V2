"""Splits of the RAVEn-parity track (PROTOCOL §3.33).

RAVEn's from-scratch/low-resource ASR trains on the official LRS3 ``trainval`` (30 h)
minus the 1,200 validation clips listed in AV-HuBERT's ``lrs3-valid.id`` and tests on
the official 1,321-clip test set. The validation clips are NOT video-disjoint from
the rest of trainval - that is how RAVEn (inheriting AV-HuBERT's split) did it.

Ids are the ``utterance_id`` of ``data_index.parquet`` (``<video_id>/<stem>`` for
trainval, the bare stem for test), i.e. exactly what ``dataset.load_id_list`` /
``dataset.load_records`` consume. The licensed id file is read, hashed and recorded
but never copied into the repo. Track-B split files are untouched.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from parity.text import VENDORED_TEST_CSV_PATH, reference_text

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPLITS_DIR = PROJECT_ROOT / "parity" / "splits"

VALID_ID_SHA256 = "4e24da968352dedfe11dd5bbd3efa77c96dd26e062019acf7a00c9264aab884a"
VALID_ID_REPO = "https://github.com/facebookresearch/av_hubert"
VALID_ID_COMMIT = "b4fa4af9b437757b8649dc8a226e6d4d8e780f6f"
VALID_ID_PATH = "avhubert/preparation/data/lrs3-valid.id"
VALID_ID_URL = (f"https://raw.githubusercontent.com/facebookresearch/av_hubert/"
                f"{VALID_ID_COMMIT}/{VALID_ID_PATH}")
RAVEN_REPO = "https://github.com/ahaliassos/raven"
RAVEN_COMMIT = "9fd9a3743c41ba93532258b340d123c20fe25432"
RAVEN_TEST_CSV_PATH = "data_paths/test_with_tags_counts_unigram1000.csv"

SPLIT_FILES = {
    "train": "parity_train_ids.txt",
    "val": "parity_val_ids.txt",
    "test": "parity_test_ids.txt",
}
PROVENANCE_FILE = "provenance.json"
SCHEMA = "spell-rq2-parity-splits-v1"
REQUIRED_COLUMNS = ("utterance_id", "split", "video_id", "duration_s", "text_raw")


class SplitError(RuntimeError):
    """A split invariant is violated - always loud."""


class ChecksumError(SplitError):
    """A pinned sha256 does not match."""


@dataclass(frozen=True)
class Expectations:
    n_train: int
    n_val: int
    n_test: int
    train_hours: float
    val_hours: float
    test_hours: float
    hours_tolerance: float


# Hours are quoted to two decimals (29.25 / 1.15 / 0.85) in the plan; the real values
# are 29.2478 / 1.1539 / 0.8532, so the tolerance is "within rounding".
PINNED_EXPECTATIONS = Expectations(
    n_train=30782, n_val=1200, n_test=1321,
    train_hours=29.25, val_hours=1.15, test_hours=0.85, hours_tolerance=0.005)


@dataclass
class SplitResult:
    train_ids: list[str]
    val_ids: list[str]
    test_ids: list[str]
    stats: dict = field(default_factory=dict)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_valid_id_file(path: str | Path, *, expect_sha256: str = VALID_ID_SHA256) -> list[str]:
    """Index-form ids (``<video_id>/<stem>``) of AV-HuBERT's validation clips."""
    raw = Path(path).read_bytes()
    got = sha256_bytes(raw)
    if got != expect_sha256:
        raise ChecksumError(f"{path}: sha256 {got} != expected {expect_sha256}")
    ids = []
    for n, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        if not line.startswith("trainval/"):
            raise SplitError(f"{path}:{n}: expected a `trainval/<video>/<clip>` id, got {line!r}")
        ids.append(line[len("trainval/"):])
    if not ids:
        raise SplitError(f"{path}: no ids")
    return ids


def _hours(df: pd.DataFrame) -> float:
    return float(df["duration_s"].sum()) / 3600.0


def build_splits(index: pd.DataFrame, valid_ids: list[str], *,
                 expected: Expectations | None = PINNED_EXPECTATIONS,
                 raven_test_texts: list[str] | None = None) -> SplitResult:
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in index.columns]
    if missing_cols:
        raise SplitError(f"index lacks columns {missing_cols}")
    if index["utterance_id"].duplicated().any():
        raise SplitError("utterance_id must be unique in the index")
    trainval = index[index["split"] == "trainval"]
    test = index[index["split"] == "test"]
    if trainval.empty or test.empty:
        raise SplitError("index must contain both trainval and test rows")
    if not trainval["utterance_id"].str.contains("/").all() or test["utterance_id"].str.contains("/").any():
        raise SplitError("id namespaces violated: trainval ids are <video>/<stem>, test ids are bare stems")

    if len(valid_ids) != len(set(valid_ids)):
        dups = [k for k, c in Counter(valid_ids).items() if c > 1]
        raise SplitError(f"duplicate valid ids: {dups[:5]}")
    unknown = sorted(set(valid_ids) - set(trainval["utterance_id"]))
    if unknown:
        raise SplitError(f"{len(unknown)} valid ids are not in the index (trainval): {unknown[:5]}")

    is_val = trainval["utterance_id"].isin(set(valid_ids))
    val_df, train_df = trainval[is_val], trainval[~is_val]
    frames = {"train": train_df, "val": val_df, "test": test}

    for name, df in frames.items():
        if (df["text_raw"].fillna("").str.strip() == "").any():
            raise SplitError(f"{name}: empty reference text present")

    ids = {name: sorted(df["utterance_id"]) for name, df in frames.items()}
    if (set(ids["train"]) & set(ids["val"])) or (set(ids["train"]) | set(ids["val"])) & set(ids["test"]):
        raise SplitError("splits are not pairwise disjoint")

    counts = {name: len(v) for name, v in ids.items()}
    hours = {name: _hours(df) for name, df in frames.items()}
    if expected is not None:
        want_n = {"train": expected.n_train, "val": expected.n_val, "test": expected.n_test}
        want_h = {"train": expected.train_hours, "val": expected.val_hours, "test": expected.test_hours}
        for name in frames:
            if counts[name] != want_n[name]:
                raise SplitError(f"{name} count {counts[name]} != expected {want_n[name]}")
            if abs(hours[name] - want_h[name]) > expected.hours_tolerance:
                raise SplitError(f"{name} hours {hours[name]:.4f} not within "
                                 f"{expected.hours_tolerance} of {want_h[name]}")

    stats: dict = {
        "counts": counts,
        "hours": hours,
        "val_videos": {
            "n_videos": int(val_df["video_id"].nunique()),
            "n_also_in_train": len(set(val_df["video_id"]) & set(train_df["video_id"])),
        },
        "test_text_check": None,
    }
    if raven_test_texts is not None:
        ours = Counter(reference_text(t) for t in test["text_raw"])
        theirs = Counter(t.upper() for t in raven_test_texts)
        if ours != theirs:
            only_ours = list((ours - theirs).elements())
            only_theirs = list((theirs - ours).elements())
            raise SplitError(
                f"test references differ from RAVEn's csv: {len(only_ours)} only in the index "
                f"(e.g. {only_ours[:2]}), {len(only_theirs)} only in RAVEn's csv (e.g. {only_theirs[:2]})")
        stats["test_text_check"] = {"n": sum(ours.values()), "equal_to_raven_csv": True}
    return SplitResult(train_ids=ids["train"], val_ids=ids["val"], test_ids=ids["test"], stats=stats)


def pinned_sources(valid_id_path: str | Path, index_path: str | Path | None = None) -> dict:
    """Provenance of the real inputs (hashes are recomputed from the files)."""
    index_path = PROJECT_ROOT / "data_index.parquet" if index_path is None else Path(index_path)
    valid_raw = Path(valid_id_path).read_bytes()
    return {
        "valid_id_file": {
            "repo": VALID_ID_REPO, "commit": VALID_ID_COMMIT, "path": VALID_ID_PATH,
            "url": VALID_ID_URL, "sha256": sha256_bytes(valid_raw),
            "n_lines": len(valid_raw.decode("utf-8").splitlines()),
            "licence_note": "custom Meta agreement; the file is NOT copied into this repo",
        },
        "raven_test_csv": {
            "repo": RAVEN_REPO, "commit": RAVEN_COMMIT, "path": RAVEN_TEST_CSV_PATH,
            "sha256": sha256_bytes(VENDORED_TEST_CSV_PATH.read_bytes()),
        },
        "data_index": {"path": "data_index.parquet", "sha256": sha256_bytes(index_path.read_bytes())},
    }


def write_splits(result: SplitResult, out_dir: str | Path, *, provenance_sources: dict) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, filename in SPLIT_FILES.items():
        ids = getattr(result, f"{name}_ids")
        data = "".join(i + "\n" for i in ids).encode("utf-8")
        (out_dir / filename).write_bytes(data)
        files[filename] = {"n": len(ids), "sha256": sha256_bytes(data)}
    provenance = {
        "schema": SCHEMA,
        "definition": ("train = official LRS3 trainval minus AV-HuBERT lrs3-valid.id (RAVEn's training "
                       "data); val = those 1,200 clips (not video-disjoint from train); "
                       "test = official LRS3 test"),
        "sources": provenance_sources,
        "counts": result.stats["counts"],
        "hours": result.stats["hours"],
        "val_videos": result.stats["val_videos"],
        "test_text_check": result.stats["test_text_check"],
        "files": files,
    }
    (out_dir / PROVENANCE_FILE).write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return provenance


def verify_split_files(directory: str | Path = SPLITS_DIR, *, index_path: str | Path | None = None) -> dict:
    """Check the id lists against provenance.json (sha256 and line counts) and, if
    ``index_path`` is given, that the index is the one they were built from."""
    directory = Path(directory)
    prov_path = directory / PROVENANCE_FILE
    if not prov_path.is_file():
        raise SplitError(f"{prov_path} missing: run scripts/parity_build_splits.py first")
    prov = json.loads(prov_path.read_text(encoding="utf-8"))
    for filename, meta in prov["files"].items():
        path = directory / filename
        if not path.is_file():
            raise SplitError(f"{path} missing (listed in provenance.json)")
        data = path.read_bytes()
        if sha256_bytes(data) != meta["sha256"] or data.count(b"\n") != meta["n"]:
            raise SplitError(f"{filename} does not match provenance.json (edited after it was built)")
    if index_path is not None:
        built_from = prov.get("sources", {}).get("data_index", {}).get("sha256")
        if built_from is None or sha256_bytes(Path(index_path).read_bytes()) != built_from:
            raise SplitError(
                f"{index_path} is not the data_index.parquet these splits were built from "
                "(INDEX IS TRUTH, PROTOCOL §10 item 5): rebuild with scripts/parity_build_splits.py")
    return prov
