#!/usr/bin/env python3
"""Offline tokenizer preparation for the RAVEn-parity track (PROTOCOL §3.34).

    # laptop, in a THROWAWAY venv that has sentencepiece (never in ~/envs/spell, never on Ada)
    python scripts/parity_train_spm.py --valid-id-file /path/to/lrs3-valid.id --out-dir parity/spm

Trains a SentencePiece unigram model (vocab 1000, character_coverage 1.0, <unk> kept, no
bos/eos/pad unit) on the upper-cased reference text of the parity TRAIN split only - the
split is rebuilt here exactly as scripts/parity_build_splits.py builds it - and writes:

  unigram1000.model / .vocab    the SentencePiece model
  unigram1000_units.txt         `piece id` lines, id = index in the token list (first piece = 1);
                                token list = ["<blank>"] + units + ["<eos>"]
  token_ids_trainval.tsv        `utterance_id<TAB>ids` for EVERY trainval utterance (train and
                                val; one file, so it does not reveal the validation membership)
  tokenizer.json                provenance + sha256 of the files + fidelity vs RAVEn's units list

The trainer reads these without sentencepiece. Whether to commit them is the user's licence
decision (parity/REVIEW.md); the val ids are the Meta-licensed lrs3-valid.id content, which is
why the token-id file covers all of trainval.

Exit codes: 0 ok, 2 input missing, 3 checksum refusal, 4 split/tokenizer invariant violated.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd                                     # noqa: E402

from config import load_paths                           # noqa: E402
from parity import spm_tokens as K                      # noqa: E402
from parity import splits as S                          # noqa: E402
from parity import text as T                            # noqa: E402

DEFAULT_OUT_DIR = Path(__file__).resolve().parents[1] / "parity" / "spm"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--valid-id-file", required=True, type=Path,
                    help="AV-HuBERT lrs3-valid.id (sha256-verified, same file as parity_build_splits.py)")
    ap.add_argument("--index", type=Path, default=None, help="default: data_index.parquet")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--overwrite", action="store_true", help="replace existing artifacts in --out-dir")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        valid_ids = S.read_valid_id_file(args.valid_id_file)
    except OSError as e:
        print(f"ERROR: cannot read {args.valid_id_file}: {e}", file=sys.stderr)
        return 2
    except S.ChecksumError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 3
    except S.SplitError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 4

    out = args.out_dir
    targets = [out / n for n in (K.MODEL_FILENAME, K.VOCAB_FILENAME, K.UNITS_FILENAME,
                                 K.TOKEN_IDS_FILENAME, K.PROVENANCE_FILENAME)]
    if any(p.exists() for p in targets) and not args.overwrite:
        print(f"REFUSED: artifacts already exist in {out}; pass --overwrite to replace them", file=sys.stderr)
        return 2

    index_path = args.index if args.index is not None else Path(load_paths().index_path)
    index = pd.read_parquet(index_path)
    raven_rows = T.read_raven_test_csv()
    try:
        result = S.build_splits(index, valid_ids, expected=S.PINNED_EXPECTATIONS,
                                raven_test_texts=[r.text for r in raven_rows])
    except S.SplitError as e:
        print(f"SPLIT INVARIANT VIOLATED: {e}", file=sys.stderr)
        return 4

    text_of = dict(zip(index["utterance_id"], (T.reference_text(t) for t in index["text_raw"])))
    train_refs = {u: text_of[u] for u in result.train_ids}
    val_refs = {u: text_of[u] for u in result.val_ids}
    test_refs = {u: text_of[u] for u in result.test_ids}

    out.mkdir(parents=True, exist_ok=True)
    prefix = out / "unigram1000"
    K.train_unigram(train_refs.values(), prefix)
    sp = K.load_processor(prefix.with_suffix(".model"))
    units = K.units_from_processor(sp)
    K.write_units_file(units, out / K.UNITS_FILENAME)

    trainval_refs = {**train_refs, **val_refs}
    rows = {u: K.encode_token_ids(sp, ref) for u, ref in trainval_refs.items()}
    K.write_token_ids(rows, out / K.TOKEN_IDS_FILENAME)

    roundtrip = {name: K.roundtrip_mismatches(sp, units, refs)
                 for name, refs in (("train", train_refs), ("val", val_refs), ("test", test_refs))}
    if roundtrip["train"]:
        print(f"TOKENIZER INVARIANT VIOLATED: {len(roundtrip['train'])} train references do not "
              f"survive encode/decode, e.g. {roundtrip['train'][:3]}", file=sys.stderr)
        return 4

    fidelity = K.fidelity_report(sp, units, T.load_units(), raven_rows)
    files = {n: {"sha256": K.sha256_path(out / n)}
             for n in (K.MODEL_FILENAME, K.VOCAB_FILENAME, K.UNITS_FILENAME, K.TOKEN_IDS_FILENAME)}
    import sentencepiece
    doc = K.provenance_document(
        sources={
            "valid_id_file_sha256": K.sha256_path(args.valid_id_file),
            "data_index_sha256": K.sha256_path(index_path),
            "raven_units_sha256": T.UNITS_SHA256,
            "n_train_texts": len(train_refs),
            "n_trainval_token_rows": len(rows),
        },
        units=units, files=files, fidelity=fidelity,
        roundtrip={k: {"n": len(v), "first": v[:5]} for k, v in roundtrip.items()},
        spm_version=sentencepiece.__version__)
    K.write_json_atomic(doc, out / K.PROVENANCE_FILENAME)

    print(f"[parity-spm] units={len(units)} train_texts={len(train_refs)} trainval_rows={len(rows)} -> {out}")
    print(f"[parity-spm] round-trip mismatches: " + ", ".join(f"{k}={len(v)}" for k, v in roundtrip.items()))
    print(f"[parity-spm] vs RAVEn units: common={fidelity['n_common']} "
          f"({100 * fidelity['common_fraction_of_mine']:.1f}% of mine, "
          f"{100 * fidelity['common_fraction_of_raven']:.1f}% of RAVEn's {fidelity['n_raven']}); "
          f"identical piece sequence on {fidelity['test_rows_with_identical_piece_sequence']}/"
          f"{fidelity['raven_test_rows']} RAVEn test rows "
          f"({100 * fidelity['test_rows_identical_fraction']:.1f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
