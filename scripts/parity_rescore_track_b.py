#!/usr/bin/env python3
"""Phase-0 diagnostic (PROTOCOL §3.33): re-score a Track-B ``predictions.jsonl``.

    python scripts/parity_rescore_track_b.py \
        --predictions <eval_dir>/predictions.jsonl --summary <eval_dir>/summary.json \
        --out outputs/parity/rescore_ceiling.json

Two statistics over the same decoded text:

* macro-mean WER - mean of per-utterance jiwer WERs, exactly what Track B reports
  (``ctc.py``, ``scripts/evaluate_track_b.py``);
* CORPUS WER - total word-level edits / total reference words, what RAVEn and
  Auto-AVSR report. Edits come from jiwer's alignment (substitutions + deletions +
  insertions) and must agree, row by row, with the independent edit distance in
  ``parity.metrics``; both statistics are also given inside reference-length
  quartiles.

Given the bundle's ``summary.json`` the macro mean must reproduce its stored
``mean_wer`` to 1e-9 (and ``n_utterances`` must match), otherwise the script exits
non-zero and writes no result. Exit codes: 0 ok, 2 bad input, 4 check failed.
Logs to W&B per PROTOCOL §3.23-§3.25 (job_type ``parity_rescore``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import jiwer                                            # noqa: E402

import ctc as ctc_lib                                   # noqa: E402  Track B's own wer_one
from parity import metrics, runinfo, wandb_run          # noqa: E402

TOLERANCE = 1e-9
ROW_KEYS = ("utterance_id", "ref", "hyp")

DEFINITIONS = {
    "macro_wer": "mean over scoreable utterances of jiwer.wer(ref, hyp) (Track B, ctc.py)",
    "corpus_wer": "sum of word-level edits / sum of reference words (RAVEn, metrics.WER)",
}


class InputError(Exception):
    """Unreadable or malformed input (exit 2)."""


class CheckFailed(Exception):
    """The re-score contradicts a stored number or an independent computation (exit 4)."""


def read_predictions(path: Path) -> list[dict]:
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as e:
        raise InputError(f"cannot read predictions {path}: {e}") from e
    rows = []
    for n, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except ValueError as e:
            raise InputError(f"{path}:{n}: not JSON ({e})") from e
        missing = [k for k in ROW_KEYS if k not in row]
        if missing:
            raise InputError(f"{path}:{n}: row lacks {missing}")
        if not isinstance(row["ref"], str) or not isinstance(row["hyp"], str):
            raise InputError(f"{path}:{n}: ref/hyp must be strings")
        rows.append(row)
    if not rows:
        raise InputError(f"{path}: no rows")
    return rows


def rescore(rows: list[dict]) -> dict:
    """Pure re-scoring of Track-B rows; raises CheckFailed on any inconsistency."""
    per_utt: list[metrics.UttScore] = []
    wers: list[float] = []
    n_unscorable = 0
    for row in rows:
        ref, hyp = row["ref"], row["hyp"]
        if not ref.strip():                       # Track B leaves wer=None for these
            n_unscorable += 1
            continue
        wer = ctc_lib.wer_one(ref, hyp)
        stored = row.get("wer")
        if stored is not None and abs(stored - wer) > TOLERANCE:
            raise CheckFailed(f"row {row['utterance_id']}: stored wer {stored!r} != recomputed {wer!r}")
        out = jiwer.process_words(ref, hyp)
        edits = out.substitutions + out.deletions + out.insertions
        ref_words = out.hits + out.substitutions + out.deletions
        if edits != metrics.pair_edits(ref, hyp) or ref_words != len(ref.split()):
            raise CheckFailed(
                f"row {row['utterance_id']}: jiwer alignment (edits={edits}, ref_words={ref_words}) "
                f"and the independent edit distance disagree")
        per_utt.append(metrics.UttScore(edits=edits, ref_words=ref_words))
        wers.append(wer)
    if not per_utt:
        raise InputError("no scoreable rows (every reference is empty)")
    corpus = metrics.aggregate(per_utt)
    macro = sum(wers) / len(wers)                 # Track B's exact formula and order
    if abs(macro - corpus.macro_wer) > 1e-12:
        raise CheckFailed(f"macro mean {macro!r} != edits/ref-words macro {corpus.macro_wer!r}")
    quartiles = (
        [vars(b) for b in metrics.length_quartiles(corpus)] if corpus.n_scored >= 4 else None)
    return {
        "n_rows": len(rows),
        "n_scored": corpus.n_scored,
        "n_unscorable": n_unscorable,
        "macro_wer": macro,
        "corpus_wer": corpus.corpus_wer,
        "total_edits": corpus.total_edits,
        "total_ref_words": corpus.total_ref_words,
        "quartiles": quartiles,
        "definitions": DEFINITIONS,
    }


def check_against_summary(result: dict, summary_path: Path) -> dict:
    try:
        summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise InputError(f"cannot read summary {summary_path}: {e}") from e
    stored = summary.get("mean_wer")
    if stored is None:
        raise InputError(f"{summary_path} has no mean_wer to reproduce")
    n_stored = summary.get("n_utterances")
    if n_stored is not None and n_stored != result["n_rows"]:
        raise CheckFailed(f"summary n_utterances={n_stored} but predictions has {result['n_rows']} rows")
    diff = abs(result["macro_wer"] - stored)
    if diff > TOLERANCE:
        raise CheckFailed(
            f"recomputed macro mean_wer {result['macro_wer']!r} does not reproduce the stored "
            f"mean_wer {stored!r} (|diff|={diff:.3g} > {TOLERANCE:g})")
    return {"stored_mean_wer": stored, "abs_diff_vs_stored": diff}


def format_table(res: dict) -> str:
    lines = [
        f"rows={res['n_rows']} scored={res['n_scored']} unscorable={res['n_unscorable']}",
        f"  macro-mean WER : {res['macro_wer']:.6f}   (Track B definition)",
        f"  corpus WER     : {res['corpus_wer']:.6f}   (RAVEn definition; "
        f"{res['total_edits']} edits / {res['total_ref_words']} reference words)",
    ]
    if res["quartiles"]:
        lines.append("  reference-length quartiles (words)      n   corpus WER   macro WER")
        for b in res["quartiles"]:
            lines.append(f"    Q{b['index'] + 1}  {b['lo_words']:>3}-{b['hi_words']:<3}"
                         f"{b['n_utts']:>26} {b['corpus_wer']:>12.6f} {b['macro_wer']:>11.6f}")
    return "\n".join(lines)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--summary", type=Path, default=None,
                    help="the bundle's summary.json; its mean_wer must be reproduced to 1e-9")
    ap.add_argument("--out", type=Path,
                    default=runinfo.PROJECT_ROOT / "outputs" / "parity" / "rescore.json")
    ap.add_argument("--wandb-project", default="spell-rq2")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    ap.add_argument("--job-id", default="local", help="SLURM job id or tag for the W&B run name")
    return ap.parse_args(argv)


def _execute(args, run) -> tuple[int, str | None]:
    try:
        rows = read_predictions(args.predictions)
        wandb_run.log(run, "parity_rescore", {"rows_loaded": len(rows)})
        result = rescore(rows)
        wandb_run.log(run, "parity_rescore", {
            "macro_wer": result["macro_wer"], "corpus_wer": result["corpus_wer"],
            "n_scored": result["n_scored"], "total_ref_words": result["total_ref_words"]})
        result["stored_mean_wer"] = None
        result["abs_diff_vs_stored"] = None
        if args.summary is not None:
            result.update(check_against_summary(result, args.summary))
    except InputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2, str(e)
    except CheckFailed as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        return 4, str(e)

    result.update({
        "predictions": str(args.predictions),
        "predictions_sha256": runinfo.sha256_file(args.predictions),
        "summary": None if args.summary is None else str(args.summary),
        "finished_utc": runinfo.utc_now_iso(),
        **runinfo.git_state(),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(format_table(result))
    print(f"[parity-rescore] wrote {args.out}")
    for key in ("macro_wer", "corpus_wer", "n_scored", "total_edits", "total_ref_words"):
        run.summary[f"parity_rescore/{key}"] = result[key]
    return 0, None


def main(argv=None) -> int:
    args = parse_args(argv)
    run = wandb_run.init_run(
        job_type="parity_rescore", name=f"parity-rescore-{args.job_id}",
        project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
        config={"predictions": str(args.predictions), "summary": None if args.summary is None
                else str(args.summary), "tolerance": TOLERANCE},
        tags=["parity", "phase-0"])
    outcome = (1, "interrupted")
    try:
        outcome = _execute(args, run)
        return outcome[0]
    finally:
        wandb_run.finish_run(run, rc=outcome[0], message=outcome[1])


if __name__ == "__main__":
    raise SystemExit(main())
