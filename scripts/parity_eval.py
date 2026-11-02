#!/usr/bin/env python3
"""Phase-1 evaluation-harness gate (PROTOCOL §3.33): the released RAVEn checkpoint,
decoded exactly as upstream decodes, over a parity split.

    # no-quarantine sanity number (the released checkpoint never trained on these clips)
    python scripts/parity_eval.py --checkpoint $HOME/spell/ckpts/raven_base_lrs3_asr.pth --split val
    # official test; allowed only with the unlock, after the val run looked sane
    SPELL_FINAL_EVAL=1 python scripts/parity_eval.py --checkpoint ... --split test

Decoding follows ``finetune_learner.calculate_wer`` (see ``parity/decode.py``): raw
16 kHz waveform in [-1, 1], model in eval(), joint CTC/attention ``BatchBeamSearch``
(beam 40, decoder 0.9 / CTC 0.1, no LM, no length bonus, maxlenratio = minlenratio =
0), 1-best. Headline metric = CORPUS WER (sum of word edits / sum of reference words);
the macro mean of per-utterance WERs is reported next to it, never instead of it.

Writes ``predictions.jsonl`` (one row per utterance, flushed as it is decoded) and, on
completion only, ``summary.json``. Logs to W&B live per PROTOCOL §3.23-§3.25 (job_type
``parity_eval``, series ``parity_eval/*``).

A checkpoint written by ``scripts/parity_train.py`` (``final.pt``) is evaluated with the SAME
decoding recipe by naming its tokenizer and switching the released-file pin off:

    python scripts/parity_eval.py --checkpoint outputs/parity/train_R-B_<job>/final.pt --split val \
        --units parity/spm/unigram1000_units.txt --expect-sha256 none

(``--units`` sets the token list and so the model's output size; the released-architecture
fingerprint, which only describes the 1,049-output released file, is then not asserted.)

Exit codes: 0 ok, 2 setup refused (checkpoint/ids/sha256/strict load), 3 test quarantine.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from torch.utils.data import DataLoader                 # noqa: E402

import paths as data_paths                              # noqa: E402  DATA LAYOUT LAW
from config import PROTOCOL_REVISION, load_paths        # noqa: E402
from dataset import filter_records, load_id_list, load_records  # noqa: E402
from hardware_guard import enforce_gpu_policy           # noqa: E402
from parity import data as parity_data                  # noqa: E402
from parity import decode, metrics, runinfo, splits, wandb_run  # noqa: E402
from parity import model as parity_model                # noqa: E402
from parity import text as parity_text                  # noqa: E402
from parity.progress import ProgressReporter            # noqa: E402

SCHEMA = "spell-rq2-parity-eval-v1"
MAXLENRATIO = 0.0
MINLENRATIO = 0.0
LM_WEIGHT = 0.0
LENGTH_BONUS = 0.0


class SetupError(Exception):
    """Inputs or checkpoint are not acceptable (exit 2)."""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--ids", type=Path, default=None,
                    help="id list (default: parity/splits/parity_<split>_ids.txt, verified "
                         "against provenance.json)")
    ap.add_argument("--index", type=Path, default=None, help="default: data_index.parquet")
    ap.add_argument("--limit", type=int, default=None,
                    help="decode only the first N utterances (smoke; flagged in summary.json)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: outputs/parity/eval_<split>_<job-id>")
    ap.add_argument("--overwrite", action="store_true",
                    help="allow replacing a finished result in --out-dir")
    ap.add_argument("--beam-size", type=int, default=40)
    ap.add_argument("--ctc-weight", type=float, default=0.1)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                    help="cuda goes through the hardware drift guard; cpu is for tests/smoke")
    ap.add_argument("--num-workers", type=int, default=2, help="audio-loading workers")
    ap.add_argument("--arch-config", type=Path, default=parity_model.ARCH_CONFIG_PATH)
    ap.add_argument("--expect-sha256", default=parity_model.RELEASED_CKPT_SHA256,
                    help="checkpoint sha256 to require (default: the released RAVEn Base ASR file); "
                         "'none' accepts any file (a trainer checkpoint) and records sha256_verified=false")
    ap.add_argument("--units", type=Path, default=None,
                    help="units file (`piece id` lines) of the checkpoint's tokenizer; default: the vendored "
                         "RAVEn unigram-1000 list that the released checkpoint uses (pinned by sha256)")
    ap.add_argument("--skip-sha-check", action="store_true",
                    help="do NOT verify the checkpoint sha256 (recorded in summary.json)")
    ap.add_argument("--wandb-project", default="spell-rq2")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    ap.add_argument("--job-id", default="local", help="SLURM job id or tag for the W&B run name")
    ap.add_argument("--report-every-s", type=float, default=30.0,
                    help="minimum wall-clock seconds between progress reports")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be >= 1")
    if args.beam_size < 1:
        ap.error("--beam-size must be >= 1")
    if not 0.0 <= args.ctc_weight <= 1.0:
        ap.error("--ctc-weight must be in [0, 1]")
    return args


def _resolve_ids(args, index_path: Path) -> tuple[Path, bool]:
    if args.ids is not None:
        if not args.ids.is_file():
            raise SetupError(f"id list not found: {args.ids}")
        return args.ids, False
    try:
        splits.verify_split_files(splits.SPLITS_DIR, index_path=index_path)
    except splits.SplitError as e:
        raise SetupError(f"parity splits unusable: {e}") from e
    return splits.SPLITS_DIR / splits.SPLIT_FILES[args.split], True


def _load_model(args, ckpt: Path, device: str, odim: int, custom_units: bool):
    ckpt_sha = runinfo.sha256_file(ckpt)
    expect = None if args.expect_sha256.lower() == "none" else args.expect_sha256
    verified = expect is not None and not args.skip_sha_check
    if verified and ckpt_sha != expect:
        raise SetupError(
            f"checkpoint sha256 {ckpt_sha} != expected {expect} "
            "(wrong or corrupted download; --skip-sha-check overrides, recorded in summary.json; "
            "a trainer checkpoint needs --expect-sha256 none)")
    if not verified:
        print("WARNING: checkpoint sha256 NOT verified", file=sys.stderr, flush=True)
    try:
        arch = parity_model.load_arch_config(args.arch_config)
        model = parity_model.load_released_checkpoint(ckpt, strict=True, arch_cfg=arch, odim=odim)
        if not custom_units:
            parity_model.assert_released_architecture(model)
    except (RuntimeError, ValueError) as e:
        raise SetupError(f"checkpoint does not load strictly into the architecture (odim {odim}): {e}") from e
    model.to(device)
    model.eval()
    info = {
        "path": str(ckpt),
        "bytes": ckpt.stat().st_size,
        "sha256": ckpt_sha,
        "sha256_verified": verified,
        "pinned_sha256": parity_model.RELEASED_CKPT_SHA256,
        "n_parameters": parity_model.count_parameters(model),
        "state_dict_fingerprint": parity_model.state_dict_fingerprint(model.state_dict()),
        "released_architecture_asserted": not custom_units,
    }
    return model, info


def _decode_split(args, model, beam_search, tokens, records, pred_path: Path, run) -> dict:
    dataset = parity_data.ParityEvalDataset(records)
    loader = DataLoader(dataset, batch_size=None, shuffle=False, num_workers=args.num_workers)
    per_utt: list[metrics.UttScore] = []
    state = {"edits": 0, "words": 0}

    def on_report(payload: dict) -> None:
        wandb_run.log(run, "parity_eval", payload)
        eta = payload["eta_s"]
        eta_txt = f"{eta / 60:.1f} min" if eta == eta else "unknown"
        print(f"[parity-eval] {payload['utts_done']}/{payload['utts_total']} "
              f"({100 * payload['progress']:.1f}%) {payload['elapsed_s'] / 60:.1f} min elapsed, "
              f"last utt {payload['last_utt_s']:.2f}s (window max {payload['window_max_utt_s']:.2f}s), "
              f"{payload['utts_per_s']:.2f} utt/s, ETA {eta_txt}, "
              f"running corpus WER {payload['running_corpus_wer']:.4f}", flush=True)

    reporter = ProgressReporter(len(records), report_every_s=args.report_every_s, on_report=on_report)
    decode_time = 0.0
    with open(pred_path, "w", encoding="utf-8") as pf:
        for uid, wave, ref in loader:
            t0 = time.perf_counter()
            hyp = decode.transcribe(model, beam_search, wave, tokens)
            dt = time.perf_counter() - t0
            decode_time += dt
            edits = metrics.pair_edits(ref, hyp)
            ref_words = len(ref.split())
            per_utt.append(metrics.UttScore(edits=edits, ref_words=ref_words))
            state["edits"] += edits
            state["words"] += ref_words
            pf.write(json.dumps({
                "utterance_id": uid, "ref": ref, "hyp": hyp, "edits": edits,
                "ref_words": ref_words, "audio_s": wave.shape[0] / parity_data.SAMPLE_RATE,
                "decode_s": dt}, ensure_ascii=False) + "\n")
            pf.flush()
            reporter.update(dt, running_corpus_wer=state["edits"] / state["words"])
    reporter.finish()
    result = metrics.aggregate(per_utt)
    return {"result": result, "decode_time_s": decode_time}


def _execute(args, run, t_main: float) -> tuple[int, str | None]:
    try:
        ckpt = args.checkpoint.resolve()
        if not ckpt.is_file():
            raise SetupError(f"checkpoint not found: {ckpt}")
        job = args.job_id
        out_dir = args.out_dir or (runinfo.PROJECT_ROOT / "outputs" / "parity" / f"eval_{args.split}_{job}")
        summary_path = out_dir / "summary.json"
        if summary_path.exists() and not args.overwrite:
            raise SetupError(f"{summary_path} already holds a finished result; "
                             "use a new --out-dir or pass --overwrite")

        index_path = args.index if args.index is not None else Path(load_paths().index_path)
        ids_path, ids_verified = _resolve_ids(args, index_path)

        gpu = enforce_gpu_policy() if args.device == "cuda" else {"gpu_name": None}
        if args.units is not None and not args.units.is_file():
            raise SetupError(f"units file not found: {args.units}")
        units_path = args.units if args.units is not None else parity_text.VENDORED_UNITS_PATH
        try:
            units = parity_text.load_units(args.units)
        except ValueError as e:
            raise SetupError(f"units file unusable: {e}") from e
        tokens = parity_text.token_list(units)
        model, ckpt_info = _load_model(args, ckpt, args.device, odim=len(tokens),
                                       custom_units=args.units is not None)
        beam_search = decode.build_beam_search(
            model, tokens, beam_size=args.beam_size, ctc_weight=args.ctc_weight)

        index_split = "trainval" if args.split == "val" else "test"
        ids = load_id_list(ids_path)
        try:
            records = filter_records(load_records(index_path, split=index_split), ids, strict=True)
        except ValueError as e:
            raise SetupError(f"ids do not belong to split {args.split!r} of {index_path}: {e}") from e
        if not records:
            raise SetupError(f"{ids_path} selects no utterances")
        n_full = len(records)
        if args.limit is not None:
            records = records[: args.limit]
        dp = data_paths.current()
        print(f"[spell] data_root={dp.root} layout={dp.layout}", flush=True)
        data_paths.preflight_resolve(records, k=min(50, len(records)), kind="audio")
        subtype = parity_data.audio_subtype(data_paths.resolve_audio_path(records[0]))
        print(f"[parity-eval] split={args.split} n={len(records)}/{n_full} beam={args.beam_size} "
              f"ctc_weight={args.ctc_weight} device={args.device} audio_subtype={subtype}", flush=True)
    except SetupError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2, str(e)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path.unlink(missing_ok=True)              # an interrupted re-run must never look complete
    pred_path = out_dir / "predictions.jsonl"
    decoded = _decode_split(args, model, beam_search, tokens, records, pred_path, run)
    res: metrics.CorpusWer = decoded["result"]

    summary = {
        "schema": SCHEMA,
        "complete": True,
        "protocol_revision": PROTOCOL_REVISION,
        "split": args.split,
        "final_eval_unlocked": args.split == "test",
        "n_utts": res.n_utts,
        "limit": args.limit,
        "is_subset_smoke": len(records) < n_full,
        "corpus_wer": res.corpus_wer,
        "macro_wer": res.macro_wer,
        "total_edits": res.total_edits,
        "total_ref_words": res.total_ref_words,
        "config": {
            "beam_size": args.beam_size, "ctc_weight": args.ctc_weight, "lm_weight": LM_WEIGHT,
            "length_bonus": LENGTH_BONUS, "maxlenratio": MAXLENRATIO, "minlenratio": MINLENRATIO,
            "device": args.device,
        },
        "checkpoint": ckpt_info,
        "arch_config": {"path": str(args.arch_config), "sha256": runinfo.sha256_file(args.arch_config)},
        "units_sha256": runinfo.sha256_file(units_path),
        "units": {"path": str(units_path), "n_units": len(units), "custom": args.units is not None},
        "ids": {"path": str(ids_path), "sha256": runinfo.sha256_file(ids_path), "n": len(ids),
                "verified_against_provenance": ids_verified},
        "data_root": str(dp.root),
        "audio_subtype_sample": subtype,
        "gpu": gpu,
        "wall_time_s": time.perf_counter() - t_main,
        "decode_time_s": decoded["decode_time_s"],
        "mean_utt_s": decoded["decode_time_s"] / res.n_utts,
        "predictions_sha256": runinfo.sha256_file(pred_path),
        "finished_utc": runinfo.utc_now_iso(),
        **runinfo.git_state(),
    }
    tmp = summary_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, summary_path)
    for key in ("corpus_wer", "macro_wer", "n_utts", "total_edits", "total_ref_words",
                "wall_time_s", "decode_time_s", "mean_utt_s"):
        run.summary[f"parity_eval/{key}"] = summary[key]
    print(f"[parity-eval] split={args.split} n={res.n_utts} CORPUS WER={100 * res.corpus_wer:.2f}% "
          f"(macro {100 * res.macro_wer:.2f}%) edits={res.total_edits} ref_words={res.total_ref_words} "
          f"-> {out_dir}", flush=True)
    return 0, None


def main(argv=None) -> int:
    t_main = time.perf_counter()
    args = parse_args(argv)
    if args.split == "test" and os.environ.get("SPELL_FINAL_EVAL") != "1":
        print(
            "REFUSED: the official LRS3 test split is quarantined until the parity Phase-1 gate "
            "has been run on val (PROTOCOL §3.5, §3.33). Re-run with SPELL_FINAL_EVAL=1.",
            file=sys.stderr)
        return 3

    run = wandb_run.init_run(
        job_type="parity_eval", name=f"parity-eval-{args.split}-{args.job_id}",
        project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
        config={**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                **runinfo.git_state()},
        tags=["parity", "phase-1", args.split])
    outcome: tuple[int, str | None] = (1, "interrupted")
    try:
        outcome = _execute(args, run, t_main)
        return outcome[0]
    finally:
        wandb_run.finish_run(run, rc=outcome[0], message=outcome[1])


if __name__ == "__main__":
    raise SystemExit(main())
