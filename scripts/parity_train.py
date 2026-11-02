#!/usr/bin/env python3
"""From-scratch RAVEn-parity training run (PROTOCOL §3.34).

    python scripts/parity_train.py --recipe R-B --job-id $SLURM_JOB_ID          # via slurm/parity_train.sbatch
    python scripts/parity_train.py --recipe R-B --job-id 123 --resume            # continue outputs/parity/train_R-B_123
    python scripts/parity_train.py --recipe R-B --stop-after-epochs 1 ...        # 1-epoch timing run

Trains the vendored RAVEn Base audio-only model (raw 16 kHz waveform -> joint CTC/attention)
from scratch on the parity TRAIN split under one of the two pre-registered recipes in
``configs/parity/recipes.yaml`` (R-A, R-B; the file is pinned by sha256). Validation is the
parity VAL split; the official test split is never read here (it is scored once, by
``scripts/parity_eval.py``, after the Phase-1 gate and for the recipe chosen on val WER).

Needs the tokenizer artifacts of ``scripts/parity_train_spm.py`` (``--tokens-dir``) and the
parity splits (``parity/splits``, built by ``scripts/parity_build_splits.py``).

Outputs (``outputs/parity/train_<recipe>_<job-id>/``): ``last.pt`` (full state, atomic),
``weights_epochNNNN.pt`` (last ``--keep-last-weights`` epochs), ``final.pt`` (selected weights, a
plain state_dict that ``scripts/parity_eval.py`` loads), ``metrics.jsonl``, ``summary.json``,
``val_beam_predictions.jsonl``. Logs to W&B LIVE (job_type ``parity_train``, series
``parity_train/*``) per PROTOCOL §3.23-§3.25.

Exit codes: 0 ok, 1 failed/interrupted, 2 setup refused.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import soundfile as sf                                  # noqa: E402

import paths as data_paths                              # noqa: E402  DATA LAYOUT LAW
from config import PROTOCOL_REVISION, load_paths        # noqa: E402
from dataset import filter_records, load_id_list, load_records  # noqa: E402
from hardware_guard import enforce_gpu_policy           # noqa: E402
from parity import model as parity_model                # noqa: E402
from parity import optim, recipes, runinfo, splits, trainer, wandb_run  # noqa: E402
from parity import spm_tokens                           # noqa: E402
from parity import train_data                           # noqa: E402
from parity.data import SAMPLE_RATE                     # noqa: E402

DEFAULT_TOKENS_DIR = Path(__file__).resolve().parents[1] / "parity" / "spm"
DURATION_CHECK_N = 50
DURATION_WARN_S = 0.1


class SetupError(Exception):
    """Inputs are not acceptable (exit 2)."""


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True, choices=list(recipes.RECIPE_NAMES))
    ap.add_argument("--tokens-dir", type=Path, default=DEFAULT_TOKENS_DIR,
                    help="artifacts of scripts/parity_train_spm.py (default: parity/spm)")
    ap.add_argument("--train-ids", type=Path, default=None,
                    help="default: parity/splits/parity_train_ids.txt, verified against provenance.json")
    ap.add_argument("--val-ids", type=Path, default=None,
                    help="default: parity/splits/parity_val_ids.txt, verified against provenance.json")
    ap.add_argument("--index", type=Path, default=None, help="default: data_index.parquet")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default: outputs/parity/train_<recipe>_<job-id>")
    ap.add_argument("--job-id", default="local", help="run id for the output directory and the W&B run name")
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                    help="cuda goes through the hardware drift guard; cpu is for tests/smoke")
    ap.add_argument("--num-workers", type=int, default=4, help="audio-loading workers")
    ap.add_argument("--resume", action="store_true", help="continue from <out-dir>/last.pt")
    ap.add_argument("--stop-after-epochs", type=int, default=None,
                    help="stop after N epochs of the recipe's schedule (timing run; no final evaluation)")
    ap.add_argument("--beam-val-every", type=int, default=0,
                    help="also run the (slow) beam-search val WER every N epochs; 0 = only at the end")
    ap.add_argument("--beam-val-limit", type=int, default=None,
                    help="beam-decode only the first N val utterances (flagged as a subset)")
    ap.add_argument("--keep-last-weights", type=int, default=10,
                    help="weights-only checkpoints kept (needed >= average_last_k for that selection rule)")
    ap.add_argument("--micro-batch-seconds", type=float, default=None,
                    help="override the recipe's micro-batch size (memory only; recorded as a deviation)")
    ap.add_argument("--no-deterministic-kernels", action="store_true",
                    help="keep fp32 but let cuDNN pick its fastest algorithms (PROTOCOL §10 item 1 asks for "
                         "deterministic warn_only; the default here). Recorded in summary.json")
    ap.add_argument("--arch-config", type=Path, default=parity_model.ARCH_CONFIG_PATH)
    ap.add_argument("--wandb-project", default="spell-rq2")
    ap.add_argument("--wandb-entity", default=None)
    ap.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="online")
    ap.add_argument("--report-every-s", type=float, default=30.0,
                    help="minimum wall-clock seconds between progress reports")
    args = ap.parse_args(argv)
    if args.num_workers < 0 or args.keep_last_weights < 0:
        ap.error("--num-workers and --keep-last-weights must be >= 0")
    if args.beam_val_every < 0 or (args.beam_val_limit is not None and args.beam_val_limit < 1):
        ap.error("--beam-val-every must be >= 0 and --beam-val-limit >= 1")
    return args


def _resolve_ids(args, index_path: Path) -> tuple[Path, Path, bool]:
    if args.train_ids is not None and args.val_ids is not None:
        for p in (args.train_ids, args.val_ids):
            if not p.is_file():
                raise SetupError(f"id list not found: {p}")
        return args.train_ids, args.val_ids, False
    if args.train_ids is not None or args.val_ids is not None:
        raise SetupError("pass both --train-ids and --val-ids, or neither")
    try:
        splits.verify_split_files(splits.SPLITS_DIR, index_path=index_path)
    except splits.SplitError as e:
        raise SetupError(f"parity splits unusable: {e}") from e
    return (splits.SPLITS_DIR / splits.SPLIT_FILES["train"], splits.SPLITS_DIR / splits.SPLIT_FILES["val"], True)


def _records(ids_path: Path, index_path: Path, what: str):
    ids = load_id_list(ids_path)
    try:
        recs = filter_records(load_records(index_path, split="trainval"), ids, strict=True)
    except ValueError as e:
        raise SetupError(f"{what} ids are not all in the index's trainval split (the test split is "
                         f"never used for training): {e}") from e
    if not recs:
        raise SetupError(f"{ids_path} selects no utterances")
    return recs


def _duration_check(records, resolver) -> dict:
    step = max(1, len(records) // DURATION_CHECK_N)
    diffs = []
    for rec in records[::step][:DURATION_CHECK_N]:
        info = sf.info(str(resolver(rec)))
        diffs.append(abs(info.frames / info.samplerate - rec.duration_s))
    return {"n_checked": len(diffs), "max_abs_diff_s": max(diffs)}


def _setup(args, recipe: recipes.Recipe, run, gpu: dict):
    index_path = args.index if args.index is not None else Path(load_paths().index_path)
    train_ids, val_ids, ids_verified = _resolve_ids(args, index_path)
    try:
        art = spm_tokens.load_artifacts(args.tokens_dir)
    except (FileNotFoundError, ValueError) as e:
        raise SetupError(f"tokenizer artifacts unusable: {e}") from e
    train_recs = _records(train_ids, index_path, "train")
    val_recs = _records(val_ids, index_path, "val")

    def dataset(recs, augment):
        try:
            return train_data.ParityTrainDataset(
                recs, art["token_ids"], seed=recipe.seed, time_mask_window_s=recipe.time_mask_window_s,
                time_mask_stride_s=recipe.time_mask_stride_s, augment=augment)
        except ValueError as e:
            raise SetupError(str(e)) from e

    train_ds, val_ds = dataset(train_recs, True), dataset(val_recs, False)
    dp = data_paths.current()
    print(f"[spell] data_root={dp.root} layout={dp.layout}", flush=True)
    data_paths.preflight_resolve(train_recs, k=min(50, len(train_recs)), kind="audio")
    data_paths.preflight_resolve(val_recs, k=min(50, len(val_recs)), kind="audio")
    duration = _duration_check(train_recs, data_paths.resolve_audio_path)
    n_infeasible = sum(1 for f, rec in zip(train_ds.frames, train_recs)
                       if f < len(art["token_ids"][rec.utterance_id]))
    print(f"[parity-train] index-vs-file duration check: {duration}; "
          f"CTC-infeasible train utterances (frames < tokens): {n_infeasible}", flush=True)
    if duration["max_abs_diff_s"] > DURATION_WARN_S:
        print(f"WARNING: index durations differ from the staged files by up to "
              f"{duration['max_abs_diff_s']:.3f} s; micro-batch budgets are approximate", file=sys.stderr, flush=True)

    base_arch = parity_model.load_arch_config(args.arch_config)
    arch = recipe.arch_config(base_arch)
    trainer.seed_everything(recipe.seed)                              # the initial weights are a function of the seed
    model = parity_model.build_model(arch, odim=len(art["tokens"]))
    model.to(args.device)

    provenance = {
        "recipes_sha256": recipe.sha256,
        "arch_config_sha256": runinfo.sha256_file(args.arch_config),
        "train_ids_sha256": runinfo.sha256_file(train_ids),
        "val_ids_sha256": runinfo.sha256_file(val_ids),
        "data_index_sha256": runinfo.sha256_file(index_path),
        "tokenizer_json_sha256": runinfo.sha256_file(args.tokens_dir / spm_tokens.PROVENANCE_FILENAME),
        "units_sha256": art["provenance"]["files"][spm_tokens.UNITS_FILENAME]["sha256"],
        "token_ids_sha256": art["provenance"]["files"][spm_tokens.TOKEN_IDS_FILENAME]["sha256"],
        "n_units": len(art["units"]),
        "n_train_utts": len(train_recs), "n_val_utts": len(val_recs),
    }
    out_dir = args.out_dir or (runinfo.PROJECT_ROOT / "outputs" / "parity"
                               / f"train_{recipe.name}_{args.job_id}")
    environment = {"data_root": str(dp.root), "ids_verified_against_provenance": ids_verified,
                   "duration_check": duration, "n_ctc_infeasible_train_utts": n_infeasible,
                   "protocol_revision": PROTOCOL_REVISION, "arch_config": str(args.arch_config),
                   "tokens_dir": str(args.tokens_dir), "gpu": gpu,
                   "deterministic_kernels": not args.no_deterministic_kernels}
    options = trainer.TrainerOptions(
        out_dir=out_dir, device=args.device, num_workers=args.num_workers, resume=args.resume,
        stop_after_epochs=args.stop_after_epochs, beam_val_every=args.beam_val_every,
        beam_val_limit=args.beam_val_limit, keep_last_weights=args.keep_last_weights,
        report_every_s=args.report_every_s, micro_batch_seconds=args.micro_batch_seconds,
        provenance=provenance, environment=environment)
    try:
        tr = trainer.Trainer(recipe=recipe, model=model, train_dataset=train_ds, val_dataset=val_ds,
                             tokens=art["tokens"], options=options, run=run)
    except trainer.TrainerError as e:
        raise SetupError(str(e)) from e
    return tr


def _execute(args, recipe, run) -> tuple[int, str | None]:
    try:
        gpu = enforce_gpu_policy() if args.device == "cuda" else {"gpu_name": None}
        tr = _setup(args, recipe, run, gpu)
        for g in optim.group_table(tr.optimizer):
            print(f"[parity-train] group {g['name']:<10} peak_lr {g['peak_lr']:.3e} params {g['n_params']:,}", flush=True)
        print(f"[parity-train] recipe {recipe.name} (sha256 {recipe.sha256[:12]}) epochs {recipe.epochs} "
              f"steps/epoch {tr.steps_per_epoch} micro {tr.micro_s:g}s effective {recipe.effective_batch_seconds:g}s "
              f"device {args.device} -> {tr.out_dir}", flush=True)
        run.summary["parity_train/steps_per_epoch"] = tr.steps_per_epoch
        run.summary["parity_train/n_parameters"] = sum(p.numel() for p in tr.model.parameters())
    except SetupError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2, str(e)

    try:
        with trainer.fp32_deterministic_mode(not args.no_deterministic_kernels):
            summary = tr.fit()
    except trainer.TrainerError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2, str(e)
    last = summary["history"][-1]
    run.summary["parity_train/epochs_done"] = summary["epochs_done"]
    run.summary["parity_train/complete"] = summary["complete"]
    run.summary["parity_train/last_val_ctc_greedy_wer"] = last["val_ctc_greedy_wer"]
    if summary["complete"]:
        run.summary["parity_train/final_val_beam_wer"] = summary["final"]["val_beam"]["corpus_wer"]
        print(f"[parity-train] COMPLETE: val beam corpus WER {100 * summary['final']['val_beam']['corpus_wer']:.2f}% "
              f"(CTC-greedy {100 * summary['final']['val']['ctc_greedy_wer']:.2f}%) -> {tr.out_dir}", flush=True)
    else:
        print(f"[parity-train] stopped after {summary['epochs_done']}/{recipe.epochs} epochs "
              f"(no final evaluation) -> {tr.out_dir}", flush=True)
    return 0, None


def main(argv=None) -> int:
    t_main = time.perf_counter()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # before any CUDA call (deterministic cuBLAS)
    args = parse_args(argv)
    try:
        recipe = recipes.load_recipe(args.recipe)
    except recipes.RecipeError as e:
        print(f"FATAL: {e}", file=sys.stderr)
        return 2

    run = wandb_run.init_run(
        job_type="parity_train", name=f"parity-train-{recipe.name}-{args.job_id}",
        project=args.wandb_project, entity=args.wandb_entity, mode=args.wandb_mode,
        config={**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
                "recipe_config": recipe.to_dict(), **runinfo.git_state()},
        tags=["parity", "phase-2", recipe.name])
    outcome: tuple[int, str | None] = (1, "interrupted")
    try:
        outcome = _execute(args, recipe, run)
        run.summary["parity_train/wall_time_s"] = time.perf_counter() - t_main
        return outcome[0]
    finally:
        wandb_run.finish_run(run, rc=outcome[0], message=outcome[1])


if __name__ == "__main__":
    raise SystemExit(main())
