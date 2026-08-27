#!/usr/bin/env python3
"""Track B training entrypoint (single CLI contract with slurm/template.sbatch).

    python scripts/train_track_b.py \
        --config configs/pilot_100pct.yaml \
        --subset-manifest subsets/splits/train_ids.txt \
        --seed 101 \
        --output-dir runs/track_b/pilot100_job42 \
        [--run-manifest-out outputs/run_manifest_track_b.json]

Gates (PROTOCOL): storage check (strict iff SPELL_STRICT_STORAGE=1), then
``hardware_guard.enforce_gpu_policy()`` — drift-guard abort on Ada nodes;
the loud DEV banner is allowed ONLY under ``SPELL_DEV_GPU=1`` (laptop bring-up,
§3.16).

Seed semantics (§3.11): ``--seed`` is the SUBSET identity (bookkeeping only);
ALL training randomness keys off the FIXED ``training.train_seed`` so every run
shares one training stochastic regime while subsets stay distinct identities.
$SPELL_DATA_ROOT relocation keeps Ada runs byte-identical in logic to laptop
smoke runs — both exercise this exact code path (build amendment 2).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from functools import partial
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning.pytorch as pl                      # noqa: E402
from lightning.pytorch.callbacks import LearningRateMonitor   # noqa: E402
from lightning.pytorch.loggers import WandbLogger   # noqa: E402

from ckpt_bundle import BundleCallback              # noqa: E402
from config import PROJECT_ROOT, load_config, load_paths  # noqa: E402
from dataset import (                               # noqa: E402
    TokenDataset,
    collate_token_batch,
    filter_records,
    load_id_list,
    load_records,
)
from hardware_guard import DEV_GPU_ENV, enforce_gpu_policy  # noqa: E402
import paths as data_paths                          # noqa: E402  DATA LAYOUT LAW
from lit_track_b import LitConformerCTC             # noqa: E402
from manifest import _utc_now_iso, write_run_manifest  # noqa: E402
from vocab import build_char_vocab                  # noqa: E402


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--subset-manifest", required=True)
    ap.add_argument("--seed", type=int, required=True,
                    help="subset identity seed (bookkeeping; NOT a training RNG input)")
    ap.add_argument("--output-dir", required=True, help="bundle dir under runs/<track>/")
    ap.add_argument("--run-manifest-out", type=Path, default=None,
                    help="additional copy of run_manifest.json (slurm template path)")
    # ---- DEV-loop-only overrides; refused outside SPELL_DEV_GPU=1 --------------
    ap.add_argument("--dev-max-train-utts", type=int, default=None)
    ap.add_argument("--dev-max-val-utts", type=int, default=None)
    ap.add_argument("--dev-max-epochs", type=int, default=None)
    return ap.parse_args(argv)


def _guard_dev_flags(args) -> None:
    any_dev = any(getattr(args, f) is not None
                  for f in ("dev_max_train_utts", "dev_max_val_utts", "dev_max_epochs"))
    if any_dev and os.environ.get(DEV_GPU_ENV, "") not in ("1", "true"):
        sys.exit(
            "FATAL: --dev-* overrides require SPELL_DEV_GPU=1 (laptop bring-up only, "
            "PROTOCOL §3.16). Formal runs take their epochs/config from the YAML."
        )


def _run_storage_gate(strict_env: bool) -> None:
    cmd = [sys.executable, str(PROJECT_ROOT / "scripts" / "check_storage.py")]
    if strict_env:
        cmd.append("--strict")
    if subprocess.run(cmd, cwd=PROJECT_ROOT).returncode != 0:
        sys.exit(f"FATAL: storage gate failed ({'strict' if strict_env else 'soft'} mode)")


def attest_completed(run_dir: Path) -> Path:
    """THE single COMPLETED writer (single-writer law, PROTOCOL §10 item 9).

    Called ONLY as this entrypoint's last success-path statement — i.e. after
    ``trainer.fit()`` returned cleanly and ``run_manifest.json`` was rewritten
    with status="finished". Before touching the marker it demands the bundle's
    content-provenance already be DURABLE on disk:

      * metrics.parquet loads AND holds >= 1 epoch row for BOTH splits;
      * ``last.ckpt`` exists;

    then fsyncs the metrics file and the bundle directory so the marker can
    never hit disk ordered ahead of the numbers it attests. Any shortfall
    raises LOUDLY and leaves the bundle unmarked (a marker without provenance
    is exactly how dead bypass bundles once posed as good ones). Shell EXIT
    traps NEVER touch this filename; drain_runs.sh independently re-validates
    everything checked here.
    """
    run_dir = Path(run_dir)
    metrics_path = run_dir / "metrics.parquet"
    missing: list[str] = []
    split_counts: dict[str, int] | None = None
    if not metrics_path.is_file():
        missing.append("metrics.parquet (missing)")
    else:
        import pandas as pd                       # deferred: only needed on this path

        try:
            df = pd.read_parquet(metrics_path)
            split_counts = {s: int((df["split"] == s).sum()) for s in ("train", "val")}
            if min(split_counts.values()) < 1:
                missing.append(f"metrics.parquet needs >=1 epoch row per split "
                               f"(got {split_counts})")
        except Exception as exc:                  # noqa: BLE001 — refuse on ANY probe failure
            missing.append(f"metrics.parquet unreadable: {exc}")
    last_ckpt = run_dir / "last.ckpt"
    if not last_ckpt.is_file():
        missing.append("last.ckpt (missing)")
    if missing:
        raise RuntimeError(
            "[train_track_b] REFUSING to mark COMPLETED — bundle lacks provenance:\n  - "
            + "\n  - ".join(missing)
        )

    # durability BEFORE attestation: fsync the parquet bytes, then the directory
    # entries holding both files, ordering marker strictly after data.
    fd = os.open(metrics_path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(run_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)

    # THE sanctioned writer context — deliberately kept as ONE physical line
    # so tests/test_single_writer_law.py can anchor the repo-wide invariant.
    (run_dir / "COMPLETED").touch()
    return run_dir / "COMPLETED"


def main(argv=None) -> int:
    args = parse_args(argv)
    _guard_dev_flags(args)

    paths = load_paths()
    _run_storage_gate(os.environ.get("SPELL_STRICT_STORAGE", "") == "1")
    gpu_info = enforce_gpu_policy()          # drift guard OR loud dev banner

    cfg = load_config(args.config)

    n_epochs_cfg = cfg.get("training", {}).get("n_epochs")
    n_epochs = int(n_epochs_cfg) if n_epochs_cfg is not None else None
    if args.dev_max_epochs is not None:
        print(f"[DEV] overriding n_epochs -> {args.dev_max_epochs}", flush=True)
        n_epochs = int(args.dev_max_epochs)
    if not n_epochs or n_epochs < 1:
        sys.exit("FATAL: n_epochs unset — pilot yamls pin it; never train with null.")

    train_seed = int(cfg["training"]["train_seed"])     # FIXED constant (§3.11)
    pl.seed_everything(train_seed, workers=True)
    print(f"[train_track_b] subset={args.subset_manifest} subset_seed={args.seed} "
          f"train_seed={train_seed} n_epochs={n_epochs} gpu={gpu_info.get('gpu_name')}")

    vocab = build_char_vocab()

    # -------- records: manifest ∩ frozen train split; val monitoring only ----
    recs = load_records(paths.index_path, split="trainval")
    subset_ids = load_id_list(args.subset_manifest)
    train_recs = filter_records(recs, subset_ids, strict=True)      # raises on foreign ids
    val_ids = load_id_list(paths.splits_dir / "val_ids.txt")
    val_recs = filter_records(recs, val_ids, strict=False)

    if args.dev_max_train_utts:
        print(f"[DEV] capping TRAIN utterances {len(train_recs)} -> {args.dev_max_train_utts}")
        train_recs = train_recs[: args.dev_max_train_utts]
    if args.dev_max_val_utts:
        print(f"[DEV] capping VAL utterances {len(val_recs)} -> {args.dev_max_val_utts}")
        val_recs = val_recs[: args.dev_max_val_utts]

    # -------- PREFLIGHT (§10 item 4): resolve random manifest utts BEFORE epoch 1
    dp = data_paths.current()
    print(f"[spell] data_root={dp.root} layout={dp.layout}", flush=True)
    preflight_pairs = data_paths.preflight_resolve(
        train_recs, k=min(50, len(train_recs)), seed=train_seed)
    print(f"[spell] preflight: sampled {len(preflight_pairs)}/{len(train_recs)} "
          f"manifest utts -> {len(preflight_pairs)}/{len(preflight_pairs)} token files present",
          flush=True)

    return run_training(
        args=args, cfg=cfg, train_recs=train_recs, val_recs=val_recs,
        vocab=vocab, train_seed=train_seed, n_epochs=n_epochs,
    )


def run_training(*, args, cfg, train_recs, val_recs, vocab, train_seed,
                 n_epochs: int) -> int:
    tcfg, lcfg = cfg["training"], cfg.get("logging", {})
    wcfg = lcfg.get("wandb", {})

    bundle = BundleCallback(
        args.output_dir,
        ckpt_every_epochs=int(lcfg.get("ckpt_every_epochs", 5)),
        metrics_filename=lcfg.get("metrics_file", "metrics.parquet"),
    )
    run_dir = bundle.run_dir

    started_iso = _utc_now_iso()
    dev_active = os.environ.get(DEV_GPU_ENV, "") == "1"
    write_run_manifest(
        run_dir / "run_manifest.json",
        cfg=cfg, config_path=args.config,
        subset_manifest=args.subset_manifest,
        train_seed=train_seed, subset_seed=int(args.seed),
        started_iso=started_iso, status="running",
        extra={"data_root": str(data_paths.current().root),
               "data_layout": data_paths.current().layout,
               "dev_gpu_bypass": dev_active},
    )

    mode = str(wcfg.get("mode", "online"))
    # §5.2 inode discipline: bundles hold ONLY contract files (≤20 asserted);
    # W&B's own directory tree must never materialize inside them.
    wandb_dir = PROJECT_ROOT / "outputs" / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    logger = WandbLogger(
        offline=(mode == "offline"),
        project=wcfg.get("project", "spell-rq2"),
        entity=os.environ.get("WANDB_ENTITY") or wcfg.get("entity"),
        name=run_dir.name,
        save_dir=str(wandb_dir),
    )

    batch_size = int(tcfg["batch_size"])
    workers = int(tcfg.get("num_workers", 4))
    collate = partial(collate_token_batch, token_pad_id=-1, text_pad_id=vocab.pad_id)
    train_dl = pl_validation_safe_loader(
        TokenDataset(train_recs, vocab=vocab, include_text=True),
        batch_size=batch_size, shuffle=True, workers=workers, collate=collate)
    val_dl = pl_validation_safe_loader(
        TokenDataset(val_recs, vocab=vocab, include_text=True),
        batch_size=batch_size, shuffle=False, workers=workers, collate=collate)

    lit = LitConformerCTC(cfg, use_augment=True)

    trainer = pl.Trainer(
        max_epochs=n_epochs,
        accelerator="auto",
        devices=1,                                   # single-GPU protocol (Phase 1)
        # §3.11 pins the SEEDS (data order, init, augment); bit-exact CUDA
        # reproducibility is impossible anyway — F.ctc_loss's backward has no
        # deterministic GPU kernel (verified: crashes under strict mode on both
        # laptop 4060 and pool 2080 Ti code path), so 'warn_only' logs such ops
        # instead of aborting after a full seeding-consistent setup.
        deterministic="warn_only",
        gradient_clip_val=float(tcfg.get("grad_clip_norm", 1.0)),
        # §3.17 logging contract: LR monitor streams the realized schedule
        # (warmup+decay, §3.15 ABSOLUTE constants) as a live series — without
        # it lightning never logs learning_rate even with save_hyperparameters.
        callbacks=[bundle, LearningRateMonitor(logging_interval="step")],
        logger=logger,
        log_every_n_steps=50,
        enable_checkpointing=False,                  # BundleCallback owns all files
        benchmark=False,
    )
    try:
        trainer.fit(lit, train_dl, val_dl)
    finally:
        if args.run_manifest_out is not None:
            shutil.copyfile(run_dir / "run_manifest.json", args.run_manifest_out)

    final_wer = trainer.callback_metrics.get("val/wer")
    write_run_manifest(
        run_dir / "run_manifest.json",
        cfg=cfg, config_path=args.config,
        subset_manifest=args.subset_manifest,
        train_seed=train_seed, subset_seed=int(args.seed),
        started_iso=started_iso, finished_iso=_utc_now_iso(),
        status="finished",
        extra={"final_val_wer": float(final_wer) if final_wer is not None else None,
               "data_root": str(data_paths.current().root),
               "data_layout": data_paths.current().layout,
               "dev_gpu_bypass": dev_active},
    )
    # SINGLE-WRITER LAW (PROTOCOL §10 item 9): the ONLY COMPLETED writer in the
    # repo — reached solely after fit returned cleanly + final manifest write;
    # attest_completed itself refuses unless metrics/ckpt provenance is durable.
    marker = attest_completed(run_dir)
    print(f"[train_track_b] done — bundle at {run_dir} ({marker.name}); "
          "drain later via scripts/drain_runs.sh")
    return 0


def pl_validation_safe_loader(ds, *, batch_size, shuffle, workers, collate):
    """DataLoader whose persistent_workers stays False (bundles are short-lived)."""
    from torch.utils.data import DataLoader

    kwargs = dict(batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
                  num_workers=workers, persistent_workers=False)
    if workers:
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


if __name__ == "__main__":
    raise SystemExit(main())
