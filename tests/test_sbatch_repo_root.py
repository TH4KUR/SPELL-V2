"""REPO_ROOT resolution regression (2026-09-09 incident, PROTOCOL §10):

score_proxy job 2691902 died instantly ("mkdir: cannot create directory
'logs': Permission denied") because score_{dnsmos,kmeans,dsir}.sbatch (and
the first score_proxy.sbatch draft, copied from them) derived REPO_ROOT via
``$(cd "$(dirname "$0")/.." && pwd)``. This cluster's slurmd executes batch
scripts from a per-job spool copy, so ``$0`` does not point back into the
submitter's clone — the resolved path is unwritable, and ``set -e`` aborts on
the very first ``mkdir -p logs``. ``template.sbatch`` and ``setup_env.sbatch``
already used the reliable value, ``SLURM_SUBMIT_DIR`` (set by sbatch to the
submission directory) — every scorer sbatch must match.
"""

from __future__ import annotations

from pathlib import Path

from config import PROJECT_ROOT

SBATCH_FILES = sorted((PROJECT_ROOT / "slurm").glob("*.sbatch"))


def test_slurm_dir_is_not_empty():
    assert len(SBATCH_FILES) >= 5


def test_no_sbatch_derives_repo_root_from_dollar_zero():
    offenders = [p.name for p in SBATCH_FILES
                 if 'dirname "$0"' in p.read_text()]
    assert offenders == [], (
        f"{offenders} derive REPO_ROOT from $0, which resolves to slurmd's "
        "spool copy on this cluster, not the submission clone — use "
        "SLURM_SUBMIT_DIR instead (2026-09-09 incident)")


def test_every_sbatch_with_repo_root_uses_slurm_submit_dir():
    for p in SBATCH_FILES:
        text = p.read_text()
        if "REPO_ROOT=" not in text:
            continue
        assert 'REPO_ROOT="${SLURM_SUBMIT_DIR}"' in text, (
            f"{p.name} sets REPO_ROOT without using SLURM_SUBMIT_DIR")
