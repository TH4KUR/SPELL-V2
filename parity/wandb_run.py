"""W&B surface shared by the parity scripts (PROTOCOL §3.23-§3.25).

* §3.23: research-crucial non-training runs log to W&B by default, under their own
  ``job_type`` and ``<tool>/``-prefixed series; ``train/*`` and ``val/*`` belong to
  the formal-run contract (§3.17) and are refused here, as is any run name in the
  formal id namespace ``<track>/<stem>_seed<seed>_job<id>_t<task>`` (§3.17(d)).
* §3.24: the run is opened BEFORE the expensive work and progress is logged as it
  happens (the callers own the cadence).
* §3.25(b): :func:`finish_run` is meant to be called from a ``finally:`` block; it
  reads the in-flight exception so a crashed run is marked failed, with the error
  and traceback on ``run.summary``.
"""

from __future__ import annotations

import re
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WANDB_DIR = PROJECT_ROOT / "outputs" / "wandb"

_FORMAL_RUN_NAME = re.compile(r"^track_[a-z]+/")
_FORMAL_SERIES_ROOTS = ("train", "val")


def check_run_name(name: str) -> None:
    if _FORMAL_RUN_NAME.match(name):
        raise ValueError(
            f"run name {name!r} is in the formal run-id namespace "
            "(<track>/<stem>_seed.._job.._t.., PROTOCOL §3.17(d)); parity runs use a flat name")


def init_run(*, job_type: str, name: str, project: str, entity: str | None, mode: str,
             config: dict, tags=()):
    check_run_name(name)
    import wandb

    WANDB_DIR.mkdir(parents=True, exist_ok=True)
    return wandb.init(project=project, entity=entity, mode=mode, job_type=job_type,
                      name=name, config=config, tags=list(tags), dir=str(WANDB_DIR))


def log(run, prefix: str, payload: dict) -> None:
    if prefix.split("/")[0] in _FORMAL_SERIES_ROOTS:
        raise ValueError(
            f"series prefix {prefix!r} belongs to the formal training contract "
            "(PROTOCOL §3.17); non-training runs use their own <tool>/ prefix (§3.23)")
    run.log({f"{prefix}/{k}": v for k, v in payload.items()})


def finish_run(run, *, rc: int = 0, message: str | None = None) -> None:
    """Call from ``finally:``. An exception in flight, or a non-zero ``rc`` the body
    is about to return, marks the run failed (exit_code=1) and records why on
    ``run.summary`` (the exception's error + traceback, or ``message``)."""
    if run is None:
        return
    exc_type, exc_value, exc_tb = sys.exc_info()
    if exc_type is not None:
        run.summary["error"] = f"{exc_type.__name__}: {exc_value}"
        run.summary["traceback"] = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    elif rc != 0:
        run.summary["error"] = message or f"exited with code {rc}"
    run.finish(exit_code=1 if (exc_type is not None or rc != 0) else 0)
