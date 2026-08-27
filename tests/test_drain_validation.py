"""Drain-gate regressions for scripts/drain_runs.sh (PROTOCOL §10 item 9).

2026-08-27 incident: crashed bypass bundles carried shell-trap-written
COMPLETED markers, and the drainer trusted the marker ALONE — worse, its
auto-discovery depth was silently wrong and had NEVER been exercised by a
test. These subprocess suites drive THE production script end-to-end:

* refusals NAME the failed provenance check (marker alone proves nothing);
* a trainer-shaped valid bundle drains byte-identically and empties the relay;
* mixed batches never let a refused bundle leave;
* depth-3 discovery is EXACT — a valid-at-depth-2 decoy outside the documented
  layout is neither discovered nor touched;
* --verify-only validates loudly and moves NOTHING.
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
import torch

from config import PROJECT_ROOT

DRAIN = PROJECT_ROOT / "scripts" / "drain_runs.sh"

ROWS = [
    {"epoch": 1, "split": "train", "utterance_id": "u1",
     "metric": "loss", "value": 0.5},
    {"epoch": 1, "split": "val", "utterance_id": None,
     "metric": "wer", "value": 0.4},
]


def _mkbundle(runs: Path, rid: str, *, mode: str = "healthy") -> Path:
    """Build a marker-carrying bundle whose CONTENT matches ``mode``. The
    COMPLETED file is always present (trainer-shaped or forged) — proving the
    drainer rejects on content-provenance, never on the marker alone."""
    b = runs / rid
    b.mkdir(parents=True, exist_ok=True)
    (b / "COMPLETED").touch()

    parquet_modes = ("healthy", "bad_ckpt", "no_ckpt",
                     "manifest_missing_key", "no_manifest")
    if mode == "no_parquet":
        pass                                             # no table written
    elif mode == "empty_parquet":
        pd.DataFrame(columns=["epoch", "split", "utterance_id",
                              "metric", "value"]).to_parquet(b / "metrics.parquet")
    elif mode == "train_only":
        pd.DataFrame(ROWS[:1]).to_parquet(b / "metrics.parquet")
    elif mode in parquet_modes:
        pd.DataFrame(ROWS).to_parquet(b / "metrics.parquet")

    if mode == "bad_ckpt":
        (b / "last.ckpt").write_bytes(b"definitely-not-a-torch-archive")
    elif mode in ("healthy", "empty_parquet", "train_only",
                  "manifest_missing_key", "no_manifest"):
        torch.save({"state_dict": {}}, b / "last.ckpt")
    elif mode == "no_ckpt":
        pass                                             # table fine, ckpt absent

    if mode == "no_manifest":
        pass
    else:
        man = {"config_hash": "0" * 64, "git_sha": "a" * 40,
               "gpu_name": "RTX 2080 Ti",
               "subset_manifest": "subsets/splits/train_ids.txt",
               "train_seed": 42}
        if mode == "manifest_missing_key":
            del man["gpu_name"]
        (b / "run_manifest.json").write_text(json.dumps(man))
    return b


def _drain(*args: str, expect_timeout: int = 300,
           env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run the production drainer hermetically: tmp archive + runs-dir,
    PYTHON_BIN pinned to this interpreter (overridable via ``env_extra``)."""
    env = os.environ.copy()
    env.pop("SPELL_ARCHIVE_ROOT", None)                 # defensive vs exported var
    env["PYTHON_BIN"] = sys.executable
    if env_extra:
        env.update(env_extra)
    return subprocess.run(["bash", str(DRAIN), *args],
                          capture_output=True, text=True,
                          env=env, timeout=expect_timeout)


@pytest.fixture
def layout(tmp_path):
    """Pre-created archive + fresh runs relay (absolute paths both)."""
    archive = tmp_path / "archive"
    archive.mkdir()
    runs = tmp_path / "relay_runs"
    runs.mkdir()
    return tmp_path, archive, runs


# ------------------------------------------------------------ refusal matrix --

@pytest.mark.parametrize("mode,fragment", [
    ("no_parquet", "metrics.parquet missing"),
    ("empty_parquet", "epoch row"),
    ("train_only", "epoch row"),
    ("bad_ckpt", "last.ckpt unreadable"),
    ("no_ckpt", "last.ckpt missing"),
    ("manifest_missing_key", "missing key 'gpu_name'"),
    ("no_manifest", "run_manifest.json missing"),
])
def test_forged_marker_with_defect_refused_by_name(layout, mode, fragment):
    _, archive, runs = layout
    rid = f"track_b/{mode}"
    _mkbundle(runs, rid, mode=mode)

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs),
                  "--verify-only")

    assert proc.returncode != 0, proc.stdout
    assert f"REFUSE {rid}" in proc.stderr, proc.stderr
    assert fragment in proc.stderr, proc.stderr         # WHICH check failed
    assert "VALIDATE ok" not in proc.stdout
    assert (runs / rid).is_dir()                        # stays in the relay


def test_marker_alone_is_insufficient_provenance(layout):
    """The original bug shaped exactly this: marker present, nothing behind
    it — must refuse, never drain."""
    _, archive, runs = layout
    b = runs / "track_b/markeronly"
    b.mkdir(parents=True)
    (b / "COMPLETED").touch()

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs),
                  "--verify-only")

    assert proc.returncode != 0
    assert "REFUSE track_b/markeronly" in proc.stderr
    assert "metrics.parquet missing" in proc.stderr
    assert (b / "COMPLETED").is_file()                  # untouched


# ------------------------------------------------------------- drain success --

@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync unavailable")
def test_valid_bundle_drains_byte_identical_and_empties_relay(layout):
    _, archive, runs = layout
    b = _mkbundle(runs, "track_b/good", mode="healthy")
    snapshot = {p.relative_to(b).as_posix(): p.read_bytes()
                for p in sorted(b.rglob("*")) if p.is_file()}
    assert len(snapshot) >= 4                           # marker+parquet+ckpt+manifest

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs))

    assert proc.returncode == 0, proc.stderr
    assert "VALIDATE ok track_b/good" in proc.stdout
    assert "drain: OK track_b/good ->" in proc.stdout
    assert not b.exists()                               # relay emptied
    dst = archive / "track_b" / "good"
    drained = {p.relative_to(dst).as_posix(): p.read_bytes()
               for p in sorted(dst.rglob("*")) if p.is_file()}
    assert drained == snapshot                          # byte-identical relay


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync unavailable")
def test_mixed_batch_drains_valid_and_refuses_defective(layout):
    _, archive, runs = layout
    _mkbundle(runs, "track_b/good", mode="healthy")
    _mkbundle(runs, "track_b/nok", mode="no_ckpt")

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs),
                  "track_b/good", "track_b/nok")

    assert proc.returncode == 1                         # SOME refusal ⇒ non-zero
    assert "VALIDATE ok track_b/good" in proc.stdout
    assert "REFUSE track_b/nok" in proc.stderr
    assert not (runs / "track_b/good").exists()         # valid left the relay
    assert (runs / "track_b/nok").is_dir()              # defective stayed
    assert (archive / "track_b/good" / "last.ckpt").is_file()


# ------------------------------------------------------ discovery-depth laws --

def test_discovery_finds_depth3_exactly_not_depth2_decoy(layout):
    """First-ever exercise of auto-discovery: valid + forged live at documentd
    depth <runs>/<track>/<id>/; an equally-VALID bundle sitting one level too
    shallow must NOT be auto-discovered NOR touched."""
    _, archive, runs = layout
    _mkbundle(runs, "track_b/v1", mode="healthy")
    _mkbundle(runs, "track_b/bad", mode="no_parquet")
    _mkbundle(runs, "decoy", mode="healthy")            # VALID yet depth-2 shape

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs), "-n")

    joined = proc.stdout + proc.stderr
    assert "VALIDATE ok track_b/v1" in proc.stdout
    assert "REFUSE track_b/bad" in proc.stderr          # reason named above it
    assert "decoy" not in joined                        # invisible to discovery
    assert (runs / "decoy" / "COMPLETED").is_file()     # and fully untouched
    assert proc.returncode != 0                         # refusal keeps rc loud


def test_explicit_id_reaches_outside_documented_depth(layout):
    """Documented escape hatch: explicit RUN_ID args bypass the depth filter
    deliberately — an unusual bundle can be validated/drained on purpose."""
    _, archive, runs = layout
    _mkbundle(runs, "decoy", mode="healthy")

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs),
                  "--verify-only", "decoy")

    assert proc.returncode == 0, proc.stderr
    assert "VALIDATE ok decoy" in proc.stdout
    assert (runs / "decoy" / "COMPLETED").is_file()


# ------------------------------------------------------------------- gating --

def test_verify_only_prints_verdicts_but_moves_nothing(layout):
    _, archive, runs = layout
    good = _mkbundle(runs, "track_b/good", mode="healthy")
    _mkbundle(runs, "track_b/nok", mode="bad_ckpt")

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs),
                  "--verify-only")

    assert proc.returncode != 0                         # defect present ⇒ rc≠0
    assert "VALIDATE ok track_b/good" in proc.stdout
    assert "REFUSE track_b/nok" in proc.stderr
    assert "--verify-only — moved nothing" in proc.stdout
    assert good.is_dir()                                # relay untouched
    assert not any(archive.rglob("*"))                  # archive still empty


def test_nonexistent_archive_is_loud_error(layout):
    _, archive, runs = layout
    _mkbundle(runs, "track_b/x", mode="healthy")
    missing = archive.parent / "not-an-archive"

    proc = _drain("--archive", str(missing), "--runs-dir", str(runs))

    assert proc.returncode == 1
    assert "not reachable" in proc.stderr


# ------------------------------------------------- probe-env laws (2026-08-27) --

def test_non_python3_interpreter_refused_at_gate(layout):
    """Ada login shells: bare `python` = CentOS-7 Python 2.7 — it cannot parse
    the probe (f-strings), which is exactly how the 2026-08-27 --verify-only
    attempt died. Non-python3 PYTHON_BIN must fail LOUDLY at startup, before
    any bundle is even considered."""
    _, archive, runs = layout
    _mkbundle(runs, "track_b/x", mode="healthy")

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs),
                  env_extra={"PYTHON_BIN": "/bin/false"})   # exits 1 on -c probe

    assert proc.returncode == 1
    assert "not a python3 interpreter" in proc.stderr
    assert "FROZEN" in proc.stderr                           # remedy is named
    assert (runs / "track_b/x").is_dir()                     # nothing considered


def test_crashed_probe_never_validates(layout):
    """THE hole this incident exposed: a probe that dies with empty stdout
    (python2 SyntaxError, missing interpreter, crash) used to fall through to
    VALIDATE ok — silent provenance bypass. Empty/no verdict ⇒ refusal."""
    _, archive, runs = layout
    _mkbundle(runs, "track_b/x", mode="healthy")

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs),
                  env_extra={"PYTHON_BIN": "/bin/true"})    # passes gate (rc 0),
                                                            # yields NO verdict
    assert proc.returncode != 0
    assert "REFUSE track_b/x" in proc.stderr
    assert "no verdict" in proc.stderr
    assert "VALIDATE ok" not in proc.stdout
    assert (runs / "track_b/x").is_dir()
