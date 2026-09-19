"""Drain-gate regressions for scripts/drain_runs.sh (PROTOCOL §10 item 9).

2026-08-27 incident: crashed bypass bundles carried shell-trap-written
COMPLETED markers, and the drainer trusted the marker ALONE — worse, its
auto-discovery depth was silently wrong and had NEVER been exercised by a
test. These subprocess suites drive THE production script end-to-end:

* refusals NAME the failed provenance check (marker alone proves nothing);
* a trainer-shaped valid bundle drains byte-identically and, by default
  (2026-09-19: copy, not move — other scripts sometimes need the relay copy
  after draining), KEEPS the relay bundle; --prune-relay opts into deleting it;
* mixed batches never let a refused bundle leave;
* depth-3 discovery is EXACT — a valid-at-depth-2 decoy outside the documented
  layout is neither discovered nor touched;
* --verify-only validates loudly and copies/moves NOTHING.
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
    PYTHON_BIN pinned to this interpreter (overridable via ``env_extra``).
    SPELL_PROBE=local is pinned so suites never depend on a slurm client;
    srun-transport suites below opt back in explicitly."""
    env = os.environ.copy()
    env.pop("SPELL_ARCHIVE_ROOT", None)                 # defensive vs exported var
    env["PYTHON_BIN"] = sys.executable
    env["SPELL_PROBE"] = "local"
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
def test_valid_bundle_drains_byte_identical_and_keeps_relay_by_default(layout):
    """COPY, not move, is the default (2026-09-19): other scripts sometimes
    need the relay copy to still be there after draining (e.g.
    summarize_track_b.py, run from a compute node that cannot see /share1 at
    all). The relay bundle must survive a successful drain unless
    --prune-relay is explicitly passed (see the test below)."""
    _, archive, runs = layout
    b = _mkbundle(runs, "track_b/good", mode="healthy")
    snapshot = {p.relative_to(b).as_posix(): p.read_bytes()
                for p in sorted(b.rglob("*")) if p.is_file()}
    assert len(snapshot) >= 4                           # marker+parquet+ckpt+manifest

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs))

    assert proc.returncode == 0, proc.stderr
    assert "VALIDATE ok track_b/good" in proc.stdout
    assert "drain: OK track_b/good ->" in proc.stdout
    assert "relay copy KEPT" in proc.stdout
    assert b.exists()                                   # relay copy KEPT (default)
    relay = {p.relative_to(b).as_posix(): p.read_bytes()
             for p in sorted(b.rglob("*")) if p.is_file()}
    assert relay == snapshot                            # untouched, byte-identical
    dst = archive / "track_b" / "good"
    drained = {p.relative_to(dst).as_posix(): p.read_bytes()
               for p in sorted(dst.rglob("*")) if p.is_file()}
    assert drained == snapshot                          # byte-identical archive copy


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync unavailable")
def test_prune_relay_flag_restores_old_delete_after_verify_behavior(layout):
    _, archive, runs = layout
    b = _mkbundle(runs, "track_b/good", mode="healthy")

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs), "--prune-relay")

    assert proc.returncode == 0, proc.stderr
    assert "drain: OK track_b/good ->" in proc.stdout
    assert "relay pruned" in proc.stdout
    assert not b.exists()                               # --prune-relay empties it
    assert (archive / "track_b" / "good" / "last.ckpt").is_file()


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
    assert (runs / "track_b/good").exists()             # valid KEPT in relay (copy default)
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
    assert "--verify-only — copied/moved nothing" in proc.stdout
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
    """Unusable interpreters (python2, broken venv symlink, missing base
    interpreter) must fail LOUDLY at startup — with the REAL interpreter error
    shown and a remedy named — before any bundle is even considered. This is
    the 2026-08-27 lesson x2: first bare `python` (py2 parse death), then the
    frozen venv python refusing to start without its module on login shells."""
    _, archive, runs = layout
    _mkbundle(runs, "track_b/x", mode="healthy")

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs),
                  env_extra={"PYTHON_BIN": "/bin/false"})   # exits 1 on -c probe

    assert proc.returncode == 1
    assert "not a usable python3" in proc.stderr
    assert "remedy A: module load u22/python/3.12.4" in proc.stderr
    assert "PYTHON_BIN=" in proc.stderr                      # remedy B is named
    assert (runs / "track_b/x").is_dir()                     # nothing considered


def test_missing_interpreter_named_not_generic(layout):
    _, archive, runs = layout
    _mkbundle(runs, "track_b/x", mode="healthy")

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs),
                  env_extra={"PYTHON_BIN": str(runs / "no-such-python")})

    assert proc.returncode == 1
    assert "does not exist on this node" in proc.stderr


# ------------------------------------------------------- srun-transport laws --
#
# 2026-08-28: /usr/local/apps/python-3.12.4 (the frozen venv's base) is mounted
# ONLY on compute nodes — module load succeeds on the Ada login node but
# exposes no python. Content checks therefore run as ONE CPU srun job (full
# §5.8 string) executing scripts/drain_probe.py under the frozen venv. These
# suites drive that transport end-to-end through an srun SHIM (executes the
# bash -lc payload locally) — the laptop cannot submit real slurm jobs, but
# payload assembly, verdict parsing, and failure handling all run for real.

def _mk_srun_shim(tmp_path: Path, behaviour: str = "exec") -> Path:
    """A fake `srun` on PATH: skips the scheduling flags, finds the inner
    `bash -lc <payload>` and runs it HERE — proving the payload is complete
    and correct without a cluster."""
    shim = tmp_path / "srun-shim"
    shim.mkdir(exist_ok=True)
    exe = shim / "srun"
    if behaviour == "exec":
        body = (
            '#!/bin/bash\n'
            'args=("$@")\n'
            'for i in "${!args[@]}"; do\n'
            '    if [[ "${args[$i]}" == "-lc" ]]; then\n'
            '        exec bash -lc "${args[$((i+1))]}"\n'
            '    fi\n'
            'done\n'
            'echo "shim: no bash -lc payload found" >&2\n'
            'exit 99\n'
        )
    elif behaviour == "fail":
        body = '#!/bin/bash\necho "srun: error: shimmed allocation failure" >&2\nexit 7\n'
    else:                                                   # junk: rc 0, no verdicts
        body = '#!/bin/bash\necho "hello from a noisy batch system"\nexit 0\n'
    exe.write_text(body)
    exe.chmod(0o755)
    return shim


def _srun_env(shim_dir: Path) -> dict:
    return {
        "PATH": f"{shim_dir}:{os.environ['PATH']}",
        "SPELL_PROBE": "srun",
        "PYTHON_BIN": sys.executable,       # exported through the payload
    }


def test_srun_payload_carries_full_scheduling_string_and_module_recipe():
    """§5.8 law, static pin: the srun probe job must carry the FROZEN
    scheduling string in full and the module->venv recipe that makes the
    compute-node interpreter usable."""
    body = DRAIN.read_text(encoding="utf-8")
    for token in ("-p u22", "-A research", "--qos=medium",
                  "--constraint=2080ti", "--exclude=gnode066",
                  "--gres=gpu:0", "module load u22/python/3.12.4",
                  "drain_probe.py", "envs/spell/bin/python"):
        assert token in body, f"drainer lost §5.8/recipe token: {token!r}"


def test_srun_transport_end_to_end_via_shim(layout):
    """Verdicts flow through the srun payload: healthy validates, defective is
    refused BY NAME — proving probe transport, not just the local path."""
    _, archive, runs = layout
    _mkbundle(runs, "track_b/good", mode="healthy")
    _mkbundle(runs, "track_b/nok", mode="no_ckpt")

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs),
                  "--verify-only",
                  env_extra=_srun_env(_mk_srun_shim(layout[0])))

    assert proc.returncode != 0
    assert "VALIDATE ok track_b/good" in proc.stdout, proc.stderr
    assert "REFUSE track_b/nok" in proc.stderr, proc.stderr
    assert "last.ckpt missing" in proc.stderr


@pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync unavailable")
def test_srun_transport_drains_for_real_via_shim(layout):
    _, archive, runs = layout
    _mkbundle(runs, "track_b/good", mode="healthy")

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs),
                  env_extra=_srun_env(_mk_srun_shim(layout[0])))

    assert proc.returncode == 0, proc.stderr
    assert "drain: OK track_b/good ->" in proc.stdout
    assert (runs / "track_b/good").exists()             # copy default: relay kept
    assert (archive / "track_b/good" / "metrics.parquet").is_file()


def test_failed_srun_job_refuses_everything_loudly(layout):
    """A dead allocation must never look like valid bundles: srun rc!=0 with
    no verdicts => every candidate REFUSED, named as a probe-job failure."""
    _, archive, runs = layout
    _mkbundle(runs, "track_b/x", mode="healthy")

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs),
                  env_extra=_srun_env(_mk_srun_shim(layout[0], "fail")))

    assert proc.returncode != 0
    assert "REFUSE track_b/x" in proc.stderr
    assert "probe job failed (rc=7)" in proc.stderr
    assert (runs / "track_b/x").is_dir()


def test_noisy_srun_output_counts_as_protocol_violation(layout):
    """Junk on the probe's stdout (batch-system banners, profile noise) may
    never be mistaken for verdicts: missing verdict lines => refusal."""
    _, archive, runs = layout
    _mkbundle(runs, "track_b/x", mode="healthy")

    proc = _drain("--archive", str(archive), "--runs-dir", str(runs),
                  env_extra=_srun_env(_mk_srun_shim(layout[0], "junk")))

    assert proc.returncode != 0
    assert "REFUSE track_b/x" in proc.stderr
    assert "probe protocol violation" in proc.stderr
    assert (runs / "track_b/x").is_dir()


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
    assert "probe protocol violation" in proc.stderr
    assert "VALIDATE ok" not in proc.stdout
    assert (runs / "track_b/x").is_dir()
