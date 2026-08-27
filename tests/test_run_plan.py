"""Run-plan pipeline regressions (the Phase-1 pilot-A lessons, PROTOCOL §10):

* every committed *.tsv parses and covers its task range under the REAL
  production reader (scripts/plan_reader.awk), not a lookalike;
* out-of-range array ids fail LOUDLY (exit 3);
* malformed plans fail LOUDLY with structural exit 2;
* slurm/template.sbatch stays greppable-clean: no inline #SBATCH comments,
  no hardcoded array range, no --ntasks-per-node;
* the sanctioned generator reproduces every committed plan byte-for-byte.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import gen_run_plan as grp  # noqa: E402  (flat-repo dev import)

AWK = Path(__file__).resolve().parents[1] / "scripts" / "plan_reader.awk"
TEMPLATE = PROJECT_ROOT / "slurm" / "template.sbatch"


def run_reader(plan_text: str | Path, task_id: str) -> subprocess.CompletedProcess:
    """Invoke THE production awk reader exactly as template.sbatch does."""
    if isinstance(plan_text, str):
        import tempfile
        with tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False) as f:
            f.write(plan_text)
            path = f.name
    else:
        path = str(plan_text)
    return subprocess.run(
        ["awk", "-v", f"t={task_id}", "-f", str(AWK), path],
        capture_output=True, text=True,
    )


# ------------------------------------------------- committed plan identities --

COMMITTED_PLANS = sorted(str(p.relative_to(PROJECT_ROOT))
                         for p in (PROJECT_ROOT / "slurm").glob("*.tsv"))


def test_committed_plans_exist_and_parse():
    assert COMMITTED_PLANS, "run-plan TSVs vanished from slurm/"
    for key in COMMITTED_PLANS:
        rows = grp.parse_plan(PROJECT_ROOT / key)
        assert rows, f"{key} has no data rows"
        ids = [int(r[0]) for r in rows]
        assert ids == sorted(ids)
        assert len(set(ids)) == len(ids)


def test_pilot_plan_ids_cover_array_range_0_to_2():
    rows = grp.parse_plan(PROJECT_ROOT / "slurm" / "run_plan_pilots.tsv")
    ids = [int(r[0]) for r in rows]
    assert ids == list(range(min(ids), max(ids) + 1))          # contiguous
    assert min(ids) == 0                                       # ceiling is task 0
    ceiling = next(r for r in rows if int(r[0]) == 0)
    assert ceiling[2] == "subsets/splits/train_ids.txt"        # FROZEN identity
    assert ceiling[4] == "configs/pilot_100pct.yaml"
    floors = [r for r in rows if int(r[0]) != 0]
    assert {r[3] for r in floors} == {"101", "102"}            # subset seeds ARE identity


def test_committed_plans_match_sanctioned_generator_byte_for_byte():
    for key in COMMITTED_PLANS:
        want = grp.render(key)
        got = (PROJECT_ROOT / key).read_text(encoding="utf-8")
        assert got == want, f"{key} was hand-edited — regenerate via scripts/gen_run_plan.py"


@pytest.mark.parametrize("plan_key", COMMITTED_PLANS)
def test_production_reader_parses_every_id_of_committed_plans(plan_key):
    """Task range of each committed plan flows through the real awk reader."""
    for row in grp.parse_plan(PROJECT_ROOT / plan_key):
        proc = run_reader(PROJECT_ROOT / plan_key, str(row[0]))
        assert proc.returncode == 0, f"{plan_key} id {row[0]}: {proc.stderr}"
        assert proc.stdout.strip() == "\t".join(row)


def test_out_of_range_task_id_fails_loudly():
    proc = run_reader(PROJECT_ROOT / "slurm" / "run_plan_pilots.tsv", "99")
    assert proc.returncode == 3
    assert "FATAL[run_plan]" in proc.stderr
    assert "not found" in proc.stderr
    assert proc.stdout == ""                                   # no half output


# ------------------------------------------------------------ malformed input --

GOOD_ROW = "7\ttrack_b\tsubsets/m.txt\t9\tconfigs/c.yaml\t0\n"
GOOD_HEADER = grp.HEADER + "\n"


@pytest.mark.parametrize("text,why", [
    ("# only comments here\n", "empty plan after comments"),
    ("track\twrong\theader\na\tb\tc\td\te\tf\n", "missing/incorrect header"),
    (GOOD_HEADER + "0\ttrack_b\tm.txt\t1\tc.yaml\n", "five fields instead of six"),
    (GOOD_HEADER + "0 track_b m.txt 1 c.yaml 0\n", "space-separated single field"),
    (GOOD_HEADER + GOOD_ROW + GOOD_ROW.replace("\t7\t", "\t8\t").replace(
        "8\ttrack_b", "7\ttrack_b"), "duplicate task id"),
    (GOOD_HEADER + "zz\ttrack_b\tm.txt\t1\tc.yaml\t0\n", "non-integer task id"),
    (GOOD_HEADER + "0\t\tm.txt\t1\tc.yaml\t0\n", "empty field inside row"),
])
def test_malformed_plans_fail_loudly(text, why):
    proc = run_reader(text, "0")
    assert proc.returncode == 2, f"{why}: expected structural fatal, rc={proc.returncode}"
    assert "FATAL[run_plan]" in proc.stderr
    assert proc.stdout == ""


def test_field_value_guards_live_in_template_not_reader():
    """Seed/keep/track sanity is the LAUNCHING script's job (awk stays schema-
    pure); the template must carry those guards explicitly."""
    body = TEMPLATE.read_text(encoding="utf-8")
    assert '[[ "$SEED" =~ ^[0-9]+$ ]]' in body
    assert '[[ "$KEEP_LOCAL_TRAJ" =~ ^[01]$ ]]' in body
    assert 'case "$TRACK" in track_a|track_b)' in body


def test_exit_code_precedence_structural_beats_not_found():
    """A structural violation must keep exit 2 even when t is also absent
    (awk `exit` in a rule still runs END — regression-guarded)."""
    proc = run_reader(GOOD_HEADER + "0\t\tm\t1\tc\t0\n", "5")
    assert proc.returncode == 2


# -------------------------------------------------------- template hygiene ----

SBATCH_LINES: list[str] = []


def _sbatch_directives() -> list[str]:
    global SBATCH_LINES
    if not SBATCH_LINES:
        SBATCH_LINES = [ln for ln in TEMPLATE.read_text(encoding="utf-8").splitlines()
                        if ln.startswith("#SBATCH")]
    return SBATCH_LINES


def test_template_has_no_inline_sbatch_comments():
    for ln in _sbatch_directives():
        content = ln[len("#SBATCH"):].strip()
        assert "#" not in content, f"inline comment on directive: {ln!r}"


def test_template_does_not_pin_array_range_or_ntasks():
    joined = "\n".join(_sbatch_directives())
    assert "--array" not in joined, "array selection belongs on the sbatch CLI"
    assert "--ntasks-per-node" not in joined, "single-task jobs default to 1"


def test_template_carries_frozen_constants_exactly_once():
    joined = [ln for ln in TEMPLATE.read_text(encoding="utf-8").splitlines()]
    for flag in ("-p u22", "-A research", "--qos=medium",
                 "--constraint=2080ti", "--exclude=gnode066", "--gres=gpu:1"):
        hits = [ln for ln in joined if ln.startswith("#SBATCH") and flag in ln]
        assert len(hits) == 1, f"{flag!r} appears {len(hits)} time(s)"


def test_template_routes_through_index_free_reader_with_guards():
    body = TEMPLATE.read_text(encoding="utf-8")
    assert "plan_reader.awk" in body                       # single-source reader
    assert 'SLURM_ARRAY_TASK_ID:?array context required' in body
    assert 'readarray' not in body and '$((SLURM_ARRAY_TASK_ID + ' not in body
    assert "--run-manifest-out" in body                    # bypass parity, §10
    # single-writer law (§10 item 9): the EXIT trap NEVER writes COMPLETED —
    # success attestation lives inside the training entrypoint, not shell rc.
    for ln in body.splitlines():
        if "touch" in ln:
            assert "COMPLETED" not in ln, f"trap must never mark bundles: {ln!r}"
        if "COMPLETED" in ln and "#SBATCH" not in ln:
            assert "touch" not in ln
    assert "FAILED rc=" in body                            # loud failure instead
    assert "run_manifest" in body


def test_gen_run_plan_check_cli_is_green():
    proc = subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts/gen_run_plan.py"),
                           "--check"], capture_output=True, text=True,
                          cwd=str(PROJECT_ROOT))
    assert proc.returncode == 0, proc.stderr
