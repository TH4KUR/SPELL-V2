"""W&B run-id naming law (PROTOCOL §3.17(d), 2026-09-08).

Every formal run's id is derived EXACTLY as ``slurm/template.sbatch`` derives it:
``<track>/<manifest-stem>_seed<seed>_job<SLURM_JOB_ID>_t<task>``. This suite pins
the template's derivation line byte-for-byte (the law predates the test — this
is a guardrail, same discipline as the template pins in test_run_plan.py) and
EXECUTES the pinned expression so the documented id shape is proven, not assumed.
§3.17(b): renaming this pattern requires a PROTOCOL revision note first.
"""

from __future__ import annotations

import subprocess

from config import PROJECT_ROOT

TEMPLATE = PROJECT_ROOT / "slurm" / "template.sbatch"

# The frozen derivation line (template.sbatch:92) — byte-for-byte.
RUN_ID_LINE = (
    'RUN_ID="${TRACK}/$(basename "$MANIFEST" .txt)'
    '_seed${SEED}_job${SLURM_JOB_ID}_t${TASK_ID}"'
)


def test_template_pins_run_id_pattern_exactly_once():
    lines = TEMPLATE.read_text(encoding="utf-8").splitlines()
    hits = [ln for ln in lines if ln.startswith("RUN_ID=")]
    assert hits == [RUN_ID_LINE], f"run-id derivation drifted: {hits}"


def test_run_id_expression_produces_documented_shape():
    """Execute the template's own expression against a known bundle identity and
    assert the id every consumer (W&B naming, drainer, summarizer) will see."""
    expr = RUN_ID_LINE[len("RUN_ID="):]
    proc = subprocess.run(
        ["bash", "-c",
         f'TRACK=track_b; MANIFEST=subsets/random_25pct_seed101.txt; SEED=101; '
         f'SLURM_JOB_ID=2680643; TASK_ID=3; echo {expr}'],
        capture_output=True, text=True, timeout=10, check=True,
    )
    # NOTE: the manifest stem itself embeds the subset seed (§3.11), so it
    # appears twice by design — <stem>_seed<seed> is the frozen convention.
    assert proc.stdout.strip() == (
        "track_b/random_25pct_seed101_seed101_job2680643_t3"
    )


def test_run_id_shape_is_track_anchored_and_seed_embedded():
    """The id must remain parseable into track/manifest/seed/job/task segments —
    the summarizer and backfiller key on exactly this shape. Generic inputs
    (manifest stem without its own seed marker) this time."""
    proc = subprocess.run(
        ["bash", "-c",
         'TRACK=track_a; MANIFEST=x/y/dnsmos_25pct.txt; SEED=0; '
         'SLURM_JOB_ID=42; TASK_ID=7; '
         'RUN_ID="${TRACK}/$(basename "$MANIFEST" .txt)_seed${SEED}'
         '_job${SLURM_JOB_ID}_t${TASK_ID}"; echo "$RUN_ID"'],
        capture_output=True, text=True, timeout=10, check=True,
    )
    assert proc.stdout.strip() == "track_a/dnsmos_25pct_seed0_job42_t7"