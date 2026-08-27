"""MECHANICAL single-writer enforcement (PROTOCOL §10 item 9).

Exactly ONE writer context for the filename ``COMPLETED`` may exist across the
production surface: the trainer-attested touch inside
``scripts/train_track_b.py`` — reached only after a clean fit, durable metrics,
and the finished-manifest rewrite. The BundleCallback and every shell trap are
cleanup-only. History: the bypass script and template EXIT traps wrote the
marker unconditionally, then an rc-gate still attested mere process exit, and
BundleCallback.on_train_end held a third status-sniffing writer whose guard was
dead code (lightning 2.x has no STOPPED state and skips on_train_end on
exceptions by design of fit loops). Any reintroduction of a writer ANYWHERE is
a suite failure, never a silent re-corruption of provenance.
"""

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Production surface only: src shells + python. tests/ is EXEMPT because its
# drain/bundle fixtures legitimately FORGE markers into tmp bundles to prove
# the drainer refuses them.
SCAN_SUFFIXES = {".py", ".sh", ".sbatch"}
SKIP_DIRS = {"tests", ".git", "runs", "outputs", "logs", "checkpoints",
             ".pytest_cache", "__pycache__", "wandb", "datasets"}

# A writer context = one line that both names COMPLETED and performs a touch /
# shell `touch`. Name references for validation or prose are fine anywhere;
# creating the filename is what this law rations to a single site.
WRITER_LINE = re.compile(
    r"\btouch\b[^\n]*\bCOMPLETED\b"          # shell: touch .../COMPLETED ...
    r"|\bCOMPLETED\b[^\n]*\btouch\s*\("      # python: (…/"COMPLETED").touch()
)


def _iter_scan_files():
    for p in sorted(PROJECT_ROOT.rglob("*")):
        if not p.is_file() or p.suffix not in SCAN_SUFFIXES:
            continue
        rel_parts = set(p.relative_to(PROJECT_ROOT).parts[:-1])
        if rel_parts & SKIP_DIRS:
            continue
        yield p


def _writer_hits():
    hits: list[tuple[str, str]] = []
    for p in _iter_scan_files():
        for ln in p.read_text(encoding="utf-8").splitlines():
            if WRITER_LINE.search(ln):
                hits.append((str(p.relative_to(PROJECT_ROOT)), ln.strip()))
    return hits


def test_exactly_one_completed_writer_context_exists_repo_wide():
    hits = _writer_hits()
    assert len(hits) == 1, (
        "COMPLETED single-writer law violated (§10 item 9) — writer contexts:\n  "
        + "\n  ".join(f"{f}: {ln}" for f, ln in hits)
    )
    fname, line = hits[0]
    assert fname == str(Path("scripts") / "train_track_b.py"), fname
    assert '(run_dir / "COMPLETED").touch()' in line


def test_bundle_callback_never_creates_the_filename():
    src_lines = (PROJECT_ROOT / "ckpt_bundle.py").read_text(
        encoding="utf-8").splitlines()
    for ln in src_lines:
        assert '"COMPLETED").touch' not in ln           # the deleted third writer
        assert not WRITER_LINE.search(ln), f"callback writer context: {ln!r}"


def test_shell_scripts_never_create_the_filename():
    for rel in ("slurm/template.sbatch", "scripts/drain_runs.sh"):
        text = (PROJECT_ROOT / rel).read_text(encoding="utf-8")
        for ln in text.splitlines():
            if WRITER_LINE.search(ln):
                raise AssertionError(f"{rel}: shell writer context: {ln!r}")
    # ...and the loud-failure replacement exists where the trap once marked:
    template_body = (PROJECT_ROOT / "slurm" / "template.sbatch").read_text(
        encoding="utf-8")
    assert "FAILED rc=" in template_body              # named non-zero exit echo
