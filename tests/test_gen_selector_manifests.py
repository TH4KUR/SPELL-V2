"""scripts/gen_selector_manifests.py contract — batch driver over
scripts/make_selector_manifest.py's roster, not a second sanctioned writer.

Laws pinned here:
  * every roster entry gets exactly one call for deterministic selectors,
    one PER §3.20 seed (201/202/203) for stochastic ones — derived from the
    REAL msm.ROSTER, so a future roster change is picked up automatically;
  * one selector's failure (e.g. a score table not pulled from Ada yet)
    does not stop the others -- every call is attempted;
  * the exit code reflects whether ALL calls succeeded.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import gen_selector_manifests as G  # noqa: E402
import make_selector_manifest as msm  # noqa: E402


def test_build_calls_covers_every_roster_entry_with_right_seed_count():
    calls = G.build_calls(0.25)
    by_selector: dict[str, list[list[str]]] = {}
    for call in calls:
        name = call[call.index("--selector") + 1]
        by_selector.setdefault(name, []).append(call)

    assert set(by_selector) == set(msm.ROSTER)
    for name, spec in msm.ROSTER.items():
        entries = by_selector[name]
        if spec["stochastic"]:
            assert len(entries) == 3
            seeds = {c[c.index("--seed") + 1] for c in entries}
            assert seeds == {"201", "202", "203"}
        else:
            assert len(entries) == 1
            assert "--seed" not in entries[0]


def test_build_calls_threads_fraction_through():
    calls = G.build_calls(0.10)
    assert all("0.1" in c[c.index("--fraction") + 1] for c in calls)


def test_run_all_attempts_every_call_and_reports_status(monkeypatch):
    calls = [["--selector", "dnsmos", "--fraction", "0.25"],
            ["--selector", "kmeans", "--fraction", "0.25", "--seed", "201"],
            ["--selector", "dsir", "--fraction", "0.25", "--seed", "201"]]
    seen = []

    def fake_main(argv):
        seen.append(argv)
        if argv[1] == "kmeans":
            raise FileNotFoundError("score table missing: scores/kmeans_assignments_seed201.parquet")
        return 0

    monkeypatch.setattr(G.msm, "main", fake_main)
    results = G.run_all(calls)

    assert seen == calls   # ALL three attempted, kmeans's failure didn't stop dsir
    assert [status for _, status in results] == ["OK", "FAILED: score table missing: "
                                                  "scores/kmeans_assignments_seed201.parquet",
                                                  "OK"]


def test_main_all_succeed_returns_0(monkeypatch, capsys):
    monkeypatch.setattr(G.msm, "main", lambda argv: 0)
    rc = G.main(["--fraction", "0.25"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "13/13" in out


def test_main_partial_failure_returns_nonzero(monkeypatch, capsys):
    def fake_main(argv):
        if argv[1] == "el2n":
            raise ValueError("budget exceeds pool")
        return 0

    monkeypatch.setattr(G.msm, "main", fake_main)
    rc = G.main(["--fraction", "0.25"])
    assert rc == 1
    out = capsys.readouterr().out
    assert "12/13" in out
    assert "FAIL" in out


def test_main_rejects_missing_fraction():
    with pytest.raises(SystemExit):
        G.main([])
