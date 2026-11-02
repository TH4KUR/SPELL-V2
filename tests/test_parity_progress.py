"""parity/progress.py: wall-clock-cadence progress reports (PROTOCOL §3.24/§3.25).

The cadence is TIME-based, never row-count-based (a count threshold silently
assumes a throughput rate); every report carries the progress fraction, that
unit's OWN duration (an average hides a single slow or stuck unit), an ETA, and the
window's slowest unit.
"""

from __future__ import annotations

import pytest

from parity.progress import ProgressReporter


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def _drive(costs, report_every_s=10.0, total=None):
    clock = Clock()
    seen = []
    rep = ProgressReporter(total or len(costs), report_every_s=report_every_s,
                           on_report=seen.append, clock=clock)
    for c in costs:
        clock.t += c
        rep.update(c)
    rep.finish()
    return seen


def test_reports_are_scheduled_by_elapsed_time_not_by_row_count():
    # cumulative: 5, 10, 25, 26, 27, 28 -> reports at done=2 (t=10), done=3 (t=25), forced done=6
    seen = _drive([5.0, 5.0, 15.0, 1.0, 1.0, 1.0])
    assert [s["utts_done"] for s in seen] == [2, 3, 6]
    assert [s["elapsed_s"] for s in seen] == pytest.approx([10.0, 25.0, 28.0])


def test_each_report_carries_the_windows_own_duration_and_the_last_and_slowest_unit():
    seen = _drive([5.0, 5.0, 15.0, 1.0, 1.0, 1.0])
    assert [s["window_s"] for s in seen] == pytest.approx([10.0, 15.0, 3.0])
    assert [s["last_utt_s"] for s in seen] == pytest.approx([5.0, 15.0, 1.0])
    assert [s["window_max_utt_s"] for s in seen] == pytest.approx([5.0, 15.0, 1.0])
    # a slow unit hidden inside an otherwise quick window is still surfaced:
    seen = _drive([1.0, 20.0, 1.0, 1.0], report_every_s=10.0)
    assert max(s["window_max_utt_s"] for s in seen) == pytest.approx(20.0)


def test_report_has_progress_fraction_rate_and_eta():
    seen = _drive([5.0, 5.0, 5.0, 5.0], report_every_s=10.0)
    first = seen[0]
    assert first["utts_total"] == 4 and first["progress"] == pytest.approx(0.5)
    assert first["utts_per_s"] == pytest.approx(0.2)
    assert first["eta_s"] == pytest.approx(10.0)
    assert seen[-1]["utts_done"] == 4 and seen[-1]["eta_s"] == pytest.approx(0.0)


def test_final_report_is_forced_even_off_cadence_and_not_duplicated():
    seen = _drive([1.0, 1.0, 1.0], report_every_s=100.0)
    assert [s["utts_done"] for s in seen] == [3]
    clock = Clock()
    out = []
    rep = ProgressReporter(2, report_every_s=1.0, on_report=out.append, clock=clock)
    clock.t = 5.0
    rep.update(5.0)
    clock.t = 6.0
    rep.update(1.0)
    rep.finish()
    rep.finish()
    assert [s["utts_done"] for s in out] == [1, 2]            # finish() twice does not repeat done=2


def test_extra_fields_ride_along_with_the_report():
    clock = Clock()
    out = []
    rep = ProgressReporter(2, report_every_s=0.0, on_report=out.append, clock=clock)
    clock.t = 1.0
    rep.update(1.0, running_corpus_wer=0.25)
    assert out[0]["running_corpus_wer"] == 0.25


def test_unit_count_is_validated():
    with pytest.raises(ValueError):
        ProgressReporter(0, report_every_s=1.0, on_report=lambda p: None)
    clock = Clock()
    rep = ProgressReporter(1, report_every_s=1.0, on_report=lambda p: None, clock=clock)
    rep.update(0.1)
    with pytest.raises(ValueError, match="more units"):
        rep.update(0.1)
