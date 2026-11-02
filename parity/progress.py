"""Wall-clock-cadence progress reporting (PROTOCOL §3.24/§3.25).

Reports fire when ``report_every_s`` seconds of WALL-CLOCK time have passed since
the previous report (never on a row-count threshold, which silently assumes a
throughput rate), plus one forced report at the end. Each report carries the
progress fraction, the last unit's own duration, the slowest unit of the window,
the rate and an ETA.
"""

from __future__ import annotations

import time
from typing import Callable


class ProgressReporter:
    def __init__(self, total: int, *, report_every_s: float,
                 on_report: Callable[[dict], None], clock: Callable[[], float] = time.monotonic):
        if total < 1:
            raise ValueError(f"total must be >= 1, got {total}")
        self.total = total
        self.report_every_s = report_every_s
        self.on_report = on_report
        self.clock = clock
        self.done = 0
        self._t_start = clock()
        self._t_last_report = self._t_start
        self._reported_done = 0
        self._window_max = 0.0
        self._last_unit_s = 0.0
        self._extra: dict = {}

    def update(self, unit_s: float, **extra) -> None:
        if self.done >= self.total:
            raise ValueError("update() called for more units than the declared total")
        self.done += 1
        self._last_unit_s = unit_s
        self._window_max = max(self._window_max, unit_s)
        self._extra = extra
        if self.clock() - self._t_last_report >= self.report_every_s:
            self._report()

    def finish(self) -> None:
        if self.done != self._reported_done:
            self._report()

    def _report(self) -> None:
        now = self.clock()
        elapsed = now - self._t_start
        rate = self.done / elapsed if elapsed > 0 else 0.0
        eta = (self.total - self.done) / rate if rate > 0 else float("nan")
        payload = {
            "utts_done": self.done,
            "utts_total": self.total,
            "progress": self.done / self.total,
            "elapsed_s": elapsed,
            "window_s": now - self._t_last_report,
            "last_utt_s": self._last_unit_s,
            "window_max_utt_s": self._window_max,
            "utts_per_s": rate,
            "eta_s": eta,
            **self._extra,
        }
        self._t_last_report = now
        self._reported_done = self.done
        self._window_max = 0.0
        self.on_report(payload)
