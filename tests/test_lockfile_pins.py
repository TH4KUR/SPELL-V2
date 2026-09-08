"""Cross-package lockfile-consistency regression (2026-09-09 incident):

An interactive `pip install onnxruntime==1.18.1` on Ada silently downgraded
numpy 2.5.2 -> 1.26.4 (onnxruntime's resolver pulled an old numpy), and the
subsequent `pip freeze > requirements.lock` faithfully committed that BROKEN
combination as the project's binding description of the env (§5.5). scipy
1.18.1 (and lightning/torchmetrics beneath it) require numpy>=2.0 at RUNTIME
(scipy/sparse/_sputils.py uses np.long/np.ulong, reintroduced only in numpy
2.0) — pip's installer never caught this because nothing in either package's
declared metadata conflicted; only import-time code did. `pip install -r
requirements.lock` (scripts/setup_env.sbatch) cannot self-heal a lock that is
internally inconsistent — the lock's CONTENT must be correct BEFORE
reconciling the venv to it. This test pins the one compatibility fact that
bit us: whatever else `requirements.lock` says, numpy must stay >=2.0.
"""

from __future__ import annotations

import re

from config import PROJECT_ROOT

LOCK = PROJECT_ROOT / "requirements.lock"


def _pin(pkg: str) -> str:
    for line in LOCK.read_text().splitlines():
        m = re.match(rf"^{re.escape(pkg)}==([\w.+-]+)\s*$", line)
        if m:
            return m.group(1)
    raise AssertionError(f"{pkg} not pinned in {LOCK}")


def test_numpy_stays_v2_for_scipy_lightning_compat():
    numpy_version = _pin("numpy")
    major = int(numpy_version.split(".")[0])
    assert major >= 2, (
        f"requirements.lock pins numpy=={numpy_version}, but the pinned "
        f"scipy=={_pin('scipy')} (and lightning/torchmetrics above it) "
        "require numpy>=2.0 at import time -- this exact downgrade broke "
        "score_proxy.py on Ada (2026-09-09, job 2691962)")


def test_onnxruntime_never_reenters_the_lock():
    """onnxruntime==1.18.1 declares numpy<2.0 -- irreconcilable with
    numpy==2.5.2/scipy==1.18.1 in one `pip install -r` (2026-09-09). Resolved
    by dropping onnxruntime entirely: score_dnsmos.py now runs
    dnsmos_model.DNSMOSTorch, a verified pure-PyTorch port of the vendored
    ONNX graph (scripts/port_dnsmos_to_torch.py). A future hand-install
    re-adding onnxruntime to the lock would silently resurrect this exact
    conflict -- catch it here instead."""
    for line in LOCK.read_text().splitlines():
        assert not line.startswith("onnxruntime=="), (
            "onnxruntime must not be in requirements.lock -- see "
            "dnsmos_model.py / PROTOCOL §5.5 for why (2026-09-09)")
