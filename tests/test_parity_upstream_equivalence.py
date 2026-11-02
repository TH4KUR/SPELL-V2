"""Opt-in proof that the vendor patches changed no numerics.

With ``RAVEN_UPSTREAM_CLONE`` pointing at a clone of ahaliassos/raven checked out at
9fd9a37, the UNPATCHED upstream tree and the vendored tree are each run in their own
process on identical weights and input; the encoder features, the 1-best token ids and
every beam-search score must be bit-identical. The unpatched tree needs ``six`` and
``distutils`` (e.g. a Python 3.12 environment that still has setuptools and six), so the
test skips when either is missing. Weights come from ``RAVEN_ASR_CKPT`` when set, else a
fixed-seed random initialisation (same RNG consumption in both processes).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from parity import upstream

CLONE = os.environ.get("RAVEN_UPSTREAM_CLONE")
CKPT = os.environ.get("RAVEN_ASR_CKPT")
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCH = PROJECT_ROOT / "configs" / "parity" / "arch_base.yaml"

WORKER = textwrap.dedent("""
    import sys, types
    import torch, yaml

    root, arch_yaml, ckpt, out = sys.argv[1:5]
    sys.path.insert(0, root)
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
    from espnet.nets.scorers.length_bonus import LengthBonus
    import espnet
    assert all(str(p).startswith(root) for p in espnet.__path__), espnet.__path__

    units = [l.split()[0] for l in open(root + "/labels/unigram1000_units.txt").read().splitlines()]
    tokens = ["<blank>"] + units + ["<eos>"]
    odim = len(tokens)
    cfg = yaml.safe_load(open(arch_yaml))
    torch.manual_seed(1234)
    model = E2E(odim, types.SimpleNamespace(**cfg))
    if ckpt != "-":
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
    model.eval()

    scorers = model.scorers()
    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(odim)
    search = BatchBeamSearch(
        beam_size=4, vocab_size=odim,
        weights=dict(decoder=1.0 - 0.1, ctc=0.1, lm=0.0, length_bonus=0.0),
        scorers=scorers, sos=odim - 1, eos=odim - 1, token_list=tokens,
        pre_beam_score_key="decoder")

    g = torch.Generator().manual_seed(7)
    wave = 0.1 * torch.randn(24000, generator=g)
    with torch.no_grad():
        feat, _ = model.encoder(wave.reshape(1, -1, 1), None)
        best = search(x=feat.squeeze(0), maxlenratio=0.0, minlenratio=0.0)[0].asdict()
    torch.save({"feat": feat, "yseq": best["yseq"], "score": best["score"],
                "scores": best["scores"]}, out)
""")


def _have_unpatched_prerequisites() -> bool:
    try:
        import distutils.version  # noqa: F401
        import six  # noqa: F401
    except ImportError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    not CLONE or not Path(CLONE, "espnet").is_dir() or not _have_unpatched_prerequisites(),
    reason="set RAVEN_UPSTREAM_CLONE to a clone at 9fd9a37 (needs six + distutils for the unpatched tree)")


def _run(root: Path, out: Path):
    ckpt = CKPT if CKPT and Path(CKPT).is_file() else "-"
    res = subprocess.run([sys.executable, "-W", "ignore", "-c", WORKER, str(root), str(ARCH), ckpt, str(out)],
                         capture_output=True, text=True, cwd=out.parent)
    assert res.returncode == 0, res.stderr[-3000:]
    return torch.load(out, weights_only=True)


def test_unpatched_upstream_and_vendored_tree_agree_bitwise(tmp_path):
    clone_head = subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True,
                                text=True, check=True).stdout.strip()
    assert clone_head == "9fd9a3743c41ba93532258b340d123c20fe25432"
    a = _run(Path(CLONE).resolve(), tmp_path / "upstream.pt")
    b = _run(upstream.VENDOR_ROOT, tmp_path / "vendored.pt")
    assert a["feat"].shape == (1, 37, 512)
    assert torch.equal(a["feat"], b["feat"])
    assert a["yseq"] == b["yseq"]
    assert a["score"] == b["score"]
    assert a["scores"] == b["scores"]
