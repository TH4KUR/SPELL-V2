"""The one explicit way into the vendored RAVEn tree (``parity/vendor/raven``).

Upstream modules import each other absolutely (``from espnet.nets... import``),
so the vendor root has to be on ``sys.path``. :func:`activate` does that and
proves it worked: ``espnet`` must resolve to the vendored copy, never to a
pip-installed one (a regular package would silently beat the vendored namespace
package) and never to something imported earlier.

The three upstream root-level modules are vendored under collision-free names
(``raven_utils``, ``raven_metrics``, ``raven_data_transforms``) and are imported
by those names after :func:`activate`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "raven"


def _under(path, root: Path) -> bool:
    p = Path(path).resolve()
    return p == root.resolve() or root.resolve() in p.parents


def activate() -> Path:
    """Make ``import espnet...`` resolve to the vendored copy; idempotent."""
    root = str(VENDOR_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)

    loaded = sys.modules.get("espnet")
    if loaded is not None:
        locations = list(getattr(loaded, "__path__", []))
    else:
        spec = importlib.util.find_spec("espnet")
        locations = list(spec.submodule_search_locations or []) if spec else []
    if not locations or not all(_under(p, VENDOR_ROOT) for p in locations):
        raise RuntimeError(
            f"`espnet` resolves to {locations}, not {VENDOR_ROOT}: another espnet "
            "installation shadows (or was imported before) the vendored copy; the "
            "parity track must run on the vendored RAVEn tree")
    return VENDOR_ROOT
