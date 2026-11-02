"""The vendored RAVEn tree is reached through exactly one explicit mechanism
(parity.upstream.activate) and its import closure is free of every package that
is absent from the Ada env (or that the parity track refuses on principle)."""

from __future__ import annotations

import ast
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from parity import upstream

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Never imported anywhere under parity/ (vendored files included) nor in scripts/parity_*.py.
FORBIDDEN = ("torchaudio", "torchmetrics", "hydra", "fairseq", "sentencepiece",
             "six", "distutils", "lightning", "pytorch_lightning", "torchvision")
# Upstream dead-code paths import these lazily (inside functions that the parity
# track never calls): ESPnet trainer extensions (chainer), plotting (matplotlib),
# ErrorCalculator (editdistance) and the warpctc CTC branch, which
# CTC.__init__ disables on torch >= 1.7. Documented in PATCHES.md "not patched".
LAZY_DEAD_CODE_IMPORTS = {"chainer", "matplotlib", "editdistance", "warpctc_pytorch"}
# The offline tokenizer step (PROTOCOL §3.34) runs on the laptop, not in Ada's frozen env: the two files
# below are the ONLY first-party code that may import sentencepiece, and only inside a function.
SENTENCEPIECE_ALLOWED = {"parity/spm_tokens.py", "scripts/parity_train_spm.py"}


def _under(path: Path, root: Path) -> bool:
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def test_activate_resolves_espnet_to_the_vendored_copy():
    upstream.activate()
    import espnet.nets.pytorch_backend.e2e_asr_transformer as e2e  # noqa: F401
    import espnet
    assert list(espnet.__path__), "espnet must resolve to a package"
    assert all(_under(Path(p), upstream.VENDOR_ROOT) for p in espnet.__path__)
    assert _under(Path(e2e.__file__), upstream.VENDOR_ROOT)


def test_activate_is_idempotent():
    upstream.activate()
    n = sys.path.count(str(upstream.VENDOR_ROOT))
    upstream.activate()
    assert n == 1 and sys.path.count(str(upstream.VENDOR_ROOT)) == 1


def test_activate_restores_the_vendor_root_if_it_fell_off_sys_path(monkeypatch):
    upstream.activate()
    monkeypatch.setattr(sys, "path", [p for p in sys.path if p != str(upstream.VENDOR_ROOT)])
    upstream.activate()
    assert str(upstream.VENDOR_ROOT) in sys.path


def test_activate_refuses_a_foreign_espnet_already_imported(monkeypatch):
    class _Foreign:
        __path__ = ["/definitely/not/the/vendor/dir/espnet"]
    monkeypatch.setitem(sys.modules, "espnet", _Foreign())
    with pytest.raises(RuntimeError, match="espnet"):
        upstream.activate()


def test_activate_refuses_when_a_regular_espnet_package_would_shadow_it(tmp_path):
    """A pip-installed espnet is a regular package and beats the vendored
    namespace package on sys.path; resolution must be checked, not assumed."""
    fake = tmp_path / "espnet"
    fake.mkdir()
    (fake / "__init__.py").write_text("")
    prog = textwrap.dedent(f"""
        import sys
        sys.path.append({str(tmp_path)!r})
        from parity import upstream
        try:
            upstream.activate()
        except RuntimeError as e:
            print("REFUSED", e)
        else:
            import espnet
            print("ACTIVATED", list(espnet.__path__))
    """)
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                         cwd=PROJECT_ROOT, check=True)
    assert "REFUSED" in out.stdout, out.stdout + out.stderr


def test_root_level_upstream_modules_have_collision_free_names():
    upstream.activate()
    import raven_data_transforms
    import raven_metrics
    import raven_utils
    for mod in (raven_data_transforms, raven_metrics, raven_utils):
        assert _under(Path(mod.__file__), upstream.VENDOR_ROOT)
    assert hasattr(raven_utils, "get_param_groups")
    assert hasattr(raven_data_transforms, "AdaptiveLengthTimeMask")
    assert hasattr(raven_metrics, "get_wer")


# ------------------------------------------------------------ import hygiene --

def _first_party_python_files():
    files = [p for p in (PROJECT_ROOT / "parity").rglob("*.py")]
    files += sorted((PROJECT_ROOT / "scripts").glob("parity_*.py"))
    return files


def _imports(path: Path) -> list[tuple[str, bool]]:
    """(top-level package name, is_module_level) for every import statement."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    module_level = {id(n) for n in tree.body}
    found: list[tuple[str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((a.name.split(".")[0], id(node) in module_level) for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            found.append((node.module.split(".")[0], id(node) in module_level))
    return found


def test_no_parity_module_or_vendored_file_imports_a_forbidden_package():
    """Static scan of every import statement (module level or lazy) under parity/
    (vendored files included -- the patches exist precisely to make this true) and
    in scripts/parity_*.py."""
    offenders = []
    for path in _first_party_python_files():
        rel = path.relative_to(PROJECT_ROOT).as_posix()
        bad = {n for n, _ in _imports(path)} & set(FORBIDDEN)
        if rel in SENTENCEPIECE_ALLOWED:
            bad -= {"sentencepiece"}
        if bad:
            offenders.append((rel, sorted(bad)))
    assert offenders == []


def test_sentencepiece_is_imported_only_lazily_and_only_by_the_offline_tokenizer_step():
    importers = {}
    for path in _first_party_python_files():
        hits = [module_level for name, module_level in _imports(path) if name == "sentencepiece"]
        if hits:
            importers[path.relative_to(PROJECT_ROOT).as_posix()] = hits
    assert set(importers) == SENTENCEPIECE_ALLOWED
    assert all(not any(levels) for levels in importers.values()), "sentencepiece must be imported inside functions"


def test_lazy_dead_code_imports_are_confined_to_the_documented_vendored_files():
    """chainer/matplotlib/editdistance/warpctc_pytorch may appear ONLY as lazy
    (function-level) imports, ONLY in vendored files; our own modules never use them."""
    vendor_py = set(upstream.VENDOR_ROOT.rglob("*.py"))
    seen = set()
    for path in _first_party_python_files():
        for name, module_level in _imports(path):
            if name in LAZY_DEAD_CODE_IMPORTS:
                assert path in vendor_py, f"{path} imports {name}"
                assert not module_level, f"{path} imports {name} at import time"
                seen.add(name)
    assert seen == LAZY_DEAD_CODE_IMPORTS


def test_full_import_closure_works_with_forbidden_packages_blocked():
    """Dynamic proof: import every parity module in a subprocess whose import
    system refuses the forbidden packages outright."""
    prog = textwrap.dedent(f"""
        import importlib.abc, sys
        FORBIDDEN = {FORBIDDEN + tuple(sorted(LAZY_DEAD_CODE_IMPORTS))!r}
        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name.split(".")[0] in FORBIDDEN:
                    raise ImportError("blocked: " + name)
        sys.meta_path.insert(0, Block())
        from parity import upstream
        upstream.activate()
        import parity.model, parity.text, parity.metrics, parity.data, parity.decode
        import parity.trainer, parity.train_data, parity.optim, parity.recipes, parity.checkpoint_avg
        import parity.spm_tokens
        import espnet.nets.pytorch_backend.e2e_asr_transformer
        import espnet.nets.batch_beam_search
        import espnet.nets.scorers.ctc, espnet.nets.scorers.length_bonus
        import espnet.asr.asr_utils
        import raven_utils, raven_metrics, raven_data_transforms
        print("CLOSURE-OK")
    """)
    out = subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True,
                         cwd=PROJECT_ROOT)
    assert out.returncode == 0 and "CLOSURE-OK" in out.stdout, out.stderr[-2000:]


def test_vendored_python_compiles_without_syntax_warnings():
    import warnings
    for path in (upstream.VENDOR_ROOT).rglob("*.py"):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            compile(path.read_bytes(), str(path), "exec")
