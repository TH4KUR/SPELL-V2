"""Offline proof that parity/vendor/raven == upstream RAVEn @ 9fd9a37 + vendor.patch.

The proof needs no network and no upstream clone: the manifest stores, per file,
the upstream sha256 and git blob sha1; every patched file's UPSTREAM bytes are
re-derived by reverse-applying the stored unified diff to the vendored bytes and
compared against those hashes. Unpatched files must hash-equal upstream directly.
A reviewer can re-check the same hashes against a fresh clone with
``python scripts/parity_vendor_manifest.py check --upstream <clone>``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from parity import vendor_provenance as vp

VENDOR = vp.VENDOR_ROOT
PROJECT_ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------
# reverse_apply: the strict unified-diff applier the proof relies on
# --------------------------------------------------------------------------

def _diff(a: bytes, b: bytes, tmp_path: Path) -> bytes:
    if shutil.which("diff") is None:
        pytest.skip("GNU diff not available")
    pa, pb = tmp_path / "a.txt", tmp_path / "b.txt"
    pa.write_bytes(a)
    pb.write_bytes(b)
    out = subprocess.run(
        ["diff", "-u", "--label", "a/x", "--label", "b/x", str(pa), str(pb)],
        capture_output=True, check=False)
    assert out.returncode in (0, 1), out.stderr
    return out.stdout


@pytest.mark.parametrize("a,b", [
    (b"l1\nl2\nl3\n", b"l1\nL2\nl3\n"),                       # replace
    (b"l1\nl2\nl3\n", b"l1\nl2\nnew\nl3\n"),                  # insert
    (b"l1\nl2\nl3\n", b"l1\nl3\n"),                           # delete
    (b"l1\nl2\nl3\n", b"l2\nl3\n"),                           # delete at start
    (b"l1\nl2\nl3\n", b"l1\nl2\n"),                           # delete at end
    (b"l1\nl2\nl3", b"l1\nl2\nl3\n"),                         # newline added at EOF
    (b"l1\nl2\nl3\n", b"l1\nl2\nl3"),                         # newline removed at EOF
    (b"l1\nl2\nl3", b"l1\nl2\nL3"),                           # both lack final newline
    (b"".join(b"line%d\n" % i for i in range(60)),
     b"".join(b"line%d\n" % i if i not in (3, 30, 57) else b"CHANGED%d\n" % i
              for i in range(60))),                           # three separate hunks
    (b"a\r\nb\x0c\nc\n", b"a\r\nB\x0c\nc\n"),                 # CR / form-feed are not line breaks
])
def test_reverse_apply_roundtrips_gnu_diff(a, b, tmp_path):
    patch = _diff(a, b, tmp_path)
    assert patch, "test inputs must differ"
    assert vp.reverse_apply(b, patch) == a


def test_reverse_apply_rejects_context_mismatch(tmp_path):
    a, b = b"l1\nl2\nl3\n", b"l1\nL2\nl3\n"
    patch = _diff(a, b, tmp_path)
    with pytest.raises(vp.ProvenanceError):
        vp.reverse_apply(b"l1\nTAMPERED\nl3\n", patch)


def test_reverse_apply_rejects_added_line_mismatch(tmp_path):
    a, b = b"l1\nl2\nl3\n", b"l1\nL2\nl3\n"
    patch = _diff(a, b, tmp_path)
    with pytest.raises(vp.ProvenanceError):
        vp.reverse_apply(b"l1\nl2\nl3\n", patch)             # vendored lacks the '+' line


def test_reverse_apply_rejects_hunk_count_lie():
    bad = (b"--- a/x\n+++ b/x\n@@ -1,3 +1,3 @@\n l1\n-l2\n+L2\n")   # claims 3 lines, has 2
    with pytest.raises(vp.ProvenanceError):
        vp.reverse_apply(b"l1\nL2\nl3\n", bad)


def test_split_patch_by_file_keys_on_upstream_and_vendored_paths():
    patch = (b"--- a/up/one.py\n+++ b/ven/one.py\n@@ -1 +1 @@\n-x\n+y\n"
             b"--- a/two.py\n+++ b/two.py\n@@ -1 +1 @@\n-p\n+q\n")
    parts = vp.split_patch(patch)
    assert set(parts) == {("up/one.py", "ven/one.py"), ("two.py", "two.py")}
    assert parts[("two.py", "two.py")].startswith(b"--- a/two.py\n")


# --------------------------------------------------------------------------
# the real vendored tree
# --------------------------------------------------------------------------

def test_vendored_tree_is_upstream_plus_listed_patches():
    assert vp.verify(VENDOR) == []


def test_manifest_pins_the_expected_upstream_commit():
    m = vp.load_manifest(VENDOR)
    assert m["upstream"]["commit"] == "9fd9a3743c41ba93532258b340d123c20fe25432"
    assert m["upstream"]["repo"] == "https://github.com/ahaliassos/raven"
    assert m["upstream"]["license"] == "MIT"


def test_manifest_file_list_matches_the_declared_vendoring_plan():
    m = vp.load_manifest(VENDOR)
    declared = sorted(vp.VENDORED_FILES)
    listed = sorted((f["upstream_path"], f["vendored_path"]) for f in m["files"])
    assert listed == declared


def test_vendor_dir_holds_exactly_manifest_files_plus_our_three_documents():
    m = vp.load_manifest(VENDOR)
    expected = {f["vendored_path"] for f in m["files"]} | set(vp.NON_UPSTREAM_FILES)
    on_disk = {p.relative_to(VENDOR).as_posix() for p in VENDOR.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts}
    assert on_disk == expected


def test_every_patched_file_is_explained_in_patches_md():
    m = vp.load_manifest(VENDOR)
    doc = (VENDOR / vp.PATCHES_DOC).read_text(encoding="utf-8")
    patched = [f for f in m["files"] if f["upstream_sha256"] != f["vendored_sha256"]]
    assert patched, "expected at least the six/distutils patches"
    for f in patched:
        assert f["vendored_path"] in doc, f"{f['vendored_path']} patched but undocumented"


def test_patch_touches_only_listed_files():
    m = vp.load_manifest(VENDOR)
    patched = {(f["upstream_path"], f["vendored_path"]) for f in m["files"]
               if f["upstream_sha256"] != f["vendored_sha256"]}
    parts = vp.split_patch((VENDOR / vp.PATCH_NAME).read_bytes())
    assert set(parts) == patched


def test_patch_only_makes_the_documented_numerics_neutral_edits():
    """Every changed line must be one of: (a) the distutils LooseVersion import
    swapped for packaging's Version under the same name, (b) `import six` removed,
    (c) `six.moves.range(` rewritten to `range(` on an otherwise identical line,
    (d) `dtype=np.int` rewritten to `dtype=int` on an otherwise identical line (the
    removed numpy alias was the builtin), or (e) a pure deletion in the two extraction
    files (no '+' lines at all)."""
    extraction = {"raven_metrics.py", "raven_data_transforms.py"}
    rewrites = (("six.moves.range(", "range("), ("dtype=np.int", "dtype=int"))
    parts = vp.split_patch((VENDOR / vp.PATCH_NAME).read_bytes())
    for (_, vendored_path), file_patch in parts.items():
        removed, added = [], []
        for line in file_patch.split(b"\n"):
            if line.startswith(b"---") or line.startswith(b"+++"):
                continue
            if line.startswith(b"-"):
                removed.append(line[1:].decode())
            elif line.startswith(b"+"):
                added.append(line[1:].decode())
        if vendored_path in extraction:
            assert added == [], f"{vendored_path}: extraction must be deletion-only"
            continue
        remaining_removed = list(removed)
        for a in added:
            if a == "from packaging.version import Version as LooseVersion":
                remaining_removed.remove("from distutils.version import LooseVersion")
                continue
            for old, new in rewrites:
                original = a.replace(new, old, 1)
                if original != a and original in remaining_removed:
                    remaining_removed.remove(original)
                    break
            else:
                raise AssertionError(f"{vendored_path}: unexplained added line {a!r}")
        assert all(r == "import six" for r in remaining_removed), (
            f"{vendored_path}: unexplained removed lines {remaining_removed}")


def test_license_and_apache_headers_survive():
    lic = (VENDOR / "LICENSE").read_text(encoding="utf-8")
    assert lic.startswith("MIT License") and "ahaliassos" in lic
    m = vp.load_manifest(VENDOR)
    checked = 0
    for f in m["files"]:
        data = (VENDOR / f["vendored_path"]).read_bytes()
        upstream = vp.upstream_bytes(VENDOR, f)
        if b"Apache 2.0" in upstream[:600]:
            assert b"Apache 2.0" in data[:600], f"{f['vendored_path']} lost its Apache-2.0 header"
            checked += 1
    assert checked >= 20


def test_tampering_with_a_vendored_byte_is_detected(tmp_path):
    work = tmp_path / "raven"
    shutil.copytree(VENDOR, work)
    target = work / "espnet/nets/pytorch_backend/transformer/attention.py"
    target.write_bytes(target.read_bytes().replace(b"math.sqrt(self.d_k)", b"math.sqrt(self.d_k + 1)", 1))
    problems = vp.verify(work)
    assert any("attention.py" in p for p in problems)


def test_tampering_with_the_patch_is_detected(tmp_path):
    work = tmp_path / "raven"
    shutil.copytree(VENDOR, work)
    patch = work / vp.PATCH_NAME
    patch.write_bytes(patch.read_bytes().replace(b"-import six", b"-import sixx", 1))
    assert vp.verify(work) != []


def test_an_extra_unlisted_file_is_detected(tmp_path):
    work = tmp_path / "raven"
    shutil.copytree(VENDOR, work)
    (work / "espnet" / "stray.py").write_text("x = 1\n")
    assert any("stray.py" in p for p in vp.verify(work))


def test_manifest_json_is_sorted_and_stable():
    raw = (VENDOR / vp.MANIFEST_NAME).read_text(encoding="utf-8")
    m = json.loads(raw)
    paths = [f["vendored_path"] for f in m["files"]]
    assert paths == sorted(paths)
    assert raw == json.dumps(m, indent=2, sort_keys=True) + "\n"


def test_vendored_files_are_not_gitignored():
    """Round-1 trap: .gitignore carries `data/`, which would have silently dropped
    upstream's data/transforms.py from the first commit."""
    if shutil.which("git") is None or not (PROJECT_ROOT / ".git").exists():
        pytest.skip("git checkout not available")
    files = [p.relative_to(PROJECT_ROOT).as_posix() for p in (PROJECT_ROOT / "parity").rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    res = subprocess.run(["git", "check-ignore", "--no-index", "--stdin"],
                         input="\n".join(files), text=True, capture_output=True,
                         cwd=PROJECT_ROOT, check=False)
    assert res.returncode == 1 and res.stdout == "", f"git would ignore: {res.stdout}"
