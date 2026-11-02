"""Provenance machinery for ``parity/vendor/raven`` (upstream RAVEn @ pinned commit).

Claim proven by :func:`verify`, offline: every vendored file equals the upstream
file at ``UPSTREAM_COMMIT`` plus the unified diff stored in ``vendor.patch``.

* The manifest records, per file, the upstream sha256 and git blob sha1.
* Unpatched files must hash to those values directly.
* Patched files are reverse-patched back to their upstream bytes and must hash
  to the same values.
* Nothing may exist in the vendor directory that the manifest does not list
  (besides the three documents we author).

``reverse_apply`` is a deliberately strict, zero-fuzz unified-diff applier: any
context or hunk-count mismatch raises, so a tampered patch or file cannot pass.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

UPSTREAM_REPO = "https://github.com/ahaliassos/raven"
UPSTREAM_COMMIT = "9fd9a3743c41ba93532258b340d123c20fe25432"
UPSTREAM_LICENSE = "MIT"

VENDOR_ROOT = Path(__file__).resolve().parent / "vendor" / "raven"

MANIFEST_NAME = "UPSTREAM_MANIFEST.json"
PATCH_NAME = "vendor.patch"
PATCHES_DOC = "PATCHES.md"
NON_UPSTREAM_FILES = (MANIFEST_NAME, PATCH_NAME, PATCHES_DOC)

# (upstream path, vendored path). Three upstream root modules are renamed on
# vendoring: generic names (`utils`, `metrics`) would shadow unrelated modules
# once the vendor root is on sys.path, and `data/` is matched by the repo's
# `.gitignore` (`data/`), which would silently drop the file from a commit.
VENDORED_FILES: tuple[tuple[str, str], ...] = tuple(sorted(
    [(p, p) for p in (
        "LICENSE",
        "conf/model/audio_backbone/resnet_transformer_base.yaml",
        "data_paths/test_with_tags_counts_unigram1000.csv",
        "espnet/asr/asr_utils.py",
        "espnet/nets/batch_beam_search.py",
        "espnet/nets/beam_search.py",
        "espnet/nets/ctc_prefix_score.py",
        "espnet/nets/e2e_asr_common.py",
        "espnet/nets/pytorch_backend/backbones/conv1d_extractor.py",
        "espnet/nets/pytorch_backend/backbones/conv3d_extractor.py",
        "espnet/nets/pytorch_backend/backbones/modules/resnet.py",
        "espnet/nets/pytorch_backend/backbones/modules/resnet1d.py",
        "espnet/nets/pytorch_backend/ctc.py",
        "espnet/nets/pytorch_backend/e2e_asr_transformer.py",
        "espnet/nets/pytorch_backend/nets_utils.py",
        "espnet/nets/pytorch_backend/transformer/__init__.py",
        "espnet/nets/pytorch_backend/transformer/add_sos_eos.py",
        "espnet/nets/pytorch_backend/transformer/attention.py",
        "espnet/nets/pytorch_backend/transformer/convolution.py",
        "espnet/nets/pytorch_backend/transformer/decoder.py",
        "espnet/nets/pytorch_backend/transformer/decoder_layer.py",
        "espnet/nets/pytorch_backend/transformer/embedding.py",
        "espnet/nets/pytorch_backend/transformer/encoder.py",
        "espnet/nets/pytorch_backend/transformer/encoder_layer.py",
        "espnet/nets/pytorch_backend/transformer/label_smoothing_loss.py",
        "espnet/nets/pytorch_backend/transformer/mask.py",
        "espnet/nets/pytorch_backend/transformer/multi_layer_conv.py",
        "espnet/nets/pytorch_backend/transformer/norm.py",
        "espnet/nets/pytorch_backend/transformer/positionwise_feed_forward.py",
        "espnet/nets/pytorch_backend/transformer/repeat.py",
        "espnet/nets/scorer_interface.py",
        "espnet/nets/scorers/__init__.py",
        "espnet/nets/scorers/ctc.py",
        "espnet/nets/scorers/length_bonus.py",
        "labels/unigram1000_units.txt",
    )] + [
        ("data/transforms.py", "raven_data_transforms.py"),
        ("metrics.py", "raven_metrics.py"),
        ("utils.py", "raven_utils.py"),
    ]))


class ProvenanceError(RuntimeError):
    """The vendored tree, manifest or patch is inconsistent."""


# ------------------------------------------------------------------ hashing --

def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_blob_sha1(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()


# --------------------------------------------------------- unified diff I/O --

_HUNK_RE = re.compile(rb"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def _split_lines(data: bytes) -> list[bytes]:
    """Split on b'\\n' only (keeping it); CR and form feed are ordinary bytes."""
    lines, start = [], 0
    while start < len(data):
        i = data.find(b"\n", start)
        if i < 0:
            lines.append(data[start:])
            break
        lines.append(data[start:i + 1])
        start = i + 1
    return lines


def split_patch(patch: bytes) -> dict[tuple[str, str], bytes]:
    """{(upstream_path, vendored_path): that file's section of the patch}."""
    sections: dict[tuple[str, str], bytes] = {}
    lines = _split_lines(patch)
    starts = [i for i, ln in enumerate(lines)
              if ln.startswith(b"--- a/") and i + 1 < len(lines)
              and lines[i + 1].startswith(b"+++ b/")]
    for n, s in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        old = lines[s][len(b"--- a/"):].rstrip(b"\n").decode()
        new = lines[s + 1][len(b"+++ b/"):].rstrip(b"\n").decode()
        if (old, new) in sections:
            raise ProvenanceError(f"patch lists {old} -> {new} twice")
        sections[(old, new)] = b"".join(lines[s:end])
    return sections


def _parse_hunks(file_patch: bytes) -> list[dict]:
    lines = _split_lines(file_patch)
    if len(lines) < 2 or not lines[0].startswith(b"--- ") or not lines[1].startswith(b"+++ "):
        raise ProvenanceError("patch section lacks ---/+++ headers")
    hunks: list[dict] = []
    cur = None
    for ln in lines[2:]:
        m = _HUNK_RE.match(ln)
        if m:
            cur = {"old_start": int(m.group(1)), "old_len": int(m.group(2) or 1),
                   "new_start": int(m.group(3)), "new_len": int(m.group(4) or 1),
                   "lines": []}
            hunks.append(cur)
            continue
        if cur is None:
            raise ProvenanceError(f"unexpected line before first hunk: {ln[:60]!r}")
        if ln.startswith(b"\\"):                       # "\ No newline at end of file"
            if not cur["lines"]:
                raise ProvenanceError("dangling no-newline marker")
            tag, text = cur["lines"][-1]
            if not text.endswith(b"\n"):
                raise ProvenanceError("duplicate no-newline marker")
            cur["lines"][-1] = (tag, text[:-1])
            continue
        tag, text = ln[:1], ln[1:]
        if tag not in (b" ", b"-", b"+"):
            raise ProvenanceError(f"bad hunk line {ln[:60]!r}")
        cur["lines"].append((tag, text))
    for h in hunks:
        n_old = sum(1 for t, _ in h["lines"] if t in (b" ", b"-"))
        n_new = sum(1 for t, _ in h["lines"] if t in (b" ", b"+"))
        if (n_old, n_new) != (h["old_len"], h["new_len"]):
            raise ProvenanceError(
                f"hunk header says -{h['old_len']} +{h['new_len']} but body has "
                f"-{n_old} +{n_new}")
    return hunks


def reverse_apply(vendored: bytes, file_patch: bytes) -> bytes:
    """Undo ``file_patch`` (upstream -> vendored) on ``vendored``; strict, no fuzz."""
    src = _split_lines(vendored)
    out: list[bytes] = []
    pos = 0
    for h in _parse_hunks(file_patch):
        start = h["new_start"] if h["new_len"] == 0 else h["new_start"] - 1
        if start < pos or start > len(src):
            raise ProvenanceError(f"hunk @ +{h['new_start']} out of order/range")
        out.extend(src[pos:start])
        pos = start
        for tag, text in h["lines"]:
            if tag in (b" ", b"+"):
                if pos >= len(src) or src[pos] != text:
                    got = src[pos][:60] if pos < len(src) else b"<EOF>"
                    raise ProvenanceError(
                        f"vendored line {pos + 1} is {got!r}, patch expects {text[:60]!r}")
                pos += 1
                if tag == b" ":
                    out.append(text)
            else:
                out.append(text)
    out.extend(src[pos:])
    return b"".join(out)


# ----------------------------------------------------------------- manifest --

def load_manifest(vendor_dir: Path = VENDOR_ROOT) -> dict:
    return json.loads((Path(vendor_dir) / MANIFEST_NAME).read_text(encoding="utf-8"))


def render_manifest(manifest: dict) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def _patch_sections(vendor_dir: Path) -> dict[tuple[str, str], bytes]:
    return split_patch((Path(vendor_dir) / PATCH_NAME).read_bytes())


def upstream_bytes(vendor_dir: Path, entry: dict) -> bytes:
    """Upstream content of one manifest entry, re-derived from the vendored copy."""
    data = (Path(vendor_dir) / entry["vendored_path"]).read_bytes()
    if entry["upstream_sha256"] == entry["vendored_sha256"]:
        return data
    section = _patch_sections(vendor_dir).get((entry["upstream_path"], entry["vendored_path"]))
    if section is None:
        raise ProvenanceError(f"{entry['vendored_path']} differs from upstream but has no patch section")
    return reverse_apply(data, section)


def verify(vendor_dir: Path = VENDOR_ROOT) -> list[str]:
    """Return a list of problems (empty == provenance proven)."""
    vendor_dir = Path(vendor_dir)
    problems: list[str] = []
    try:
        manifest = load_manifest(vendor_dir)
    except (OSError, ValueError) as e:
        return [f"manifest unreadable: {e}"]

    up = manifest.get("upstream", {})
    if up.get("commit") != UPSTREAM_COMMIT:
        problems.append(f"manifest commit {up.get('commit')} != pinned {UPSTREAM_COMMIT}")

    listed = set()
    try:
        sections = _patch_sections(vendor_dir)
    except (OSError, ProvenanceError) as e:
        return problems + [f"patch unreadable: {e}"]
    used_sections = set()

    for entry in manifest.get("files", []):
        vp_ = entry["vendored_path"]
        listed.add(vp_)
        path = vendor_dir / vp_
        if not path.is_file():
            problems.append(f"{vp_}: listed in manifest but missing")
            continue
        data = path.read_bytes()
        if sha256_bytes(data) != entry["vendored_sha256"]:
            problems.append(f"{vp_}: vendored sha256 differs from manifest (file edited after manifest was made)")
            continue
        if entry["upstream_sha256"] == entry["vendored_sha256"]:
            upstream = data
        else:
            key = (entry["upstream_path"], vp_)
            if key not in sections:
                problems.append(f"{vp_}: differs from upstream but vendor.patch has no section for it")
                continue
            used_sections.add(key)
            try:
                upstream = reverse_apply(data, sections[key])
            except ProvenanceError as e:
                problems.append(f"{vp_}: reverse patch failed: {e}")
                continue
        if sha256_bytes(upstream) != entry["upstream_sha256"]:
            problems.append(f"{vp_}: reverse-patched sha256 != recorded upstream sha256")
        if git_blob_sha1(upstream) != entry["upstream_git_blob_sha1"]:
            problems.append(f"{vp_}: reverse-patched git blob sha1 != recorded upstream blob")

    for key in sections:
        if key not in used_sections:
            problems.append(f"vendor.patch has a section for unlisted/unchanged file {key}")

    on_disk = {p.relative_to(vendor_dir).as_posix() for p in vendor_dir.rglob("*")
               if p.is_file() and "__pycache__" not in p.parts}
    for extra in sorted(on_disk - listed - set(NON_UPSTREAM_FILES)):
        problems.append(f"{extra}: present in vendor dir but not in manifest")
    for doc in NON_UPSTREAM_FILES:
        if doc not in on_disk:
            problems.append(f"{doc}: required document missing")
    return problems


def render_provenance_table(manifest: dict) -> str:
    """Markdown table of the manifest (embedded verbatim in parity/REVIEW.md)."""
    rows = ["| upstream file | vendored file | upstream sha256 | status |", "|---|---|---|---|"]
    for f in sorted(manifest["files"], key=lambda f: f["vendored_path"]):
        if f["upstream_sha256"] == f["vendored_sha256"]:
            status = "identical"
        else:
            status = f"patched (vendored sha256 `{f['vendored_sha256']}`)"
        rows.append(f"| `{f['upstream_path']}` | `{f['vendored_path']}` | "
                    f"`{f['upstream_sha256']}` | {status} |")
    return "\n".join(rows)


# ---------------------------------------------- generation (needs a clone) --

def _git(upstream_dir: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(upstream_dir), *args], check=True,
                          capture_output=True, text=True).stdout.strip()


def check_upstream_clone(upstream_dir: Path) -> None:
    """The clone must be exactly UPSTREAM_COMMIT with an unmodified tree."""
    head = _git(upstream_dir, "rev-parse", "HEAD")
    if head != UPSTREAM_COMMIT:
        raise ProvenanceError(f"upstream clone is at {head}, expected {UPSTREAM_COMMIT}")
    dirty = _git(upstream_dir, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ProvenanceError(f"upstream clone has local modifications:\n{dirty}")


def _diff_u(upstream_file: Path, vendored_file: Path, up_rel: str, ven_rel: str) -> bytes:
    res = subprocess.run(
        ["diff", "-u", "--label", f"a/{up_rel}", "--label", f"b/{ven_rel}",
         str(upstream_file), str(vendored_file)],
        capture_output=True, check=False, env={"LC_ALL": "C", "PATH": "/usr/bin:/bin"})
    if res.returncode not in (0, 1):
        raise ProvenanceError(f"diff failed for {up_rel}: {res.stderr.decode()}")
    return res.stdout


def build_manifest_and_patch(upstream_dir: Path, vendor_dir: Path = VENDOR_ROOT
                             ) -> tuple[dict, bytes]:
    """Recompute manifest + patch from a real upstream clone and the vendored tree."""
    upstream_dir, vendor_dir = Path(upstream_dir), Path(vendor_dir)
    check_upstream_clone(upstream_dir)
    files, sections = [], {}
    for up_rel, ven_rel in VENDORED_FILES:
        up_bytes = (upstream_dir / up_rel).read_bytes()
        ven_bytes = (vendor_dir / ven_rel).read_bytes()
        blob = git_blob_sha1(up_bytes)
        if _git(upstream_dir, "rev-parse", f"{UPSTREAM_COMMIT}:{up_rel}") != blob:
            raise ProvenanceError(f"{up_rel}: working-tree blob != commit blob")
        files.append({
            "upstream_path": up_rel,
            "vendored_path": ven_rel,
            "upstream_sha256": sha256_bytes(up_bytes),
            "upstream_git_blob_sha1": blob,
            "vendored_sha256": sha256_bytes(ven_bytes),
            "upstream_size": len(up_bytes),
            "vendored_size": len(ven_bytes),
        })
        if up_bytes != ven_bytes:
            sections[ven_rel] = _diff_u(upstream_dir / up_rel, vendor_dir / ven_rel, up_rel, ven_rel)
    files.sort(key=lambda f: f["vendored_path"])
    patch = b"".join(sections[k] for k in sorted(sections))   # byte order == `LC_ALL=C sort` of vendored paths
    manifest = {
        "upstream": {"repo": UPSTREAM_REPO, "commit": UPSTREAM_COMMIT,
                     "license": UPSTREAM_LICENSE},
        "non_upstream_files": sorted(NON_UPSTREAM_FILES),
        "files": files,
    }
    return manifest, patch
