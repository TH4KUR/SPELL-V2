#!/usr/bin/env python3
"""Provenance tool for parity/vendor/raven (upstream RAVEn @ 9fd9a37).

    python scripts/parity_vendor_manifest.py verify
    python scripts/parity_vendor_manifest.py table
    python scripts/parity_vendor_manifest.py check    --upstream /path/to/raven_clone
    python scripts/parity_vendor_manifest.py generate --upstream /path/to/raven_clone

``verify``   offline: manifest + patch reproduce upstream's bytes (what pytest runs).
``table``    print the per-file provenance table embedded in parity/REVIEW.md.
``check``    against a real clone pinned at the commit: recomputes manifest and patch
             and requires them to equal the stored files byte-for-byte.
``generate`` rewrites UPSTREAM_MANIFEST.json and vendor.patch from the clone
             (run after editing a vendored file; then update PATCHES.md).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from parity import vendor_provenance as vp  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["verify", "table", "check", "generate"])
    ap.add_argument("--upstream", type=Path,
                    help="git clone of https://github.com/ahaliassos/raven at the pinned commit")
    ap.add_argument("--vendor-dir", type=Path, default=vp.VENDOR_ROOT)
    args = ap.parse_args(argv)

    if args.mode == "table":
        print(vp.render_provenance_table(vp.load_manifest(args.vendor_dir)))
        return 0

    if args.mode == "verify":
        problems = vp.verify(args.vendor_dir)
        for p in problems:
            print(f"PROVENANCE PROBLEM: {p}", file=sys.stderr)
        if problems:
            return 1
        print(f"[parity-vendor] OK: {args.vendor_dir} == upstream@{vp.UPSTREAM_COMMIT[:7]} + vendor.patch")
        return 0

    if args.upstream is None:
        ap.error(f"--upstream is required for {args.mode}")
    manifest, patch = vp.build_manifest_and_patch(args.upstream, args.vendor_dir)
    manifest_text = vp.render_manifest(manifest)
    manifest_path = args.vendor_dir / vp.MANIFEST_NAME
    patch_path = args.vendor_dir / vp.PATCH_NAME

    if args.mode == "generate":
        manifest_path.write_text(manifest_text, encoding="utf-8")
        patch_path.write_bytes(patch)
        print(f"[parity-vendor] wrote {manifest_path} ({len(manifest['files'])} files) and {patch_path}")
        return 0

    same_manifest = manifest_path.read_text(encoding="utf-8") == manifest_text
    same_patch = patch_path.read_bytes() == patch
    if not (same_manifest and same_patch):
        print(f"MISMATCH vs clone: manifest_equal={same_manifest} patch_equal={same_patch}",
              file=sys.stderr)
        return 1
    print(f"[parity-vendor] OK: stored manifest and patch equal a fresh recomputation from {args.upstream}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
