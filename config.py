"""Config loading utilities.

Plain-YAML configs (user decision: no Hydra). Locked protocol constants live in
``configs/protocol.yaml`` and are loaded into a frozen dataclass so nothing in
code relies on magic numbers. Every training run later hashes its config via
:func:`config_hash` into ``run_manifest.json``.

All scripts/tests import this module flat from the project root.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent

# Protocol revision string embedded in every run_manifest.json — bump whenever
# PROTOCOL.md gains binding rules. History: universe-v2, ada-storage-rev1,
# contamination-guard (§3.13–16: halo re-masking, GroupNorm, absolute HPs, dev GPU).
PROTOCOL_REVISION = "universe-v2+ada-storage-rev1+contamination-guard"


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem layout + Ada storage policy (see configs/paths.yaml)."""

    dataset_root: Path
    subsets_dir: Path
    index_path: Path
    splits_dir: Path
    runs_dir: Path
    archive_mode: str                 # LOCKED to "relay" (protocol revision)
    archive_root: Path                # drain destination; NEVER touched by compute jobs
    home_warn_gb: float               # $HOME usage gates (quota 30G)
    home_abort_gb: float
    inode_warn_k: int                 # $HOME inode warn threshold (quota 300k)

    @classmethod
    def from_dict(cls, d: dict, base: Path = PROJECT_ROOT) -> "PathsConfig":
        import os

        def _p(v: str) -> Path:
            p = Path(v)
            return p if p.is_absolute() else (base / p).resolve()

        cfg = cls(
            dataset_root=_p(os.environ.get("SPELL_DATA_ROOT", d["dataset_root"])),
            subsets_dir=_p(d["subsets_dir"]),
            index_path=_p(d["index_path"]),
            splits_dir=_p(d["splits_dir"]),
            runs_dir=_p(d["runs_dir"]),
            archive_mode=str(d["archive_mode"]),
            archive_root=Path(os.environ.get("SPELL_ARCHIVE_ROOT", d["archive_root"])),
            home_warn_gb=float(d["home_warn_gb"]),
            home_abort_gb=float(d["home_abort_gb"]),
            inode_warn_k=int(d["inode_warn_k"]),
        )
        if cfg.archive_mode != "relay":
            raise ValueError(
                f"archive_mode={cfg.archive_mode!r}: LOCKED to 'relay' — /share1 is "
                "unreachable from Ada compute nodes (verified); direct writes are "
                "rejected by policy, not merely discouraged"
            )
        if not 0 < cfg.home_warn_gb < cfg.home_abort_gb:
            raise ValueError("home gates must satisfy 0 < warn < abort")
        return cfg


@dataclass(frozen=True)
class ProtocolConfig:
    """LOCKED experiment constants. Frozen after Phase 0 — see PROTOCOL.md."""

    sample_rate: int          # Hz, audio domain
    token_hz: int             # RVQ frame rate
    codebook_size: int        # entries per RVQ codebook
    n_rvq_streams: int        # stacked codebook streams per frame
    crop_frames: int          # canonical crop length in token frames
    crop_samples: int         # canonical crop length in audio samples
    min_crop_frames: int      # utterances shorter than this cannot be crop-sampled
    token_pad_id: int         # collate padding sentinel; MUST be outside [0, codebook_size)
    budget_fraction: float    # subset budget (25% of corpus)
    val_size_utts: int        # internal validation set size (utterances)
    split_seed: int           # seed for the frozen internal split

    @classmethod
    def from_dict(cls, d: dict) -> "ProtocolConfig":
        cfg = cls(**d)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        # The one law everything else depends on: crops map frames↔samples exactly.
        expected_samples = self.crop_frames * self.sample_rate // self.token_hz
        if self.crop_samples != expected_samples:
            raise ValueError(
                f"crop_samples={self.crop_samples} inconsistent with "
                f"crop_frames={self.crop_frames} @ {self.sample_rate}Hz/{self.token_hz}Hz "
                f"(expected {expected_samples})"
            )
        if 0 <= self.token_pad_id < self.codebook_size:
            raise ValueError(
                f"token_pad_id={self.token_pad_id} must be negative (real codes span "
                f"[0, {self.codebook_size})); padding with an in-range id would corrupt CTC/vocoder targets"
            )
        if self.min_crop_frames > self.crop_frames:
            raise ValueError("min_crop_frames cannot exceed crop_frames")


def load_protocol(path: str | Path = PROJECT_ROOT / "configs" / "protocol.yaml") -> ProtocolConfig:
    return ProtocolConfig.from_dict(load_yaml(path))


def load_paths(path: str | Path = PROJECT_ROOT / "configs" / "paths.yaml") -> PathsConfig:
    return PathsConfig.from_dict(load_yaml(path))


def universe_budget(n_selectable: int, fraction: float) -> int:
    """Subset budget in UTTERANCES over the selectable universe (locked rule).

    Budget percentages are defined BY UTTERANCE COUNT over the selectable
    universe (trainval utts with n_tokens >= min_crop_frames), never by hours;
    realized hours are reported per subset afterwards. Fractional results round
    HALF UP (not Python's banker's round) so the rule is unambiguous.
    """
    if not 0 < fraction <= 1:
        raise ValueError(f"budget fraction must be in (0, 1], got {fraction}")
    if n_selectable <= 0:
        raise ValueError("universe is empty")
    return int(n_selectable * fraction + 0.5)


def config_hash(cfg: Any) -> str:
    """Stable sha256 of any YAML-ish mapping/dataclass — goes into run_manifest.json."""
    if dataclasses.is_dataclass(cfg):
        cfg = dataclasses.asdict(cfg)
    canonical = json.dumps(cfg, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ------------------------- YAML overlay loading -------------------------
# Track-B pilot configs inherit the pinned track_b.yaml via a top-level ``base:``
# key instead of duplicating YAML (Decision, Phase-1 plan): small overlays, one
# source of truth for frozen constants.


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive merge; ``override`` wins leaf-for-leaf. Lists/scalars replace wholesale."""
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(
    path: str | Path,
    configs_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load a YAML config, resolving an optional ``base: <file>`` inheritance chain.

    Relative ``base:`` values resolve against ``configs_dir`` (default
    ``<repo>/configs``). Cycles raise. The returned dict is freshly allocated —
    callers may mutate freely (e.g. seed injection) without touching disk state.
    """
    configs_dir = Path(configs_dir) if configs_dir else PROJECT_ROOT / "configs"
    seen: set[Path] = set()

    def _load(p: Path) -> dict[str, Any]:
        rp = p.resolve()
        if rp in seen:
            raise ValueError(f"config 'base:' cycle at {rp}")
        seen.add(rp)
        raw = load_yaml(rp)
        if not isinstance(raw, dict):
            raise ValueError(f"config {rp} must be a mapping")
        base_ref = raw.pop("base", None)
        merged: dict[str, Any] = {}
        if base_ref is not None:
            bp = Path(base_ref)
            if not bp.is_absolute():
                bp = configs_dir / bp
            merged = _load(bp)
        return deep_merge(merged, raw)

    given = Path(path)
    if not given.is_absolute():
        # repo-root-relative first (the documented invocation cwd), CWD fallback
        cand = (PROJECT_ROOT / given)
        given = cand if cand.exists() else given
    return _load(given)
