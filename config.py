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


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass(frozen=True)
class PathsConfig:
    """Filesystem layout + storage-policy thresholds (see configs/paths.yaml)."""

    dataset_root: Path
    subsets_dir: Path
    index_path: Path
    splits_dir: Path
    nas_root: Path
    min_free_gb_local: float
    min_free_gb_nas: float

    @classmethod
    def from_dict(cls, d: dict, base: Path = PROJECT_ROOT) -> "PathsConfig":
        import os

        def _p(v: str) -> Path:
            p = Path(v)
            return p if p.is_absolute() else (base / p).resolve()

        nas = Path(os.environ.get("SPELL_NAS_ROOT", d["nas_root"]))
        return cls(
            dataset_root=_p(d["dataset_root"]),
            subsets_dir=_p(d["subsets_dir"]),
            index_path=_p(d["index_path"]),
            splits_dir=_p(d["splits_dir"]),
            nas_root=nas,
            min_free_gb_local=float(d["min_free_gb_local"]),
            min_free_gb_nas=float(d["min_free_gb_nas"]),
        )


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
