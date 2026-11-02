"""Pre-registered training recipes of the parity track (PROTOCOL §3.34).

``configs/parity/recipes.yaml`` holds the two recipes (R-A, R-B) and is pinned by
sha256: a changed byte makes :func:`load_recipe` raise until ``RECIPES_SHA256`` is
updated deliberately. Unknown or missing keys are errors; nothing falls back to a
hidden default.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RECIPES_PATH = PROJECT_ROOT / "configs" / "parity" / "recipes.yaml"
RECIPES_SHA256 = "4f0ea8572664c7fa9066831c450036ebd2d361e7173c982ed123de08c9a8dcdd"

RECIPE_NAMES = ("R-A", "R-B")
PROVENANCE_TAGS = ("paper-stated", "usr-stated", "checkpoint-derived", "inferred", "assumed")
MODEL_SELECTIONS = ("final_epoch", "average_last_k")

COMMON_KEYS = (
    "seed", "betas", "ctc_weight", "label_smoothing", "time_mask_window_s",
    "time_mask_stride_s", "waveform_normalisation", "precision", "effective_batch_seconds",
    "micro_batch_seconds", "model_selection", "average_last_k", "beam_size", "decode_ctc_weight",
)
RECIPE_KEYS = (
    "description", "epochs", "warmup_epochs", "lr_encoder", "lr_decoder", "weight_decay",
    "layer_decay", "min_lr", "grad_clip", "drop_path",
)


class RecipeError(ValueError):
    """The recipe file is not the pinned one, or a recipe is malformed."""


@dataclass(frozen=True)
class Recipe:
    name: str
    description: str
    seed: int
    epochs: int
    warmup_epochs: int
    lr_encoder: float
    lr_decoder: float
    weight_decay: float
    layer_decay: float | None
    min_lr: float
    grad_clip: float
    drop_path: float
    betas: tuple[float, float]
    ctc_weight: float
    label_smoothing: float
    time_mask_window_s: float
    time_mask_stride_s: float
    waveform_normalisation: str
    precision: str
    effective_batch_seconds: float
    micro_batch_seconds: float
    model_selection: str
    average_last_k: int
    beam_size: int
    decode_ctc_weight: float
    sha256: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["betas"] = list(self.betas)
        return d

    def arch_config(self, base: dict) -> dict:
        """``base`` (configs/parity/arch_base.yaml) with the three recipe-owned model keys."""
        cfg = dict(base)
        cfg["drop_path"] = self.drop_path
        cfg["mtlalpha"] = self.ctc_weight
        cfg["lsm_weight"] = self.label_smoothing
        return cfg


def recipes_sha256(path: str | Path = RECIPES_PATH) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _exact_keys(section: dict, expected: tuple[str, ...], where: str) -> None:
    missing = [k for k in expected if k not in section]
    unknown = sorted(set(section) - set(expected))
    if missing or unknown:
        raise RecipeError(f"{where}: missing keys {missing}, unknown keys {unknown}")


def load_recipe(name: str, path: str | Path = RECIPES_PATH, *,
                expect_sha256: str | None = RECIPES_SHA256) -> Recipe:
    path = Path(path)
    raw = path.read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if expect_sha256 is not None and got != expect_sha256:
        raise RecipeError(
            f"{path}: sha256 {got} != pinned RECIPES_SHA256 {expect_sha256}. The recipes are "
            "pre-registered before any run (PROTOCOL §3.34); changing them requires updating "
            "parity.recipes.RECIPES_SHA256 on purpose and logging the change in parity/CLAIMS.md")
    doc = yaml.safe_load(raw)
    if not isinstance(doc, dict) or set(doc) != {"common", "recipes"}:
        raise RecipeError(f"{path}: expected exactly the top-level keys 'common' and 'recipes'")
    common, recipes = doc["common"], doc["recipes"]
    if sorted(recipes) != sorted(RECIPE_NAMES):
        raise RecipeError(f"{path}: recipes must be exactly {list(RECIPE_NAMES)}, got {sorted(recipes)}")
    if name not in recipes:
        raise RecipeError(f"unknown recipe {name!r}; choose one of {list(RECIPE_NAMES)}")
    _exact_keys(common, COMMON_KEYS, "common")
    _exact_keys(recipes[name], RECIPE_KEYS, f"recipes.{name}")
    merged = {**common, **recipes[name]}

    layer_decay = merged["layer_decay"]
    recipe = Recipe(
        name=name,
        description=str(merged["description"]),
        seed=int(merged["seed"]),
        epochs=int(merged["epochs"]),
        warmup_epochs=int(merged["warmup_epochs"]),
        lr_encoder=float(merged["lr_encoder"]),
        lr_decoder=float(merged["lr_decoder"]),
        weight_decay=float(merged["weight_decay"]),
        layer_decay=None if layer_decay is None else float(layer_decay),
        min_lr=float(merged["min_lr"]),
        grad_clip=float(merged["grad_clip"]),
        drop_path=float(merged["drop_path"]),
        betas=(float(merged["betas"][0]), float(merged["betas"][1])),
        ctc_weight=float(merged["ctc_weight"]),
        label_smoothing=float(merged["label_smoothing"]),
        time_mask_window_s=float(merged["time_mask_window_s"]),
        time_mask_stride_s=float(merged["time_mask_stride_s"]),
        waveform_normalisation=str(merged["waveform_normalisation"]),
        precision=str(merged["precision"]),
        effective_batch_seconds=float(merged["effective_batch_seconds"]),
        micro_batch_seconds=float(merged["micro_batch_seconds"]),
        model_selection=str(merged["model_selection"]),
        average_last_k=int(merged["average_last_k"]),
        beam_size=int(merged["beam_size"]),
        decode_ctc_weight=float(merged["decode_ctc_weight"]),
        sha256=got,
    )
    _validate(recipe)
    return recipe


def _validate(r: Recipe) -> None:
    problems = []
    if r.waveform_normalisation != "none":
        problems.append("waveform_normalisation must be 'none' (RAVEn App. A.2)")
    if r.precision != "fp32":
        problems.append("precision must be 'fp32'")
    if r.model_selection not in MODEL_SELECTIONS:
        problems.append(f"model_selection must be one of {MODEL_SELECTIONS}")
    if not 0 <= r.warmup_epochs < r.epochs:
        problems.append("need 0 <= warmup_epochs < epochs")
    if r.layer_decay is None and r.lr_encoder != r.lr_decoder:
        problems.append("without layer_decay every parameter shares one LR: lr_encoder must equal lr_decoder")
    if r.layer_decay is not None and not 0 < r.layer_decay <= 1:
        problems.append("layer_decay must be in (0, 1]")
    if not 0 <= r.min_lr <= min(r.lr_encoder, r.lr_decoder):
        problems.append("min_lr must lie in [0, min(lr_encoder, lr_decoder)]")
    if not 0 <= r.drop_path < 1:
        problems.append("drop_path must be in [0, 1)")
    if r.grad_clip <= 0 or r.weight_decay < 0:
        problems.append("grad_clip must be > 0 and weight_decay >= 0")
    if not 0 <= r.ctc_weight <= 1 or not 0 <= r.decode_ctc_weight <= 1:
        problems.append("ctc weights must lie in [0, 1]")
    if r.micro_batch_seconds <= 0 or r.effective_batch_seconds < r.micro_batch_seconds:
        problems.append("need 0 < micro_batch_seconds <= effective_batch_seconds")
    if r.average_last_k < 1 or r.beam_size < 1:
        problems.append("average_last_k and beam_size must be >= 1")
    if r.time_mask_window_s < 0 or r.time_mask_stride_s <= 0:
        problems.append("time-mask window must be >= 0 and stride > 0")
    if problems:
        raise RecipeError(f"recipe {r.name}: " + "; ".join(problems))
