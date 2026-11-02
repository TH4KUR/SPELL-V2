"""RAVEn Base audio-only ASR model: the vendored upstream ``E2E`` built from
``configs/parity/arch_base.yaml`` and the released-checkpoint loader.

The released checkpoint (README row "Base / LRS3 / 4.7", Google Drive id
1UJXGo9qUZ0VxPNlJfL2-JD5_428JqxNv) is a plain ``state_dict`` of 577 tensors /
55,791,794 parameters; the vendored model must load it with ``strict=True``.
"""

from __future__ import annotations

import hashlib
import types
from pathlib import Path

import torch
import yaml

from parity import upstream
from parity.runinfo import sha256_file
from parity.text import ODIM

upstream.activate()
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARCH_CONFIG_PATH = PROJECT_ROOT / "configs" / "parity" / "arch_base.yaml"

# Every attribute E2E.__init__ (and the Encoder/Decoder it builds) reads from `args`.
ARCH_KEYS = (
    "idim", "adim", "aheads", "eunits", "elayers",
    "transformer_frontend", "transformer_input_layer",
    "dropout_rate", "transformer_attn_dropout_rate",
    "transformer_encoder_attn_layer_type", "macaron_style", "use_cnn_module",
    "cnn_module_kernel", "zero_triu", "a_upsample_ratio", "relu_type",
    "ddim", "dheads", "dunits", "dlayers", "lsm_weight",
    "transformer_length_normalized_loss", "rel_pos_type", "layerscale",
    "init_values", "ff_bn_pre", "post_norm", "gamma_zero", "gamma_init",
    "mask_init_type", "ctc_type", "drop_path", "mtlalpha",
)

RELEASED_CKPT_SHA256 = "78212d2b008a3636f5798745c8c0f60e86b9bed71f280fcd2a603102c4a01f4f"
RELEASED_CKPT_BYTES = 223_444_287
RELEASED_PARAM_COUNT = 55_791_794
RELEASED_STATE_DICT_KEYS = 577
# sha256 over the sorted lines "<key>|<comma-joined shape>|<dtype>" of the released
# state_dict: lets the architecture be checked against the checkpoint's structure
# without the 223 MB file (recompute from the file with state_dict_fingerprint).
RELEASED_STATE_DICT_FINGERPRINT = "675f27df81514b55739d18f1bc6ffbc0fa5608ce597cac059bf7465230eefa85"


def load_arch_config(path: str | Path = ARCH_CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: architecture config must be a mapping")
    return cfg


def _check_keys(cfg: dict) -> None:
    missing = [k for k in ARCH_KEYS if k not in cfg]
    unknown = sorted(set(cfg) - set(ARCH_KEYS))
    if missing:
        raise ValueError(f"architecture config is missing keys: {missing}")
    if unknown:
        raise ValueError(f"architecture config has unknown keys: {unknown}")


def build_model(arch_cfg: dict | None = None, *, odim: int = ODIM) -> E2E:
    """Upstream ``E2E`` for ``odim`` outputs (1,049 = blank + 1,047 units + eos).
    Returned in train mode as constructed; callers evaluating must call ``.eval()``."""
    cfg = dict(load_arch_config() if arch_cfg is None else arch_cfg)
    _check_keys(cfg)
    return E2E(odim, types.SimpleNamespace(**cfg))


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def state_dict_fingerprint(state_dict: dict) -> str:
    lines = [
        f"{k}|{','.join(str(d) for d in v.shape)}|{str(v.dtype).replace('torch.', '')}"
        for k, v in sorted(state_dict.items())
    ]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def assert_released_architecture(model: torch.nn.Module) -> None:
    """Raise unless ``model`` has the released checkpoint's parameter count,
    state_dict size and tensor-structure fingerprint."""
    problems = []
    n = count_parameters(model)
    if n != RELEASED_PARAM_COUNT:
        problems.append(f"parameter count {n} != {RELEASED_PARAM_COUNT}")
    sd = model.state_dict()
    if len(sd) != RELEASED_STATE_DICT_KEYS:
        problems.append(f"state_dict has {len(sd)} tensors != {RELEASED_STATE_DICT_KEYS}")
    fp = state_dict_fingerprint(sd)
    if fp != RELEASED_STATE_DICT_FINGERPRINT:
        problems.append(f"state_dict fingerprint {fp} != {RELEASED_STATE_DICT_FINGERPRINT}")
    if problems:
        raise ValueError("model does not match the released architecture: " + "; ".join(problems))


def load_released_checkpoint(path: str | Path, strict: bool = True,
                             expect_sha256: str | None = None,
                             arch_cfg: dict | None = None,
                             odim: int = ODIM) -> E2E:
    """Build the model and load a plain ``state_dict`` checkpoint into it (CPU, eval mode).
    ``odim`` is 1,049 for the released checkpoint; a checkpoint written by the parity trainer
    has the odim of its own tokenizer (``len(units) + 2``).

    The file is read with ``weights_only=True`` (a downloaded pickle never gets to
    run code). ``strict=True`` makes any missing/unexpected key an error.
    """
    path = Path(path)
    if expect_sha256 is not None:
        got = sha256_file(path)
        if got != expect_sha256:
            raise ValueError(f"{path}: sha256 {got} != expected {expect_sha256}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict) or not state or not all(
            isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state.items()):
        raise ValueError(f"{path}: expected a plain state_dict (str -> Tensor), got {type(state).__name__}")
    model = build_model(arch_cfg, odim=odim)
    model.load_state_dict(state, strict=strict)
    return model.eval()
