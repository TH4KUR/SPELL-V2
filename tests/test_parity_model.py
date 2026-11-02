"""parity/model.py + parity/decode.py: the vendored upstream E2E at the released
Base audio-only ASR architecture, and the exact upstream decoding recipe.

Tests that need the 223 MB released checkpoint read its path from the env var
``RAVEN_ASR_CKPT`` and skip cleanly when it is unset. Everything else runs on a
freshly built, randomly initialised model and needs no external file.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path

import pytest
import torch
import yaml

from parity import decode as D
from parity import model as M
from parity import text as T
from parity import upstream

CKPT = os.environ.get("RAVEN_ASR_CKPT")
needs_ckpt = pytest.mark.skipif(
    not CKPT or not Path(CKPT).is_file(),
    reason="set RAVEN_ASR_CKPT to the released checkpoint to run")

# Shapes quoted from the released checkpoint (577 tensors).
KNOWN_SHAPES = {
    "encoder.frontend.trunk.conv1.weight": (64, 1, 80),
    "encoder.embed.0.weight": (512, 512),
    "encoder.encoders.0.gamma_ff": (512,),
    "encoder.encoders.0.gamma_mha": (512,),
    "encoder.encoders.0.self_attn.pos_bias_u": (8, 64),
    "encoder.encoders.0.feed_forward.w_1.weight": (2048, 512),
    "encoder.encoders.11.feed_forward.w_1.weight": (2048, 512),
    "encoder.encoders.0.norm_ff.running_mean": (512,),
    "decoder.proj_decoder.weight": (256, 512),
    "decoder.embed.0.weight": (1049, 256),
    "decoder.decoders.0.feed_forward.w_1.weight": (2048, 256),
    "decoder.decoders.5.feed_forward.w_1.weight": (2048, 256),
    "decoder.output_layer.weight": (1049, 256),
    "ctc.ctc_lo.weight": (1049, 512),
}


@pytest.fixture(scope="module")
def fresh_model():
    torch.manual_seed(0)
    return M.build_model(M.load_arch_config())


# ------------------------------------------------------------ configuration --

def test_arch_config_lists_exactly_the_keys_the_constructor_reads():
    cfg = M.load_arch_config()
    assert set(cfg) == set(M.ARCH_KEYS)
    assert len(M.ARCH_KEYS) == len(set(M.ARCH_KEYS))


def test_arch_config_equals_upstream_yaml_except_the_documented_decoder_overrides():
    up = yaml.safe_load((upstream.VENDOR_ROOT / "conf/model/audio_backbone/resnet_transformer_base.yaml").read_text())
    ours = M.load_arch_config()
    interpolated = {k for k, v in up.items() if isinstance(v, str) and v.startswith("${")}
    assert interpolated == {"ddim", "dheads", "dunits"}
    for k, v in up.items():
        if k in interpolated:
            continue
        assert ours[k] == v, f"{k}: ours {ours[k]!r} != upstream {v!r}"
    assert (ours["ddim"], ours["dheads"], ours["dunits"], ours["dlayers"]) == (256, 4, 2048, 6)


@pytest.mark.parametrize("drop", ["dlayers", "relu_type", "rel_pos_type", "mask_init_type"])
def test_build_model_refuses_a_missing_key(drop):
    cfg = M.load_arch_config()
    del cfg[drop]
    with pytest.raises(ValueError, match="missing"):
        M.build_model(cfg)


def test_build_model_refuses_an_unknown_key():
    cfg = M.load_arch_config()
    cfg["surprise"] = 1
    with pytest.raises(ValueError, match="unknown"):
        M.build_model(cfg)


def test_build_model_does_not_mutate_the_callers_config():
    cfg = M.load_arch_config()
    before = copy.deepcopy(cfg)
    M.build_model(cfg)
    assert cfg == before


# ------------------------------------------------------- freshly built model --

def test_fresh_model_has_the_released_parameter_count_and_state_dict_size(fresh_model):
    assert M.count_parameters(fresh_model) == M.RELEASED_PARAM_COUNT == 55_791_794
    sd = fresh_model.state_dict()
    assert len(sd) == M.RELEASED_STATE_DICT_KEYS == 577


def test_fresh_model_component_breakdown_matches_the_checkpoint(fresh_model):
    def n(prefix):
        return sum(p.numel() for k, p in fresh_model.named_parameters() if k.startswith(prefix))
    assert n("encoder.frontend") == 3_848_576
    assert n("encoder.") - n("encoder.frontend") == 41_262_592
    assert n("decoder.") == 10_142_489
    assert n("ctc.") == 538_137


def test_fresh_model_shapes_match_known_released_shapes(fresh_model):
    sd = fresh_model.state_dict()
    for key, shape in KNOWN_SHAPES.items():
        assert tuple(sd[key].shape) == shape, key


def test_fresh_model_state_dict_fingerprint_equals_the_released_one(fresh_model):
    assert M.state_dict_fingerprint(fresh_model.state_dict()) == M.RELEASED_STATE_DICT_FINGERPRINT


def test_assert_released_architecture_passes_and_catches_a_wrong_model(fresh_model):
    M.assert_released_architecture(fresh_model)
    wrong = M.load_arch_config()
    wrong["dlayers"] = 5
    with pytest.raises(ValueError, match="architecture"):
        M.assert_released_architecture(M.build_model(wrong))


def test_token_geometry_is_blank0_sos_eos_last(fresh_model):
    assert fresh_model.odim == 1049 == T.ODIM
    assert fresh_model.blank == T.BLANK_ID == 0
    assert fresh_model.sos == fresh_model.eos == T.EOS_ID == 1048


def test_encoder_truncates_to_a_multiple_of_640_samples_and_emits_25_fps(fresh_model):
    fresh_model.eval()
    with torch.no_grad():
        feat, _ = fresh_model.encoder(torch.zeros(1, 32000, 1), None)
        assert feat.shape == (1, 50, 512)                        # 2 s -> 50 frames
        feat2, _ = fresh_model.encoder(torch.zeros(1, 32000 + 639, 1), None)
        assert feat2.shape == (1, 50, 512)                       # the 639 extra samples are dropped
        feat3, _ = fresh_model.encoder(torch.zeros(1, 32000 + 640, 1), None)
        assert feat3.shape == (1, 51, 512)


def test_eval_forward_leaves_batchnorm_statistics_untouched(fresh_model):
    fresh_model.eval()
    before = {k: v.clone() for k, v in fresh_model.state_dict().items()
              if "running_" in k or "num_batches_tracked" in k}
    with torch.no_grad():
        fresh_model.encoder(torch.randn(1, 16000, 1), None)
    after = {k: v for k, v in fresh_model.state_dict().items() if k in before}
    assert before.keys() == after.keys() and len(before) > 0
    assert all(torch.equal(before[k], after[k]) for k in before)


# ----------------------------------------------------------- checkpoint loader --

def test_load_released_checkpoint_roundtrip_from_a_synthetic_file(tmp_path, fresh_model):
    p = tmp_path / "synthetic.pth"
    torch.save(fresh_model.state_dict(), p)
    model = M.load_released_checkpoint(p)
    assert not model.training
    for k, v in fresh_model.state_dict().items():
        assert torch.equal(model.state_dict()[k], v)


def test_load_released_checkpoint_enforces_sha256_when_given(tmp_path, fresh_model):
    p = tmp_path / "synthetic.pth"
    torch.save(fresh_model.state_dict(), p)
    with pytest.raises(ValueError, match="sha256"):
        M.load_released_checkpoint(p, expect_sha256="0" * 64)
    assert M.sha256_file(p)
    M.load_released_checkpoint(p, expect_sha256=M.sha256_file(p))


def test_load_released_checkpoint_strict_rejects_missing_and_unexpected_keys(tmp_path, fresh_model):
    sd = fresh_model.state_dict()
    missing = {k: v for k, v in sd.items() if k != "ctc.ctc_lo.bias"}
    p = tmp_path / "missing.pth"
    torch.save(missing, p)
    with pytest.raises(RuntimeError, match="ctc.ctc_lo.bias"):
        M.load_released_checkpoint(p, strict=True)
    extra = dict(sd)
    extra["encoder.surprise"] = torch.zeros(1)
    p2 = tmp_path / "extra.pth"
    torch.save(extra, p2)
    with pytest.raises(RuntimeError, match="surprise"):
        M.load_released_checkpoint(p2, strict=True)


def test_load_released_checkpoint_rejects_non_state_dict_payloads(tmp_path):
    p = tmp_path / "bad.pth"
    torch.save({"state_dict": 1}, p)
    with pytest.raises(ValueError, match="state_dict"):
        M.load_released_checkpoint(p)


def test_load_released_checkpoint_uses_weights_only(tmp_path):
    """A checkpoint whose pickle would run code must be refused before it runs
    (the released file is an untrusted download)."""
    import pickle

    marker = tmp_path / "executed"

    class Evil:
        def __reduce__(self):
            return (open, (str(marker), "w"))

    p = tmp_path / "evil.pth"
    torch.save({"w": Evil()}, p)
    with pytest.raises(pickle.UnpicklingError):
        M.load_released_checkpoint(p)
    assert not marker.exists()


# ------------------------------------------------------------------ decoding --

def _synthetic_wave(seconds: float = 2.0, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return 0.1 * torch.randn(int(16000 * seconds), generator=g)


def test_beam_search_is_built_exactly_as_upstream_builds_it(fresh_model):
    tokens = T.token_list(T.load_units())
    bs = D.build_beam_search(fresh_model, tokens, beam_size=40, ctc_weight=0.1)
    assert bs.beam_size == 40 and bs.n_vocab == 1049
    assert bs.sos == bs.eos == 1048
    assert bs.weights == {"decoder": 1.0 - 0.1, "ctc": 0.1, "lm": 0.0, "length_bonus": 0.0}
    assert set(bs.scorers) == {"decoder", "ctc"}                  # lm / length_bonus weight 0 -> dropped
    assert set(bs.part_scorers) == {"ctc"} and set(bs.full_scorers) == {"decoder"}
    assert bs.pre_beam_score_key == "decoder" and bs.pre_beam_size == 60 and bs.do_pre_beam


def test_ctc_weight_one_disables_the_pre_beam_like_upstream(fresh_model):
    tokens = T.token_list(T.load_units())
    bs = D.build_beam_search(fresh_model, tokens, beam_size=4, ctc_weight=1.0)
    assert bs.pre_beam_score_key is None and not bs.do_pre_beam


def test_transcribe_refuses_a_model_left_in_train_mode(fresh_model):
    tokens = T.token_list(T.load_units())
    fresh_model.train()
    try:
        bs = D.build_beam_search(fresh_model, tokens, beam_size=2, ctc_weight=0.1)
        with pytest.raises(RuntimeError, match="eval"):
            D.transcribe(fresh_model, bs, _synthetic_wave(1.0), tokens)
    finally:
        fresh_model.eval()


def test_transcribe_refuses_audio_shorter_than_one_frame(fresh_model):
    fresh_model.eval()
    tokens = T.token_list(T.load_units())
    bs = D.build_beam_search(fresh_model, tokens, beam_size=2, ctc_weight=0.1)
    with pytest.raises(ValueError, match="640"):
        D.transcribe(fresh_model, bs, torch.zeros(639), tokens)
    with pytest.raises(ValueError, match="1-D"):
        D.transcribe(fresh_model, bs, torch.zeros(2, 1280), tokens)


def test_transcribe_matches_a_literal_replay_of_upstream_calculate_wer(fresh_model):
    """Replays finetune_learner.calculate_wer's per-utterance body statement by
    statement (shape [1, T, 1] into the encoder, maxlenratio=minlenratio=0.0, 1-best,
    add_results_to_json, <eos> removal, boundary replacement) and requires the same
    string from parity.decode.transcribe."""
    upstream.activate()
    from espnet.asr.asr_utils import add_results_to_json
    from espnet.nets.batch_beam_search import BatchBeamSearch
    from espnet.nets.scorers.length_bonus import LengthBonus

    fresh_model.eval()
    tokens = T.token_list(T.load_units())
    odim = len(tokens)
    wave = _synthetic_wave(1.5, seed=3)

    scorers = fresh_model.scorers()
    scorers["lm"] = None
    scorers["length_bonus"] = LengthBonus(odim)
    ref_search = BatchBeamSearch(
        beam_size=3, vocab_size=odim,
        weights=dict(decoder=1.0 - 0.1, ctc=0.1, lm=0.0, length_bonus=0.0),
        scorers=scorers, sos=odim - 1, eos=odim - 1, token_list=tokens,
        pre_beam_score_key="decoder")
    with torch.no_grad():
        x = wave.reshape(-1, 1).unsqueeze(0)
        feat, _ = fresh_model.encoder(x, None)
        nbest = ref_search(x=feat.squeeze(0), maxlenratio=0.0, minlenratio=0.0)
        nbest = [h.asdict() for h in nbest[: min(len(nbest), 1)]]
        expected = add_results_to_json(nbest, tokens).replace("<eos>", "").replace("▁", " ").strip()

    bs = D.build_beam_search(fresh_model, tokens, beam_size=3, ctc_weight=0.1)
    assert D.transcribe(fresh_model, bs, wave, tokens) == expected


def test_transcribe_is_deterministic_in_eval_mode(fresh_model):
    fresh_model.eval()
    tokens = T.token_list(T.load_units())
    bs = D.build_beam_search(fresh_model, tokens, beam_size=3, ctc_weight=0.1)
    wave = _synthetic_wave(1.0, seed=5)
    assert D.transcribe(fresh_model, bs, wave, tokens) == D.transcribe(fresh_model, bs, wave, tokens)


# ------------------------------------------------- released checkpoint (opt-in) --

@needs_ckpt
def test_released_checkpoint_has_the_pinned_sha256_and_size():
    assert Path(CKPT).stat().st_size == M.RELEASED_CKPT_BYTES == 223_444_287
    assert M.sha256_file(CKPT) == M.RELEASED_CKPT_SHA256


@needs_ckpt
def test_released_checkpoint_loads_strictly_and_matches_the_architecture_fingerprint():
    model = M.load_released_checkpoint(CKPT, strict=True, expect_sha256=M.RELEASED_CKPT_SHA256)
    assert not model.training
    M.assert_released_architecture(model)
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    assert len(sd) == 577
    assert M.state_dict_fingerprint(sd) == M.RELEASED_STATE_DICT_FINGERPRINT
    n_int = sum(1 for v in sd.values() if not v.is_floating_point())
    assert n_int == 32                                            # BatchNorm num_batches_tracked scalars
    assert sum(v.numel() for v in sd.values()) == 55_813_714      # = 55,791,794 params + BN running stats + 32


@needs_ckpt
def test_released_checkpoint_cpu_smoke_decode_returns_a_string():
    model = M.load_released_checkpoint(CKPT, strict=True)
    tokens = T.token_list(T.load_units())
    bs = D.build_beam_search(model, tokens, beam_size=40, ctc_weight=0.1)
    out = D.transcribe(model, bs, _synthetic_wave(2.0), tokens)
    assert isinstance(out, str)
    assert "<eos>" not in out and "▁" not in out
    assert out == D.transcribe(model, bs, _synthetic_wave(2.0), tokens)   # deterministic
