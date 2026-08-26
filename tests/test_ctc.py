"""CTC layer: greedy decode collapse semantics, §3.12 input-length rule,
jiwer scoring primitives and the zero_infinity safety net."""

import torch

import ctc as ctc_lib
from vocab import build_char_vocab


V = len(build_char_vocab())                              # frozen alphabet: 30


def _log_probs_from_ids(id_seq: list[list[int]]) -> torch.Tensor:
    """One-hot-ish log-prob tensor [B,T,V] whose argmax reproduces ``id_seq``."""
    tmax = max(len(s) for s in id_seq)
    lp = torch.full((len(id_seq), tmax, V), -10.0)
    for i, s in enumerate(id_seq):
        for t, tok in enumerate(s):
            lp[i, t, tok] = 0.0
    return lp


def _ids(text: str, vocab=build_char_vocab()) -> list[int]:
    return vocab.encode(text, warn_unknown=False)


def test_greedy_decode_all_blanks_is_empty():
    vocab = build_char_vocab()
    blank = vocab.blank_id
    lp = _log_probs_from_ids([[blank] * 7])
    hyps = ctc_lib.greedy_decode(lp, torch.tensor([7]), vocab)
    assert hyps == [""]


def test_greedy_decode_repeats_collapse_blanks_break():
    """CTC semantics via the frozen vocab decode: [a,a,a,b,b]→"ab", but a blank
    interrupts the run so [a,blank,a]→"aa"."""
    vocab = build_char_vocab()
    a = vocab.stoi["a"]
    b = vocab.stoi["b"]
    blank = vocab.blank_id

    seqs = [[a, a, a, b, b], [a, blank, a]]
    lengths = torch.tensor([5, 3])
    hyps = ctc_lib.greedy_decode(_log_probs_from_ids(seqs), lengths, vocab)
    assert hyps == ["ab", "aa"]


def test_greedy_decode_respects_true_lengths():
    """Frames beyond the true input length must never leak into hypotheses."""
    vocab = build_char_vocab()
    c = vocab.stoi["c"]
    seq = [vocab.stoi["x"]] + [c] * 4                    # trailing garbage past length=1
    lp = _log_probs_from_ids([seq])
    hyps = ctc_lib.greedy_decode(lp, torch.tensor([1]), vocab)
    assert hyps == ["x"]


def test_input_length_keep_mask_drops_rule_312_rows():
    keep = ctc_lib.input_length_keep_mask(
        tok_lengths=torch.tensor([10, 3, 7]),
        tgt_lengths=torch.tensor([5, 4, 8]),
    )
    assert keep.tolist() == [True, False, False]


def test_ctc_loss_per_utt_shapes_and_zero_infinity():
    """reduction='none' gives per-utterance vectors; an ILLEGAL alignment
    (in_len < tgt_len) yields loss 0 via zero_infinity instead of NaN/inf."""
    torch.manual_seed(0)
    vocab = build_char_vocab()
    bsz, t_in, v = 2, 8, V
    raw = torch.randn(bsz, t_in, v)
    log_probs = raw.log_softmax(dim=-1).transpose(0, 1)  # [T,B,V]

    pad = vocab.pad_id
    row0 = _ids("cat")                                  # 3 ids
    row1 = _ids("elephant")                             # 8 ids
    width = max(len(row0), len(row1))
    targets = torch.tensor([row0 + [pad] * (width - len(row0)),
                            row1 + [pad] * (width - len(row1))], dtype=torch.int64)
    tgt_lens = torch.tensor([3, 8])
    in_lens = torch.tensor([8, 8])
    losses = ctc_lib.ctc_loss_per_utt(log_probs, targets, in_lens, tgt_lens,
                                      vocab.blank_id)
    assert losses.shape == (2,)
    assert all(torch.isfinite(losses))

    illegal_in_lens = torch.tensor([8, 5])               # 2nd row now has no valid path
    bad = ctc_lib.ctc_loss_per_utt(log_probs, targets, illegal_in_lens, tgt_lens,
                                   vocab.blank_id)
    assert float(bad[1]) == 0.0                          # zero_infinity swallowed it


def test_wer_cer_primitives():
    assert ctc_lib.wer_one("hello world", "hello world") == 0.0
    # insertion: 1 edit / REFERENCE word count (jiwer denominator = len(ref.split()))
    assert abs(ctc_lib.wer_one("a b", "a b c") - 0.5) < 1e-9
    # deletion mirrors at the same rule
    assert abs(ctc_lib.wer_one("a b c", "a b") - 1 / 3) < 1e-9
    # substitution
    assert abs(ctc_lib.wer_one("a b", "a z") - 0.5) < 1e-9

    assert ctc_lib.cer_one("abc", "abc") == 0.0
    assert abs(ctc_lib.cer_one("abc", "abd") - 1 / 3) < 1e-9   # one substitution
    # insertion into a 3-char reference: jiwer CER divides by REFERENCE length
    assert abs(ctc_lib.cer_one("abc", "abcd") - 1 / 3) < 1e-9


def test_compute_batch_means_and_per_utterance():
    refs = ["one two three", "four five"]
    hyps = ["one two three", "four six"]
    mean_wer, per_wer = ctc_lib.compute_wer(refs, hyps)
    assert per_wer == [0.0, 0.5]
    assert abs(mean_wer - 0.25) < 1e-9

    mean_cer, per_cer = ctc_lib.compute_cer(refs, hyps)
    assert abs(per_cer[1] - ctc_lib.cer_one("four five", "four six")) < 1e-9
    assert abs(mean_cer - sum(per_cer) / 2) < 1e-12


def test_compute_empty_inputs_safe():
    mean_wer, per_wer = ctc_lib.compute_wer([], [])
    assert mean_wer == 0.0 and per_wer == []


def test_encode_decode_roundtrip_with_apostrophe():
    vocab = build_char_vocab()
    text = "don't stop me"
    ids = ctc_lib.ids_from_text(text, vocab)
    assert vocab.decode(ids, collapse_repeats=False) == text
    # whitespace normalization is upstream of encode — encode keeps literal spaces
    assert vocab.pad_id not in ids and vocab.blank_id not in ids


def test_char_accuracy_against_decoded_refs():
    vocab = build_char_vocab()
    ref_text = "abc def"
    ref_ids = [_ids(ref_text)]
    perfect = [ref_text]
    assert abs(ctc_lib.char_accuracy(ref_ids, perfect, vocab) - 1.0) < 1e-9

    wrong = ["xbc dex"]
    acc = ctc_lib.char_accuracy(ref_ids, wrong, vocab)
    assert 0.0 < acc < 1.0

    garbage = ["zzzzz zz"]                               # maximally wrong-ish
    acc_bad = ctc_lib.char_accuracy(ref_ids, garbage, vocab)
    assert 0.0 <= acc_bad < acc
