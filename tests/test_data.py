import pytest

from lmscratch.data import (
    BOS,
    CONTEXT_LEN,
    CORPUS,
    CORPUS_LARGE,
    EOS,
    PAD,
    Vocab,
    encode_pairs,
    make_training_pairs,
    train_val_split,
)


def test_special_tokens_fixed_at_0_1_2():
    vocab = Vocab.from_corpus()
    assert (vocab.encode(PAD), vocab.encode(BOS), vocab.encode(EOS)) == (0, 1, 2)


def test_vocab_roundtrip():
    vocab = Vocab.from_corpus()
    for i, tok in enumerate(vocab.itos):
        assert vocab.encode(tok) == i
        assert vocab.decode(i) == tok


def test_pair_count_matches_formula():
    # each sentence -> (len(words) + 1) pairs  (the +1 is the <eos> target)
    pairs = make_training_pairs()
    expected = sum(len(s.split()) + 1 for s in CORPUS)
    assert len(pairs) == expected


def test_encoded_shapes_and_range():
    vocab = Vocab.from_corpus()
    X, y = encode_pairs(make_training_pairs(), vocab)
    assert X.shape == (len(make_training_pairs()), CONTEXT_LEN)
    assert y.shape[0] == X.shape[0]
    assert X.min() >= 0 and X.max() < len(vocab)
    assert y.min() >= 0 and y.max() < len(vocab)


# --- train_val_split tests ------------------------------------------------

def test_split_sizes_sum_to_total():
    train, val = train_val_split(CORPUS_LARGE, val_frac=0.2, seed=0)
    assert len(train) + len(val) == len(CORPUS_LARGE)


def test_split_no_overlap():
    train, val = train_val_split(CORPUS_LARGE, val_frac=0.2, seed=0)
    assert set(train).isdisjoint(set(val))


def test_split_val_size_respects_frac():
    train, val = train_val_split(CORPUS_LARGE, val_frac=0.2, seed=0)
    expected = round(len(CORPUS_LARGE) * 0.2)
    assert len(val) == expected


def test_split_deterministic():
    a_tr, a_val = train_val_split(CORPUS_LARGE, val_frac=0.2, seed=42)
    b_tr, b_val = train_val_split(CORPUS_LARGE, val_frac=0.2, seed=42)
    assert a_tr == b_tr and a_val == b_val


def test_split_different_seeds_differ():
    _, val0 = train_val_split(CORPUS_LARGE, val_frac=0.2, seed=0)
    _, val1 = train_val_split(CORPUS_LARGE, val_frac=0.2, seed=1)
    assert val0 != val1


def test_split_invalid_frac_raises():
    with pytest.raises(ValueError):
        train_val_split(CORPUS_LARGE, val_frac=0.0)
    with pytest.raises(ValueError):
        train_val_split(CORPUS_LARGE, val_frac=1.0)
