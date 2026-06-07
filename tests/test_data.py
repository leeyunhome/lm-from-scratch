from lmscratch.data import (
    BOS,
    CONTEXT_LEN,
    CORPUS,
    EOS,
    PAD,
    Vocab,
    encode_pairs,
    make_training_pairs,
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
