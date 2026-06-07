import numpy as np

from lmscratch.data import Vocab, encode_pairs, make_training_pairs
from lmscratch.ngram import NGramLM


def _fit_model():
    vocab = Vocab.from_corpus()
    X, y = encode_pairs(make_training_pairs(), vocab)
    return NGramLM(vocab_size=len(vocab)).fit(X, y), vocab


def test_distribution_sums_to_one():
    model, vocab = _fit_model()
    for k in (0.0, 0.01, 0.5):
        p = model.distribution((vocab.bos, vocab.encode("alice")), smoothing_k=k)
        assert np.isclose(p.sum(), 1.0)


def test_mle_matches_hand_count():
    # After <bos>, the subjects appear: alice x2, bob x3, carol x1  -> totals 6.
    model, vocab = _fit_model()
    p = model.distribution((vocab.pad, vocab.bos), smoothing_k=0.0)
    assert np.isclose(p[vocab.encode("alice")], 2 / 6)
    assert np.isclose(p[vocab.encode("bob")], 3 / 6)
    assert np.isclose(p[vocab.encode("carol")], 1 / 6)


def test_unseen_context_is_uniform():
    model, vocab = _fit_model()
    ctx = (vocab.encode("alice"), vocab.encode("banana"))  # never seen
    assert not model.seen_context(ctx)
    p = model.distribution(ctx)
    assert np.allclose(p, 1.0 / len(vocab))


def test_mle_perplexity_is_finite_on_train():
    model, vocab = _fit_model()
    X, y = encode_pairs(make_training_pairs(), vocab)
    assert np.isfinite(model.perplexity(X, y))
