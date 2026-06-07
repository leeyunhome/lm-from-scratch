import numpy as np

from lmscratch.data import Vocab, encode_pairs, make_training_pairs
from lmscratch.nlm import NeuralLM, gradient_check, softmax


def test_gradient_check_matches_finite_differences():
    """The headline correctness guarantee: hand-derived grads == numerical."""
    worst = gradient_check(seed=0)
    for name, diff in worst.items():
        assert diff < 1e-6, f"gradient mismatch for {name}: {diff:.2e}"


def test_softmax_rows_sum_to_one():
    z = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]])
    p = softmax(z)
    assert np.allclose(p.sum(axis=-1), 1.0)


def test_training_reduces_loss():
    vocab = Vocab.from_corpus()
    X, y = encode_pairs(make_training_pairs(), vocab)
    model = NeuralLM(vocab_size=len(vocab), emb_dim=2, hidden=8, seed=123)
    history = model.fit(X, y, lr=0.3, epochs=300)
    assert history[-1] < history[0]


def test_generalizes_to_held_out_context():
    """The crux: P(good | alice, banana) should be high though never trained."""
    vocab = Vocab.from_corpus()
    X, y = encode_pairs(make_training_pairs(), vocab)
    model = NeuralLM(vocab_size=len(vocab), emb_dim=2, hidden=8, seed=123)
    model.fit(X, y, lr=0.3, epochs=3000)
    ctx = np.array([[vocab.encode("alice"), vocab.encode("banana")]], dtype=np.int64)
    p = model.predict_proba(ctx)[0]
    assert p[vocab.encode("good")] > 0.8
