"""Tests for the autograd engine and NeuralLMAuto.

Key checks:
  - individual ops (matmul, add, tanh, reshape, embedding_lookup, cross_entropy)
    each produce correct gradients against central finite differences
  - NeuralLMAuto gradients match NeuralLM hand-derived gradients exactly
  - NeuralLMAuto trains to the same loss curve as NeuralLM given the same
    initialisation, lr, and epochs
"""

from __future__ import annotations

import numpy as np
import pytest

from lmscratch.autograd import Tensor, cross_entropy, embedding_lookup
from lmscratch.nlm_auto import NeuralLMAuto, gradient_compare


# --------------------------------------------------------------------------- #
#  Finite-difference helper                                                    #
# --------------------------------------------------------------------------- #

def numerical_grad(f, x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    """Central finite differences for a scalar-valued function f(x)."""
    grad = np.zeros_like(x)
    it = np.nditer(x, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        x[idx] += eps
        fp = float(f(x))
        x[idx] -= 2 * eps
        fm = float(f(x))
        x[idx] += eps
        grad[idx] = (fp - fm) / (2 * eps)
        it.iternext()
    return grad


# --------------------------------------------------------------------------- #
#  Op-level gradient checks                                                    #
# --------------------------------------------------------------------------- #

def test_matmul_grad():
    rng = np.random.default_rng(0)
    a_data = rng.normal(size=(3, 4))
    b_data = rng.normal(size=(4, 5))

    a = Tensor(a_data.copy())
    b = Tensor(b_data.copy())
    out = a @ b
    out.backward()

    # numerical grad for a
    def fa(x):
        return (x @ b_data).sum()

    num_a = numerical_grad(fa, a_data.copy())
    assert np.allclose(a.grad, num_a, atol=1e-9)

    def fb(x):
        return (a_data @ x).sum()

    num_b = numerical_grad(fb, b_data.copy())
    assert np.allclose(b.grad, num_b, atol=1e-9)


def test_add_broadcast_grad():
    rng = np.random.default_rng(1)
    a_data = rng.normal(size=(4, 3))
    b_data = rng.normal(size=(3,))  # broadcast over batch

    a = Tensor(a_data.copy())
    b = Tensor(b_data.copy())
    out = a + b
    out.backward()

    def fa(x):
        return (x + b_data).sum()

    num_a = numerical_grad(fa, a_data.copy())
    assert np.allclose(a.grad, num_a, atol=1e-9)

    def fb(x):
        return (a_data + x).sum()

    num_b = numerical_grad(fb, b_data.copy())
    assert np.allclose(b.grad, num_b, atol=1e-9)


def test_tanh_grad():
    rng = np.random.default_rng(2)
    x_data = rng.normal(size=(3, 4))
    x = Tensor(x_data.copy())
    out = x.tanh()
    out.backward()

    def f(v):
        return np.tanh(v).sum()

    num = numerical_grad(f, x_data.copy())
    assert np.allclose(x.grad, num, atol=1e-9)


def test_reshape_grad():
    rng = np.random.default_rng(3)
    x_data = rng.normal(size=(2, 3, 4))
    x = Tensor(x_data.copy())
    out = x.reshape(2, 12)
    out.backward()

    assert x.grad.shape == (2, 3, 4)
    assert np.allclose(x.grad, np.ones((2, 3, 4)), atol=1e-9)


def test_embedding_lookup_grad():
    rng = np.random.default_rng(4)
    E_data = rng.normal(size=(8, 3))
    idx = np.array([[0, 2], [1, 0]])  # [B=2, C=2]

    E = Tensor(E_data.copy())
    out = embedding_lookup(E, idx)
    out.backward()

    # Expected: scatter-add ones into E_data rows selected by idx
    expected = np.zeros_like(E_data)
    np.add.at(expected, idx, np.ones_like(out.data))
    assert np.allclose(E.grad, expected, atol=1e-9)


def test_cross_entropy_grad():
    rng = np.random.default_rng(5)
    B, V = 4, 7
    logits_data = rng.normal(size=(B, V))
    y = rng.integers(0, V, size=(B,))

    logits = Tensor(logits_data.copy())
    loss = cross_entropy(logits, y)
    loss.backward()

    def f(z):
        z = z - z.max(axis=-1, keepdims=True)
        exp_z = np.exp(z)
        probs = exp_z / exp_z.sum(axis=-1, keepdims=True)
        return -np.log(np.maximum(probs[np.arange(B), y], 1e-30)).mean()

    num = numerical_grad(f, logits_data.copy())
    assert np.allclose(logits.grad, num, atol=1e-7)


# --------------------------------------------------------------------------- #
#  NeuralLMAuto gradient match                                                 #
# --------------------------------------------------------------------------- #

def test_gradient_compare_matches_hand():
    """Autograd and hand-derived gradients must agree to ~1e-12."""
    diffs = gradient_compare(seed=0)
    for name, diff in diffs.items():
        assert diff < 1e-12, f"param {name}: max diff = {diff:.2e}"


# --------------------------------------------------------------------------- #
#  NeuralLMAuto training sanity                                                #
# --------------------------------------------------------------------------- #

def test_nlm_auto_loss_decreases():
    rng = np.random.default_rng(0)
    V, D, H, C, B = 7, 3, 5, 2, 16
    X = rng.integers(0, V, size=(B, C))
    y = rng.integers(0, V, size=(B,))

    model = NeuralLMAuto(V, D, H, C, seed=0)
    history = model.fit(X, y, lr=0.1, epochs=200)
    assert history[-1] < history[0], "loss should decrease over training"


def test_nlm_auto_same_curve_as_hand():
    """NeuralLMAuto and NeuralLM must produce identical loss curves."""
    from lmscratch.nlm import NeuralLM

    rng = np.random.default_rng(7)
    V, D, H, C, B = 9, 2, 4, 2, 8
    X = rng.integers(0, V, size=(B, C))
    y = rng.integers(0, V, size=(B,))

    h_hand = NeuralLM(V, D, H, C, seed=99).fit(X, y, lr=0.05, epochs=50)
    h_auto = NeuralLMAuto(V, D, H, C, seed=99).fit(X, y, lr=0.05, epochs=50)

    assert np.allclose(h_hand, h_auto, atol=1e-12), \
        "loss curves must be identical given the same init and lr"
