"""Minimal tensor autograd engine — pure NumPy, no external dependencies.

Every ``Tensor`` wraps a NumPy array and records a ``_backward`` closure that
propagates gradients one step up the computation graph.  Calling
``loss.backward()`` does a reverse topological traversal and accumulates
``grad`` on every leaf.

Supported ops (enough to run the Bengio-style NeuralLM):
    matmul  (@)       embedding_lookup   tanh
    add     (+)       reshape            cross_entropy
    mean

The design mirrors Karpathy's micrograd but works on NumPy arrays instead of
scalars — so every op handles batching and broadcasting natively.
"""

from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- #
#  Broadcasting helper                                                         #
# --------------------------------------------------------------------------- #

def _unbroadcast(grad: np.ndarray, shape: tuple) -> np.ndarray:
    """Sum ``grad`` over any axes that were broadcast to match ``shape``."""
    # Leading dims that do not exist in the original shape
    for _ in range(grad.ndim - len(shape)):
        grad = grad.sum(axis=0)
    # Axes where shape has size 1 but grad does not
    for axis, size in enumerate(shape):
        if size == 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


# --------------------------------------------------------------------------- #
#  Tensor                                                                      #
# --------------------------------------------------------------------------- #

class Tensor:
    """A NumPy array that records its computation history for backprop."""

    def __init__(
        self,
        data,
        _children: tuple["Tensor", ...] = (),
        label: str = "",
    ) -> None:
        self.data: np.ndarray = np.asarray(data, dtype=np.float64)
        self.grad: np.ndarray = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self.label = label

    # ---------------------------------------------------------------- helpers

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, label={self.label!r})"

    # --------------------------------------------------------------- forward ops

    def __matmul__(self, other: "Tensor") -> "Tensor":
        out = Tensor(self.data @ other.data, (self, other))

        def _backward() -> None:
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def __add__(self, other: "Tensor") -> "Tensor":
        if not isinstance(other, Tensor):
            other = Tensor(other)
        out = Tensor(self.data + other.data, (self, other))

        def _backward() -> None:
            self.grad += _unbroadcast(out.grad, self.data.shape)
            other.grad += _unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __radd__(self, other) -> "Tensor":
        return self.__add__(other)

    def tanh(self) -> "Tensor":
        t = np.tanh(self.data)
        out = Tensor(t, (self,))

        def _backward() -> None:
            self.grad += (1.0 - t * t) * out.grad

        out._backward = _backward
        return out

    def reshape(self, *shape) -> "Tensor":
        orig_shape = self.data.shape
        out = Tensor(self.data.reshape(*shape), (self,))

        def _backward() -> None:
            self.grad += out.grad.reshape(orig_shape)

        out._backward = _backward
        return out

    # --------------------------------------------------------------- backward

    def backward(self) -> None:
        """Reverse-mode autodiff from this (scalar or array) node."""
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build(v: Tensor) -> None:
            if id(v) not in visited:
                visited.add(id(v))
                for child in v._prev:
                    build(child)
                topo.append(v)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()


# --------------------------------------------------------------------------- #
#  Standalone ops that operate on Tensors but need extra (non-Tensor) args    #
# --------------------------------------------------------------------------- #

def embedding_lookup(E: Tensor, idx: np.ndarray) -> Tensor:
    """Gather rows of ``E`` indexed by integer array ``idx``.

    ``idx`` carries no gradient (it is an integer index).  The backward
    scatter-adds ``out.grad`` back into the rows of ``E.grad``.

    Args:
        E:   Embedding matrix Tensor of shape ``(V, D)``.
        idx: Integer array of shape ``(B, C)`` (or any integer array).

    Returns:
        Tensor of shape ``(*idx.shape, D)``.
    """
    out = Tensor(E.data[idx], (E,))

    def _backward() -> None:
        np.add.at(E.grad, idx, out.grad)

    out._backward = _backward
    return out


def cross_entropy(logits: Tensor, y: np.ndarray) -> Tensor:
    """Numerically stable softmax cross-entropy, averaged over the batch.

    Combines softmax and NLL into one op for numerical stability — the
    ``-log(softmax(z))`` simplification cancels the exp/log.

    Args:
        logits: ``[B, V]`` pre-softmax scores.
        y:      ``[B]`` integer class labels.

    Returns:
        Scalar Tensor (mean loss).
    """
    z = logits.data - logits.data.max(axis=-1, keepdims=True)
    exp_z = np.exp(z)
    probs = exp_z / exp_z.sum(axis=-1, keepdims=True)
    B = len(y)
    loss_val = float(-np.log(np.maximum(probs[np.arange(B), y], 1e-30)).mean())
    out = Tensor(loss_val, (logits,))

    def _backward() -> None:
        dl_dz = probs.copy()
        dl_dz[np.arange(B), y] -= 1.0
        dl_dz /= B
        logits.grad += dl_dz * out.grad

    out._backward = _backward
    return out
