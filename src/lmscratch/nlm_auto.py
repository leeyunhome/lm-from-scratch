"""NeuralLM reimplemented with the autograd engine.

``NeuralLMAuto`` has the same architecture (embedding → tanh → softmax) and
the same hyper-parameter interface as ``NeuralLM``, but the backward pass is
computed automatically by ``autograd.Tensor`` instead of by hand.

The main use of this module is the ``gradient_compare`` function, which trains
both variants from the same random initialisation and checks that the gradients
they produce are bit-for-bit equivalent — confirming both implementations are
correct.
"""

from __future__ import annotations

import numpy as np

from lmscratch.autograd import Tensor, cross_entropy, embedding_lookup
from lmscratch.data import Vocab


class NeuralLMAuto:
    """Bengio-style trigram LM whose backward pass uses the autograd engine."""

    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 2,
        hidden: int = 8,
        context_len: int = 2,
        seed: int = 123,
    ) -> None:
        self.V, self.D, self.H, self.C = vocab_size, emb_dim, hidden, context_len
        rng = np.random.default_rng(seed)
        self.E  = Tensor(rng.normal(0, 0.5, (self.V, self.D)),          label="E")
        self.W1 = Tensor(rng.normal(0, np.sqrt(2.0 / (self.C * self.D + self.H)),
                                    (self.C * self.D, self.H)),           label="W1")
        self.b1 = Tensor(np.zeros(self.H),                               label="b1")
        self.W2 = Tensor(rng.normal(0, np.sqrt(2.0 / (self.H + self.V)),
                                    (self.H, self.V)),                    label="W2")
        self.b2 = Tensor(np.zeros(self.V),                               label="b2")

    @property
    def params(self) -> list[Tensor]:
        return [self.E, self.W1, self.b1, self.W2, self.b2]

    # ------------------------------------------------------------------ forward

    def forward_loss(self, X: np.ndarray, y: np.ndarray) -> Tensor:
        B = X.shape[0]
        e    = embedding_lookup(self.E, X)           # [B, C, D]
        flat = e.reshape(B, self.C * self.D)         # [B, C*D]
        h    = (flat @ self.W1 + self.b1).tanh()     # [B, H]
        z    = h @ self.W2 + self.b2                 # [B, V]
        return cross_entropy(z, y)                   # scalar

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        B = X.shape[0]
        e    = self.E.data[X]
        flat = e.reshape(B, self.C * self.D)
        h    = np.tanh(flat @ self.W1.data + self.b1.data)
        z    = h @ self.W2.data + self.b2.data
        z    = z - z.max(axis=-1, keepdims=True)
        exp_z = np.exp(z)
        return exp_z / exp_z.sum(axis=-1, keepdims=True)

    # --------------------------------------------------------------------- train

    def fit(
        self, X: np.ndarray, y: np.ndarray, lr: float = 0.3, epochs: int = 3000
    ) -> list[float]:
        history: list[float] = []
        for _ in range(epochs):
            for p in self.params:
                p.zero_grad()
            loss = self.forward_loss(X, y)
            loss.backward()
            for p in self.params:
                p.data -= lr * p.grad
            history.append(float(loss.data))
        return history

    # ---------------------------------------------------------------- inference

    def perplexity(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self.predict_proba(X)
        pw = np.maximum(p[np.arange(len(y)), y], 1e-30)
        return float(np.exp(-np.log(pw).mean()))

    def generate(self, vocab: Vocab, rng: np.random.Generator, max_len: int = 20) -> list[str]:
        context = [vocab.pad, vocab.bos]
        out: list[int] = []
        for _ in range(max_len):
            p = self.predict_proba(np.array([context], dtype=np.int64))[0]
            word = int(rng.choice(self.V, p=p / p.sum()))
            if word == vocab.eos:
                break
            out.append(word)
            context = [context[-1], word]
        return [vocab.decode(t) for t in out]


# --------------------------------------------------------------------------- #
#  Gradient comparison                                                         #
# --------------------------------------------------------------------------- #

def gradient_compare(seed: int = 0) -> dict[str, float]:
    """Return the max absolute difference between hand-derived and autograd grads.

    Both models are initialised from the same seed and evaluated on the same
    random batch, so their gradients must match to floating-point precision
    (~1e-15) if both implementations are correct.
    """
    from lmscratch.nlm import NeuralLM

    rng = np.random.default_rng(seed)
    V, D, H, C, B = 7, 3, 5, 2, 4
    X = rng.integers(0, V, size=(B, C))
    y = rng.integers(0, V, size=(B,))

    hand = NeuralLM(V, D, H, C, seed=42)
    _, g_hand = hand.loss_and_grads(X, y)

    auto = NeuralLMAuto(V, D, H, C, seed=42)
    auto.forward_loss(X, y).backward()
    g_auto = {"E": auto.E.grad, "W1": auto.W1.grad,
               "b1": auto.b1.grad, "W2": auto.W2.grad, "b2": auto.b2.grad}

    return {name: float(np.abs(g_hand[name] - g_auto[name]).max())
            for name in g_hand}
