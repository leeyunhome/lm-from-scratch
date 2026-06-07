"""Neural language model: ``embedding -> tanh hidden -> softmax`` (a simplified
Bengio-2003 model), with backpropagation **derived by hand** - no autograd.

The whole point of ``loss_and_grads`` is that the forward pass and every
gradient sit in one function, top to bottom, so you can read the chain rule as
code. Shapes (B=batch, C=context, D=embedding, H=hidden, V=vocab):

    X     [B, C]    -> embedding lookup
    e     [B, C, D] -> flatten
    flat  [B, C*D]  -> @ W1 + b1, tanh
    h     [B, H]    -> @ W2 + b2
    z     [B, V]    -> softmax
    p     [B, V]    -> cross-entropy -> scalar loss

Correctness is verified by :func:`gradient_check` against finite differences.
"""

from __future__ import annotations

import numpy as np

from lmscratch.data import Vocab


def softmax(z: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class NeuralLM:
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
        # Xavier-ish init, friendly to tanh.
        self.E = rng.normal(0, 0.5, (self.V, self.D))
        self.W1 = rng.normal(0, np.sqrt(2.0 / (self.C * self.D + self.H)), (self.C * self.D, self.H))
        self.b1 = np.zeros(self.H)
        self.W2 = rng.normal(0, np.sqrt(2.0 / (self.H + self.V)), (self.H, self.V))
        self.b2 = np.zeros(self.V)

    @property
    def params(self) -> dict[str, np.ndarray]:
        return {"E": self.E, "W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    # ------------------------------------------------------------------ forward
    def _forward(self, X: np.ndarray):
        B = X.shape[0]
        e = self.E[X]                                  # [B, C, D]  embedding lookup
        flat = e.reshape(B, self.C * self.D)           # [B, C*D]   concatenate
        h = np.tanh(flat @ self.W1 + self.b1)          # [B, H]     hidden
        z = h @ self.W2 + self.b2                      # [B, V]     logits
        p = softmax(z)                                 # [B, V]     probabilities
        return flat, h, p

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._forward(X)[2]

    # --------------------------------------------------- forward + hand backprop
    def loss_and_grads(self, X: np.ndarray, y: np.ndarray):
        B = X.shape[0]
        flat, h, p = self._forward(X)

        # cross-entropy loss = mean -log p[correct]
        loss = float(-np.log(np.maximum(p[np.arange(B), y], 1e-30)).mean())

        # dL/dz for softmax + cross-entropy collapses to (p - onehot(y)) / B
        dz = p.copy()
        dz[np.arange(B), y] -= 1.0
        dz /= B                                        # [B, V]

        gW2 = h.T @ dz                                 # [H, V]
        gb2 = dz.sum(axis=0)                           # [V]

        dh = dz @ self.W2.T                            # [B, H]
        dpre = dh * (1.0 - h * h)                      # tanh'(x) = 1 - tanh(x)^2

        gW1 = flat.T @ dpre                            # [C*D, H]
        gb1 = dpre.sum(axis=0)                         # [H]

        dflat = dpre @ self.W1.T                       # [B, C*D]
        gE = np.zeros_like(self.E)                     # [V, D]
        np.add.at(gE, X, dflat.reshape(B, self.C, self.D))  # scatter-add into rows

        return loss, {"E": gE, "W1": gW1, "b1": gb1, "W2": gW2, "b2": gb2}

    # --------------------------------------------------------------------- train
    def fit(self, X: np.ndarray, y: np.ndarray, lr: float = 0.3, epochs: int = 3000) -> list[float]:
        history: list[float] = []
        for _ in range(epochs):
            loss, grads = self.loss_and_grads(X, y)
            for name, param in self.params.items():
                param -= lr * grads[name]              # full-batch gradient descent
            history.append(loss)
        return history

    # ---------------------------------------------------------------- inference
    def perplexity(self, X: np.ndarray, y: np.ndarray) -> float:
        p = self.predict_proba(X)
        pw = np.maximum(p[np.arange(len(y)), y], 1e-30)
        return float(np.exp(-np.log(pw).mean()))

    def sample(self, context, rng: np.random.Generator) -> int:
        p = self.predict_proba(np.array([list(context)], dtype=np.int64))[0]
        p = p / p.sum()
        return int(rng.choice(self.V, p=p))

    def generate(self, vocab: Vocab, rng: np.random.Generator, max_len: int = 20) -> list[str]:
        context = [vocab.pad, vocab.bos]
        out: list[int] = []
        for _ in range(max_len):
            word = self.sample(tuple(context), rng)
            if word == vocab.eos:
                break
            out.append(word)
            context = [context[-1], word]
        return [vocab.decode(t) for t in out]


def gradient_check(seed: int = 0, eps: float = 1e-5) -> dict[str, float]:
    """Compare hand-derived gradients against central finite differences.

    Returns the worst absolute difference per parameter; all values should be
    ~1e-9 or smaller for a correct implementation.
    """
    rng = np.random.default_rng(seed)
    V, D, H, C, B = 7, 3, 5, 2, 4
    X = rng.integers(0, V, size=(B, C))
    y = rng.integers(0, V, size=(B,))

    model = NeuralLM(V, D, H, C, seed=42)
    _, grads = model.loss_and_grads(X, y)

    worst: dict[str, float] = {}
    for name, param in model.params.items():
        numerical = np.zeros_like(param)
        it = np.nditer(param, flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            param[idx] += eps
            loss_plus, _ = model.loss_and_grads(X, y)
            param[idx] -= 2 * eps
            loss_minus, _ = model.loss_and_grads(X, y)
            param[idx] += eps
            numerical[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()
        worst[name] = float(np.abs(numerical - grads[name]).max())
    return worst
