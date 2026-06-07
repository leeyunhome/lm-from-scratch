"""Count-based trigram language model with add-k smoothing.

Training is just *counting*: ``P(w | ctx) = count(ctx, w) / count(ctx)`` (MLE).
add-k smoothing spreads a little mass onto unseen continuations, and an unseen
context falls back to a uniform distribution. There is no gradient anywhere.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from lmscratch.data import Vocab


class NGramLM:
    def __init__(self, vocab_size: int, smoothing_k: float = 0.0) -> None:
        self.vocab_size = vocab_size
        self.smoothing_k = smoothing_k
        self._counts: dict[tuple[int, ...], np.ndarray] = defaultdict(
            lambda: np.zeros(vocab_size, dtype=np.float64)
        )
        self._totals: dict[tuple[int, ...], float] = defaultdict(float)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NGramLM":
        for context, word in zip(X, y):
            key = tuple(int(t) for t in context)
            self._counts[key][int(word)] += 1.0
            self._totals[key] += 1.0
        return self

    def distribution(self, context, smoothing_k: float | None = None) -> np.ndarray:
        """Probability distribution over the next token for ``context``."""
        k = self.smoothing_k if smoothing_k is None else smoothing_k
        key = tuple(int(t) for t in context)
        counts = self._counts.get(key)
        if counts is None:
            # Unseen context: nothing to count, fall back to uniform.
            return np.full(self.vocab_size, 1.0 / self.vocab_size)
        return (counts + k) / (self._totals[key] + k * self.vocab_size)

    def seen_context(self, context) -> bool:
        return tuple(int(t) for t in context) in self._counts

    def perplexity(self, X: np.ndarray, y: np.ndarray, smoothing_k: float | None = None) -> float:
        log_probs = []
        for context, word in zip(X, y):
            p = self.distribution(context, smoothing_k)
            log_probs.append(np.log(max(float(p[int(word)]), 1e-30)))
        return float(np.exp(-np.mean(log_probs)))

    def sample(self, context, rng: np.random.Generator) -> int:
        p = self.distribution(context)
        p = p / p.sum()
        return int(rng.choice(self.vocab_size, p=p))

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
