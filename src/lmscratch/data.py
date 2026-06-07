"""Toy corpus, vocabulary, and trigram training-pair construction.

The corpus is a tiny toy grammar of the form ``<subject> <item> <verdict>`` -
read it as terse review notes, e.g. "alice apple good" = "alice finds apples
good". The verdict (the token we predict) depends on the (subject, item) pair:

    alice -> {apple, berry}          good
    bob   -> {apple, berry, banana}  good
    carol -> {banana}                bad

The order matters: the *verdict is the last token*, so the model learns the
schema ``(subject, item) -> verdict``. Crucially, ``(alice, banana)`` never
appears in training, yet a model that learns *distributed representations*
should still predict ``good``: ``alice`` behaves like ``bob`` (both give "good"),
and ``banana`` is an item like the others. A counting model has nothing to say
about that unseen pair - that single held-out context is the crux of the project.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
SPECIAL_TOKENS: tuple[str, ...] = (PAD, BOS, EOS)

CORPUS: tuple[str, ...] = (
    "alice apple good",
    "alice berry good",
    "bob apple good",
    "bob berry good",
    "bob banana good",
    "carol banana bad",
)

CONTEXT_LEN = 2  # trigram: predict the next token from the previous two


@dataclass(frozen=True)
class Vocab:
    """Bidirectional token <-> id map with the special tokens fixed at 0, 1, 2."""

    itos: tuple[str, ...]
    stoi: dict[str, int]

    @classmethod
    def from_corpus(cls, sentences: tuple[str, ...] = CORPUS) -> "Vocab":
        words: list[str] = []
        for sentence in sentences:
            for word in sentence.split():
                if word not in words:
                    words.append(word)
        itos = tuple(SPECIAL_TOKENS) + tuple(words)
        return cls(itos=itos, stoi={tok: i for i, tok in enumerate(itos)})

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, token: str) -> int:
        return self.stoi[token]

    def decode(self, idx: int) -> str:
        return self.itos[idx]

    @property
    def pad(self) -> int:
        return self.stoi[PAD]

    @property
    def bos(self) -> int:
        return self.stoi[BOS]

    @property
    def eos(self) -> int:
        return self.stoi[EOS]


def make_training_pairs(
    sentences: tuple[str, ...] = CORPUS, context_len: int = CONTEXT_LEN
) -> list[tuple[tuple[str, ...], str]]:
    """Wrap each sentence as ``[<pad> x (C-1)] <bos> w1 .. wn <eos>`` and slide a
    window of length ``context_len`` to yield ``(context, next_token)`` pairs."""
    pairs: list[tuple[tuple[str, ...], str]] = []
    for sentence in sentences:
        tokens = [PAD] * (context_len - 1) + [BOS] + sentence.split() + [EOS]
        for i in range(len(tokens) - context_len):
            context = tuple(tokens[i : i + context_len])
            nxt = tokens[i + context_len]
            pairs.append((context, nxt))
    return pairs


def encode_pairs(
    pairs: list[tuple[tuple[str, ...], str]], vocab: Vocab
) -> tuple[np.ndarray, np.ndarray]:
    """Turn string pairs into integer arrays ``X[N, context_len]``, ``y[N]``."""
    X = np.array([[vocab.encode(t) for t in ctx] for ctx, _ in pairs], dtype=np.int64)
    y = np.array([vocab.encode(nxt) for _, nxt in pairs], dtype=np.int64)
    return X, y
