"""Headline comparison: n-gram vs neural LM on the held-out context.

Run with ``make demo`` or ``python examples/compare.py``.
"""

from __future__ import annotations

import numpy as np

from lmscratch.data import Vocab, encode_pairs, make_training_pairs
from lmscratch.ngram import NGramLM
from lmscratch.nlm import NeuralLM, gradient_check


def main() -> None:
    vocab = Vocab.from_corpus()
    X, y = encode_pairs(make_training_pairs(), vocab)

    print("=" * 60)
    print(f" vocab size = {len(vocab)} | training pairs = {len(y)}")
    print("=" * 60)

    # --- gradient check (proves the hand-derived backprop is correct) ---------
    worst = gradient_check()
    print("\n[gradient check] max |hand - numerical| per parameter:")
    for name, diff in worst.items():
        print(f"  {name:>2}: {diff:.2e}")

    # --- train both models ----------------------------------------------------
    ngram = NGramLM(vocab_size=len(vocab), smoothing_k=0.0).fit(X, y)
    nlm = NeuralLM(vocab_size=len(vocab), emb_dim=2, hidden=8, seed=123)
    nlm.fit(X, y, lr=0.3, epochs=3000)

    print(f"\ntrain perplexity   n-gram(MLE)={ngram.perplexity(X, y):.4f}"
          f"   neural-LM={nlm.perplexity(X, y):.4f}")

    # --- the held-out context: (alice, banana) -> ? --------------------------
    ctx = (vocab.encode("alice"), vocab.encode("banana"))
    good = vocab.encode("good")
    p_ngram = ngram.distribution(ctx)[good]
    p_nlm = nlm.predict_proba(np.array([list(ctx)], dtype=np.int64))[0][good]

    print("\n--- held-out context  (alice, banana) -> good ---")
    print(f"  seen in training?  {ngram.seen_context(ctx)}")
    print(f"  P(good)  n-gram = {p_ngram:.4f}   (uniform 1/V = {1/len(vocab):.4f})")
    print(f"  P(good)  neural = {p_nlm:.4f}")
    print("\n=> The neural LM generalizes to a context counting never saw.")


if __name__ == "__main__":
    main()
