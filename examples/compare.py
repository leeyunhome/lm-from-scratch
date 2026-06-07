"""Headline comparison: n-gram vs neural LM on the held-out context.

Run with ``make demo`` or ``python examples/compare.py``.
"""

from __future__ import annotations

import numpy as np

from lmscratch.data import (
    CORPUS_LARGE,
    Vocab,
    encode_pairs,
    make_training_pairs,
    train_val_split,
)
from lmscratch.ngram import NGramLM
from lmscratch.nlm import NeuralLM, gradient_check


def main() -> None:
    # Build vocab from the full corpus so val tokens are never OOV.
    vocab = Vocab.from_corpus(CORPUS_LARGE)

    train_sents, val_sents = train_val_split(CORPUS_LARGE, val_frac=0.2, seed=0)

    X_tr, y_tr = encode_pairs(make_training_pairs(train_sents), vocab)
    X_val, y_val = encode_pairs(make_training_pairs(val_sents), vocab)

    print("=" * 60)
    print(f" vocab size = {len(vocab)}")
    print(f" train sentences = {len(train_sents)} | val sentences = {len(val_sents)}")
    print(f" train pairs = {len(y_tr)} | val pairs = {len(y_val)}")
    print("=" * 60)

    # --- gradient check (proves the hand-derived backprop is correct) ----------
    worst = gradient_check()
    print("\n[gradient check] max |hand - numerical| per parameter:")
    for name, diff in worst.items():
        print(f"  {name:>2}: {diff:.2e}")

    # --- train both models on the training split --------------------------------
    ngram = NGramLM(vocab_size=len(vocab), smoothing_k=0.0).fit(X_tr, y_tr)
    nlm = NeuralLM(vocab_size=len(vocab), emb_dim=4, hidden=16, seed=123)
    nlm.fit(X_tr, y_tr, lr=0.3, epochs=3000)

    print("\n--- perplexity ---")
    print(f"  {'':10s}  {'train':>10s}  {'val':>10s}")
    print(f"  {'n-gram':10s}  {ngram.perplexity(X_tr, y_tr):>10.4f}"
          f"  {ngram.perplexity(X_val, y_val, smoothing_k=0.1):>10.4f}")
    print(f"  {'neural-LM':10s}  {nlm.perplexity(X_tr, y_tr):>10.4f}"
          f"  {nlm.perplexity(X_val, y_val):>10.4f}")

    # --- the held-out context: (alice, banana) -> ? ----------------------------
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
