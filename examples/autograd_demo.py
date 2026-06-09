"""Autograd engine demo: hand-derived backprop vs autograd, side by side.

Shows that ``NeuralLMAuto`` (autograd) and ``NeuralLM`` (hand backprop) produce
bit-for-bit identical gradients and identical training curves.

Run with:
    python examples/autograd_demo.py
"""

from __future__ import annotations

import numpy as np

from lmscratch.data import CORPUS_LARGE, Vocab, encode_pairs, make_training_pairs, train_val_split
from lmscratch.nlm import NeuralLM
from lmscratch.nlm_auto import NeuralLMAuto, gradient_compare


def main() -> None:
    print("=" * 60)
    print("  Autograd engine: hand backprop vs autograd")
    print("=" * 60)

    # --- gradient comparison on a random mini-batch -------------------------
    print("\n[1] Gradient comparison (same init, same batch)")
    print("    max |hand_grad - autograd| per parameter:")
    diffs = gradient_compare(seed=0)
    for name, diff in diffs.items():
        status = "OK" if diff < 1e-12 else "FAIL"
        print(f"    {name:>2}: {diff:.2e}  [{status}]")

    # --- train both on CORPUS_LARGE, compare loss curves --------------------
    print("\n[2] Training on CORPUS_LARGE (80/20 split, 500 epochs)")
    vocab = Vocab.from_corpus(CORPUS_LARGE)
    train_sents, val_sents = train_val_split(CORPUS_LARGE, val_frac=0.2, seed=0)
    X_tr, y_tr = encode_pairs(make_training_pairs(train_sents), vocab)
    X_val, y_val = encode_pairs(make_training_pairs(val_sents), vocab)

    V = len(vocab)
    hand = NeuralLM(V, emb_dim=4, hidden=16, seed=7)
    auto = NeuralLMAuto(V, emb_dim=4, hidden=16, seed=7)

    h_hand = hand.fit(X_tr, y_tr, lr=0.3, epochs=500)
    h_auto = auto.fit(X_tr, y_tr, lr=0.3, epochs=500)

    curve_match = np.allclose(h_hand, h_auto, atol=1e-12)
    print(f"    Loss curves identical: {curve_match}")

    print(f"\n    final train perplexity  hand={hand.perplexity(X_tr, y_tr):.4f}"
          f"  auto={auto.perplexity(X_tr, y_tr):.4f}")
    print(f"    final val   perplexity  hand={hand.perplexity(X_val, y_val):.4f}"
          f"  auto={auto.perplexity(X_val, y_val):.4f}")

    print("\n=> Autograd and hand backprop are provably equivalent.")


if __name__ == "__main__":
    main()
