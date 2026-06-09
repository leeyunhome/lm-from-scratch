# lm-from-scratch

Building language models **from scratch** — a count-based **n-gram** model and a
Bengio-style **neural language model** — in pure NumPy, with backpropagation
**derived by hand** (no autograd engine). A small, readable study of *why
representation learning generalizes where counting cannot*.

[![CI](https://github.com/leeyunhome/lm-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/leeyunhome/lm-from-scratch/actions/workflows/ci.yml)

> Inspired by the honglab_llm Part 2 course; **re-implemented from scratch** as
> my own work (no course code reused). See [CREDITS.md](CREDITS.md).

---

## The one question this repo answers

Train both models on a `<subject> <item> <verdict>` toy grammar
(read "alice apple good" as "alice finds apples good"), split 80 / 20
into train and validation sets:

| subjects | items | verdicts | sentences |
|---|---|---|---:|
| alice, bob, carol, dave, eve | apple, berry, banana, cherry, grape | good / bad | 25 |

Now ask each model about a context **it never saw in training** —
`(alice, banana)` (alice was never paired with banana in train):

| Model | P(`good` \| alice, banana) |
|---|---:|
| n-gram (counting) | **0.09** (uniform guess, 1/V) |
| **neural LM** | **≈ 0.99** |

The n-gram has never seen `(alice, banana)`, so it falls back to a uniform guess.
The neural LM places `alice` near `bob` in embedding space (both give "good"), and
`banana` near the other items — so it *infers* `good`. That single contrast is the
whole point. (Reproduce with `make demo`.)

## Install

```bash
conda env create -f environment.yml && conda activate lm-from-scratch
pip install -e ".[dev,viz]"
```

## Quickstart

```bash
make demo     # n-gram vs neural-LM comparison, incl. the held-out context
make test     # run the suite (includes an automated gradient check)
```

```python
from lmscratch.data import Vocab, make_training_pairs, encode_pairs
from lmscratch.nlm import NeuralLM

vocab = Vocab.from_corpus()
X, y = encode_pairs(make_training_pairs(), vocab)
model = NeuralLM(vocab_size=len(vocab), emb_dim=2, hidden=8)
model.fit(X, y, lr=0.3, epochs=3000)
print("perplexity:", model.perplexity(X, y))
```

## Why it's interesting (engineering notes)

- **Hand-derived backprop.** `NeuralLM.loss_and_grads` writes the forward pass
  and every gradient out explicitly — no autograd. Correctness is enforced by an
  automated finite-difference **gradient check** in the test suite.
- **Minimal autograd engine.** `autograd.Tensor` is ~120 lines of pure NumPy: a
  `_backward` closure per op, reverse topological traversal. `NeuralLMAuto` uses
  it to train the same model — and `gradient_compare()` proves both paths produce
  bit-for-bit identical gradients.
- **No magic.** Pure NumPy, small enough to read end to end.
- **Tested & CI'd.** `pytest` covers the gradient check, n-gram probability
  identities, autograd op correctness, and data-pipeline invariants; GitHub
  Actions runs it on every push.

## Project layout

```
src/lmscratch/
  data.py      # toy corpus, Vocab, trigram training pairs, train/val split
  ngram.py     # NGramLM: counts + add-k smoothing + uniform backoff
  nlm.py       # NeuralLM: embedding -> tanh -> softmax + hand-derived backprop
  autograd.py  # minimal tensor autograd engine (~120 lines, pure NumPy)
  nlm_auto.py  # NeuralLMAuto: same model, backward via autograd engine
tests/         # gradient checks, model/data invariants, autograd op tests
examples/
  compare.py       # headline n-gram vs neural-LM comparison
  autograd_demo.py # hand backprop vs autograd: proves bit-for-bit equivalence
```

## Roadmap

- [x] Scale to a larger corpus; add train/val split and held-out perplexity
- [x] Add a minimal autograd engine and contrast it with the hand-derived path
- [ ] Add a tiny self-attention block (n-gram → MLP-LM → attention)
- [ ] Interactive browser demo (export weights to JSON, run in JS on GitHub Pages)

## License

[MIT](LICENSE) © 2026 Yunhome Lee
