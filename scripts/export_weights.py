"""Export trained model weights to docs/model.json for the GitHub Pages demo.

Run from the repo root:
    python scripts/export_weights.py
or:
    make export

The output file is committed to the repo so GitHub Pages can serve it
without a build step.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

from lmscratch.data import CORPUS, Vocab, encode_pairs, make_training_pairs
from lmscratch.ngram import NGramLM
from lmscratch.nlm import NeuralLM

DOCS = pathlib.Path(__file__).parent.parent / "docs"
DOCS.mkdir(exist_ok=True)

# ── train ─────────────────────────────────────────────────────────────────────
vocab = Vocab.from_corpus(CORPUS)
X, y = encode_pairs(make_training_pairs(CORPUS), vocab)

ngram = NGramLM(vocab_size=len(vocab), smoothing_k=0.0).fit(X, y)
nlm = NeuralLM(vocab_size=len(vocab), emb_dim=2, hidden=8, seed=123)
nlm.fit(X, y, lr=0.3, epochs=3000)

# ── subjects / items extracted from CORPUS ───────────────────────────────────
subjects: list[str] = []
items: list[str] = []
for sentence in CORPUS:
    parts = sentence.split()
    if parts[0] not in subjects:
        subjects.append(parts[0])
    if parts[1] not in items:
        items.append(parts[1])

# ── precompute all n-gram distributions ──────────────────────────────────────
ngram_dists: dict[str, list[float]] = {}
ngram_seen: list[str] = []

for s in subjects:
    for it in items:
        sid, iid = vocab.encode(s), vocab.encode(it)
        key = f"{sid},{iid}"
        ngram_dists[key] = ngram.distribution((sid, iid)).tolist()
        if ngram.seen_context((sid, iid)):
            ngram_seen.append(key)

# ── assemble payload ─────────────────────────────────────────────────────────
data = {
    "vocab": vocab.stoi,
    "itos": list(vocab.itos),
    "subjects": subjects,
    "items": items,
    "verdicts": ["good", "bad"],
    "train_sentences": list(CORPUS),
    "context_len": 2,
    "emb_dim": nlm.D,
    "neural": {
        "E":  nlm.E.tolist(),
        "W1": nlm.W1.tolist(),
        "b1": nlm.b1.tolist(),
        "W2": nlm.W2.tolist(),
        "b2": nlm.b2.tolist(),
    },
    "ngram": {
        "distributions": ngram_dists,
        "seen": ngram_seen,
    },
}

# ── write ─────────────────────────────────────────────────────────────────────
out = DOCS / "model.json"
out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
print(f"Wrote {out}  ({out.stat().st_size / 1024:.1f} KB)")

# ── quick sanity check ────────────────────────────────────────────────────────
ctx = (vocab.encode("alice"), vocab.encode("banana"))
good = vocab.encode("good")
p_nlm = nlm.predict_proba(np.array([list(ctx)], dtype=np.int64))[0][good]
print(f"P(good | alice, banana)  neural={p_nlm:.4f}  "
      f"ngram={ngram.distribution(ctx)[good]:.4f}  "
      f"[seen={ngram.seen_context(ctx)}]")
