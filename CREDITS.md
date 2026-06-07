# Credits & Attribution

## Inspiration

The core idea of this project — contrasting a count-based **n-gram** language
model with a small **neural language model** to show *why representation
learning generalizes* — was inspired by Part 2 of the **honglab_llm** course
(<https://www.honglab.ai>).

I followed that course to learn the concepts, then **re-implemented everything
from scratch** as my own work. No source code from the course is reused here.

## What is original to this repository

- **Independent re-implementation** of the data pipeline, n-gram model, and
  neural LM (different API design, English throughout, type hints, docstrings).
- **A new, self-contained English toy corpus** (subject–verb–object grammar)
  designed to reproduce the same held-out-generalization lesson.
- **Engineering not present in the original**: a `pytest` test suite (including
  an automated gradient check), GitHub Actions CI, packaging (`pyproject.toml`),
  a reproducible conda environment, and a `Makefile`.

## Foundational references

- Y. Bengio, R. Ducharme, P. Vincent, C. Jauvin. *A Neural Probabilistic
  Language Model.* JMLR, 2003.
- C. E. Shannon. *A Mathematical Theory of Communication.* 1948.
