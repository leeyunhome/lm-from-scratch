"""lm-from-scratch: n-gram and neural language models in pure NumPy."""

from lmscratch.data import (
    CORPUS,
    CONTEXT_LEN,
    Vocab,
    make_training_pairs,
    encode_pairs,
)
from lmscratch.ngram import NGramLM
from lmscratch.nlm import NeuralLM, gradient_check, softmax

__all__ = [
    "CORPUS",
    "CONTEXT_LEN",
    "Vocab",
    "make_training_pairs",
    "encode_pairs",
    "NGramLM",
    "NeuralLM",
    "gradient_check",
    "softmax",
]

__version__ = "0.1.0"
