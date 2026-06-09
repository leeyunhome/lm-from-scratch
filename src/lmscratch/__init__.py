"""lm-from-scratch: n-gram and neural language models in pure NumPy."""

from lmscratch.data import (
    CORPUS,
    CORPUS_LARGE,
    CONTEXT_LEN,
    Vocab,
    make_training_pairs,
    encode_pairs,
    train_val_split,
)
from lmscratch.ngram import NGramLM
from lmscratch.nlm import NeuralLM, gradient_check, softmax
from lmscratch.nlm_auto import NeuralLMAuto, gradient_compare
from lmscratch.autograd import Tensor, embedding_lookup, cross_entropy

__all__ = [
    "CORPUS",
    "CORPUS_LARGE",
    "CONTEXT_LEN",
    "Vocab",
    "make_training_pairs",
    "encode_pairs",
    "train_val_split",
    "NGramLM",
    "NeuralLM",
    "gradient_check",
    "softmax",
    "NeuralLMAuto",
    "gradient_compare",
    "Tensor",
    "embedding_lookup",
    "cross_entropy",
]

__version__ = "0.2.0"
