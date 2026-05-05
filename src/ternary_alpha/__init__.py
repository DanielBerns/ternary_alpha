"""
Initialize the ternary_alpha package.
"""

from ternary_alpha.network import (
    TernaryWeight,
    TernaryLayer,
    TernaryNeuralNetwork
)
from ternary_alpha.utils import (
    one_hot_encode,
    one_hot_decode,
    normalize_features,
    split_train_val,
    create_synthetic_dataset,
    evaluate_model
)
from ternary_alpha.inference import TernaryInferenceOptimized

__version__ = "0.1.0"
__author__ = "DanielBerns"

__all__ = [
    "TernaryWeight",
    "TernaryLayer",
    "TernaryNeuralNetwork",
    "one_hot_encode",
    "one_hot_decode",
    "normalize_features",
    "split_train_val",
    "create_synthetic_dataset",
    "evaluate_model",
    "TernaryInferenceOptimized",
]
