"""
Utility functions for ternary neural networks.
Includes encoding/decoding utilities and data preparation.
"""

import numpy as np
from typing import Tuple, Union


def one_hot_encode(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """
    Convert class labels to one-hot encoding.

    Args:
        labels: Array of class indices (shape: n_samples,)
        num_classes: Number of classes

    Returns:
        One-hot encoded array (shape: n_samples, num_classes)
    """
    n_samples = labels.shape[0]
    one_hot = np.zeros((n_samples, num_classes), dtype=np.float32)
    one_hot[np.arange(n_samples), labels] = 1
    return one_hot


def one_hot_decode(one_hot: np.ndarray) -> np.ndarray:
    """
    Convert one-hot encoding back to class labels.

    Args:
        one_hot: One-hot encoded array (shape: n_samples, num_classes)

    Returns:
        Class labels (shape: n_samples,)
    """
    return np.argmax(one_hot, axis=1)


def normalize_features(x: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize features to zero mean and unit variance.

    Args:
        x: Input array (shape: n_samples, n_features)
        axis: Axis along which to normalize

    Returns:
        Tuple of (normalized_x, mean, std)
    """
    mean = np.mean(x, axis=axis, keepdims=True)
    std = np.std(x, axis=axis, keepdims=True)
    std[std == 0] = 1  # Avoid division by zero
    normalized = (x - mean) / std
    return normalized, mean, std


def denormalize_features(x_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Denormalize features back to original scale.

    Args:
        x_norm: Normalized array
        mean: Mean used in normalization
        std: Standard deviation used in normalization

    Returns:
        Denormalized array
    """
    return x_norm * std + mean


def split_train_val(x: np.ndarray, y: np.ndarray,
                    val_split: float = 0.2,
                    shuffle: bool = True,
                    random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into training and validation sets.

    Args:
        x: Input features
        y: Target labels
        val_split: Fraction of data for validation (0.0 to 1.0)
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (x_train, x_val, y_train, y_val)
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_samples = x.shape[0]
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    split_idx = int(n_samples * (1 - val_split))

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    return x[train_idx], x[val_idx], y[train_idx], y[val_idx]


def create_synthetic_dataset(n_samples: int = 1000,
                            input_dim: int = 10,
                            output_dim: int = 5,
                            random_seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a synthetic dataset with one-hot encoded inputs and outputs.

    Args:
        n_samples: Number of samples
        input_dim: Input dimension (number of classes for input)
        output_dim: Output dimension (number of classes for output)
        random_seed: Random seed

    Returns:
        Tuple of (x, y) where both are one-hot encoded
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random class indices
    x_indices = np.random.randint(0, input_dim, n_samples)
    y_indices = np.random.randint(0, output_dim, n_samples)

    # One-hot encode
    x = one_hot_encode(x_indices, input_dim)
    y = one_hot_encode(y_indices, output_dim)

    return x, y


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute various evaluation metrics.

    Args:
        y_true: True one-hot encoded labels
        y_pred: Predicted probabilities

    Returns:
        Dictionary with metrics (accuracy, precision, recall, f1)
    """
    # Decode predictions
    y_true_labels = one_hot_decode(y_true)
    y_pred_labels = np.argmax(y_pred, axis=1)

    # Accuracy
    accuracy = np.mean(y_true_labels == y_pred_labels)

    # Per-class metrics
    n_classes = y_true.shape[1]
    precision_list = []
    recall_list = []

    for c in range(n_classes):
        true_positives = np.sum((y_pred_labels == c) & (y_true_labels == c))
        false_positives = np.sum((y_pred_labels == c) & (y_true_labels != c))
        false_negatives = np.sum((y_pred_labels != c) & (y_true_labels == c))

        precision = true_positives / (true_positives + false_positives + 1e-7)
        recall = true_positives / (true_positives + false_negatives + 1e-7)
        precision_list.append(precision)
        recall_list.append(recall)

    avg_precision = np.mean(precision_list)
    avg_recall = np.mean(recall_list)
    f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall + 1e-7)

    return {
        'accuracy': accuracy,
        'precision': avg_precision,
        'recall': avg_recall,
        'f1_score': f1_score,
        'per_class_precision': precision_list,
        'per_class_recall': recall_list
    }


def print_metrics(metrics: dict, prefix: str = ""):
    """Print evaluation metrics in a readable format."""
    print(f"{prefix}Accuracy:  {metrics['accuracy']:.4f}")
    print(f"{prefix}Precision: {metrics['precision']:.4f}")
    print(f"{prefix}Recall:    {metrics['recall']:.4f}")
    print(f"{prefix}F1-Score:  {metrics['f1_score']:.4f}")
