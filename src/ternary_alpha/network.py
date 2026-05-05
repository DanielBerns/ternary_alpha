"""
Ternary Neural Network with 2-bit weight encoding.
Supports one-hot encoded inputs and outputs.
"""

import numpy as np
from typing import Tuple, Optional


class TernaryWeight:
    """
    Ternary weight encoding/decoding.

    Encoding scheme:
    - -1 (negative) → 0b11 (3)
    -  0 (zero)     → 0b00 (0)
    -  1 (positive) → 0b01 (1)
    """

    # Mapping from ternary value to 2-bit encoding
    TERNARY_TO_BITS = {-1: 0b11, 0: 0b00, 1: 0b01}
    BITS_TO_TERNARY = {0b11: -1, 0b00: 0, 0b01: 1}

    @staticmethod
    def encode(weights: np.ndarray) -> np.ndarray:
        """Convert float weights to ternary (-1, 0, 1) with 2-bit encoding."""
        ternary = np.sign(weights)  # -1, 0, or 1
        ternary[ternary == 0] = 0   # Ensure zeros stay zero

        # Vectorized encoding to 2-bit representation
        encoded = np.vectorize(TernaryWeight.TERNARY_TO_BITS.get)(ternary)
        return encoded.astype(np.uint8)

    @staticmethod
    def decode(encoded: np.ndarray) -> np.ndarray:
        """Convert 2-bit encoded weights back to ternary values."""
        decoded = np.vectorize(TernaryWeight.BITS_TO_TERNARY.get)(encoded)
        return decoded.astype(np.float32)

    @staticmethod
    def quantize(weights: np.ndarray) -> np.ndarray:
        """Quantize float weights to nearest ternary value (-1, 0, 1)."""
        quantized = np.sign(weights)
        # For near-zero values, decide based on threshold
        # Weights with |w| < 0.5 become 0
        quantized[np.abs(weights) < 0.5] = 0
        return quantized


class TernaryLayer:
    """Single layer with ternary weights."""

    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = True):
        """
        Initialize a ternary layer.

        Args:
            input_dim: Number of input neurons
            output_dim: Number of output neurons
            use_bias: Whether to include bias terms
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_bias = use_bias

        # Initialize weights with small random values
        # Will be quantized during forward pass
        self.weights = np.random.randn(input_dim, output_dim) * 0.1
        self.ternary_weights = TernaryWeight.quantize(self.weights)

        if use_bias:
            self.bias = np.zeros(output_dim)
        else:
            self.bias = None

        # For backpropagation
        self.input_cache = None
        self.weight_gradients = None
        self.bias_gradients = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the layer.

        Args:
            x: Input array of shape (batch_size, input_dim)

        Returns:
            Output array of shape (batch_size, output_dim)
        """
        self.input_cache = x

        # Quantize weights to ternary values
        self.ternary_weights = TernaryWeight.quantize(self.weights)

        # Compute: z = x @ w + b
        z = np.dot(x, self.ternary_weights)

        if self.use_bias:
            z = z + self.bias

        return z

    def backward(self, dz: np.ndarray, learning_rate: float) -> np.ndarray:
        """
        Backward pass (compute gradients).

        Args:
            dz: Gradient of loss w.r.t. output (batch_size, output_dim)
            learning_rate: Learning rate for weight updates

        Returns:
            Gradient w.r.t. input (batch_size, input_dim)
        """
        batch_size = self.input_cache.shape[0]

        # Gradient w.r.t. input: dx = dz @ w^T
        dx = np.dot(dz, self.ternary_weights.T)

        # Gradient w.r.t. weights: dw = x^T @ dz
        self.weight_gradients = np.dot(self.input_cache.T, dz) / batch_size

        # Gradient w.r.t. bias: db = sum(dz)
        if self.use_bias:
            self.bias_gradients = np.sum(dz, axis=0) / batch_size

        return dx

    def update_weights(self, learning_rate: float):
        """Update weights using computed gradients."""
        if self.weight_gradients is None:
            return

        # Update continuous weights
        self.weights -= learning_rate * self.weight_gradients

        # Quantize to ternary
        self.ternary_weights = TernaryWeight.quantize(self.weights)

        # Update bias if present
        if self.use_bias and self.bias_gradients is not None:
            self.bias -= learning_rate * self.bias_gradients


class TernaryNeuralNetwork:
    """Two-layer ternary neural network."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 activation: str = 'relu', use_bias: bool = True):
        """
        Initialize a two-layer ternary network.

        Args:
            input_dim: Input dimension (for one-hot encoded inputs)
            hidden_dim: Hidden layer dimension
            output_dim: Output dimension (for one-hot encoded outputs)
            activation: Activation function ('relu', 'tanh', or 'linear')
            use_bias: Whether to use bias terms
        """
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias

        # Layers
        self.layer1 = TernaryLayer(input_dim, hidden_dim, use_bias)
        self.layer2 = TernaryLayer(hidden_dim, output_dim, use_bias)

        # Cache for backpropagation
        self.z1 = None  # Output of layer 1 (before activation)
        self.a1 = None  # Output of layer 1 (after activation)
        self.z2 = None  # Output of layer 2 (logits)
        self.a2 = None  # Output of layer 2 (softmax)

    def _activation_fn(self, z: np.ndarray) -> np.ndarray:
        """Apply activation function."""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'tanh':
            return np.tanh(z)
        elif self.activation == 'linear':
            return z
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _activation_derivative(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        """Compute derivative of activation function."""
        if self.activation == 'relu':
            return (z > 0).astype(np.float32)
        elif self.activation == 'tanh':
            return 1 - a ** 2
        elif self.activation == 'linear':
            return np.ones_like(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def _softmax(self, z: np.ndarray) -> np.ndarray:
        """Compute softmax for output layer."""
        # Numerical stability: subtract max
        z_shifted = z - np.max(z, axis=1, keepdims=True)
        exp_z = np.exp(z_shifted)
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass through the network.

        Args:
            x: Input array of shape (batch_size, input_dim), one-hot encoded

        Returns:
            Output probabilities of shape (batch_size, output_dim)
        """
        # Layer 1: linear + activation
        self.z1 = self.layer1.forward(x)
        self.a1 = self._activation_fn(self.z1)

        # Layer 2: linear + softmax
        self.z2 = self.layer2.forward(self.a1)
        self.a2 = self._softmax(self.z2)

        return self.a2

    def backward(self, y: np.ndarray, learning_rate: float):
        """
        Backward pass (backpropagation).

        Args:
            y: Target output, one-hot encoded (batch_size, output_dim)
            learning_rate: Learning rate for weight updates
        """
        batch_size = y.shape[0]

        # Loss: cross-entropy
        # dz2 = a2 - y (derivative of softmax + cross-entropy)
        dz2 = self.a2 - y

        # Backprop through layer 2
        da1 = self.layer2.backward(dz2, learning_rate)

        # Backprop through activation
        dz1 = da1 * self._activation_derivative(self.z1, self.a1)

        # Backprop through layer 1
        self.layer1.backward(dz1, learning_rate)

    def update_weights(self, learning_rate: float):
        """Update all weights in the network."""
        self.layer1.update_weights(learning_rate)
        self.layer2.update_weights(learning_rate)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute cross-entropy loss.

        Args:
            y_true: True one-hot encoded labels
            y_pred: Predicted probabilities

        Returns:
            Average cross-entropy loss
        """
        # Avoid log(0)
        eps = 1e-7
        y_pred = np.clip(y_pred, eps, 1 - eps)
        loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
        return loss

    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Compute accuracy (percentage of correct predictions).

        Args:
            y_true: True one-hot encoded labels
            y_pred: Predicted probabilities

        Returns:
            Accuracy (0.0 to 1.0)
        """
        predictions = np.argmax(y_pred, axis=1)
        targets = np.argmax(y_true, axis=1)
        return np.mean(predictions == targets)

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              x_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None,
              epochs: int = 100, batch_size: int = 32,
              learning_rate: float = 0.01, verbose: bool = True) -> dict:
        """
        Train the network.

        Args:
            x_train: Training inputs, one-hot encoded (n_samples, input_dim)
            y_train: Training outputs, one-hot encoded (n_samples, output_dim)
            x_val: Validation inputs (optional)
            y_val: Validation outputs (optional)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            verbose: Whether to print progress

        Returns:
            Dictionary with training history
        """
        n_samples = x_train.shape[0]
        history = {'train_loss': [], 'train_accuracy': [],
                   'val_loss': [], 'val_accuracy': []}

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            epoch_loss = 0
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]

                # Forward pass
                y_pred = self.forward(x_batch)

                # Backward pass
                self.backward(y_batch, learning_rate)

                # Update weights
                self.update_weights(learning_rate)

                # Accumulate loss
                epoch_loss += self.compute_loss(y_batch, y_pred)

            # Average loss for epoch
            epoch_loss /= (n_samples // batch_size)
            history['train_loss'].append(epoch_loss)

            # Training accuracy
            y_pred_train = self.forward(x_train)
            train_acc = self.compute_accuracy(y_train, y_pred_train)
            history['train_accuracy'].append(train_acc)

            # Validation metrics
            if x_val is not None and y_val is not None:
                y_pred_val = self.forward(x_val)
                val_loss = self.compute_loss(y_val, y_pred_val)
                val_acc = self.compute_accuracy(y_val, y_pred_val)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {epoch_loss:.4f}, Acc: {train_acc:.4f} | "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{epochs} - "
                          f"Loss: {epoch_loss:.4f}, Acc: {train_acc:.4f}")

        return history

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.

        Args:
            x: Input data (batch_size, input_dim)

        Returns:
            Tuple of (probabilities, class_indices)
        """
        probs = self.forward(x)
        classes = np.argmax(probs, axis=1)
        return probs, classes

    def get_ternary_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the current ternary weights of both layers.

        Returns:
            Tuple of (weights_layer1, weights_layer2), each quantized to {-1, 0, 1}
        """
        w1 = TernaryWeight.quantize(self.layer1.weights)
        w2 = TernaryWeight.quantize(self.layer2.weights)
        return w1, w2

    def get_encoded_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the 2-bit encoded weights.

        Returns:
            Tuple of (encoded_weights_layer1, encoded_weights_layer2)
        """
        w1, w2 = self.get_ternary_weights()
        encoded_w1 = TernaryWeight.encode(w1)
        encoded_w2 = TernaryWeight.encode(w2)
        return encoded_w1, encoded_w2
