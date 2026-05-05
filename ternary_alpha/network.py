"""
Ternary Neural Network with 2-bit weight encoding.
Supports one-hot encoded inputs and outputs.
Supports deep networks with multiple hidden layers.
"""

import numpy as np
from typing import Tuple, Optional, List, Dict


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
    """
    Deep ternary neural network with multiple hidden layers.
    Supports flexible architecture and one-hot encoded inputs/outputs.
    """

    def __init__(
        self,
        layer_dims: List[int],
        activation: str = 'relu',
        use_bias: bool = True,
        random_seed: Optional[int] = None
    ):
        """
        Initialize a deep ternary neural network.

        Args:
            layer_dims: List of dimensions for each layer.
                       Format: [input_dim, hidden1_dim, hidden2_dim, ..., output_dim]
                       Example: [10, 32, 16, 5] creates a 3-layer network
                       (input -> 32 hidden -> 16 hidden -> 5 output)
            activation: Activation function for hidden layers ('relu', 'tanh', or 'linear')
                       Output layer always uses softmax
            use_bias: Whether to use bias terms in all layers
            random_seed: Random seed for reproducibility
        """
        if random_seed is not None:
            np.random.seed(random_seed)

        if len(layer_dims) < 2:
            raise ValueError("layer_dims must have at least 2 elements (input and output)")

        self.input_dim = layer_dims[0]
        self.output_dim = layer_dims[-1]
        self.hidden_dims = layer_dims[1:-1]
        self.activation = activation
        self.use_bias = use_bias
        self.layer_dims = layer_dims
        self.n_layers = len(layer_dims) - 1  # Number of weight matrices

        # Create layers
        self.layers = []
        for i in range(self.n_layers):
            layer = TernaryLayer(layer_dims[i], layer_dims[i + 1], use_bias)
            self.layers.append(layer)

        # Cache for backpropagation
        self.z_cache = []  # Pre-activation values
        self.a_cache = []  # Post-activation values

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
        Forward pass through the entire network.

        Args:
            x: Input array of shape (batch_size, input_dim), one-hot encoded

        Returns:
            Output probabilities of shape (batch_size, output_dim)
        """
        # Clear caches
        self.z_cache = []
        self.a_cache = []
        self.a_cache.append(x)  # a0 = x

        a = x
        # Forward through all hidden layers
        for i in range(len(self.layers) - 1):
            z = self.layers[i].forward(a)
            self.z_cache.append(z)
            a = self._activation_fn(z)
            self.a_cache.append(a)

        # Output layer (linear -> softmax)
        z_output = self.layers[-1].forward(a)
        self.z_cache.append(z_output)
        a_output = self._softmax(z_output)
        self.a_cache.append(a_output)

        return a_output

    def backward(self, y: np.ndarray, learning_rate: float):
        """
        Backward pass (backpropagation through all layers).

        Args:
            y: Target output, one-hot encoded (batch_size, output_dim)
            learning_rate: Learning rate for weight updates
        """
        # Start with output layer gradient
        # For softmax + cross-entropy: dz = a - y
        dz = self.a_cache[-1] - y

        # Backprop through layers in reverse
        for i in range(len(self.layers) - 1, -1, -1):
            # Backprop through current layer
            da = self.layers[i].backward(dz, learning_rate)

            # If not the first layer, apply activation derivative
            if i > 0:
                dz = da * self._activation_derivative(self.z_cache[i - 1], self.a_cache[i])
            else:
                dz = da

    def update_weights(self, learning_rate: float):
        """Update all weights in the network."""
        for layer in self.layers:
            layer.update_weights(learning_rate)

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

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
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
        history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': []
        }

        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            epoch_loss = 0
            n_batches = 0
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
                n_batches += 1

            # Average loss for epoch
            epoch_loss /= n_batches
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

                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    print(
                        f"Epoch {epoch + 1:3d}/{epochs} - "
                        f"Loss: {epoch_loss:.4f}, Acc: {train_acc:.4f} | "
                        f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
                    )
            else:
                if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                    print(
                        f"Epoch {epoch + 1:3d}/{epochs} - "
                        f"Loss: {epoch_loss:.4f}, Acc: {train_acc:.4f}"
                    )

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

    def get_ternary_weights(self) -> List[np.ndarray]:
        """
        Get the current ternary weights of all layers.

        Returns:
            List of weight matrices, each quantized to {-1, 0, 1}
        """
        ternary_weights = []
        for layer in self.layers:
            w = TernaryWeight.quantize(layer.weights)
            ternary_weights.append(w)
        return ternary_weights

    def get_encoded_weights(self) -> List[np.ndarray]:
        """
        Get the 2-bit encoded weights of all layers.

        Returns:
            List of 2-bit encoded weight matrices
        """
        ternary_weights = self.get_ternary_weights()
        encoded_weights = []
        for w in ternary_weights:
            encoded = TernaryWeight.encode(w)
            encoded_weights.append(encoded)
        return encoded_weights

    def get_network_stats(self) -> Dict:
        """
        Get statistics about the network architecture and weights.

        Returns:
            Dictionary with network statistics
        """
        total_params = 0
        active_params = 0
        sparsity_per_layer = []

        for i, layer in enumerate(self.layers):
            w = TernaryWeight.quantize(layer.weights)
            n_total = w.size
            n_active = np.sum(w != 0)
            sparsity = 1.0 - (n_active / n_total)

            total_params += n_total
            active_params += n_active
            sparsity_per_layer.append({
                'layer': i + 1,
                'total': n_total,
                'active': n_active,
                'sparsity': sparsity
            })

        return {
            'architecture': self.layer_dims,
            'n_layers': self.n_layers,
            'total_parameters': total_params,
            'active_parameters': active_params,
            'overall_sparsity': 1.0 - (active_params / total_params) if total_params > 0 else 0.0,
            'sparsity_per_layer': sparsity_per_layer,
            'memory_reduction_factor': 16,  # 32 bits vs 2 bits
        }

    def summary(self):
        """Print a summary of the network architecture."""
        print("=" * 70)
        print("Ternary Neural Network Summary")
        print("=" * 70)
        print(f"Activation: {self.activation}")
        print(f"Use Bias: {self.use_bias}")
        print()
        print("Layer (type)                  Input Shape         Output Shape")
        print("-" * 70)

        for i, layer in enumerate(self.layers):
            if i == 0:
                input_shape = f"({layer.input_dim},)"
            else:
                input_shape = f"({self.layer_dims[i],},)"

            output_shape = f"({layer.output_dim},)"
            layer_type = "Ternary" if i < len(self.layers) - 1 else "Output (Ternary)"
            print(f"Layer {i + 1} ({layer_type:20s})  {input_shape:18s}  {output_shape:18s}")

        print("=" * 70)
        stats = self.get_network_stats()
        print(f"Total Parameters: {stats['total_parameters']:,}")
        print(f"Active Parameters: {stats['active_parameters']:,}")
        print(f"Overall Sparsity: {stats['overall_sparsity']:.2%}")
        print(f"Memory Reduction: {stats['memory_reduction_factor']}x (32-bit → 2-bit)")
        print("=" * 70)


# Backward compatibility: Keep old class for existing code
class TernaryNeuralNetworkLegacy:
    """
    Legacy two-layer ternary neural network.
    Use TernaryNeuralNetwork with layer_dims=[input_dim, hidden_dim, output_dim] instead.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation: str = 'relu',
        use_bias: bool = True
    ):
        """Initialize legacy two-layer network."""
        self.network = TernaryNeuralNetwork(
            layer_dims=[input_dim, hidden_dim, output_dim],
            activation=activation,
            use_bias=use_bias
        )
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.activation = activation
        self.use_bias = use_bias

    @property
    def layer1(self):
        """Access first layer for backward compatibility."""
        return self.network.layers[0]

    @property
    def layer2(self):
        """Access second layer for backward compatibility."""
        return self.network.layers[1]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return self.network.forward(x)

    def backward(self, y: np.ndarray, learning_rate: float):
        """Backward pass."""
        return self.network.backward(y, learning_rate)

    def update_weights(self, learning_rate: float):
        """Update weights."""
        return self.network.update_weights(learning_rate)

    def compute_loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute loss."""
        return self.network.compute_loss(y_true, y_pred)

    def compute_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute accuracy."""
        return self.network.compute_accuracy(y_true, y_pred)

    def train(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        verbose: bool = True
    ) -> Dict:
        """Train network."""
        return self.network.train(x_train, y_train, x_val, y_val, epochs, batch_size, learning_rate, verbose)

    def predict(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions."""
        return self.network.predict(x)

    def get_ternary_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get ternary weights (legacy format)."""
        weights = self.network.get_ternary_weights()
        return weights[0], weights[1]

    def get_encoded_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get encoded weights (legacy format)."""
        weights = self.network.get_encoded_weights()
        return weights[0], weights[1]
