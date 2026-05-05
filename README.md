# Ternary Alpha

A Python/NumPy implementation of neural networks with ternary weights (-1, 0, 1) and one-hot encoded inputs/outputs.

## Features

- **Ternary Weights**: 2-bit encoding for weight values {-1, 0, 1}
- **One-Hot Encoding**: Native support for categorical inputs and outputs
- **Two-Layer Architecture**: Configurable hidden dimension
- **Training Algorithm**: Full backpropagation with quantization-aware training
- **Inference Optimization**: Three inference methods with performance benchmarking
- **Pure NumPy**: No external dependencies beyond NumPy

## Quick Start

```python
from ternary_alpha.network import TernaryNeuralNetwork
from ternary_alpha.utils import create_synthetic_dataset

# Create network
network = TernaryNeuralNetwork(input_dim=10, hidden_dim=16, output_dim=5)

# Generate synthetic data
X_train, y_train = create_synthetic_dataset(1000, 10, 5)

# Train
history = network.train(X_train, y_train, epochs=50, batch_size=32, learning_rate=0.01)

# Inference
predictions, classes = network.predict(X_test)
