from ternary_alpha.network import TernaryNeuralNetwork
from ternary_alpha.utils import create_synthetic_dataset

# Create network
network = TernaryNeuralNetwork(input_dim=10, hidden_dim=16, output_dim=5)

# Generate synthetic data
X_train, y_train = create_synthetic_dataset(1000, 10, 5)

# Train
history = network.train(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    learning_rate=0.01,
    val_split=0.2
)

# Inference
predictions, classes = network.predict(X_test)
