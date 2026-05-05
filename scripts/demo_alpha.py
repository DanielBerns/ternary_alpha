"""
Example usage of the ternary neural network.
"""

import numpy as np
from ternary_alpha.network import TernaryNeuralNetwork
from ternary_alpha.utils import (
    create_synthetic_dataset, split_train_val, evaluate_model, print_metrics
)


def main():
    """Main example demonstrating training and inference."""

    print("=" * 60)
    print("Ternary Neural Network - Example")
    print("=" * 60)

    # Hyperparameters
    input_dim = 10      # 10 input classes
    hidden_dim = 32     # 16 hidden neurons
    output_dim = 5      # 5 output classes
    n_samples = 2000    # 2000 training samples
    epochs = 500
    batch_size = 32
    learning_rate = 0.01

    print(f"\nDataset Configuration:")
    print(f"  Input dimension:  {input_dim}")
    print(f"  Hidden dimension: {hidden_dim}")
    print(f"  Output dimension: {output_dim}")
    print(f"  Total samples:    {n_samples}")

    # Create synthetic dataset
    print(f"\nCreating synthetic dataset...")
    x, y = create_synthetic_dataset(n_samples, input_dim, output_dim, random_seed=42)

    # Split into train and validation
    x_train, x_val, y_train, y_val = split_train_val(
        x, y, val_split=0.2, shuffle=True, random_seed=42
    )

    print(f"  Training samples:   {x_train.shape[0]}")
    print(f"  Validation samples: {x_val.shape[0]}")

    # Create network
    print(f"\nInitializing Ternary Neural Network...")
    network = TernaryNeuralNetwork(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        activation='relu',
        use_bias=True
    )

    # Training
    print(f"\nTraining Configuration:")
    print(f"  Epochs:        {epochs}")
    print(f"  Batch size:    {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"\nTraining...")

    history = network.train(
        x_train, y_train,
        x_val=x_val, y_val=y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        verbose=True
    )

    # Evaluation
    print(f"\n" + "=" * 60)
    print("Final Evaluation")
    print("=" * 60)

    y_pred_train = network.forward(x_train)
    y_pred_val = network.forward(x_val)

    print("\nTraining Set:")
    train_metrics = evaluate_model(y_train, y_pred_train)
    print_metrics(train_metrics, "  ")

    print("\nValidation Set:")
    val_metrics = evaluate_model(y_val, y_pred_val)
    print_metrics(val_metrics, "  ")

    # Display ternary weights
    print(f"\n" + "=" * 60)
    print("Ternary Weights")
    print("=" * 60)

    w1, w2 = network.get_ternary_weights()
    print(f"\nLayer 1 weights shape: {w1.shape}")
    print(f"  Unique values: {np.unique(w1)}")
    print(f"  Value counts: ", end="")
    for val in [-1, 0, 1]:
        count = np.sum(w1 == val)
        pct = 100 * count / w1.size
        print(f"{val}: {count:5d} ({pct:5.1f}%)  ", end="")

    print(f"\n\nLayer 2 weights shape: {w2.shape}")
    print(f"  Unique values: {np.unique(w2)}")
    print(f"  Value counts: ", end="")
    for val in [-1, 0, 1]:
        count = np.sum(w2 == val)
        pct = 100 * count / w2.size
        print(f"{val}: {count:5d} ({pct:5.1f}%)  ", end="")
    print()

    # Example inference
    print(f"\n" + "=" * 60)
    print("Example Inference")
    print("=" * 60)

    # Create a test sample
    test_input = np.zeros((1, input_dim))
    test_input[0, 0] = 1  # One-hot encode: first class

    probs, pred_class = network.predict(test_input)
    print(f"\nTest input: class 0 (one-hot encoded)")
    print(f"Predicted class: {pred_class[0]}")
    print(f"Prediction probabilities: {probs[0]}")

    # Verify 2-bit encoding
    print(f"\n" + "=" * 60)
    print("2-Bit Weight Encoding Verification")
    print("=" * 60)

    enc_w1, enc_w2 = network.get_encoded_weights()
    print(f"\nLayer 1 encoded weights shape: {enc_w1.shape}")
    print(f"Sample encoded values (first 5x5):")
    print(enc_w1[:5, :5])
    print(f"\nEncoding scheme:")
    print(f"  -1 (negative) → 0b11 (3)")
    print(f"   0 (zero)     → 0b00 (0)")
    print(f"   1 (positive) → 0b01 (1)")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
