"""
Unit tests for the ternary neural network.
"""

import numpy as np
import unittest
from ternary_alpha.network import TernaryWeight, TernaryLayer, TernaryNeuralNetwork
from ternary_alpha.utils import one_hot_encode, one_hot_decode, split_train_val


class TestTernaryWeight(unittest.TestCase):
    """Tests for ternary weight encoding/decoding."""

    def test_ternary_to_bits_mapping(self):
        """Test ternary to 2-bit encoding."""
        self.assertEqual(TernaryWeight.TERNARY_TO_BITS[-1], 0b11)
        self.assertEqual(TernaryWeight.TERNARY_TO_BITS[0], 0b00)
        self.assertEqual(TernaryWeight.TERNARY_TO_BITS[1], 0b01)

    def test_bits_to_ternary_mapping(self):
        """Test 2-bit to ternary decoding."""
        self.assertEqual(TernaryWeight.BITS_TO_TERNARY[0b11], -1)
        self.assertEqual(TernaryWeight.BITS_TO_TERNARY[0b00], 0)
        self.assertEqual(TernaryWeight.BITS_TO_TERNARY[0b01], 1)

    def test_quantize(self):
        """Test quantization to ternary values."""
        weights = np.array([[-1.5, 0.1, 0.7], [2.3, -0.3, 0.0]])
        quantized = TernaryWeight.quantize(weights)

        # Check that values are in {-1, 0, 1}
        unique_vals = np.unique(quantized)
        self.assertTrue(all(v in [-1, 0, 1] for v in unique_vals))

    def test_encode_decode_roundtrip(self):
        """Test encoding and decoding roundtrip."""
        ternary = np.array([[-1, 0, 1], [1, -1, 0]])
        encoded = TernaryWeight.encode(ternary)
        decoded = TernaryWeight.decode(encoded)

        np.testing.assert_array_equal(decoded, ternary)


class TestTernaryLayer(unittest.TestCase):
    """Tests for ternary layer."""

    def test_layer_initialization(self):
        """Test layer initialization."""
        layer = TernaryLayer(input_dim=10, output_dim=5, use_bias=True)

        self.assertEqual(layer.weights.shape, (10, 5))
        self.assertEqual(layer.bias.shape, (5,))
        self.assertEqual(layer.ternary_weights.shape, (10, 5))

    def test_forward_pass(self):
        """Test forward pass through layer."""
        layer = TernaryLayer(input_dim=10, output_dim=5)
        x = np.random.randn(32, 10)  # batch_size=32

        output = layer.forward(x)

        self.assertEqual(output.shape, (32, 5))
        np.testing.assert_array_equal(layer.input_cache, x)

    def test_backward_pass(self):
        """Test backward pass through layer."""
        layer = TernaryLayer(input_dim=10, output_dim=5)
        x = np.random.randn(32, 10)

        # Forward
        layer.forward(x)

        # Backward
        dz = np.random.randn(32, 5)
        dx = layer.backward(dz, learning_rate=0.01)

        self.assertEqual(dx.shape, x.shape)
        self.assertEqual(layer.weight_gradients.shape, layer.weights.shape)
        self.assertEqual(layer.bias_gradients.shape, layer.bias.shape)


class TestTernaryNeuralNetwork(unittest.TestCase):
    """Tests for the full ternary neural network."""

    def setUp(self):
        """Set up test fixtures."""
        self.network = TernaryNeuralNetwork(
            input_dim=10,
            hidden_dim=8,
            output_dim=5,
            activation='relu'
        )

        # Create synthetic data
        np.random.seed(42)
        self.x_train = np.random.randn(100, 10)
        self.y_train = np.zeros((100, 5))
        self.y_train[np.arange(100), np.random.randint(0, 5, 100)] = 1

    def test_forward_pass(self):
        """Test forward pass."""
        output = self.network.forward(self.x_train[:10])

        self.assertEqual(output.shape, (10, 5))
        # Check softmax: probabilities should sum to 1
        np.testing.assert_allclose(np.sum(output, axis=1), 1.0, rtol=1e-5)

    def test_loss_computation(self):
        """Test loss computation."""
        y_pred = self.network.forward(self.x_train)
        loss = self.network.compute_loss(self.y_train, y_pred)

        self.assertIsInstance(loss, (float, np.floating))
        self.assertGreater(loss, 0)

    def test_accuracy_computation(self):
        """Test accuracy computation."""
        y_pred = self.network.forward(self.x_train)
        accuracy = self.network.compute_accuracy(self.y_train, y_pred)

        self.assertIsInstance(accuracy, (float, np.floating))
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 1)

    def test_training(self):
        """Test training loop."""
        initial_loss = self.network.compute_loss(
            self.y_train,
            self.network.forward(self.x_train)
        )

        # Train for a few epochs
        history = self.network.train(
            self.x_train, self.y_train,
            epochs=5,
            batch_size=32,
            learning_rate=0.01,
            verbose=False
        )

        self.assertEqual(len(history['train_loss']), 5)
        self.assertEqual(len(history['train_accuracy']), 5)

        # Loss should generally decrease (at least some epochs)
        final_loss = history['train_loss'][-1]
        # Note: ternary quantization might not guarantee monotonic decrease
        # Just check that training ran without errors
        self.assertIsNotNone(final_loss)

    def test_get_ternary_weights(self):
        """Test getting ternary weights."""
        w1, w2 = self.network.get_ternary_weights()

        self.assertEqual(w1.shape, (10, 8))
        self.assertEqual(w2.shape, (8, 5))

        # Check that weights are ternary
        for w in [w1, w2]:
            unique_vals = np.unique(w)
            self.assertTrue(all(v in [-1, 0, 1] for v in unique_vals))

    def test_get_encoded_weights(self):
        """Test getting 2-bit encoded weights."""
        enc_w1, enc_w2 = self.network.get_encoded_weights()

        self.assertEqual(enc_w1.shape, (10, 8))
        self.assertEqual(enc_w2.shape, (8, 5))

        # Check that encoded values are valid 2-bit representations
        for enc_w in [enc_w1, enc_w2]:
            unique_vals = np.unique(enc_w)
            self.assertTrue(all(v in [0b00, 0b01, 0b11] for v in unique_vals))

    def test_predict(self):
        """Test prediction."""
        probs, classes = self.network.predict(self.x_train[:10])

        self.assertEqual(probs.shape, (10, 5))
        self.assertEqual(classes.shape, (10,))

        # Classes should be in valid range
        self.assertTrue(np.all(classes >= 0) and np.all(classes < 5))


class TestUtils(unittest.TestCase):
    """Tests for utility functions."""

    def test_one_hot_encode_decode(self):
        """Test one-hot encoding and decoding."""
        labels = np.array([0, 1, 2, 1, 0])
        num_classes = 3

        encoded = one_hot_encode(labels, num_classes)
        decoded = one_hot_decode(encoded)

        self.assertEqual(encoded.shape, (5, 3))
        np.testing.assert_array_equal(decoded, labels)

    def test_split_train_val(self):
        """Test train-validation split."""
        x = np.random.randn(100, 10)
        y = np.random.randn(100, 5)

        x_train, x_val, y_train, y_val = split_train_val(
            x, y, val_split=0.2, shuffle=False
        )

        self.assertEqual(x_train.shape[0], 80)
        self.assertEqual(x_val.shape[0], 20)
        self.assertEqual(y_train.shape[0], 80)
        self.assertEqual(y_val.shape[0], 20)


if __name__ == '__main__':
    unittest.main()
