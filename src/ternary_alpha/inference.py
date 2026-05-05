"""
Inference optimization for ternary neural networks.
Implements efficient forward pass using ternary arithmetic.
"""

import numpy as np
from typing import Tuple, Optional
from ternary_alpha.network import TernaryNeuralNetwork, TernaryWeight


class TernaryInferenceOptimized:
    """
    Optimized inference engine for ternary neural networks.
    Leverages sparsity and ternary arithmetic for efficiency.
    """

    def __init__(self, network: TernaryNeuralNetwork):
        """
        Initialize inference engine.

        Args:
            network: Trained TernaryNeuralNetwork instance
        """
        self.network = network
        self.w1_ternary = None
        self.b1 = None
        self.w2_ternary = None
        self.b2 = None
        self.sparsity_w1 = None
        self.sparsity_w2 = None

        # Load and cache weights
        self._load_weights()

    def _load_weights(self):
        """Load and cache ternary weights."""
        self.w1_ternary, self.w2_ternary = self.network.get_ternary_weights()
        self.b1 = self.network.layer1.bias
        self.b2 = self.network.layer2.bias

        # Compute sparsity (proportion of zero weights)
        self.sparsity_w1 = np.sum(self.w1_ternary == 0) / self.w1_ternary.size
        self.sparsity_w2 = np.sum(self.w2_ternary == 0) / self.w2_ternary.size

    def forward_standard(self, x: np.ndarray) -> np.ndarray:
        """
        Standard forward pass (for comparison).

        Args:
            x: Input batch (batch_size, input_dim)

        Returns:
            Probabilities (batch_size, output_dim)
        """
        return self.network.forward(x)

    def forward_sparse(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass with sparse matrix handling.
        Skips computations for zero weights.

        Args:
            x: Input batch (batch_size, input_dim)

        Returns:
            Probabilities (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Layer 1: only compute non-zero terms
        z1 = np.zeros((batch_size, self.network.hidden_dim))

        # Find non-zero weight indices
        nonzero_idx = np.where(self.w1_ternary != 0)

        for i, j in zip(nonzero_idx[0], nonzero_idx[1]):
            w_ij = self.w1_ternary[i, j]
            z1[:, j] += w_ij * x[:, i]

        if self.b1 is not None:
            z1 += self.b1

        a1 = np.maximum(0, z1)  # ReLU

        # Layer 2: only compute non-zero terms
        z2 = np.zeros((batch_size, self.network.output_dim))

        nonzero_idx = np.where(self.w2_ternary != 0)

        for i, j in zip(nonzero_idx[0], nonzero_idx[1]):
            w_ij = self.w2_ternary[i, j]
            z2[:, j] += w_ij * a1[:, i]

        if self.b2 is not None:
            z2 += self.b2

        # Softmax
        z2_shifted = z2 - np.max(z2, axis=1, keepdims=True)
        exp_z2 = np.exp(z2_shifted)
        return exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)

    def forward_ternary_optimized(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass optimized for ternary arithmetic.
        Replaces multiplications with additions/subtractions for ±1 weights.

        Args:
            x: Input batch (batch_size, input_dim)

        Returns:
            Probabilities (batch_size, output_dim)
        """
        batch_size = x.shape[0]

        # Layer 1 with ternary optimization
        z1 = np.zeros((batch_size, self.network.hidden_dim))

        for j in range(self.network.hidden_dim):
            w_col = self.w1_ternary[:, j]

            # Separate positive and negative weights
            pos_mask = w_col == 1
            neg_mask = w_col == -1

            if np.any(pos_mask):
                z1[:, j] += np.sum(x[:, pos_mask], axis=1)

            if np.any(neg_mask):
                z1[:, j] -= np.sum(x[:, neg_mask], axis=1)

        if self.b1 is not None:
            z1 += self.b1

        a1 = np.maximum(0, z1)  # ReLU

        # Layer 2 with ternary optimization
        z2 = np.zeros((batch_size, self.network.output_dim))

        for j in range(self.network.output_dim):
            w_col = self.w2_ternary[:, j]

            pos_mask = w_col == 1
            neg_mask = w_col == -1

            if np.any(pos_mask):
                z2[:, j] += np.sum(a1[:, pos_mask], axis=1)

            if np.any(neg_mask):
                z2[:, j] -= np.sum(a1[:, neg_mask], axis=1)

        if self.b2 is not None:
            z2 += self.b2

        # Softmax
        z2_shifted = z2 - np.max(z2, axis=1, keepdims=True)
        exp_z2 = np.exp(z2_shifted)
        return exp_z2 / np.sum(exp_z2, axis=1, keepdims=True)

    def get_statistics(self) -> dict:
        """
        Get statistics about the network for efficiency analysis.

        Returns:
            Dictionary with statistics
        """
        total_params = self.w1_ternary.size + self.w2_ternary.size
        active_params = (np.sum(self.w1_ternary != 0) +
                         np.sum(self.w2_ternary != 0))

        return {
            'total_parameters': total_params,
            'active_parameters': active_params,
            'sparsity': 1.0 - (active_params / total_params),
            'layer1_sparsity': self.sparsity_w1,
            'layer2_sparsity': self.sparsity_w2,
            'memory_reduction_factor': 16,  # 32 bits vs 2 bits
            'parameter_reduction_factor': total_params / active_params if active_params > 0 else float('inf'),
        }

    def benchmark_inference(self, x: np.ndarray, n_runs: int = 100) -> dict:
        """
        Benchmark different inference methods.

        Args:
            x: Test input batch
            n_runs: Number of runs per method

        Returns:
            Dictionary with timing results
        """
        import time

        results = {}

        # Standard forward pass
        start = time.perf_counter()
        for _ in range(n_runs):
            self.forward_standard(x)
        results['standard'] = (time.perf_counter() - start) / n_runs

        # Sparse forward pass
        start = time.perf_counter()
        for _ in range(n_runs):
            self.forward_sparse(x)
        results['sparse'] = (time.perf_counter() - start) / n_runs

        # Ternary optimized forward pass
        start = time.perf_counter()
        for _ in range(n_runs):
            self.forward_ternary_optimized(x)
        results['ternary_optimized'] = (time.perf_counter() - start) / n_runs

        return results


def compare_inference_methods():
    """Compare different inference methods on a test network."""
    from ternary_alpha.utils import create_synthetic_dataset

    # Create test network
    network = TernaryNeuralNetwork(
        input_dim=20,
        hidden_dim=32,
        output_dim=10
    )

    # Create test data
    x_test, _ = create_synthetic_dataset(1000, 20, 10, random_seed=42)

    # Initialize inference engine
    inf_engine = TernaryInferenceOptimized(network)

    # Get statistics
    stats = inf_engine.get_statistics()
    print("Network Statistics:")
    print(f"  Total parameters: {stats['total_parameters']}")
    print(f"  Active parameters: {stats['active_parameters']}")
    print(f"  Sparsity: {stats['sparsity']:.2%}")
    print(f"  Layer 1 sparsity: {stats['layer1_sparsity']:.2%}")
    print(f"  Layer 2 sparsity: {stats['layer2_sparsity']:.2%}")
    print(f"  Memory reduction factor: {stats['memory_reduction_factor']}x")

    # Benchmark
    print("\nInference Timing (100 runs, 1000 samples):")
    timings = inf_engine.benchmark_inference(x_test, n_runs=100)
    for method, time_per_run in timings.items():
        print(f"  {method:20s}: {time_per_run*1000:7.3f} ms")

    # Speedup relative to standard
    baseline = timings['standard']
    print("\nSpeedup vs Standard:")
    for method, time_per_run in timings.items():
        speedup = baseline / time_per_run
        print(f"  {method:20s}: {speedup:6.2f}x")


if __name__ == "__main__":
    compare_inference_methods()
