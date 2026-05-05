#!/usr/bin/env python3
"""
Example: Deep Ternary Neural Network with Synthetic Data

Demonstrates:
- Creating a deep network with multiple hidden layers
- Generating synthetic classification data
- Training and evaluating the network
- Analyzing sparsity and performance metrics
- Comparing performance across different architectures
"""

import numpy as np
import matplotlib.pyplot as plt
from ternary_alpha.network import TernaryNeuralNetwork
from ternary_alpha.utils import (
    create_synthetic_dataset,
    split_train_val,
    evaluate_model
)


def plot_training_history(history, title="Training History"):
    """Plot training and validation metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Cross-Entropy Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history['train_accuracy'], label='Train Accuracy', linewidth=2)
    axes[1].plot(history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Classification Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def plot_sparsity_analysis(network, title="Weight Sparsity Analysis"):
    """Plot sparsity per layer."""
    stats = network.get_network_stats()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Sparsity per layer
    layers = [s['layer'] for s in stats['sparsity_per_layer']]
    sparsities = [s['sparsity'] for s in stats['sparsity_per_layer']]
    
    axes[0].bar(layers, sparsities, color='steelblue', alpha=0.7)
    axes[0].axhline(y=stats['overall_sparsity'], color='red', linestyle='--', 
                    label=f"Overall: {stats['overall_sparsity']:.2%}", linewidth=2)
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Sparsity')
    axes[0].set_title('Sparsity per Layer')
    axes[0].set_ylim([0, 1])
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')

    # Parameters per layer
    params = [s['total'] for s in stats['sparsity_per_layer']]
    active = [s['active'] for s in stats['sparsity_per_layer']]
    
    x = np.arange(len(layers))
    width = 0.35
    
    axes[1].bar(x - width/2, params, width, label='Total Parameters', alpha=0.7)
    axes[1].bar(x + width/2, active, width, label='Active Parameters', alpha=0.7)
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Number of Parameters')
    axes[1].set_title('Parameters per Layer')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(layers)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def example_basic_deep_network():
    """
    Example 1: Basic Deep Network
    A simple 4-layer network with synthetic data
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 1: Basic Deep Network (4 layers)")
    print("=" * 80)

    # Network architecture: 20 -> 64 -> 32 -> 16 -> 10
    layer_dims = [20, 64, 32, 16, 10]
    network = TernaryNeuralNetwork(
        layer_dims=layer_dims,
        activation='relu',
        use_bias=True,
        random_seed=42
    )

    print(f"\nNetwork Architecture: {layer_dims}")
    network.summary()

    # Generate synthetic data
    print("\nGenerating synthetic dataset...")
    n_samples = 2000
    X, y = create_synthetic_dataset(n_samples, 20, 10, random_seed=42)
    X_train, X_val, y_train, y_val = split_train_val(X, y, val_split=0.2, random_seed=42)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Output classes: {y_train.shape[1]}")

    # Train
    print("\nTraining network for 100 epochs...")
    history = network.train(
        X_train, y_train,
        x_val=X_val, y_val=y_val,
        epochs=100,
        batch_size=32,
        learning_rate=0.01,
        verbose=True
    )

    # Evaluate
    print("\nEvaluating on validation set...")
    y_val_pred, y_val_classes = network.predict(X_val)
    metrics = evaluate_model(y_val, y_val_pred, one_hot_encoded=True)

    print(f"Validation Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Validation Precision: {metrics['precision']:.4f}")
    print(f"Validation Recall:    {metrics['recall']:.4f}")
    print(f"Validation F1 Score:  {metrics['f1']:.4f}")

    # Network statistics
    print("\nNetwork Statistics:")
    stats = network.get_network_stats()
    print(f"Total Parameters:      {stats['total_parameters']:,}")
    print(f"Active Parameters:     {stats['active_parameters']:,}")
    print(f"Overall Sparsity:      {stats['overall_sparsity']:.2%}")
    print(f"Memory Reduction:      {stats['memory_reduction_factor']}x")

    print("\nSparsity per Layer:")
    for layer_stat in stats['sparsity_per_layer']:
        print(f"  Layer {layer_stat['layer']}: {layer_stat['sparsity']:6.2%} "
              f"({layer_stat['active']:4d}/{layer_stat['total']:4d} active)")

    return network, history, X_val, y_val


def example_very_deep_network():
    """
    Example 2: Very Deep Network
    A 7-layer deep network for comparison
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Very Deep Network (7 layers)")
    print("=" * 80)

    # Network architecture: 20 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 10
    layer_dims = [20, 128, 64, 32, 16, 8, 4, 10]
    network = TernaryNeuralNetwork(
        layer_dims=layer_dims,
        activation='relu',
        use_bias=True,
        random_seed=42
    )

    print(f"\nNetwork Architecture: {layer_dims}")
    network.summary()

    # Generate synthetic data
    print("\nGenerating synthetic dataset...")
    n_samples = 3000
    X, y = create_synthetic_dataset(n_samples, 20, 10, random_seed=43)
    X_train, X_val, y_train, y_val = split_train_val(X, y, val_split=0.2, random_seed=43)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")

    # Train
    print("\nTraining network for 150 epochs...")
    history = network.train(
        X_train, y_train,
        x_val=X_val, y_val=y_val,
        epochs=150,
        batch_size=32,
        learning_rate=0.01,
        verbose=True
    )

    # Evaluate
    print("\nEvaluating on validation set...")
    y_val_pred, y_val_classes = network.predict(X_val)
    metrics = evaluate_model(y_val, y_val_pred, one_hot_encoded=True)

    print(f"Validation Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Validation Precision: {metrics['precision']:.4f}")
    print(f"Validation Recall:    {metrics['recall']:.4f}")
    print(f"Validation F1 Score:  {metrics['f1']:.4f}")

    # Network statistics
    print("\nNetwork Statistics:")
    stats = network.get_network_stats()
    print(f"Total Parameters:      {stats['total_parameters']:,}")
    print(f"Active Parameters:     {stats['active_parameters']:,}")
    print(f"Overall Sparsity:      {stats['overall_sparsity']:.2%}")

    print("\nSparsity per Layer:")
    for layer_stat in stats['sparsity_per_layer']:
        print(f"  Layer {layer_stat['layer']}: {layer_stat['sparsity']:6.2%} "
              f"({layer_stat['active']:5d}/{layer_stat['total']:5d} active)")

    return network, history, X_val, y_val


def example_architecture_comparison():
    """
    Example 3: Compare Different Architectures
    Train multiple networks with different depths
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Architecture Comparison")
    print("=" * 80)

    # Generate common dataset
    print("\nGenerating synthetic dataset...")
    n_samples = 2000
    X, y = create_synthetic_dataset(n_samples, 20, 10, random_seed=42)
    X_train, X_val, y_train, y_val = split_train_val(X, y, val_split=0.2, random_seed=42)

    architectures = [
        [20, 32, 10],           # Shallow (2 layers)
        [20, 64, 32, 10],       # Medium (3 layers)
        [20, 64, 32, 16, 10],   # Deep (4 layers)
        [20, 128, 64, 32, 16, 8, 10],  # Very deep (6 layers)
    ]

    results = []

    for arch in architectures:
        print(f"\n{'=' * 80}")
        print(f"Training architecture: {arch}")
        print(f"{'=' * 80}")

        network = TernaryNeuralNetwork(
            layer_dims=arch,
            activation='relu',
            use_bias=True,
            random_seed=42
        )

        # Quick summary
        stats = network.get_network_stats()
        print(f"Total Parameters: {stats['total_parameters']:,}")

        # Train
        print(f"Training for 80 epochs...")
        history = network.train(
            X_train, y_train,
            x_val=X_val, y_val=y_val,
            epochs=80,
            batch_size=32,
            learning_rate=0.01,
            verbose=False
        )

        # Evaluate
        y_val_pred, _ = network.predict(X_val)
        metrics = evaluate_model(y_val, y_val_pred, one_hot_encoded=True)

        final_train_loss = history['train_loss'][-1]
        final_train_acc = history['train_accuracy'][-1]
        final_val_loss = history['val_loss'][-1]
        final_val_acc = history['val_accuracy'][-1]

        print(f"Final Train Loss: {final_train_loss:.4f}, Acc: {final_train_acc:.4f}")
        print(f"Final Val Loss:   {final_val_loss:.4f}, Acc: {final_val_acc:.4f}")
        print(f"Sparsity:         {stats['overall_sparsity']:.2%}")

        results.append({
            'architecture': arch,
            'n_layers': len(arch) - 1,
            'total_params': stats['total_parameters'],
            'sparsity': stats['overall_sparsity'],
            'train_loss': final_train_loss,
            'train_acc': final_train_acc,
            'val_loss': final_val_loss,
            'val_acc': final_val_acc,
            'history': history
        })

    # Summary table
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print(f"{'Architecture':<30} {'Layers':<8} {'Params':<10} {'Sparsity':<10} {'Val Acc':<10}")
    print("-" * 80)

    for r in results:
        arch_str = str(r['architecture'])
        print(f"{arch_str:<30} {r['n_layers']:<8} {r['total_params']:<10} "
              f"{r['sparsity']:>8.2%}  {r['val_acc']:>8.4f}")

    return results


def example_deep_network_with_tanh():
    """
    Example 4: Deep Network with Different Activation Function
    Using tanh instead of relu
    """
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Deep Network with Tanh Activation")
    print("=" * 80)

    # Network architecture
    layer_dims = [20, 64, 32, 16, 10]
    network = TernaryNeuralNetwork(
        layer_dims=layer_dims,
        activation='tanh',  # Use tanh instead of relu
        use_bias=True,
        random_seed=42
    )

    print(f"\nNetwork Architecture: {layer_dims}")
    print(f"Activation Function: tanh")
    network.summary()

    # Generate data
    print("\nGenerating synthetic dataset...")
    n_samples = 2000
    X, y = create_synthetic_dataset(n_samples, 20, 10, random_seed=44)
    X_train, X_val, y_train, y_val = split_train_val(X, y, val_split=0.2, random_seed=44)

    # Train
    print("\nTraining network for 100 epochs...")
    history = network.train(
        X_train, y_train,
        x_val=X_val, y_val=y_val,
        epochs=100,
        batch_size=32,
        learning_rate=0.01,
        verbose=True
    )

    # Evaluate
    print("\nEvaluating on validation set...")
    y_val_pred, _ = network.predict(X_val)
    metrics = evaluate_model(y_val, y_val_pred, one_hot_encoded=True)

    print(f"\nValidation Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")

    stats = network.get_network_stats()
    print(f"\nSparsity: {stats['overall_sparsity']:.2%}")

    return network, history


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("DEEP TERNARY NEURAL NETWORK - EXAMPLES WITH SYNTHETIC DATA")
    print("=" * 80)

    # Example 1: Basic deep network
    network1, history1, X_val1, y_val1 = example_basic_deep_network()

    # Example 2: Very deep network
    network2, history2, X_val2, y_val2 = example_very_deep_network()

    # Example 3: Architecture comparison
    results = example_architecture_comparison()

    # Example 4: Different activation function
    network4, history4 = example_deep_network_with_tanh()

    # Generate visualizations
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Plot 1: Training history for Example 1
    print("Generating training history plot for Example 1...")
    fig1 = plot_training_history(history1, "Example 1: Basic Deep Network (4 layers)")
    fig1.savefig('example1_training_history.png', dpi=150, bbox_inches='tight')
    print("  Saved: example1_training_history.png")

    # Plot 2: Sparsity analysis for Example 1
    print("Generating sparsity analysis plot for Example 1...")
    fig2 = plot_sparsity_analysis(network1, "Example 1: Weight Sparsity Analysis")
    fig2.savefig('example1_sparsity_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: example1_sparsity_analysis.png")

    # Plot 3: Training history for Example 2
    print("Generating training history plot for Example 2...")
    fig3 = plot_training_history(history2, "Example 2: Very Deep Network (7 layers)")
    fig3.savefig('example2_training_history.png', dpi=150, bbox_inches='tight')
    print("  Saved: example2_training_history.png")

    # Plot 4: Sparsity analysis for Example 2
    print("Generating sparsity analysis plot for Example 2...")
    fig4 = plot_sparsity_analysis(network2, "Example 2: Weight Sparsity Analysis")
    fig4.savefig('example2_sparsity_analysis.png', dpi=150, bbox_inches='tight')
    print("  Saved: example2_sparsity_analysis.png")

    # Plot 5: Architecture comparison
    print("Generating architecture comparison plots...")
    fig5, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Validation accuracy comparison
    arch_labels = [f"{len(r['architecture'])-1}L" for r in results]
    axes[0, 0].plot(arch_labels, [r['val_acc'] for r in results], 'o-', linewidth=2, markersize=8)
    axes[0, 0].set_ylabel('Validation Accuracy')
    axes[0, 0].set_title('Accuracy vs Network Depth')
    axes[0, 0].grid(True, alpha=0.3)

    # Parameters vs sparsity
    axes[0, 1].scatter([r['total_params'] for r in results], 
                       [r['sparsity'] for r in results], s=200, alpha=0.7)
    for i, r in enumerate(results):
        axes[0, 1].annotate(f"{len(r['architecture'])-1}L", 
                           (r['total_params'], r['sparsity']),
                           xytext=(5, 5), textcoords='offset points')
    axes[0, 1].set_xlabel('Total Parameters')
    axes[0, 1].set_ylabel('Sparsity')
    axes[0, 1].set_title('Parameters vs Sparsity')
    axes[0, 1].grid(True, alpha=0.3)

    # Validation loss comparison
    axes[1, 0].plot(arch_labels, [r['val_loss'] for r in results], 'o-', 
                   linewidth=2, markersize=8, color='orange')
    axes[1, 0].set_ylabel('Validation Loss')
    axes[1, 0].set_title('Loss vs Network Depth')
    axes[1, 0].grid(True, alpha=0.3)

    # Training curves for architectures
    for r in results:
        axes[1, 1].plot(r['history']['val_accuracy'], 
                       label=f"{len(r['architecture'])-1}L", linewidth=2, alpha=0.7)
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Validation Accuracy')
    axes[1, 1].set_title('Validation Accuracy Over Training')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.suptitle('Architecture Comparison Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    fig5.savefig('example3_architecture_comparison.png', dpi=150, bbox_inches='tight')
    print("  Saved: example3_architecture_comparison.png")

    # Plot 6: Training history for Example 4 (Tanh)
    print("Generating training history plot for Example 4...")
    fig6 = plot_training_history(history4, "Example 4: Deep Network with Tanh Activation")
    fig6.savefig('example4_training_history_tanh.png', dpi=150, bbox_inches='tight')
    print("  Saved: example4_training_history_tanh.png")

    print("\n" + "=" * 80)
    print("ALL EXAMPLES COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nGenerated plots:")
    print("  - example1_training_history.png")
    print("  - example1_sparsity_analysis.png")
    print("  - example2_training_history.png")
    print("  - example2_sparsity_analysis.png")
    print("  - example3_architecture_comparison.png")
    print("  - example4_training_history_tanh.png")


if __name__ == "__main__":
    main()
