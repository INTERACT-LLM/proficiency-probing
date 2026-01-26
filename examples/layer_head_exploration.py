"""
Example: Exploring different layers and attention heads

This example demonstrates how to:
1. Compare probes on different layers
2. Extract embeddings from specific attention heads
3. Compare different pooling strategies
"""

from proficiency_probing import ProficiencyProbingPipeline
import numpy as np

np.random.seed(42)


def create_sample_data(num_samples=100):
    """Create simple sample data."""
    texts = []
    labels = []
    
    # Level 0: Very simple sentences
    for _ in range(num_samples // 4):
        texts.append(f"I like cats.")
        labels.append(0)
    
    # Level 1: Simple sentences with more words
    for _ in range(num_samples // 4):
        texts.append(f"I have been learning English for two years.")
        labels.append(1)
    
    # Level 2: Complex sentences
    for _ in range(num_samples // 4):
        texts.append(f"The research demonstrates that environmental factors significantly influence outcomes.")
        labels.append(2)
    
    # Level 3: Very complex sentences
    for _ in range(num_samples // 4):
        texts.append(f"Notwithstanding the methodological limitations, the empirical evidence substantiates the hypothesis.")
        labels.append(3)
    
    return texts, labels


def compare_layers():
    """Compare probes on different layers."""
    print("=" * 70)
    print("Comparing Different Layers")
    print("=" * 70)
    print()
    
    # Create data
    train_texts, train_labels = create_sample_data(200)
    test_texts, test_labels = create_sample_data(100)
    
    # Try different layers
    layers_to_test = [0, 6, 11, -1]  # First, middle, second-to-last, last
    
    results = {}
    
    for layer_idx in layers_to_test:
        print(f"\n{'='*70}")
        print(f"Layer {layer_idx}")
        print(f"{'='*70}")
        
        pipeline = ProficiencyProbingPipeline(
            model_name="bert-base-uncased",
            layer_index=layer_idx,
            head_index=None,
            pooling="mean"
        )
        
        # Fit
        pipeline.fit(
            texts=train_texts,
            labels=train_labels,
            epochs=10,
            verbose=False
        )
        
        # Evaluate
        metrics = pipeline.evaluate(
            texts=test_texts,
            labels=test_labels,
            verbose=False
        )
        
        results[f"Layer {layer_idx}"] = metrics
        print(f"Accuracy: {metrics['accuracy']:.4f}, MAE: {metrics['mae']:.4f}")
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    for layer_name, metrics in results.items():
        print(f"{layer_name}: Acc={metrics['accuracy']:.4f}, MAE={metrics['mae']:.4f}")
    print()


def compare_pooling_strategies():
    """Compare different pooling strategies."""
    print("=" * 70)
    print("Comparing Pooling Strategies")
    print("=" * 70)
    print()
    
    # Create data
    train_texts, train_labels = create_sample_data(200)
    test_texts, test_labels = create_sample_data(100)
    
    # Try different pooling strategies
    pooling_strategies = ["mean", "cls", "max"]
    
    results = {}
    
    for pooling in pooling_strategies:
        print(f"\n{'='*70}")
        print(f"Pooling: {pooling}")
        print(f"{'='*70}")
        
        pipeline = ProficiencyProbingPipeline(
            model_name="bert-base-uncased",
            layer_index=-1,
            head_index=None,
            pooling=pooling
        )
        
        # Fit
        pipeline.fit(
            texts=train_texts,
            labels=train_labels,
            epochs=10,
            verbose=False
        )
        
        # Evaluate
        metrics = pipeline.evaluate(
            texts=test_texts,
            labels=test_labels,
            verbose=False
        )
        
        results[f"Pooling {pooling}"] = metrics
        print(f"Accuracy: {metrics['accuracy']:.4f}, MAE: {metrics['mae']:.4f}")
    
    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    for strategy_name, metrics in results.items():
        print(f"{strategy_name}: Acc={metrics['accuracy']:.4f}, MAE={metrics['mae']:.4f}")
    print()


def explore_attention_heads():
    """Explore specific attention heads."""
    print("=" * 70)
    print("Exploring Specific Attention Heads")
    print("=" * 70)
    print()
    
    # Create data
    train_texts, train_labels = create_sample_data(200)
    test_texts, test_labels = create_sample_data(100)
    
    # Try a specific attention head vs full representation
    print("Comparing full representation vs specific attention head...")
    print()
    
    # Full representation
    print(f"{'='*70}")
    print("Full representation (all heads)")
    print(f"{'='*70}")
    
    pipeline_full = ProficiencyProbingPipeline(
        model_name="bert-base-uncased",
        layer_index=-1,
        head_index=None,  # Use all heads
        pooling="mean"
    )
    
    pipeline_full.fit(
        texts=train_texts,
        labels=train_labels,
        epochs=10,
        verbose=False
    )
    
    metrics_full = pipeline_full.evaluate(
        texts=test_texts,
        labels=test_labels,
        verbose=False
    )
    print(f"Accuracy: {metrics_full['accuracy']:.4f}, MAE: {metrics_full['mae']:.4f}")
    print()
    
    # Specific head (head 0)
    print(f"{'='*70}")
    print("Attention head 0")
    print(f"{'='*70}")
    
    pipeline_head = ProficiencyProbingPipeline(
        model_name="bert-base-uncased",
        layer_index=-1,
        head_index=0,  # Use specific head
        pooling="mean"  # Pooling still applies to attention patterns
    )
    
    pipeline_head.fit(
        texts=train_texts,
        labels=train_labels,
        epochs=10,
        verbose=False
    )
    
    metrics_head = pipeline_head.evaluate(
        texts=test_texts,
        labels=test_labels,
        verbose=False
    )
    print(f"Accuracy: {metrics_head['accuracy']:.4f}, MAE: {metrics_head['mae']:.4f}")
    print()
    
    print(f"{'='*70}")
    print("Summary")
    print(f"{'='*70}")
    print(f"Full representation: Acc={metrics_full['accuracy']:.4f}, MAE={metrics_full['mae']:.4f}")
    print(f"Head 0: Acc={metrics_head['accuracy']:.4f}, MAE={metrics_head['mae']:.4f}")
    print()


def main():
    print("=" * 70)
    print("Proficiency Probing Pipeline - Layer & Head Exploration")
    print("=" * 70)
    print()
    
    # Compare different layers
    compare_layers()
    
    # Compare pooling strategies
    compare_pooling_strategies()
    
    # Explore attention heads
    explore_attention_heads()
    
    print("=" * 70)
    print("Exploration completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
