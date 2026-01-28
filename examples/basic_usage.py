"""
Example: Basic usage of the Proficiency Probing Pipeline

This example demonstrates how to:
1. Create sample data with ordinal proficiency labels
2. Fit a probe on the data
3. Evaluate on a different distribution
4. Save and load the pipeline
"""

from proficiency_probing import ProficiencyProbingPipeline
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)


def create_sample_data(num_samples=100, difficulty_level="easy"):
    """
    Create sample texts with ordinal proficiency labels.
    
    Labels: 0=Beginner, 1=Intermediate, 2=Advanced, 3=Expert
    """
    templates = {
        0: [  # Beginner
            "Hello, my name is {}.",
            "I like {}.",
            "The {} is nice.",
        ],
        1: [  # Intermediate
            "I have been learning {} for about {} years.",
            "Yesterday, I went to {} and saw {}.",
            "I think {} is very interesting because {}.",
        ],
        2: [  # Advanced
            "Having studied {} extensively, I believe that {} represents {}.",
            "The implications of {} on {} cannot be understated, particularly when considering {}.",
            "While {} may seem straightforward, the nuances of {} require careful consideration.",
        ],
        3: [  # Expert
            "The epistemological ramifications of {} vis-à-vis {} underscore the inherent complexities.",
            "Notwithstanding the conventional wisdom regarding {}, contemporary scholarship suggests {}.",
            "The dialectical relationship between {} and {} necessitates a paradigmatic shift in {}.",
        ]
    }
    
    words = ["science", "technology", "nature", "education", "research", "theory", "practice", 
             "development", "innovation", "analysis", "methodology", "framework", "context"]
    
    texts = []
    labels = []
    
    # Adjust distribution based on difficulty
    if difficulty_level == "easy":
        label_distribution = [0.4, 0.3, 0.2, 0.1]  # More beginners
    elif difficulty_level == "hard":
        label_distribution = [0.1, 0.2, 0.3, 0.4]  # More experts
    else:  # balanced
        label_distribution = [0.25, 0.25, 0.25, 0.25]
    
    for _ in range(num_samples):
        # Sample a proficiency level
        label = np.random.choice([0, 1, 2, 3], p=label_distribution)
        
        # Generate text for that level
        template = np.random.choice(templates[label])
        num_placeholders = template.count("{}")
        words_sample = np.random.choice(words, size=num_placeholders, replace=True)
        text = template.format(*words_sample)
        
        texts.append(text)
        labels.append(label)
    
    return texts, labels


def main():
    print("=" * 70)
    print("Proficiency Probing Pipeline - Basic Example")
    print("=" * 70)
    print()
    
    # Create training data (balanced distribution)
    print("Creating sample training data (balanced distribution)...")
    train_texts, train_labels = create_sample_data(num_samples=200, difficulty_level="balanced")
    print(f"  Generated {len(train_texts)} training samples")
    print(f"  Label distribution: {np.bincount(train_labels)}")
    print()
    
    # Create test data (similar distribution)
    print("Creating sample test data (balanced distribution)...")
    test_texts, test_labels = create_sample_data(num_samples=100, difficulty_level="balanced")
    print(f"  Generated {len(test_texts)} test samples")
    print()
    
    # Create out-of-distribution test data
    print("Creating out-of-distribution data (easy distribution)...")
    ood_easy_texts, ood_easy_labels = create_sample_data(num_samples=100, difficulty_level="easy")
    print(f"  Generated {len(ood_easy_texts)} OOD (easy) samples")
    print()
    
    print("Creating out-of-distribution data (hard distribution)...")
    ood_hard_texts, ood_hard_labels = create_sample_data(num_samples=100, difficulty_level="hard")
    print(f"  Generated {len(ood_hard_texts)} OOD (hard) samples")
    print()
    
    # Initialize pipeline
    print("Initializing pipeline with BERT...")
    pipeline = ProficiencyProbingPipeline(
        model_name="bert-base-uncased",
        layer_index=-1,  # Use last layer
        head_index=None,  # Use full representation (not specific head)
        pooling="mean"    # Mean pooling over tokens
    )
    print()
    
    # Fit the pipeline
    print("=" * 70)
    print("Training Phase")
    print("=" * 70)
    history = pipeline.fit(
        texts=train_texts,
        labels=train_labels,
        val_size=0.2,
        epochs=20,  # Reduced for demo
        batch_size=16,
        learning_rate=0.001,
        embedding_batch_size=32,
        verbose=True
    )
    print()
    
    # Evaluate on in-distribution test set
    print("=" * 70)
    print("In-Distribution Evaluation")
    print("=" * 70)
    test_metrics = pipeline.evaluate(
        texts=test_texts,
        labels=test_labels,
        verbose=True
    )
    print()
    
    # Cross-distribution evaluation
    print("=" * 70)
    print("Cross-Distribution Evaluation")
    print("=" * 70)
    distributions = {
        "Test (Balanced)": (test_texts, test_labels),
        "OOD (Easy)": (ood_easy_texts, ood_easy_labels),
        "OOD (Hard)": (ood_hard_texts, ood_hard_labels),
    }
    
    cross_dist_results = pipeline.cross_distribution_evaluation(
        distributions=distributions,
        verbose=True
    )
    print()
    
    # Example predictions
    print("=" * 70)
    print("Example Predictions")
    print("=" * 70)
    sample_texts = [
        "Hello, my name is John.",  # Should be Beginner (0)
        "I have been studying science for three years.",  # Should be Intermediate (1)
        "The implications of this theory are significant.",  # Should be Advanced (2)
        "The epistemological framework necessitates careful analysis.",  # Should be Expert (3)
    ]
    
    predictions, probabilities = pipeline.predict(
        texts=sample_texts,
        return_probabilities=True,
        verbose=False
    )
    
    label_names = ["Beginner", "Intermediate", "Advanced", "Expert"]
    
    for text, pred, probs in zip(sample_texts, predictions, probabilities):
        print(f"\nText: {text}")
        print(f"Predicted level: {label_names[pred]} ({pred})")
        print(f"Probabilities: ", end="")
        for i, p in enumerate(probs):
            print(f"{label_names[i]}={p:.3f} ", end="")
        print()
    
    print()
    
    # Save pipeline
    print("=" * 70)
    print("Saving and Loading Pipeline")
    print("=" * 70)
    save_path = "/tmp/proficiency_probe"
    pipeline.save(save_path)
    print()
    
    # Load pipeline
    loaded_pipeline = ProficiencyProbingPipeline.load(save_path)
    print()
    
    # Verify loaded pipeline works
    print("Testing loaded pipeline...")
    loaded_predictions = loaded_pipeline.predict(sample_texts, verbose=False)
    print(f"Predictions match: {np.array_equal(predictions, loaded_predictions)}")
    print()
    
    print("=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
