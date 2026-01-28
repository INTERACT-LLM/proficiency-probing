# Proficiency Probing

A flexible pipeline for embedding texts, fitting a linear probe with ordinal regression, and testing generalizability of the probe on other distributions.

## Overview

This package provides a complete pipeline for proficiency probing in NLP models. It allows you to:

- **Embed texts** using any HuggingFace transformer model
- **Extract representations** from any layer or attention head
- **Fit ordinal regression probes** to predict proficiency levels
- **Test generalizability** across different data distributions

The pipeline is designed to be modular and easy to use, with sensible defaults that work out of the box.

## Features

- 🔧 **Flexible Model Support**: Works with any HuggingFace transformer model
- 🎯 **Layer & Head Selection**: Extract embeddings from any layer or attention head
- 📊 **Ordinal Regression**: Proper handling of ordinal proficiency labels
- 🔄 **Cross-Distribution Testing**: Evaluate generalizability on different distributions
- 💾 **Save & Load**: Persist trained probes for later use
- ⚡ **GPU Support**: Automatic GPU detection and usage

## Installation

```bash
# Clone the repository
git clone https://github.com/INTERACT-LLM/Proficiency_Probing.git
cd Proficiency_Probing

# Install dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Quick Start

```python
from proficiency_probing import ProficiencyProbingPipeline

# Create sample data
texts = [
    "Hello, my name is John.",  # Beginner (0)
    "I have been studying for three years.",  # Intermediate (1)
    "The implications are significant.",  # Advanced (2)
    "The epistemological framework is complex.",  # Expert (3)
]
labels = [0, 1, 2, 3]  # Ordinal proficiency levels

# Initialize pipeline
pipeline = ProficiencyProbingPipeline(
    model_name="bert-base-uncased",
    layer_index=-1,  # Use last layer
    pooling="mean"   # Mean pooling over tokens
)

# Fit the probe
pipeline.fit(texts, labels, epochs=20)

# Make predictions
new_texts = ["I like cats.", "The research demonstrates key findings."]
predictions = pipeline.predict(new_texts)
print(predictions)  # [0, 2]

# Evaluate on different distribution
test_texts = ["Simple sentence.", "Complex analytical statement."]
test_labels = [0, 2]
metrics = pipeline.evaluate(test_texts, test_labels)
print(f"Accuracy: {metrics['accuracy']:.2f}")
```

## Usage

### Basic Pipeline

```python
from proficiency_probing import ProficiencyProbingPipeline

# Initialize with custom configuration
pipeline = ProficiencyProbingPipeline(
    model_name="bert-base-uncased",  # Any HuggingFace model
    layer_index=-1,                   # -1 for last layer, 0 for first, etc.
    head_index=None,                  # None for full representation, 0-11 for specific head
    pooling="mean",                   # "mean", "cls", or "max"
    device="cuda"                     # "cuda" or "cpu"
)

# Fit on training data
history = pipeline.fit(
    texts=train_texts,
    labels=train_labels,
    val_size=0.2,          # Validation split
    epochs=50,
    batch_size=32,
    learning_rate=0.001
)

# Evaluate on test data
metrics = pipeline.evaluate(test_texts, test_labels)
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")
```

### Cross-Distribution Evaluation

```python
# Evaluate generalizability across distributions
distributions = {
    "In-Domain": (test_texts, test_labels),
    "Domain-A": (domain_a_texts, domain_a_labels),
    "Domain-B": (domain_b_texts, domain_b_labels),
}

results = pipeline.cross_distribution_evaluation(distributions)

# Results is a dict mapping distribution names to metrics
for dist_name, metrics in results.items():
    print(f"{dist_name}: Acc={metrics['accuracy']:.4f}")
```

### Exploring Layers and Heads

```python
# Extract from a specific layer
pipeline = ProficiencyProbingPipeline(
    model_name="bert-base-uncased",
    layer_index=6,  # Middle layer
    pooling="mean"
)

# Extract from a specific attention head
pipeline = ProficiencyProbingPipeline(
    model_name="bert-base-uncased",
    layer_index=-1,  # Last layer
    head_index=0,    # First attention head
    pooling="mean"
)
```

### Save and Load

```python
# Save trained pipeline
pipeline.save("/path/to/save/directory")

# Load pipeline
from proficiency_probing import ProficiencyProbingPipeline
loaded_pipeline = ProficiencyProbingPipeline.load("/path/to/save/directory")

# Use loaded pipeline
predictions = loaded_pipeline.predict(new_texts)
```

### Predictions with Probabilities

```python
# Get class probabilities
predictions, probabilities = pipeline.predict(
    texts=new_texts,
    return_probabilities=True
)

# probabilities shape: [num_texts, num_classes]
for text, pred, probs in zip(new_texts, predictions, probabilities):
    print(f"Text: {text}")
    print(f"Predicted: {pred}")
    print(f"Probabilities: {probs}")
```

## Architecture

The pipeline consists of three main components:

### 1. TextEmbedder
Extracts embeddings from transformer models:
- Supports any HuggingFace model
- Can extract from any layer (hidden states)
- Can extract from specific attention heads
- Multiple pooling strategies (mean, cls, max)

### 2. OrdinalProbe
Linear probe with ordinal regression:
- Uses cumulative link model for ordinal labels
- Learns to respect the ordering of proficiency levels
- More appropriate than standard classification for ordinal data
- Includes learnable threshold parameters

### 3. ProficiencyProbingPipeline
Orchestrates the full workflow:
- Manages embedding extraction and probe training
- Handles train/validation splits
- Provides evaluation and prediction methods
- Supports cross-distribution testing
- Save/load functionality

## Examples

See the `examples/` directory for complete working examples:

- `basic_usage.py`: Introduction to the pipeline with synthetic data
- `layer_head_exploration.py`: Comparing different layers, heads, and pooling strategies

Run examples:
```bash
cd examples
python basic_usage.py
python layer_head_exploration.py
```

## Label Format

Proficiency labels should be integers representing ordinal levels, starting from 0:

- **0**: Beginner / A1
- **1**: Intermediate / A2-B1
- **2**: Advanced / B2-C1
- **3**: Expert / C2

The specific meaning of each level is application-dependent. The key requirement is that the labels are ordinal (i.e., level 2 > level 1 > level 0).

## API Reference

### ProficiencyProbingPipeline

#### `__init__(model_name, layer_index, head_index, pooling, device)`
Initialize the pipeline.

#### `fit(texts, labels, **kwargs)`
Fit the probe on training data. Returns training history.

#### `evaluate(texts, labels, **kwargs)`
Evaluate on test data. Returns metrics dict.

#### `predict(texts, return_probabilities, **kwargs)`
Predict labels for new texts.

#### `cross_distribution_evaluation(distributions, **kwargs)`
Evaluate across multiple distributions.

#### `save(path)` / `load(path)`
Save/load trained pipeline.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- scikit-learn 1.3+
- NumPy 1.24+

See `requirements.txt` for full dependencies.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{proficiency_probing,
  title={Proficiency Probing: A Pipeline for Ordinal Regression on Text Embeddings},
  author={INTERACT-LLM},
  year={2024},
  url={https://github.com/INTERACT-LLM/Proficiency_Probing}
}
```

## License

This project is open source. See LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
