"""
Pipeline Module: Orchestrates the full proficiency probing pipeline.
"""

from typing import Optional, List, Dict, Tuple, Union
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import os
import json

from .embedder import TextEmbedder
from .probe import OrdinalProbe


class ProficiencyProbingPipeline:
    """
    End-to-end pipeline for proficiency probing.
    
    This pipeline:
    1. Embeds texts using a transformer model
    2. Fits an ordinal regression probe on the embeddings
    3. Evaluates generalizability on different distributions
    
    Attributes:
        embedder: TextEmbedder instance
        probe: OrdinalProbe instance
        is_fitted: Whether the probe has been fitted
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        layer_index: int = -1,
        head_index: Optional[int] = None,
        pooling: str = "mean",
        device: Optional[str] = None
    ):
        """
        Initialize the pipeline.
        
        Args:
            model_name: Name or path of the HuggingFace model
            layer_index: Which layer to extract embeddings from
            head_index: Which attention head to use (None for full representation)
            pooling: Pooling strategy ('mean', 'cls', or 'max')
            device: Device to run on
        """
        self.embedder = TextEmbedder(
            model_name=model_name,
            device=device,
            layer_index=layer_index,
            head_index=head_index,
            pooling=pooling
        )
        self.probe = None
        self.is_fitted = False
        self.num_classes = None
        
    def fit(
        self,
        texts: List[str],
        labels: Union[List[int], np.ndarray],
        val_size: float = 0.2,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        embedding_batch_size: int = 32,
        max_length: int = 512,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Fit the pipeline on training data.
        
        Args:
            texts: List of input texts
            labels: Ordinal proficiency labels (0, 1, 2, ...)
            val_size: Proportion of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for probe training
            learning_rate: Learning rate for probe training
            weight_decay: L2 regularization strength
            embedding_batch_size: Batch size for embedding extraction
            max_length: Maximum sequence length for tokenization
            verbose: Whether to print progress
            
        Returns:
            Training history dictionary
        """
        if verbose:
            print(f"Pipeline Configuration:")
            print(f"  Model: {self.embedder.model_name}")
            print(f"  Layer: {self.embedder.layer_index}")
            print(f"  Head: {self.embedder.head_index}")
            print(f"  Pooling: {self.embedder.pooling}")
            print()
        
        # Convert labels to numpy array
        labels = np.array(labels)
        self.num_classes = len(np.unique(labels))
        
        if verbose:
            print(f"Embedding {len(texts)} texts...")
        
        # Extract embeddings
        embeddings = self.embedder.embed_texts(
            texts,
            batch_size=embedding_batch_size,
            max_length=max_length,
            show_progress=verbose
        )
        
        if verbose:
            print(f"Embedding dimension: {embeddings.shape[1]}")
            print(f"Number of classes: {self.num_classes}")
            print()
        
        # Split into train and validation
        X_train, X_val, y_train, y_val = train_test_split(
            embeddings,
            labels,
            test_size=val_size,
            stratify=labels,
            random_state=42
        )
        
        # Initialize probe
        self.probe = OrdinalProbe(
            input_dim=embeddings.shape[1],
            num_classes=self.num_classes
        )
        
        if verbose:
            print(f"Training probe...")
            print(f"  Train samples: {len(X_train)}")
            print(f"  Val samples: {len(X_val)}")
            print()
        
        # Train probe
        history = self.probe.fit(
            X_train,
            y_train,
            X_val,
            y_val,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            device=self.embedder.device,
            verbose=verbose
        )
        
        self.is_fitted = True
        
        return history
    
    def evaluate(
        self,
        texts: List[str],
        labels: Union[List[int], np.ndarray],
        batch_size: int = 32,
        embedding_batch_size: int = 32,
        max_length: int = 512,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the pipeline on test data.
        
        Args:
            texts: List of input texts
            labels: True ordinal labels
            batch_size: Batch size for probe evaluation
            embedding_batch_size: Batch size for embedding extraction
            max_length: Maximum sequence length
            verbose: Whether to print results
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")
        
        if verbose:
            print(f"Evaluating on {len(texts)} texts...")
        
        # Extract embeddings
        embeddings = self.embedder.embed_texts(
            texts,
            batch_size=embedding_batch_size,
            max_length=max_length,
            show_progress=verbose
        )
        
        # Evaluate probe
        labels = np.array(labels)
        metrics = self.probe.evaluate(
            embeddings,
            labels,
            batch_size=batch_size,
            device=self.embedder.device
        )
        
        if verbose:
            print(f"\nEvaluation Results:")
            print(f"  Loss: {metrics['loss']:.4f}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  MAE: {metrics['mae']:.4f}")
        
        return metrics
    
    def predict(
        self,
        texts: List[str],
        batch_size: int = 32,
        embedding_batch_size: int = 32,
        max_length: int = 512,
        return_probabilities: bool = False,
        verbose: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict proficiency labels for new texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for probe prediction
            embedding_batch_size: Batch size for embedding extraction
            max_length: Maximum sequence length
            return_probabilities: Whether to return class probabilities
            verbose: Whether to show progress
            
        Returns:
            Predicted labels, or (labels, probabilities) if return_probabilities=True
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before prediction")
        
        # Extract embeddings
        embeddings = self.embedder.embed_texts(
            texts,
            batch_size=embedding_batch_size,
            max_length=max_length,
            show_progress=verbose
        )
        
        # Predict
        if return_probabilities:
            probs = self.probe.predict_proba(
                embeddings,
                batch_size=batch_size,
                device=self.embedder.device
            )
            labels = np.argmax(probs, axis=1)
            return labels, probs
        else:
            labels = self.probe.predict_labels(
                embeddings,
                batch_size=batch_size,
                device=self.embedder.device
            )
            return labels
    
    def cross_distribution_evaluation(
        self,
        distributions: Dict[str, Tuple[List[str], Union[List[int], np.ndarray]]],
        embedding_batch_size: int = 32,
        max_length: int = 512,
        verbose: bool = True
    ) -> Dict[str, Dict[str, float]]:
        """
        Evaluate generalizability across multiple distributions.
        
        Args:
            distributions: Dictionary mapping distribution names to (texts, labels) tuples
            embedding_batch_size: Batch size for embedding extraction
            max_length: Maximum sequence length
            verbose: Whether to print results
            
        Returns:
            Dictionary mapping distribution names to metrics
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before evaluation")
        
        results = {}
        
        for dist_name, (texts, labels) in distributions.items():
            if verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating on distribution: {dist_name}")
                print(f"{'='*60}")
            
            metrics = self.evaluate(
                texts,
                labels,
                embedding_batch_size=embedding_batch_size,
                max_length=max_length,
                verbose=verbose
            )
            
            results[dist_name] = metrics
        
        if verbose:
            print(f"\n{'='*60}")
            print("Cross-Distribution Summary")
            print(f"{'='*60}")
            for dist_name, metrics in results.items():
                print(f"{dist_name}:")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  MAE: {metrics['mae']:.4f}")
        
        return results
    
    def save(self, path: str):
        """
        Save the pipeline to disk.
        
        Args:
            path: Directory path to save the pipeline
        """
        if not self.is_fitted:
            raise ValueError("Pipeline must be fitted before saving")
        
        os.makedirs(path, exist_ok=True)
        
        # Save configuration
        config = {
            "model_name": self.embedder.model_name,
            "layer_index": self.embedder.layer_index,
            "head_index": self.embedder.head_index,
            "pooling": self.embedder.pooling,
            "num_classes": self.num_classes,
            "input_dim": self.probe.input_dim
        }
        
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        
        # Save probe weights
        torch.save(self.probe.state_dict(), os.path.join(path, "probe.pt"))
        
        print(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str, device: Optional[str] = None):
        """
        Load a pipeline from disk.
        
        Args:
            path: Directory path containing the saved pipeline
            device: Device to load the model on
            
        Returns:
            Loaded ProficiencyProbingPipeline instance
        """
        # Load configuration
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        
        # Create pipeline
        pipeline = cls(
            model_name=config["model_name"],
            layer_index=config["layer_index"],
            head_index=config["head_index"],
            pooling=config["pooling"],
            device=device
        )
        
        # Load probe
        pipeline.num_classes = config["num_classes"]
        pipeline.probe = OrdinalProbe(
            input_dim=config["input_dim"],
            num_classes=config["num_classes"]
        )
        pipeline.probe.load_state_dict(
            torch.load(os.path.join(path, "probe.pt"), map_location=device or "cpu")
        )
        pipeline.is_fitted = True
        
        print(f"Pipeline loaded from {path}")
        
        return pipeline
