"""
Ordinal Probe Module: Linear probe for ordinal regression on proficiency levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from tqdm import tqdm


class OrdinalProbe(nn.Module):
    """
    Linear probe for ordinal regression.
    
    Uses a cumulative link model approach where we predict whether the input
    belongs to class k or higher for each threshold k. This respects the
    ordinal nature of proficiency labels.
    
    Attributes:
        input_dim: Dimension of input embeddings
        num_classes: Number of ordinal classes
        linear: Linear projection layer
        thresholds: Learnable threshold parameters
    """
    
    def __init__(self, input_dim: int, num_classes: int):
        """
        Initialize the ordinal probe.
        
        Args:
            input_dim: Dimension of input embeddings
            num_classes: Number of ordinal proficiency levels
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # Linear projection to a scalar
        self.linear = nn.Linear(input_dim, 1)
        
        # Learnable thresholds (num_classes - 1 thresholds for num_classes classes)
        # Initialize with evenly spaced values
        initial_thresholds = torch.linspace(-2, 2, num_classes - 1)
        self.thresholds = nn.Parameter(initial_thresholds)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            
        Returns:
            Class probabilities [batch_size, num_classes]
        """
        # Project to scalar
        projected = self.linear(x)  # [batch_size, 1]
        
        # Ensure thresholds are sorted
        sorted_thresholds = torch.sort(self.thresholds)[0]
        
        # Compute cumulative probabilities
        # P(y >= k) = sigmoid(projected - threshold_k)
        cumulative_probs = []
        for threshold in sorted_thresholds:
            prob = torch.sigmoid(projected - threshold)
            cumulative_probs.append(prob)
        
        cumulative_probs = torch.cat(cumulative_probs, dim=1)  # [batch_size, num_classes-1]
        
        # Convert to class probabilities
        # P(y = 0) = 1 - P(y >= 1)
        # P(y = k) = P(y >= k) - P(y >= k+1) for 0 < k < num_classes-1
        # P(y = num_classes-1) = P(y >= num_classes-1)
        
        probs = []
        # First class
        probs.append(1 - cumulative_probs[:, 0:1])
        # Middle classes
        for i in range(cumulative_probs.size(1) - 1):
            prob = cumulative_probs[:, i:i+1] - cumulative_probs[:, i+1:i+2]
            probs.append(prob)
        # Last class
        probs.append(cumulative_probs[:, -1:])
        
        return torch.cat(probs, dim=1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class labels.
        
        Args:
            x: Input embeddings [batch_size, input_dim]
            
        Returns:
            Predicted class labels [batch_size]
        """
        probs = self.forward(x)
        return torch.argmax(probs, dim=1)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        device: Optional[str] = None,
        verbose: bool = True
    ) -> Dict[str, list]:
        """
        Fit the probe to training data.
        
        Args:
            X_train: Training embeddings [num_samples, input_dim]
            y_train: Training labels [num_samples] (ordinal values)
            X_val: Optional validation embeddings
            y_val: Optional validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            weight_decay: L2 regularization strength
            device: Device to train on
            verbose: Whether to print training progress
            
        Returns:
            Dictionary with training history
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(device)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
        # Setup optimizer
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Training history
        history = {
            "train_loss": [],
            "train_acc": [],
            "train_mae": [],
            "train_qwk": []
        }
        
        if X_val is not None and y_val is not None:
            history["val_loss"] = []
            history["val_acc"] = []
            history["val_mae"] = []
            history["val_qwk"] = [] # quadratic weighted kappa
        
        # Training loop
        for epoch in range(epochs):
            self.train()
            train_losses = []
            train_preds = []
            train_labels = []
            
            iterator = train_loader
            if verbose:
                iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            
            for batch_X, batch_y in iterator:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                probs = self.forward(batch_X)
                
                # Cross-entropy loss
                loss = F.cross_entropy(probs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                train_losses.append(loss.item())
                
                # Predictions
                preds = torch.argmax(probs, dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())
            
            # Compute metrics
            train_loss = np.mean(train_losses)
            train_acc = accuracy_score(train_labels, train_preds)
            train_mae = mean_absolute_error(train_labels, train_preds)
            train_qwk = cohen_kappa_score(train_labels, train_preds, weights='quadratic')
            
            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["train_mae"].append(train_mae)
            history["train_qwk"].append(train_qwk)
            if verbose:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, MAE: {train_mae:.4f}, QWK: {train_qwk:.4f}")
            
            # Validation
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val, device=device)
                history["val_loss"].append(val_metrics["loss"])
                history["val_acc"].append(val_metrics["accuracy"])
                history["val_mae"].append(val_metrics["mae"])
                history["val_qwk"].append(val_metrics["qwk"])
                if verbose:
                    print(f"  Val - Loss: {val_metrics['loss']:.4f}, "
                          f"Acc: {val_metrics['accuracy']:.4f}, MAE: {val_metrics['mae']:.4f}, QWK: {val_metrics['qwk']:.4f}")
        
        return history
    
    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        device: Optional[str] = None
    ) -> Dict[str, float]:
        """
        Evaluate the probe on data.
        
        Args:
            X: Embeddings [num_samples, input_dim]
            y: Labels [num_samples]
            batch_size: Batch size for evaluation
            device: Device to evaluate on
            
        Returns:
            Dictionary with metrics (loss, accuracy, MAE)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(device)
        self.eval()
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        losses = []
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                
                probs = self.forward(batch_X)
                loss = F.cross_entropy(probs, batch_y)
                
                losses.append(loss.item())
                
                preds = torch.argmax(probs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        return {
            "loss": np.mean(losses),
            "accuracy": accuracy_score(all_labels, all_preds),
            "mae": mean_absolute_error(all_labels, all_preds),
            "qwk": cohen_kappa_score(all_labels, all_preds, weights='quadratic')
        }
    
    def predict_proba(
        self,
        X: np.ndarray,
        batch_size: int = 32,
        device: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Embeddings [num_samples, input_dim]
            batch_size: Batch size for prediction
            device: Device to predict on
            
        Returns:
            Class probabilities [num_samples, num_classes]
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(device)
        self.eval()
        
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_probs = []
        
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(device)
                probs = self.forward(batch_X)
                all_probs.append(probs.cpu().numpy())
        
        return np.vstack(all_probs)
    
    def predict_labels(
        self,
        X: np.ndarray,
        batch_size: int = 32,
        device: Optional[str] = None
    ) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Embeddings [num_samples, input_dim]
            batch_size: Batch size for prediction
            device: Device to predict on
            
        Returns:
            Predicted labels [num_samples]
        """
        probs = self.predict_proba(X, batch_size=batch_size, device=device)
        return np.argmax(probs, axis=1)
    def get_linear_scores(
        self,
        X: np.ndarray,
        batch_size: int = 32,
        device: Optional[str] = None
    ) -> np.ndarray:
        """
        Extract the raw linear projection scores (before thresholding).
        
        Args:
            X: Embeddings [num_samples, input_dim]
            batch_size: Batch size for prediction
            device: Device to predict on
            
        Returns:
            Linear scores [num_samples] - the latent variable before thresholds
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.to(device)
        self.eval()
        
        X_tensor = torch.FloatTensor(X)
        dataset = TensorDataset(X_tensor)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        all_scores = []
        
        with torch.no_grad():
            for (batch_X,) in loader:
                batch_X = batch_X.to(device)
                projected = self.linear(batch_X).squeeze(-1)  # [batch_size]
                all_scores.append(projected.cpu().numpy())
        
        return np.concatenate(all_scores)
    
    def get_thresholds(self) -> np.ndarray:
        """
        Get the learned thresholds.
        
        Returns:
            Threshold values [num_classes - 1]
        """
        with torch.no_grad():
            sorted_thresholds = torch.sort(self.thresholds)[0]
            return sorted_thresholds.cpu().numpy()
