"""
Ordinal Probe Module: Linear probe for ordinal regression on proficiency levels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Optional, Dict
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from tqdm import tqdm


class OrdinalProbe(nn.Module):
    """
    Linear probe for ordinal regression using a cumulative link model.

    Projects input embeddings to a scalar latent variable, then compares
    against learned thresholds to produce ordinal class probabilities.

    Thresholds are reparametrized to guarantee ordering: the first threshold
    is free, and subsequent thresholds are first_threshold + cumsum(exp(log_gaps)),
    which are always positive and thus always increasing.

    Attributes:
        input_dim:   Dimension of input embeddings
        num_classes: Number of ordinal classes
        linear:      Linear projection to scalar latent variable
        threshold_0: First (lowest) threshold
        log_gaps:    Log of gaps between consecutive thresholds
    """

    def __init__(self, input_dim: int, num_classes: int):
        """
        Args:
            input_dim:   Dimension of input embeddings
            num_classes: Number of ordinal proficiency levels
        """
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes

        self.linear = nn.Linear(input_dim, 1)

        # Reparametrize thresholds to enforce ordering:
        #   thresholds = [t0, t0 + exp(g0), t0 + exp(g0) + exp(g1), ...]
        self.threshold_0 = nn.Parameter(torch.tensor(0.0))
        if num_classes > 2:
            self.log_gaps = nn.Parameter(torch.zeros(num_classes - 2))
        else:
            self.log_gaps = None

    def _get_thresholds(self) -> torch.Tensor:
        """Build ordered threshold vector from reparametrized parameters."""
        if self.log_gaps is not None:
            gaps = torch.exp(self.log_gaps)
            rest = self.threshold_0 + torch.cumsum(gaps, dim=0)
            return torch.cat([self.threshold_0.unsqueeze(0), rest])
        return self.threshold_0.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input embeddings [batch_size, input_dim]

        Returns:
            Class probabilities [batch_size, num_classes]
        """
        projected = self.linear(x)  # [batch_size, 1]
        thresholds = self._get_thresholds()  # [num_classes - 1]

        # P(y >= k) = sigmoid(projected - threshold_k)
        cumulative_probs = torch.sigmoid(
            projected - thresholds.unsqueeze(0)
        )  # [batch_size, num_classes - 1]

        # Convert cumulative to class probabilities
        # P(y = 0)              = 1 - P(y >= 1)
        # P(y = k)              = P(y >= k) - P(y >= k+1)
        # P(y = num_classes-1)  = P(y >= num_classes-1)
        probs = torch.cat([
            1 - cumulative_probs[:, :1],
            cumulative_probs[:, :-1] - cumulative_probs[:, 1:],
            cumulative_probs[:, -1:],
        ], dim=1)  # [batch_size, num_classes]

        return probs

    def reset_parameters(self):
        """Reset all parameters to their initial values."""
        self.linear.reset_parameters()
        with torch.no_grad():
            self.threshold_0.data = torch.tensor(0.0)
            if self.log_gaps is not None:
                self.log_gaps.data = torch.zeros(self.num_classes - 2)

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
        verbose: bool = True,
        reset: bool = True,
    ) -> Dict[str, list]:
        """
        Fit the probe to training data.

        Args:
            X_train:       Training embeddings [num_samples, input_dim]
            y_train:       Training labels [num_samples]
            X_val:         Optional validation embeddings
            y_val:         Optional validation labels
            epochs:        Number of training epochs
            batch_size:    Batch size
            learning_rate: Learning rate
            weight_decay:  L2 regularisation strength
            device:        Device string (defaults to cuda if available)
            verbose:       Print per-epoch metrics
            reset:         Reset parameters before training (default True)

        Returns:
            Training history dict with loss/acc/mae/qwk per epoch
        """
        if reset:
            self.reset_parameters()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to(device)

        train_loader = DataLoader(
            TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train)),
            batch_size=batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.AdamW(
            self.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        history: Dict[str, list] = {
            "train_loss": [], "train_acc": [], "train_mae": [], "train_qwk": []
        }
        if X_val is not None and y_val is not None:
            history.update({"val_loss": [], "val_acc": [], "val_mae": [], "val_qwk": []})

        for epoch in range(epochs):
            self.train()
            train_losses, train_preds, train_labels = [], [], []

            iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_loader

            for batch_X, batch_y in iterator:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                optimizer.zero_grad()
                loss = F.cross_entropy(self.forward(batch_X), batch_y)
                loss.backward()
                optimizer.step()

                train_losses.append(loss.item())
                preds = torch.argmax(self.forward(batch_X), dim=1)
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(batch_y.cpu().numpy())

            self._log_metrics(history, "train", train_losses, train_labels, train_preds, verbose, epoch, epochs)

            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val, device=device)
                for k in ("loss", "acc", "mae", "qwk"):
                    history[f"val_{k}"].append(val_metrics[k if k != "acc" else "accuracy"])
                if verbose:
                    print(f"  Val  - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
                          f"MAE: {val_metrics['mae']:.4f}, QWK: {val_metrics['qwk']:.4f}")

        return history

    @staticmethod
    def _log_metrics(history, split, losses, labels, preds, verbose, epoch, epochs):
        loss = np.mean(losses)
        acc  = accuracy_score(labels, preds)
        mae  = mean_absolute_error(labels, preds)
        qwk  = cohen_kappa_score(labels, preds, weights="quadratic")
        history[f"{split}_loss"].append(loss)
        history[f"{split}_acc"].append(acc)
        history[f"{split}_mae"].append(mae)
        history[f"{split}_qwk"].append(qwk)
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {loss:.4f}, Acc: {acc:.4f}, MAE: {mae:.4f}, QWK: {qwk:.4f}")

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Evaluate the probe on data.

        Returns:
            Dict with keys: loss, accuracy, mae, qwk
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to(device)
        self.eval()

        loader = DataLoader(
            TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
            batch_size=batch_size,
            shuffle=False,
        )

        losses, all_preds, all_labels = [], [], []

        with torch.no_grad():
            for batch_X, batch_y in loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                probs = self.forward(batch_X)
                losses.append(F.cross_entropy(probs, batch_y).item())
                all_preds.extend(torch.argmax(probs, dim=1).cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        return {
            "loss":     np.mean(losses),
            "accuracy": accuracy_score(all_labels, all_preds),
            "mae":      mean_absolute_error(all_labels, all_preds),
            "qwk":      cohen_kappa_score(all_labels, all_preds, weights="quadratic"),
        }

    def predict_proba(
        self,
        X: np.ndarray,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns:
            Class probabilities [num_samples, num_classes]
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to(device)
        self.eval()

        loader = DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=batch_size, shuffle=False)
        all_probs = []

        with torch.no_grad():
            for (batch_X,) in loader:
                all_probs.append(self.forward(batch_X.to(device)).cpu().numpy())

        return np.vstack(all_probs)

    def predict_labels(self, X: np.ndarray, batch_size: int = 32, device: Optional[str] = None) -> np.ndarray:
        """Returns predicted class labels [num_samples]."""
        return np.argmax(self.predict_proba(X, batch_size=batch_size, device=device), axis=1)

    def get_linear_scores(
        self,
        X: np.ndarray,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns the raw latent scalar before thresholding [num_samples].
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to(device)
        self.eval()

        loader = DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=batch_size, shuffle=False)
        all_scores = []

        with torch.no_grad():
            for (batch_X,) in loader:
                all_scores.append(self.linear(batch_X.to(device)).squeeze(-1).cpu().numpy())

        return np.concatenate(all_scores)

    def get_thresholds(self) -> np.ndarray:
        """Returns the learned threshold values [num_classes - 1]."""
        with torch.no_grad():
            return self._get_thresholds().cpu().numpy()