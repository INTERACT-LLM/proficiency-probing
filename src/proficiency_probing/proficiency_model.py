"""
Proficiency Model: Optional backbone with pluggable ordinal or regression head.

Usage examples:

    # No backbone (linear probe directly on embeddings)
    model = ProficiencyModel(
        input_dim=768,
        head=OrdinalHead(input_dim=768, num_classes=5),
    )

    # Default MLP backbone
    model = ProficiencyModel(
        input_dim=768,
        backbone=MLPBackbone(input_dim=768, hidden_dim=256, num_layers=2, dropout=0.1),
        head=OrdinalHead(input_dim=256, num_classes=5),
    )

    # Custom backbone (any nn.Module)
    backbone = nn.Sequential(nn.Linear(768, 128), nn.GELU(), nn.Linear(128, 64))
    model = ProficiencyModel(
        input_dim=768,
        backbone=backbone,
        head=RegressionHead(input_dim=64, min_val=0.0, max_val=4.0),
    )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from typing import Optional, Dict
from abc import ABC, abstractmethod
from sklearn.metrics import accuracy_score, cohen_kappa_score, mean_absolute_error
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------

class BaseHead(ABC, nn.Module):
    """Abstract base class for prediction heads."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns predictions or probabilities."""

    @abstractmethod
    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss given forward output and targets."""

    @abstractmethod
    def predict(self, output: torch.Tensor) -> torch.Tensor:
        """Convert forward output to scalar predictions."""

    @abstractmethod
    def reset_parameters(self):
        """Reset head parameters to initial values."""


class OrdinalHead(BaseHead):
    """
    Cumulative link ordinal head.

    Thresholds are reparametrized to guarantee ordering:
        thresholds = [t0, t0 + exp(g0), t0 + exp(g0) + exp(g1), ...]
    """

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(input_dim, 1)
        self.threshold_0 = nn.Parameter(torch.tensor(0.0))
        self.log_gaps = nn.Parameter(torch.zeros(num_classes - 2)) if num_classes > 2 else None

    def _get_thresholds(self) -> torch.Tensor:
        if self.log_gaps is not None:
            gaps = torch.exp(self.log_gaps)
            rest = self.threshold_0 + torch.cumsum(gaps, dim=0)
            return torch.cat([self.threshold_0.unsqueeze(0), rest])
        return self.threshold_0.unsqueeze(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns class probabilities [batch_size, num_classes]."""
        projected = self.linear(x)
        thresholds = self._get_thresholds()
        cumulative_probs = torch.sigmoid(projected - thresholds.unsqueeze(0))
        return torch.cat([
            1 - cumulative_probs[:, :1],
            cumulative_probs[:, :-1] - cumulative_probs[:, 1:],
            cumulative_probs[:, -1:],
        ], dim=1)

    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(output, target)

    def predict(self, output: torch.Tensor) -> torch.Tensor:
        """Returns predicted class indices."""
        return torch.argmax(output, dim=1).float()

    def reset_parameters(self):
        self.linear.reset_parameters()
        with torch.no_grad():
            self.threshold_0.data = torch.tensor(0.0)
            if self.log_gaps is not None:
                self.log_gaps.data = torch.zeros(self.num_classes - 2)

    def get_thresholds(self) -> np.ndarray:
        with torch.no_grad():
            return self._get_thresholds().cpu().numpy()


class RegressionHead(BaseHead):
    """
    Linear regression head with output clamped to [min_val, max_val].

    Uses a scaled sigmoid instead of a hard clamp to keep gradients
    flowing at the boundaries:
        output = min_val + (max_val - min_val) * sigmoid(x)
    """

    def __init__(self, input_dim: int, min_val: float = 0.0, max_val: float = 4.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns scalar predictions [batch_size]."""
        raw = self.linear(x).squeeze(-1)
        return self.min_val + (self.max_val - self.min_val) * torch.sigmoid(raw)

    def loss(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(output, target.float())

    def predict(self, output: torch.Tensor) -> torch.Tensor:
        return output

    def reset_parameters(self):
        self.linear.reset_parameters()


# ---------------------------------------------------------------------------
# Backbone
# ---------------------------------------------------------------------------

class MLPBackbone(nn.Module):
    """
    Standard MLP backbone with configurable depth and width.

    Args:
        input_dim:  Input feature dimension
        hidden_dim: Width of each hidden layer
        num_layers: Number of hidden layers (minimum 1)
        dropout:    Dropout probability (0 = disabled)
        activation: Activation class to use (default: nn.ReLU)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        activation: type = nn.ReLU,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be at least 1"
        self.output_dim = hidden_dim

        layers = []
        in_dim = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(activation())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def reset_parameters(self):
        for layer in self.net:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ProficiencyModel(nn.Module):
    """
    Proficiency model with an optional backbone and pluggable head.

    If no backbone is provided, the head receives raw input embeddings
    (i.e. a linear probe). Otherwise the backbone can be any nn.Module —
    use MLPBackbone for a standard MLP or pass your own.

    Args:
        input_dim: Dimension of input embeddings (used only for validation)
        head:      An instantiated BaseHead (OrdinalHead or RegressionHead)
        backbone:  Optional nn.Module. If None, head receives raw embeddings.
    """

    def __init__(
        self,
        input_dim: int,
        head: BaseHead,
        backbone: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x) if self.backbone is not None else x
        return self.head(h)

    def reset_parameters(self):
        if self.backbone is not None:
            if hasattr(self.backbone, "reset_parameters"):
                self.backbone.reset_parameters()
            else:
                # Fallback for arbitrary nn.Module backbones
                for layer in self.backbone.modules():
                    if hasattr(layer, "reset_parameters"):
                        layer.reset_parameters()
        self.head.reset_parameters()

    # -----------------------------------------------------------------------
    # Fit
    # -----------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        device: Optional[str] = None,
        verbose: bool = True,
        reset: bool = True,
    ) -> Dict[str, list]:
        """
        Fit the model.

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
            reset:         Reset all parameters before training

        Returns:
            Training history dict
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
            "train_loss": [], "train_mae": [], "train_qwk": []
        }
        if X_val is not None and y_val is not None:
            history.update({"val_loss": [], "val_mae": [], "val_qwk": []})

        for epoch in range(epochs):
            self.train()
            losses, preds, labels = [], [], []

            iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}") if verbose else train_loader

            for batch_X, batch_y in iterator:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                optimizer.zero_grad()
                output = self.forward(batch_X)
                loss = self.head.loss(output, batch_y)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                preds.extend(self.head.predict(output).detach().cpu().numpy())
                labels.extend(batch_y.cpu().numpy())

            self._append_metrics(history, "train", losses, labels, preds)
            if verbose:
                self._print_metrics(history, "train", epoch, epochs)

            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val, device=device)
                for k in ("loss", "mae", "qwk"):
                    history[f"val_{k}"].append(val_metrics[k])
                if verbose:
                    self._print_metrics(history, "val", epoch, epochs)

        return history

    # -----------------------------------------------------------------------
    # Evaluate / Predict
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> Dict[str, float]:
        """Returns dict with loss, mae, qwk."""
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
                output = self.forward(batch_X)
                losses.append(self.head.loss(output, batch_y).item())
                all_preds.extend(self.head.predict(output).cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())

        return _compute_metrics(losses, all_labels, all_preds)

    def predict(
        self,
        X: np.ndarray,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """Returns scalar predictions [num_samples]."""
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to(device)
        self.eval()

        loader = DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=batch_size, shuffle=False)
        all_preds = []

        with torch.no_grad():
            for (batch_X,) in loader:
                output = self.forward(batch_X.to(device))
                all_preds.extend(self.head.predict(output).cpu().numpy())

        return np.array(all_preds)

    def predict_proba(
        self,
        X: np.ndarray,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> np.ndarray:
        """
        Returns raw forward output [num_samples, num_classes] for ordinal head,
        or scalar predictions [num_samples] for regression head.
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.to(device)
        self.eval()

        loader = DataLoader(TensorDataset(torch.FloatTensor(X)), batch_size=batch_size, shuffle=False)
        all_outputs = []

        with torch.no_grad():
            for (batch_X,) in loader:
                all_outputs.append(self.forward(batch_X.to(device)).cpu().numpy())

        return np.vstack(all_outputs)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _append_metrics(history, split, losses, labels, preds):
        m = _compute_metrics(losses, labels, preds)
        for k in ("loss", "mae", "qwk"):
            history[f"{split}_{k}"].append(m[k])

    @staticmethod
    def _print_metrics(history, split, epoch, epochs):
        h = history
        loss = h[f"{split}_loss"][-1]
        mae  = h[f"{split}_mae"][-1]
        qwk  = h[f"{split}_qwk"][-1]
        tag = "Train" if split == "train" else "Val  "
        print(f"  {tag} - Loss: {loss:.4f}, MAE: {mae:.4f}, QWK: {qwk:.4f}")


# ---------------------------------------------------------------------------
# Shared metric computation
# ---------------------------------------------------------------------------

def _compute_metrics(losses, labels, preds) -> Dict[str, float]:
    rounded = np.round(preds).astype(int)
    return {
        "loss": float(np.mean(losses)),
        "mae":  float(mean_absolute_error(labels, preds)),
        "qwk":  float(cohen_kappa_score(labels, rounded, weights="quadratic")),
    }