# %%
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
import mord
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.proficiency_probing import EmbeddingCache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

model_name     = "Qwen/Qwen3-Embedding-4B"
pooling        = "last"
instruct_model = True
instruction: Optional[str] = "Instruct: Assess the CEFR level of the following text.\nQuery: "
level_order    = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C+": 4}
cache_path     = f"./src/cache/CEFR_{model_name.replace('/', '_')}_{pooling}.npz"

NUM_CLASSES  = 5
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

cefr_df = pd.read_csv("./src/data/combined_cefr_data.csv")
cefr_df["cefr_level"] = cefr_df["cefr_level"].map(level_order)
cefr_df["text"] = instruction + cefr_df["text"] if instruct_model else cefr_df["text"]

cache = EmbeddingCache(cache_path=cache_path)
cached_texts = cache.cached_texts

cached_levels = []
for text in cached_texts:
    level = cefr_df[cefr_df["text"] == text]["cefr_level"]
    cached_levels.append(level.values[0] if not level.empty else None)

cached_levels = np.array(cached_levels)
layer_keys = sorted(cache.cached_embeddings.keys())
print(f"Layers found: {layer_keys}")

# ---------------------------------------------------------------------------
# Shared metric helper
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred_raw) -> dict:
    """Rounds predictions to nearest integer for QWK, keeps raw for MAE."""
    y_pred_rounded = np.round(y_pred_raw).astype(int).clip(0, NUM_CLASSES - 1)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred_raw)),
        "qwk": float(cohen_kappa_score(y_true, y_pred_rounded, weights="quadratic")),
    }

# ---------------------------------------------------------------------------
# Sklearn / mord fitters
# ---------------------------------------------------------------------------

### TODO: FIt a logistic regression / NN logistic for baseline.

def fit_linear_regression(X_train, y_train, X_test, y_test) -> dict:
    model = Ridge()
    model.fit(X_train, y_train)
    return compute_metrics(y_test, model.predict(X_test))


def fit_ordinal_cumlink(X_train, y_train, X_test, y_test) -> dict:
    model = mord.LogisticAT()
    model.fit(X_train, y_train)
    return compute_metrics(y_test, model.predict(X_test))

def fit_nn_linear_regression(
    X_train, y_train, X_test, y_test, 
    hidden_dims: list[int] = [512, 128],
    dropout: float = 0.2,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> dict:
    """MLP feature extractor + linear regression head trained end-to-end with MSE loss."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_tr = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_te = torch.tensor(X_test,  dtype=torch.float32).to(device)

    input_dim = X_tr.shape[1]

    # Build MLP: [input → hidden... → 1]
    layers = []
    in_dim = input_dim
    for h in hidden_dims:
        layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))   # linear regression head
    model = nn.Sequential(*layers).to(device)

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn   = nn.MSELoss()

    loader = DataLoader(
        TensorDataset(X_tr, y_tr.unsqueeze(1)),
        batch_size=batch_size, shuffle=True,
    )

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimiser.zero_grad()
            loss_fn(model(xb), yb).backward()
            optimiser.step()

    model.eval()
    with torch.no_grad():
        y_pred = model(X_te).squeeze(1).cpu().numpy()

    return compute_metrics(y_test, y_pred)

# ---------------------------------------------------------------------------
# Loop over layers
# ---------------------------------------------------------------------------

results = []

for layer_idx in layer_keys:
    embeddings = cache.cached_embeddings[layer_idx]
    input_dim  = embeddings.shape[1]
    print(f"\n=== Layer {layer_idx} | dim={input_dim} ===")

    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, cached_levels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=cached_levels,
    )

    # Sklearn / mord
    configs = {
        "linear_regression": fit_linear_regression,
        "ordinal_cumlink":   fit_ordinal_cumlink,
        "nn_linear_regression": fit_nn_linear_regression,
    }

    for config_name, fit_fn in configs.items():
        print(f"  Fitting {config_name}...")
        metrics = fit_fn(X_train, y_train, X_test, y_test)
        results.append({"layer": layer_idx, "config": config_name, **metrics})
        print(f"    mae={metrics['mae']:.4f}  qwk={metrics['qwk']:.4f}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["layer", "config"]).reset_index(drop=True)

output_path = f"./src/results/layer_probe_results_{model_name.replace('/', '_')}_{pooling}.csv"
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")
print(results_df.pivot(index="layer", columns="config", values="qwk").round(4))
# %%
