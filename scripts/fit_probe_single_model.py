# %%
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
import mord
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.proficiency_probing import EmbeddingCache

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

model_name     = "Qwen/Qwen3-Embedding-8B"
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

cached_levels  = []
cached_sources = []
for text in cached_texts:
    level = cefr_df[cefr_df["text"] == text]["cefr_level"]
    cached_levels.append(level.values[0] if not level.empty else None)
    source = cefr_df[cefr_df["text"] == text]["source_name"]
    cached_sources.append(source.values[0] if not source.empty else None)

cached_levels  = np.array(cached_levels)
cached_sources = np.array(cached_sources)
layer_keys = sorted(cache.cached_embeddings.keys())
print(f"Layers found: {layer_keys}")

# ---------------------------------------------------------------------------
# Build split masks
# ---------------------------------------------------------------------------

mask_merlin  = cached_sources == "merlin-de"
mask_asag    = cached_sources == "cefr-asag"
mask_train   = ~mask_merlin & ~mask_asag          # remaining data for train/test split
mask_all     = np.ones_like(mask_train, dtype=bool)  # All data

print(f"merlin-de OOD set : {mask_merlin.sum()} samples")
print(f"cefr-asag OOD set : {mask_asag.sum()} samples")
print(f"Remaining for split: {mask_train.sum()} samples")

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

def evaluate_model(model, X, y, predict_fn=None) -> dict:
    """Evaluate a fitted model on a given (X, y) split."""
    predict_fn = predict_fn or model.predict
    return compute_metrics(y, predict_fn(X))

# ---------------------------------------------------------------------------
# Sklearn / mord fitters  — each returns (model, predict_fn)
# ---------------------------------------------------------------------------

def fit_linear_regression(X_train, y_train):
    model = Ridge()
    model.fit(X_train, y_train)
    return model, model.predict

def fit_MLP_linear_regression(X_train, y_train):
    input_dim = X_train.shape[1]
    model = MLPRegressor(hidden_layer_sizes=(input_dim, input_dim), max_iter=1000)
    model.fit(X_train, y_train)
    return model, model.predict

def fit_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, model.predict

def fit_MLP_logistic_regression(X_train, y_train):
    input_dim = X_train.shape[1]
    model = MLPClassifier(hidden_layer_sizes=(input_dim, input_dim), max_iter=1000)
    model.fit(X_train, y_train)
    return model, model.predict

def fit_ordinal_cumlink(X_train, y_train):
    model = mord.LogisticAT()
    model.fit(X_train, y_train)
    return model, model.predict


# ---------------------------------------------------------------------------
# Loop over layers
# ---------------------------------------------------------------------------

results = []
for layer_idx in layer_keys:
    embeddings = cache.cached_embeddings[layer_idx]
    input_dim  = embeddings.shape[1]
    print(f"\n=== Layer {layer_idx} | dim={input_dim} ===")

    # --- per-layer OOD splits ---
    X_merlin, y_merlin = embeddings[mask_merlin], cached_levels[mask_merlin]
    X_asag,   y_asag   = embeddings[mask_asag],   cached_levels[mask_asag]

    # --- in-distribution train / test split ---
    X_remaining, y_remaining = embeddings[mask_train], cached_levels[mask_train]
    X_train, X_test, y_train, y_test = train_test_split(
        X_remaining, y_remaining,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_remaining,
    )

    # Change to fit less models - if you just want to investigate a single model.
    configs = {
        "linear_regression":    fit_linear_regression,
        "nn_linear_regression": fit_MLP_linear_regression,
        "logistic_regression":  fit_logistic_regression,
        "nn_logistic_regression": fit_MLP_logistic_regression,
        "ordinal_cumlink":      fit_ordinal_cumlink,
    }

    for config_name, fit_fn in configs.items():
        print(f"  Fitting {config_name}...")
        model, predict_fn = fit_fn(X_train, y_train)

        test_sets = {
            "iid":          (X_test,   y_test),
            "merlin-de": (X_merlin, y_merlin),
            "cefr-asag": (X_asag,   y_asag),
        }

        for split_name, (X_eval, y_eval) in test_sets.items():
            metrics = evaluate_model(model, X_eval, y_eval, predict_fn)
            results.append({
                "layer":  layer_idx,
                "config": config_name,
                "split":  split_name,
                **metrics,
            })
            print(f"    [{split_name}] mae={metrics['mae']:.4f}  qwk={metrics['qwk']:.4f}")

# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(["layer", "config", "split"]).reset_index(drop=True)

output_path = f"./src/results/layer_probe_results_{model_name.replace('/', '_')}_{pooling}.csv"
results_df.to_csv(output_path, index=False)
print(f"\nResults saved to {output_path}")

for split_name in ["iid", "merlin-de", "cefr-asag"]:
    print(f"\n=== QWK pivot — {split_name} ===")
    subset = results_df[results_df["split"] == split_name]
    print(subset.pivot(index="layer", columns="config", values="qwk").round(4))

# %%