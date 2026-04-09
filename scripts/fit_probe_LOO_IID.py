from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
import mord
from src.proficiency_probing import EmbeddingCache

# ---------------------------------------------------------------------------
# Model configs
# ---------------------------------------------------------------------------

model_configs = [
    {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "pooling": "last",
        "layer_keys": {1, 5, 10, 15, 20, 25, 27},        # 27 layers, cap at 25
    },
    {
        "model_name": "Qwen/Qwen3-Embedding-4B",
        "pooling": "last",
        "layer_keys": {1, 5, 10, 15, 20, 25, 30, 35},
    },
    {
        "model_name": "Qwen/Qwen3-Embedding-8B",
        "pooling": "last",
        "layer_keys": {1, 5, 10, 15, 20, 25, 30, 35},
    },
]

# ---------------------------------------------------------------------------
# Shared config
# ---------------------------------------------------------------------------

instruct_model = True
instruction: Optional[str] = "Instruct: Assess the CEFR level of the following text.\nQuery: "
level_order    = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C+": 4}

NUM_CLASSES  = 5
TEST_SIZE    = 0.2
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Shared metric helpers
# ---------------------------------------------------------------------------

def compute_metrics(y_true, y_pred_raw) -> dict:
    y_pred_rounded = np.round(y_pred_raw).astype(int).clip(0, NUM_CLASSES - 1)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred_raw)),
        "qwk": float(cohen_kappa_score(y_true, y_pred_rounded, weights="quadratic")),
    }

def get_predictions(model, X, predict_fn=None):
    predict_fn = predict_fn or model.predict
    return predict_fn(X)

# ---------------------------------------------------------------------------
# Fitters
# ---------------------------------------------------------------------------

def fit_linear_regression(X_train, y_train):
    model = Ridge()
    model.fit(X_train, y_train)
    return model, model.predict

def fit_MLP_linear_regression(X_train, y_train):
    model = MLPRegressor(hidden_layer_sizes=(X_train.shape[1],), max_iter=1000)
    model.fit(X_train, y_train)
    return model, model.predict

def fit_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, model.predict

def fit_MLP_logistic_regression(X_train, y_train):
    model = MLPClassifier(hidden_layer_sizes=(X_train.shape[1],), max_iter=1000)
    model.fit(X_train, y_train)
    return model, model.predict

def fit_ordinal_cumlink(X_train, y_train):
    model = mord.LogisticAT()
    model.fit(X_train, y_train)
    return model, model.predict

configs = {
    "linear_regression":     fit_linear_regression,
    "logistic_regression":   fit_logistic_regression,
    "ordinal_cumlink":       fit_ordinal_cumlink,
    "MLP_linear_regression": fit_MLP_linear_regression,
    "MLP_logistic_regression": fit_MLP_logistic_regression,
}

# ---------------------------------------------------------------------------
# Main loop over models
# ---------------------------------------------------------------------------

for mc in model_configs:
    model_name  = mc["model_name"]
    pooling     = mc["pooling"]
    target_layers = mc["layer_keys"]

    print(f"\n{'#'*60}")
    print(f"# Model: {model_name}")
    print(f"{'#'*60}")

    cache_path = f"./src/cache/CEFR_{model_name.replace('/', '_')}_{pooling}.npz"

    # Load data
    cefr_df = pd.read_csv("./src/data/combined_cefr_data.csv")
    cefr_df["cefr_level"] = cefr_df["cefr_level"].map(level_order)
    cefr_df["text"] = instruction + cefr_df["text"] if instruct_model else cefr_df["text"]

    cache = EmbeddingCache(cache_path=cache_path)
    cached_texts = cache.cached_texts

    cached_levels, cached_sources = [], []
    for text in cached_texts:
        level  = cefr_df[cefr_df["text"] == text]["cefr_level"]
        source = cefr_df[cefr_df["text"] == text]["source_name"]
        cached_levels.append(level.values[0]  if not level.empty  else None)
        cached_sources.append(source.values[0] if not source.empty else None)

    cached_levels  = np.array(cached_levels)
    cached_sources = np.array(cached_sources)
    layer_keys = [l for l in sorted(cache.cached_embeddings.keys()) if l in target_layers]

    all_sources     = [s for s in np.unique(cached_sources) if s is not None]
    MERLIN_GROUP    = [s for s in all_sources if s.startswith("merlin-")]
    GROUPED_SOURCES = {"merlin-all": MERLIN_GROUP}
    ood_configs     = [(src, [src]) for src in all_sources] + list(GROUPED_SOURCES.items())

    print(f"Layers: {layer_keys}")
    print(f"Sources: {all_sources}")

    all_indices = np.arange(len(cached_levels))
    iid_train_indices, iid_test_indices = train_test_split(
        all_indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=cached_levels,
    )

    iid_records, ood_records = [], []

    for layer_idx in layer_keys:
        embeddings = cache.cached_embeddings[layer_idx]
        print(f"\n=== Layer {layer_idx} | dim={embeddings.shape[1]} ===")

        X_iid_train  = embeddings[iid_train_indices]
        y_iid_train  = cached_levels[iid_train_indices]
        X_iid_test   = embeddings[iid_test_indices]
        y_iid_test   = cached_levels[iid_test_indices]
        src_iid_test = cached_sources[iid_test_indices]

        for ood_label, ood_source_list in ood_configs:
            mask_ood   = np.isin(cached_sources, ood_source_list)
            mask_train = ~mask_ood

            X_ood, y_ood      = embeddings[mask_ood], cached_levels[mask_ood]
            ood_source_labels = cached_sources[mask_ood]

            X_tr, _, y_tr, _ = train_test_split(
                embeddings[mask_train], cached_levels[mask_train],
                test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=cached_levels[mask_train],
            )

            iid_subset_mask = np.isin(src_iid_test, ood_source_list)
            X_iid_src       = X_iid_test[iid_subset_mask]
            y_iid_src       = y_iid_test[iid_subset_mask]
            src_iid_src     = src_iid_test[iid_subset_mask]

            print(f"  ood_label={ood_label} | ood_n={mask_ood.sum()} | ood_train_n={mask_train.sum()}")

            for config_name, fit_fn in configs.items():
                model_ood, pred_fn_ood = fit_fn(X_tr, y_tr)
                preds_ood = get_predictions(model_ood, X_ood, pred_fn_ood)

                for pred, true, src in zip(preds_ood, y_ood, ood_source_labels):
                    ood_records.append({
                        "model": model_name, "layer": layer_idx, "config": config_name,
                        "ood_group": ood_label, "source": src, "y_true": true, "y_pred": pred,
                    })

                if len(X_iid_src) > 0:
                    model_iid, pred_fn_iid = fit_fn(X_iid_train, y_iid_train)
                    preds_iid = get_predictions(model_iid, X_iid_src, pred_fn_iid)

                    for pred, true, src in zip(preds_iid, y_iid_src, src_iid_src):
                        iid_records.append({
                            "model": model_name, "layer": layer_idx, "config": config_name,
                            "ood_group": ood_label, "source": src, "y_true": true, "y_pred": pred,
                        })

                m_ood = compute_metrics(y_ood, preds_ood)
                line  = f"    [{config_name}] OOD mae={m_ood['mae']:.4f} qwk={m_ood['qwk']:.4f}"
                if len(X_iid_src) > 0:
                    m_iid = compute_metrics(y_iid_src, preds_iid)
                    line += f"  | IID mae={m_iid['mae']:.4f} qwk={m_iid['qwk']:.4f}"
                print(line)

    # Save per model
    iid_df = pd.DataFrame(iid_records)
    ood_df = pd.DataFrame(ood_records)
    base   = f"./src/results/loo_limited_{model_name.replace('/', '_')}_{pooling}"
    iid_df.to_csv(f"{base}_iid_predictions.csv", index=False)
    ood_df.to_csv(f"{base}_ood_predictions.csv", index=False)
    print(f"\nSaved -> {base}_iid/ood_predictions.csv