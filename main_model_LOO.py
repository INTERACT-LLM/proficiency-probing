# %%
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
import mord
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

all_sources = [s for s in np.unique(cached_sources) if s is not None]

# Grouped OOD: treat all merlin-* datasets as one held-out group.
# Add further groups here as needed, e.g. {"my-group": ["src-a", "src-b"]}
MERLIN_GROUP    = [s for s in all_sources if s.startswith("merlin-")]
GROUPED_SOURCES = {"merlin-all": MERLIN_GROUP}

# Combined iteration: (ood_label, [sources to hold out])
# Individual sources first, then any groups
ood_configs = [(src, [src]) for src in all_sources] + list(GROUPED_SOURCES.items())

print(f"Layers found: {layer_keys}")
print(f"Sources found: {all_sources}")
for src in all_sources:
    print(f"  {src}: {(cached_sources == src).sum()} samples")
print(f"\nGrouped OOD sets:")
for group_name, members in GROUPED_SOURCES.items():
    total = sum((cached_sources == s).sum() for s in members)
    print(f"  {group_name}: {members} -> {total} samples")

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

def fit_logistic_regression(X_train, y_train):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    return model, model.predict

def fit_ordinal_cumlink(X_train, y_train):
    model = mord.LogisticAT()
    model.fit(X_train, y_train)
    return model, model.predict

configs = {
    "linear_regression": fit_linear_regression,
    #"logistic_regression": fit_logistic_regression, 
    #"ordinal_cumlink":   fit_ordinal_cumlink,
}

# ---------------------------------------------------------------------------
# Pre-compute the fixed IID train/test split (uses ALL data, done once)
# ---------------------------------------------------------------------------

all_indices = np.arange(len(cached_levels))
iid_train_indices, iid_test_indices = train_test_split(
    all_indices,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=cached_levels,
)

# ---------------------------------------------------------------------------
# Main loop: leave-one-source-out  (individual sources + grouped)
# ---------------------------------------------------------------------------

iid_records = []
ood_records = []

for layer_idx in layer_keys:
    embeddings = cache.cached_embeddings[layer_idx]
    print(f"\n=== Layer {layer_idx} | dim={embeddings.shape[1]} ===")

    # IID matrices are the same for every ood_label — build once per layer
    X_iid_train = embeddings[iid_train_indices]
    y_iid_train = cached_levels[iid_train_indices]
    X_iid_test  = embeddings[iid_test_indices]
    y_iid_test  = cached_levels[iid_test_indices]
    src_iid_test = cached_sources[iid_test_indices]

    for ood_label, ood_source_list in ood_configs:
        # --- masks ---
        mask_ood   = np.isin(cached_sources, ood_source_list)
        mask_train = ~mask_ood

        X_ood, y_ood       = embeddings[mask_ood],  cached_levels[mask_ood]
        ood_source_labels  = cached_sources[mask_ood]   # individual name per row

        X_tr, _, y_tr, _ = train_test_split(
            embeddings[mask_train], cached_levels[mask_train],
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=cached_levels[mask_train],
        )

        # IID test rows that belong to this ood_label's source(s)
        iid_subset_mask  = np.isin(src_iid_test, ood_source_list)
        X_iid_src        = X_iid_test[iid_subset_mask]
        y_iid_src        = y_iid_test[iid_subset_mask]
        src_iid_src      = src_iid_test[iid_subset_mask]

        print(f"  ood_label={ood_label} | ood_n={mask_ood.sum()} "
              f"| ood_train_n={mask_train.sum()} | iid_test_n={iid_subset_mask.sum()}")

        for config_name, fit_fn in configs.items():

            # --- OOD model: trained WITHOUT held-out source(s) ---
            model_ood, pred_fn_ood = fit_fn(X_tr, y_tr)
            preds_ood = get_predictions(model_ood, X_ood, pred_fn_ood)

            for pred, true, src in zip(preds_ood, y_ood, ood_source_labels):
                ood_records.append({
                    "layer":     layer_idx,
                    "config":    config_name,
                    "ood_group": ood_label,  # "merlin-all" or individual name
                    "source":    src,         # always the individual source name
                    "y_true":    true,
                    "y_pred":    pred,
                })

            # --- IID model: trained WITH all sources ---
            if len(X_iid_src) > 0:
                model_iid, pred_fn_iid = fit_fn(X_iid_train, y_iid_train)
                preds_iid = get_predictions(model_iid, X_iid_src, pred_fn_iid)

                for pred, true, src in zip(preds_iid, y_iid_src, src_iid_src):
                    iid_records.append({
                        "layer":     layer_idx,
                        "config":    config_name,
                        "ood_group": ood_label,
                        "source":    src,
                        "y_true":    true,
                        "y_pred":    pred,
                    })

            # Quick summary
            m_ood = compute_metrics(y_ood, preds_ood)
            line  = f"    [{config_name}] OOD mae={m_ood['mae']:.4f} qwk={m_ood['qwk']:.4f}"
            if len(X_iid_src) > 0:
                m_iid = compute_metrics(y_iid_src, preds_iid)
                line += f"  | IID mae={m_iid['mae']:.4f} qwk={m_iid['qwk']:.4f}"
            print(line)

# ---------------------------------------------------------------------------
# Build and save DataFrames
# ---------------------------------------------------------------------------

iid_df = pd.DataFrame(iid_records)
ood_df = pd.DataFrame(ood_records)

base = f"./src/results/loo_{model_name.replace('/', '_')}_{pooling}"
iid_df.to_csv(f"{base}_iid_predictions.csv", index=False)
ood_df.to_csv(f"{base}_ood_predictions.csv", index=False)

print(f"\nSaved IID predictions ({len(iid_df)} rows) -> {base}_iid_predictions.csv")
print(f"Saved OOD predictions ({len(ood_df)} rows) -> {base}_ood_predictions.csv")

# ---------------------------------------------------------------------------
# Summary pivot: QWK per ood_group x layer
# ---------------------------------------------------------------------------

def summarise(df, label):
    print(f"\n{'='*60}\n{label} -- QWK per ood_group x layer\n{'='*60}")
    for config_name in df["config"].unique():
        sub = df[df["config"] == config_name].copy()
        sub["y_pred_r"] = sub["y_pred"].round().astype(int).clip(0, NUM_CLASSES - 1)
        grouped = (
            sub.groupby(["layer", "ood_group"])
            .apply(lambda g: cohen_kappa_score(g["y_true"], g["y_pred_r"], weights="quadratic")
                   if g["y_true"].nunique() > 1 else float("nan"))
            .reset_index(name="qwk")
        )
        pivot = grouped.pivot(index="layer", columns="ood_group", values="qwk").round(4)
        print(f"\n  [{config_name}]\n{pivot}")

summarise(iid_df, "IID (source seen in training)")
summarise(ood_df, "OOD (source held out from training)")

# %%