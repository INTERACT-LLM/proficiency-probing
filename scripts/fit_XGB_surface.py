# %%
"""
XGBoost baseline: CEFR prediction from surface features
Features: LEN (total text length), AWL (avg word length), ASL (avg sentence length)
Evaluation:
  IID — single model trained on 80% of all data, evaluated per source on held-out 20%
  OOD — LOO cross-validation: train without held-out source, predict on it
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path
from sklearn.metrics import mean_absolute_error, cohen_kappa_score
from sklearn.model_selection import train_test_split
import xgboost as xgb

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_PATH   = "./src/data/combined_cefr_data.csv"
TEXT_COL    = "text"
LABEL_COL   = "cefr_level"
SOURCE_COL  = "source_name"
NUM_CLASSES = 5
SEED        = 42

level_order = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C+": 4}

XGB_PARAMS = dict(
    objective        = "reg:squarederror",
    n_estimators     = 300,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    random_state     = SEED,
)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

df = pd.read_csv(DATA_PATH)
df[LABEL_COL] = df[LABEL_COL].map(level_order)
df = df.dropna(subset=[LABEL_COL])
df[LABEL_COL] = df[LABEL_COL].astype(int)

print(f"Loaded {len(df)} rows, {df[SOURCE_COL].nunique()} sources: {sorted(df[SOURCE_COL].unique())}")
print(f"Label distribution:\n{df[LABEL_COL].value_counts().sort_index()}\n")

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def sent_tokenize_simple(text):
    text  = text.replace("\\n", "\n")
    lines = [s.strip() for s in text.splitlines() if s.strip()]
    if len(lines) >= 2:
        return lines
    words  = text.split()
    window = 15
    return [" ".join(words[i:i+window]) for i in range(0, len(words), window)] or [text]

def extract_features(data, text_col=TEXT_COL):
    texts = data[text_col].astype(str)

    def avg_word_len(t):
        words = re.findall(r"[a-zA-Z]+", t)
        return np.mean([len(w) for w in words]) if words else 0.0

    def avg_sent_len(t):
        sents = sent_tokenize_simple(t)
        return np.mean([len(s.split()) for s in sents]) if sents else 0.0

    return pd.DataFrame({
        "LEN": texts.str.len(),
        "AWL": texts.apply(avg_word_len),
        "ASL": texts.apply(avg_sent_len),
    })

# ---------------------------------------------------------------------------
# LOO + IID evaluation
# ---------------------------------------------------------------------------

def run_loo(df, test_size=0.2):
    sources  = df[SOURCE_COL].unique()
    iid_rows = []
    ood_rows = []

    # IID: single model on 80% of all data, evaluated per source on held-out 20%
    train_all, test_all = train_test_split(
        df,
        test_size=test_size,
        random_state=SEED,
        stratify=df[LABEL_COL],
    )

    iid_model = xgb.XGBRegressor(**XGB_PARAMS)
    iid_model.fit(extract_features(train_all), train_all[LABEL_COL].values)

    for source in sources:
        iid_subset = test_all[test_all[SOURCE_COL] == source]
        if iid_subset.empty:
            continue
        preds = iid_model.predict(extract_features(iid_subset))
        for y_true, y_pred in zip(iid_subset[LABEL_COL].values, preds):
            iid_rows.append({
                "source": source,
                "y_true": y_true,
                "y_pred": float(y_pred),
            })

    # OOD: LOO — train without held-out source, predict on it
    for held_out in sources:
        train_df = df[df[SOURCE_COL] != held_out]
        test_df  = df[df[SOURCE_COL] == held_out]

        ood_model = xgb.XGBRegressor(**XGB_PARAMS)
        ood_model.fit(extract_features(train_df), train_df[LABEL_COL].values)

        preds_ood = ood_model.predict(extract_features(test_df))
        for y_true, y_pred, src in zip(test_df[LABEL_COL].values, preds_ood, test_df[SOURCE_COL]):
            ood_rows.append({
                "ood_group": held_out,
                "source":    src,
                "y_true":    y_true,
                "y_pred":    float(y_pred),
            })

    return pd.DataFrame(iid_rows), pd.DataFrame(ood_rows)

# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------

def aggregate(df, group_cols):
    df = df.copy()
    df["y_pred_r"] = df["y_pred"].round().astype(int).clip(0, NUM_CLASSES - 1)
    records = []
    for keys, g in df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        row["qwk"] = (
            cohen_kappa_score(g["y_true"], g["y_pred_r"], weights="quadratic")
            if g["y_true"].nunique() >= 2 else float("nan")
        )
        row["mae"] = mean_absolute_error(g["y_true"], g["y_pred"])
        row["n"]   = len(g)   # dataset size for weighted mean
        records.append(row)
    return pd.DataFrame(records)

def weighted_mean(metrics_df):
    """Weighted mean of QWK and MAE, weighted by number of samples per group."""
    weights = metrics_df["n"]
    return pd.Series({
        "qwk": np.average(metrics_df["qwk"].fillna(0), weights=weights),
        "mae": np.average(metrics_df["mae"],            weights=weights),
    })

# ---------------------------------------------------------------------------
# Run & save
# ---------------------------------------------------------------------------

iid_raw, ood_raw = run_loo(df)

out_dir = Path("./src/results")
out_dir.mkdir(parents=True, exist_ok=True)

iid_raw.to_csv(out_dir / "xgb_surface_iid_predictions.csv", index=False)
ood_raw.to_csv(out_dir / "xgb_surface_ood_predictions.csv", index=False)

iid_metrics = aggregate(iid_raw, ["source"]).rename(columns={"source": "group"})
ood_metrics = aggregate(ood_raw, ["ood_group"]).rename(columns={"ood_group": "group"})

iid_wmean = weighted_mean(iid_metrics)
ood_wmean = weighted_mean(ood_metrics)

print("--- IID metrics ---")
print(iid_metrics[["group", "n", "qwk", "mae"]].to_string(index=False))
print(f"\n  Weighted Mean QWK: {iid_wmean['qwk']:.3f}   Weighted Mean MAE: {iid_wmean['mae']:.3f}")

print("\n--- OOD metrics ---")
print(ood_metrics[["group", "n", "qwk", "mae"]].to_string(index=False))
print(f"\n  Weighted Mean QWK: {ood_wmean['qwk']:.3f}   Weighted Mean MAE: {ood_wmean['mae']:.3f}")

# ---------------------------------------------------------------------------
# Feature importance (full data)
# ---------------------------------------------------------------------------

final_model = xgb.XGBRegressor(**XGB_PARAMS)
final_model.fit(extract_features(df), df[LABEL_COL].values)

print("\n--- Feature importances (full data) ---")
print(pd.Series(final_model.feature_importances_, index=["LEN", "AWL", "ASL"])
      .sort_values(ascending=False).to_string())
# %%
