# %%
# This script is for development and testing purposes. It loads the CEFR and COWSL2H datasets, computes embeddings using a specified model, and evaluates a linear regression probe on the CEFR dataset.
from src.proficiency_probing import TextEmbedder
from src.proficiency_probing import EmbeddingCache
# from proficiency_probing import DecoderEmbedder, POOLING_FNS  # your new module
# from transformers import AutoTokenizer, AutoModel
# import torch
from typing import Optional, Union, List
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(889)
# %%

# Parameters for model and layer selection
model_name="intfloat/multilingual-e5-base"
# model_name="intfloat/multilingual-e5-large-instruct"
# model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
# model_name = "Qwen/Qwen3-Embedding-0.6B"
# model_name = "Qwen/Qwen3-Embedding-8B"


pooling="mean"  # "mean", "cls", "max", or "last"
instruct_model = True # Flag for instruct models like Qwen that require a prompt for embedding:
instruction: Optional[str] = "Instruct: Assess the English proficiency level of the following text.\nQuery: "

# ========== Chunk Code ===========
# Function to find probe layers based on the number of probes and total layers in the model
def find_probe_layers(embedder, num_probes=5):
    m = embedder.model
    # Decoder-only models (Qwen, LLaMA, Mistral, etc.)
    if hasattr(m, 'layers'):
        layers = m.layers
    # Encoder-based fallback
    elif hasattr(m, 'encoder') and hasattr(m.encoder, 'layer'):
        layers = m.encoder.layer
    else:
        raise AttributeError(f"Cannot find layers in model: {type(m)}")

    num_layers = len(layers)
    step = max(1, num_layers // num_probes)
    return [
        num_layers - 1 - i * step
        for i in range(num_probes)
        if num_layers - 1 - i * step >= 0
    ]


print("Loading CEFR dataset...")
cefr_df = pd.read_csv("./src/data/cefr_dataset.csv")
level_order = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
cefr_df["cefr_level"] = cefr_df["cefr_level"].map(level_order)
cefr_df["text"] = instruction + cefr_df["text"] if instruct_model else cefr_df["text"]




# Load COWSL2h dataset
print("Loading COWSL2h dataset...")
# Load COWSL2H dataset
cowsl2h_df = pd.read_csv("./src/data/COWSL2H/merged_cowsl2h.csv")
cowsl2h_df = cowsl2h_df[["essay","writing ability","prompt"]]
cowsl2h_df["essay"] = instruction + cowsl2h_df["essay"] if instruct_model else cowsl2h_df["essay"]

# %% Embedding with Caching Chunk
# Loading model
print("=" * 70)
print("Loading Model")
print("=" * 70)
print()

embedder = TextEmbedder(
            model_name=model_name,
            pooling=pooling

        )

layer_indices = find_probe_layers(embedder, num_probes=5)

# %% 
# Embedding with Caching
print("=" * 70)
print("Embeddings CEFR with Caching")
print("=" * 70)
print()

# =========== Define cache path string based on model name, layer index and pooling strategy ===========
cache_path = f"./src/cache/CEFR_{model_name.replace('/', '_')}_{pooling}.npz"
cache = EmbeddingCache(
    embedder=embedder, 
    texts= cefr_df["text"].tolist(),
    cache_path=cache_path,
    layer_indices=layer_indices
    )

cefr_embeddings = cache.get_embeddings()

print("=" * 70)
print("Embeddings COWSL2H with Caching")
print("=" * 70)
print()

cache_path_cowsl2h = f"./src/cache/cowsl2h_{model_name.replace('/', '_')}_{pooling}.npz"
cowsl2h_cache = EmbeddingCache(
    embedder=embedder, 
    texts= cowsl2h_df["essay"].tolist(),
    cache_path=cache_path_cowsl2h,
    layer_indices=layer_indices
    )

cowsl2h_embeddings = cowsl2h_cache.get_embeddings()



# %%
print("=" * 70)
print("Fit linear regression probes and evaluate")
print("=" * 70)
print()

# check if cached embeddings are a dictionary with layer indices as keys and numpy arrays as values
if isinstance(cefr_embeddings, dict):
    for layer_idx, embeddings in cefr_embeddings.items():
        print(f"Layer {layer_idx} embeddings shape: {embeddings.shape}")
else:
    print(f"Embeddings shape: {embeddings.shape}")

# %%
# 27, 22,17, 12, 7
layer_idx = 23
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    cefr_embeddings[layer_idx], 
    cefr_df["cefr_level"].values, 
    test_size=0.2, 
    random_state=889
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import cohen_kappa_score
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_val)

val_mae = mean_absolute_error(y_val, y_pred)
val_qwk = cohen_kappa_score(y_val, np.round(y_pred), weights='quadratic')
print(f"Validation MAE: {val_mae:.4f}")
print(f"Validation QWK: {val_qwk:.4f}")


cow_pred = model.predict(cowsl2h_embeddings[layer_idx])
cowsl2h_df["predicted_cefr"] = cow_pred
cowsl2h_df["predicted_cefr_rounded"] = np.round(cow_pred).astype(int)
print(cowsl2h_df[["writing ability", "predicted_cefr", "predicted_cefr_rounded"]].head(10))


# Run ANOVA for cowsl2h_df
import scipy.stats as stats
anova_result = stats.f_oneway(
    cowsl2h_df[cowsl2h_df["writing ability"] == 1]["predicted_cefr"],
    cowsl2h_df[cowsl2h_df["writing ability"] == 2]["predicted_cefr"],
    cowsl2h_df[cowsl2h_df["writing ability"] == 3]["predicted_cefr"],
    cowsl2h_df[cowsl2h_df["writing ability"] == 4]["predicted_cefr"],
    cowsl2h_df[cowsl2h_df["writing ability"] == 5]["predicted_cefr"]
)
print(f"ANOVA F-statistic: {anova_result.statistic:.4f}, p-value: {anova_result.pvalue:.4f}")


# Plot the relationship between the probe scores and the writing ability levels
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.boxplot(x="writing ability", y="predicted_cefr", data=cowsl2h_df)
plt.title("Linear Regression Scores vs Writing Ability Levels - Layer: {layer_idx}")
plt.xlabel("Self-reported Writing Ability Level")
plt.ylabel("Probe Scores")
plt.xticks(rotation=45)
plt.savefig(f"./src/plots/COWSL2H_{model_name.replace('/', '_')}_{layer_idx}_{pooling}.png")
plt.show()

# Predict probe scores for cefr_embeddings and plot grouped by CEFR levels
plot_df = pd.DataFrame({
    "y_val": y_val,
    "y_pred": y_pred
})
plt.figure(figsize=(10, 6))
sns.boxplot(x="y_val", y="y_pred", data=plot_df)
plt.suptitle(f"Linear Regression vs CEFR Levels - Layer: {layer_idx}")
plt.title(f"Loss: {val_mae:.4f}")
plt.xlabel("CEFR Level")
plt.ylabel("Probe Scores")
plt.xticks(rotation=45)
path = f"./src/plots/CEFR_{model_name.replace('/', '_')}_{layer_idx}_{pooling}.png"
plt.savefig(path)
plt.show()  



# %%
# ──  Ordinal probe ─────────────────────────────────────────────────────────────
import mord
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

model = mord.LogisticIT()
model.fit(X_train, y_train)

# Latent space: continuous projection onto model coefficients
y_latent = X_val @ model.coef_
cow_latent = cowsl2h_embeddings[layer_idx] @ model.coef_

# Thresholded class predictions
y_pred = model.predict(X_val)
cow_pred = model.predict(cowsl2h_embeddings[layer_idx])

val_mae = mean_absolute_error(y_val, y_pred)
val_qwk = cohen_kappa_score(y_val, y_pred, weights='quadratic')
print(f"Validation MAE: {val_mae:.4f}")
print(f"Validation QWK: {val_qwk:.4f}")

cowsl2h_df["predicted_cefr"] = cow_latent          # continuous latent projection
cowsl2h_df["predicted_cefr_rounded"] = cow_pred     # thresholded ordinal class

# ── Plots with latent space + thresholds ───────────────────────────────────────
theta = model.theta_
cefr_order = ["A1", "A2", "B1", "B2", "C1", "C2"]
theta_labels = cefr_order[1:]

def add_thresholds(ax, theta, theta_labels, x_max):
    for t, label in zip(theta, theta_labels):
        ax.axhline(y=t, color='red', linestyle='--', alpha=0.7)
        ax.text(x=x_max - 0.5, y=t, s=label, color='red', va='center')

# COWSL2H
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="writing ability", y="predicted_cefr", data=cowsl2h_df, ax=ax)
add_thresholds(ax, theta, theta_labels, x_max=cowsl2h_df["writing ability"].nunique())
ax.set_title(f"Latent Ordinal Scores vs Writing Ability - Layer: {layer_idx}")
ax.set_xlabel("Self-reported Writing Ability Level")
ax.set_ylabel("Latent Score (projection onto coef_)")
plt.savefig(f"./src/plots/COWSL2H_{model_name.replace('/', '_')}_{layer_idx}_{pooling}_ordinal.png")
plt.show()

# CEFR validation
plot_df = pd.DataFrame({"y_val": y_val, "y_latent": y_latent})
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x="y_val", y="y_latent", data=plot_df, ax=ax)
add_thresholds(ax, theta, theta_labels, x_max=len(cefr_order))
ax.set_title(f"Latent Ordinal Scores vs CEFR Levels - Layer: {layer_idx}\nMAE: {val_mae:.4f} | QWK: {val_qwk:.4f}")
ax.set_xlabel("CEFR Level")
ax.set_ylabel("Latent Score (projection onto coef_)")
plt.savefig(f"./src/plots/CEFR_{model_name.replace('/', '_')}_{layer_idx}_{pooling}_ordinal.png")
plt.show()

# %%
