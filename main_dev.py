# %%
# This script is for development and testing purposes. It loads the CEFR and COWSL2H datasets, computes embeddings using a specified model, and evaluates a linear regression probe on the CEFR dataset.
from proficiency_probing import TextEmbedder
from proficiency_probing import EmbeddingCache
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

# model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
model_name = "Qwen/Qwen3-Embedding-0.6B"
pooling="last"  # "mean", "cls", "max", or "last"
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

# Test text list
print("Loading CEFR dataset...")
cefr_df = pd.read_csv("./src/data/cefr_dataset.csv")
level_order = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
cefr_df["cefr_level"] = cefr_df["cefr_level"].map(level_order)
cefr_df["text"] = instruction + cefr_df["text"] if instruct_model else cefr_df["text"]
# Sample 15 of each cefr_level for testing
test = cefr_df.groupby("cefr_level").sample(n=15, random_state=889)
print("CEFR dataset loaded with shape:", test.shape)



# Load COWSL2h dataset
print("Loading COWSL2h dataset...")
# Load COWSL2H dataset
cowsl2h_df = pd.read_csv("./src/data/COWSL2H/merged_cowsl2h.csv")
cowsl2h_df = cowsl2h_df[["essay","writing ability","prompt"]]
cowsl2h_df["essay"] = instruction + cowsl2h_df["essay"] if instruct_model else cowsl2h_df["essay"]
cowsl2h_test = cowsl2h_df.groupby("writing ability").sample(n=15, random_state=889)
print("\nCOWSL2H dataset shape after keeping relevant columns:", cowsl2h_test.shape)
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
cache_path_test = f"./src/cache/test_CEFR_{model_name.replace('/', '_')}_{pooling}.npz"
test_cache = EmbeddingCache(
    embedder=embedder, 
    texts= test["text"].tolist(),
    cache_path=cache_path_test,
    layer_indices=layer_indices
    )

test_embeddings = test_cache.get_embeddings()

print("=" * 70)
print("Embeddings COWSL2H with Caching")
print("=" * 70)
print()

cache_path_cowsl2h = f"./src/cache/test_cowsl2h_{model_name.replace('/', '_')}_{pooling}.npz"
cowsl2h_cache = EmbeddingCache(
    embedder=embedder, 
    texts= cowsl2h_test["essay"].tolist(),
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
if isinstance(test_embeddings, dict):
    for layer_idx, embeddings in test_embeddings.items():
        print(f"Layer {layer_idx} embeddings shape: {embeddings.shape}")
else:
    print(f"Embeddings shape: {test_embeddings.shape}")

# %%
# 27, 22,17, 12, 7
layer_idx = 7
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    test_embeddings[layer_idx], 
    test["cefr_level"].values, 
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
cowsl2h_test["predicted_cefr"] = cow_pred
cowsl2h_test["predicted_cefr_rounded"] = np.round(cow_pred).astype(int)
print(cowsl2h_test[["writing ability", "predicted_cefr", "predicted_cefr_rounded"]].head(10))


# Run ANOVA for cowsl2h_test
import scipy.stats as stats
anova_result = stats.f_oneway(
    cowsl2h_test[cowsl2h_test["writing ability"] == 1]["predicted_cefr"],
    cowsl2h_test[cowsl2h_test["writing ability"] == 2]["predicted_cefr"],
    cowsl2h_test[cowsl2h_test["writing ability"] == 3]["predicted_cefr"],
    cowsl2h_test[cowsl2h_test["writing ability"] == 4]["predicted_cefr"],
    cowsl2h_test[cowsl2h_test["writing ability"] == 5]["predicted_cefr"]
)
print(f"ANOVA F-statistic: {anova_result.statistic:.4f}, p-value: {anova_result.pvalue:.4f}")


# Plot the relationship between the probe scores and the writing ability levels
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.boxplot(x="writing ability", y="predicted_cefr", data=cowsl2h_test)
plt.title("Linear Regression Scores vs Writing Ability Levels - Layer: {layer_idx}")
plt.xlabel("Self-reported Writing Ability Level")
plt.ylabel("Probe Scores")
plt.xticks(rotation=45)
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
plt.show()  



# %%

