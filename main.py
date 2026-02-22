# %%
from proficiency_probing import ProficiencyProbingPipeline
from proficiency_probing import TextEmbedder
from proficiency_probing import EmbeddingCache
from proficiency_probing import OrdinalProbe
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(889)

# Define model and layer for probing
model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
layer_index=0
pooling="mean"  




# %% Load Data 
print("=" * 70)
print("Loading Dataset")
print("=" * 70)
print()
import os
print("Current working directory:", os.getcwd())
# Load CEFR dataset
print("Loading CEFR dataset...")
cefr_df = pd.read_csv("./src/data/cefr_dataset.csv")
level_order = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
cefr_df["cefr_level"] = cefr_df["cefr_level"].map(level_order)
print("CEFR dataset loaded with shape:", cefr_df.shape)

# %% Embedding with Caching Chunk
print("=" * 70)
print("Loading Model")
print("=" * 70)
print()



embedder = TextEmbedder(
            model_name=model_name,
            layer_index=layer_index,
            pooling=pooling
        )

print("=" * 70)
print("Getting Embeddings with Caching")
print("=" * 70)
print()

# =========== Define cache path string based on model name, layer index and pooling strategy ===========
cache_path_cefr = f"./src/cache/CEFR_embeddings_{model_name.replace('/', '_')}_layer{layer_index}_{pooling}.npz"

cefr_cache = EmbeddingCache(
    embedder=embedder, 
    texts= cefr_df["text"].tolist(),
    cache_path=cache_path_cefr
    )

cefr_embeddings = cefr_cache.get_embeddings()

# %%
print("=" * 70)
print("Training Ordinal Probe")
print("=" * 70)
print()

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    cefr_embeddings, 
    cefr_df["cefr_level"].values, 
    test_size=0.2, 
    random_state=889
)

probe = OrdinalProbe(
    input_dim=cefr_embeddings.shape[1],
    num_classes=len(level_order)
    )

print(f"  Train samples: {len(X_train)}")
print(f"  Val samples: {len(X_val)}")
print()
        
# Train probe
history = probe.fit(
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    device=embedder.device,
    verbose=False,
    epochs=250
    )
# %%
# Print training results for 'val_loss', 'val_acc', 'val_mae', 'val_qwk'
print("Final Validation Results:")
print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
print(f"  Val Accuracy: {history['val_acc'][-1]:.4f}")
print(f"  Val Mean Absolute Error: {history['val_mae'][-1]:.4f}")
print(f"  Val Quadratic Weighted Kappa: {history['val_qwk'][-1]:.4f}")

# %%
# Test the probe on out-of-distribution data (e.g., a different dataset or a subset of the original dataset)
print("=" * 70)
print("Testing Probe on Out-of-Distribution Data")
print("=" * 70)
print()

# Load COWSL2H dataset
cowsl2h_df = pd.read_csv("./src/data/COWSL2H/merged_cowsl2h.csv")
print("COWSL2H dataset loaded with shape:", cowsl2h_df.shape)
print("Keeping only 'essay','writing ability' and 'prompt' columns")
cowsl2h_df = cowsl2h_df[["essay","writing ability","prompt"]]
print("\nCOWSL2H dataset shape after keeping relevant columns:", cowsl2h_df.shape)


# =========== Define cache path string based on model name, layer index and pooling strategy ===========
cache_path_COWSL2H = f"./src/cache/COWSL2H_embeddings_{model_name.replace('/', '_')}_layer{layer_index}_{pooling}.npz"

# Get embeddings for COWSL2H dataset
cowsl2h_cache = EmbeddingCache(
    embedder=embedder, 
    texts= cowsl2h_df["essay"].tolist(),
    cache_path=cache_path_COWSL2H
    )

print("\nGetting embeddings for COWSL2H dataset with caching...")
cowsl2h_embeddings = cowsl2h_cache.get_embeddings()

print("\nGetting linear probe scores for COWSL2H dataset...")
cowsl2h_df["scores"] = probe.get_linear_scores(cowsl2h_embeddings)

# %%
# Writing ability has multiple levels 1-5. But there is some string versions included aswell. We will convert these strings to numeric values for easier analysis.
# "Extremely uncomfortable"  -> 1
# "Uncomfortable"            -> 2
# "Neither comfortable nor uncomfortable " -> 3
# "Comfortable"              -> 4
# "Extremely comfortable"    -> 5
def convert_writing_ability(value):
    if isinstance(value, str):
        value = value.strip().lower()
        if value == "Extremely uncomfortable" or value == "1.0":
            return 1
        elif value == "Uncomfortable" or value == "2.0":
            return 2
        elif value == "Neither comfortable nor uncomfortable" or value == "3.0":
            return 3
        elif value == "Comfortable" or value == "4.0":
            return 4
        elif value == "Extremely comfortable" or value == "5.0":
            return 5
    try:
        return int(value)
    except ValueError:
        return np.nan  # Return NaN for any non-convertible values

cowsl2h_df["writing_ability_numeric"] = cowsl2h_df["writing ability"].apply(convert_writing_ability)


# %%
# Plot the relationship between the probe scores and the writing ability levels
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10, 6))
sns.boxplot(x="writing_ability_numeric", y="scores", data=cowsl2h_df)
plt.title("Probe Scores vs Writing Ability Levels")
plt.xlabel("Self-reported Writing Ability Level")
plt.ylabel("Probe Scores")
plt.xticks(rotation=45)

# Save the plot
plt.savefig(f"./src/plots/COWSL2H_{model_name.replace('/', '_')}_{layer_index}_{pooling}.png")
plt.show()


# %%
# Predict probe scores for cefr_embeddings and plot grouped by CEFR levels
cefr_df["probe_scores"] = probe.get_linear_scores(cefr_embeddings)
plt.figure(figsize=(10, 6))
sns.boxplot(x="cefr_level", y="probe_scores", data=cefr_df)
# plot thresholds as vertical lines
tresholds = probe.get_thresholds()
for threshold in tresholds:
    plt.axhline(y=threshold, color='r', linestyle='--', label=f'Threshold: {threshold:.2f}')
plt.suptitle(f"Probe Scores vs CEFR Levels - Layer: {layer_index}")
plt.title(f"Loss: {history['val_loss'][-1]:.4f}, Acc: {history['val_acc'][-1]:.4f}, MAE: {history['val_mae'][-1]:.4f}, QWK: {history['val_qwk'][-1]:.4f}")
plt.xlabel("CEFR Level")
plt.ylabel("Probe Scores")
plt.xticks(rotation=45)
path = f"./src/plots/CEFR_{model_name.replace('/', '_')}_{layer_index}_{pooling}.png"
plt.savefig(path)
plt.show()  


# %%

# %%
