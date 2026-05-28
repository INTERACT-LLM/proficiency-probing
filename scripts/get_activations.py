# %%
# This script is for development and testing purposes. 
# Set root directory to the project root
import os
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.proficiency_probing import TextEmbedder
from src.proficiency_probing import EmbeddingCache
from typing import Optional, Union, List
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(889)
# %%
# Parameters for model and layer selection
# model_names = ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B"]
model_names = ["google/embeddinggemma-300M"]
for model_name in model_names:
    pooling="last"  # "mean", "cls", "max", or "last"
    instruct_model = True # Flag for instruct models like Qwen that require a prompt for embedding:
    instruction: Optional[str] = "Instruct: Assess the CEFR level of the following text.\nQuery: "

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
    cefr_df = pd.read_csv("./src/data/combined_cefr_data.csv")
    # only sample 1000 rows for development
    # cefr_df = cefr_df.sample(n=100, random_state=889).reset_index(drop=True)
    
    level_order = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
    cefr_df["cefr_level"] = cefr_df["cefr_level"].map(level_order)
    cefr_df["text"] = instruction + cefr_df["text"] if instruct_model else cefr_df["text"]
    # Loading model
    print("=" * 70)
    print("Loading Model")
    print("=" * 70)
    print()

    embedder = TextEmbedder(
                model_name=model_name,
                pooling=pooling
            )

    # layer_indices = find_probe_layers(embedder, num_probes=5)
    # Save all layers
    layer_indices = list(range(len(embedder.model.layers)))
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

# %%
