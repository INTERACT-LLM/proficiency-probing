# %%
from proficiency_probing import ProficiencyProbingPipeline
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(889)





# %%
def main():
    # %% Load Data and Initialize Pipeline
    print("=" * 70)
    print("Proficiency Probing Pipeline - CEFR")
    print("=" * 70)
    
    # Load 
    level_order = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
    
    
    # Save merged dataframe to a new CSV file
    data["full"]["df"].to_csv("../src/data/full_cefr_dataset.csv", index=False)
    print("Merged train and test data into full_cefr_dataset.csv\n")    
    

    pipeline = ProficiencyProbingPipeline(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        layer_index=-1,  # Use last layer
        pooling="mean"    # Mean pooling over tokens      
    )
    print("Pipeline initialized.")
    pipeline.fit(
        texts=data["full"]["text"], 
        labels=data["full"]["labels"],
        cache_path="../src/cache/embeddings.npz",
        )
    
    pipeline.evaluate(
        texts=data["test"]["text"], 
        labels=data["test"]["labels"],
        cache_path="../src/cache/test_embeddings.npz",
        )
    print("Evaluation complete.")

    # %%





# %%
