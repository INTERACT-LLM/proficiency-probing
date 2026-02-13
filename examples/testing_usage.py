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
    data_paths = {
        "train": "../src/data/train_cefr_dataset.csv",
        "test": "../src/data/test_cefr_dataset.csv",}
    level_order = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C1": 4, "C2": 5}
    def load_text_classification_data(
        data_paths,
        level_order,
        text_col="text",
        label_col="cefr_level",
    ):
        data = {}
        print("Loading dataset...")

        for split, path in data_paths.items():
            print(f"  Loading {split} data from {path}...")
            df = pd.read_csv(path)
            texts = df[text_col].tolist()
            labels = [level_order[l] for l in df[label_col]]

            print(f"  Loaded {len(texts)} {split} samples")
            print(f"  Label distribution: {np.bincount(labels)}")

            data[split] = {
                "df": df,
                "text": texts,
                "labels": labels,
            }

        print("Dataset loaded successfully.\n")
        return data

    data = load_text_classification_data(data_paths, level_order)

    pipeline = ProficiencyProbingPipeline(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        layer_index=-1,  # Use last layer
        pooling="mean"    # Mean pooling over tokens
        
    )
    print("Pipeline initialized.")
    pipeline.fit(data["test"]["text"], data["test"]["labels"])
    #pipeline.evaluate(data["test"]["text"], data["test"]["labels"])
    print("Evaluation complete.")

    # %%





# %%
