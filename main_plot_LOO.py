# %%
"This script makes a plot of the results from main_model.py. It assumes that the results are saved in a CSV file in the src/results directory."
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

model_name     = ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B"]
pooling        = "last"
instruct_model = True
level_order    = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C+": 4}
results_paths = [f"./src/results/layer_probe_results_{name.replace('/', '_')}_{pooling}.csv" for name in model_name]

# ---------------------------------------------------------------------------
# Processing Data
# ---------------------------------------------------------------------------
results_dfs = [pd.read_csv(path) for path in results_paths]
for df, name in zip(results_dfs, model_name):
    df["model"] = name.split("/")[1].split("-")[2]  # Extract "0.6B" or "4B" from model name
combined_df = pd.concat(results_dfs, ignore_index=True)
combined_df["config_model"] = combined_df["config"] + " - " + combined_df["model"]

plot_qwk_df = combined_df.pivot(index="layer", columns="config_model", values="qwk").reset_index()
plot_qwk_df = plot_qwk_df[plot_qwk_df["layer"] != 0]
# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
plt.figure(figsize=(12, 6))
for column in plot_qwk_df.columns[1:]:
    print(f"Plotting column: {column}")
    plt.plot(plot_qwk_df["layer"], plot_qwk_df[column], label=column)
plt.title("Quadratic Weighted Kappa by Layer and Configuration")
plt.xlabel("Layer")
plt.ylabel("Quadratic Weighted Kappa")
plt.xticks(plot_qwk_df["layer"])
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
# %%
