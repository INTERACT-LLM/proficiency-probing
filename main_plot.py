# %%
"This script makes a plot of the results from main_model.py."
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

model_name     = ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B"]
pooling        = "last"
instruct_model = True
level_order    = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C+": 4}
results_paths  = [f"./src/results/layer_probe_results_{name.replace('/', '_')}_{pooling}.csv" for name in model_name]

# ---------------------------------------------------------------------------
# Processing Data
# ---------------------------------------------------------------------------

results_dfs = [pd.read_csv(path) for path in results_paths]
for df, name in zip(results_dfs, model_name):
    df["model"] = name.split("/")[1].split("-")[2]
combined_df = pd.concat(results_dfs, ignore_index=True)

mean_df = (
    combined_df[combined_df["layer"] != 0]
    .groupby(["layer", "split", "model"])[["qwk", "mae"]]
    .mean()
    .reset_index()
)

config_df = (
    combined_df[combined_df["layer"] != 0]
    .groupby(["layer", "split", "model", "config"])[["qwk", "mae"]]
    .mean()
    .reset_index()
)

splits     = sorted(mean_df["split"].unique())
models     = sorted(mean_df["model"].unique())
configs    = sorted(combined_df["config"].unique())
all_layers = sorted(mean_df["layer"].unique())

# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

linestyles  = {"iid": "-", "merlin-de": "--", "cefr-asag": ":"}
colors      = plt.rcParams["axes.prop_cycle"].by_key()["color"]
model_color = {m: colors[i % len(colors)] for i, m in enumerate(models)}

def make_plot(df, group_col, color_map, title, metric):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    labels = {"qwk": "Quadratic Weighted Kappa", "mae": "Mean Absolute Error"}

    for ax, met in zip(axes, [metric, "qwk"] if metric == "mae" else ["qwk", "mae"]):
        for split in splits:
            for group in sorted(df[group_col].unique()):
                subset = df[(df["split"] == split) & (df[group_col] == group)].sort_values("layer")
                ax.plot(
                    subset["layer"], subset[met],
                    linestyle=linestyles.get(split, "-"),
                    color=color_map[group],
                    marker="o",
                    label=f"{group} — {split}",
                )
        ax.set_title(f"{labels[met]}")
        ax.set_xlabel("Layer")
        ax.set_ylabel(labels[met])
        ax.set_xticks(all_layers)
        ax.grid()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Plot 1 — mean across configs
# ---------------------------------------------------------------------------

make_plot(mean_df, "model", model_color, "Mean QWK & MAE by Layer — mean across configs", "mae")

# ---------------------------------------------------------------------------
# Plot 2..N — one per config
# ---------------------------------------------------------------------------

for config in configs:
    subset = config_df[config_df["config"] == config]
    make_plot(subset, "model", model_color, f"QWK & MAE by Layer — {config}", "mae")
# %%

# %%
"This script makes a plot of the results from main_model.py."
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

model_name     = ["Qwen/Qwen3-Embedding-0.6B", "Qwen/Qwen3-Embedding-4B", "Qwen/Qwen3-Embedding-8B"]
pooling        = "last"
instruct_model = True
level_order    = {"A1": 0, "A2": 1, "B1": 2, "B2": 3, "C+": 4}
results_paths  = [f"./src/results/layer_probe_results_{name.replace('/', '_')}_{pooling}.csv" for name in model_name]

# ---------------------------------------------------------------------------
# Processing Data
# ---------------------------------------------------------------------------

results_dfs = [pd.read_csv(path) for path in results_paths]
for df, name in zip(results_dfs, model_name):
    df["model"] = name.split("/")[1].split("-")[2]
combined_df = pd.concat(results_dfs, ignore_index=True)

mean_df = (
    combined_df[combined_df["layer"] != 0]
    .groupby(["layer", "split", "model"])[["qwk", "mae"]]
    .mean()
    .reset_index()
)

config_df = (
    combined_df[combined_df["layer"] != 0]
    .groupby(["layer", "split", "model", "config"])[["qwk", "mae"]]
    .mean()
    .reset_index()
)

splits     = sorted(mean_df["split"].unique())
models     = sorted(mean_df["model"].unique())
configs    = sorted(combined_df["config"].unique())
all_layers = sorted(mean_df["layer"].unique())

# ---------------------------------------------------------------------------
# Shared style helpers
# ---------------------------------------------------------------------------

linestyles  = {"iid": "-", "merlin-de": "--", "cefr-asag": ":"}
colors      = plt.rcParams["axes.prop_cycle"].by_key()["color"]
model_color = {m: colors[i % len(colors)] for i, m in enumerate(models)}
config_color = {c: colors[i % len(colors)] for i, c in enumerate(configs)}

def make_plot(df, group_col, color_map, title, metric):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    labels = {"qwk": "Quadratic Weighted Kappa", "mae": "Mean Absolute Error"}

    for ax, met in zip(axes, [metric, "qwk"] if metric == "mae" else ["qwk", "mae"]):
        for split in splits:
            for group in sorted(df[group_col].unique()):
                subset = df[(df["split"] == split) & (df[group_col] == group)].sort_values("layer")
                ax.plot(
                    subset["layer"], subset[met],
                    linestyle=linestyles.get(split, "-"),
                    color=color_map[group],
                    marker="o",
                    label=f"{group} — {split}",
                )
        ax.set_title(f"{labels[met]}")
        ax.set_xlabel("Layer")
        ax.set_ylabel(labels[met])
        ax.set_xticks(all_layers)
        ax.legend()
        ax.grid()

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Plot 1 — mean across configs
# ---------------------------------------------------------------------------

make_plot(mean_df, "model", model_color, "Mean QWK & MAE by Layer — mean across configs", "mae")

# ---------------------------------------------------------------------------
# Plot 2..N — one per config
# ---------------------------------------------------------------------------

for config in configs:
    subset = config_df[config_df["config"] == config]
    make_plot(subset, "model", model_color, f"QWK & MAE by Layer — {config}", "mae")


# ---------------------------------------------------------------------------
# Plot N+1 — Qwen 4B: all configs on one plot
# ---------------------------------------------------------------------------

qwen4b_df = config_df[config_df["model"] == "4B"]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
labels = {"qwk": "Quadratic Weighted Kappa", "mae": "Mean Absolute Error"}

for ax, met in zip(axes, ["mae", "qwk"]):
    for split in splits:
        for config in configs:
            subset = qwen4b_df[
                (qwen4b_df["split"] == split) & (qwen4b_df["config"] == config)
            ].sort_values("layer")
            ax.plot(
                subset["layer"], subset[met],
                linestyle=linestyles.get(split, "-"),
                color=config_color[config],
                marker="o",
                label=f"{config} — {split}",
            )
    ax.set_title(labels[met])
    ax.set_xlabel("Layer")
    ax.set_ylabel(labels[met])
    ax.set_xticks(all_layers)
    ax.grid()

fig.suptitle("Qwen3-Embedding-4B — All Configs by Layer", fontsize=14)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# Plot N+2 — Qwen 4B: mean QWK across splits per config (bar chart overview)
# ---------------------------------------------------------------------------

qwen4b_mean = (
    qwen4b_df
    .groupby(["config", "split"])["qwk"]
    .mean()
    .reset_index()
)

# Add an "all splits" aggregate
all_splits_mean = (
    qwen4b_df#[qwen4b_df["layer"] > 15]  # Focus on later layers for the aggregate
    .groupby("config")["qwk"]
    .mean()
    .reset_index()
    .assign(split="all splits")
)

qwen4b_mean = pd.concat([qwen4b_mean, all_splits_mean], ignore_index=True)

all_splits_list = splits + ["all splits"]

fig, ax = plt.subplots(figsize=(12, 5))

bar_width = 0.8 / len(configs)
x = range(len(all_splits_list))

for i, config in enumerate(configs):
    config_data = qwen4b_mean[qwen4b_mean["config"] == config].set_index("split").reindex(all_splits_list)
    ax.bar(
        [xi + i * bar_width for xi in x],
        config_data["qwk"],
        width=bar_width,
        label=config,
        color=config_color[config],
    )

ax.set_title("Qwen3-Embedding-4B — Mean QWK per Split (averaged across layers)", fontsize=13)
ax.set_xlabel("Split")
ax.set_ylabel("Mean QWK")
ax.set_xticks([xi + bar_width * (len(configs) - 1) / 2 for xi in x])
ax.set_xticklabels(all_splits_list)
ax.legend(title="Config", fontsize="small")
ax.grid(axis="y")

plt.tight_layout()
plt.show()

# %%