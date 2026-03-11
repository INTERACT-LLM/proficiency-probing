# %%
"Plot LOO probe results: x=layer, lines=ood_group (IID) or ood_group (OOD), separate figures"
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, cohen_kappa_score

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

model_name  = "Qwen/Qwen3-Embedding-8B"
pooling     = "last"
NUM_CLASSES = 5

base     = f"./src/results/loo_{model_name.replace('/', '_')}_{pooling}"
iid_path = f"{base}_iid_predictions.csv"
ood_path = f"{base}_ood_predictions.csv"

# ---------------------------------------------------------------------------
# Load & aggregate metrics
# IID: group by (layer, config, source)    — source seen in training
# OOD: group by (layer, config, ood_group) — ood_group is the held-out unit
# ---------------------------------------------------------------------------

def aggregate(df, group_cols):
    df = df.copy()
    df["y_pred_r"] = df["y_pred"].round().astype(int).clip(0, NUM_CLASSES - 1)
    records = []
    for keys, g in df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        row["qwk"] = (cohen_kappa_score(g["y_true"], g["y_pred_r"], weights="quadratic")
                      if g["y_true"].nunique() >= 2 else float("nan"))
        row["mae"] = mean_absolute_error(g["y_true"], g["y_pred"])
        records.append(row)
    return pd.DataFrame(records)

iid_raw = pd.read_csv(iid_path)
ood_raw = pd.read_csv(ood_path)

iid_raw = iid_raw[iid_raw['layer'] != 0]
ood_raw = ood_raw[ood_raw['layer'] != 0]

iid_metrics = aggregate(iid_raw, ["layer", "config", "source"]).rename(columns={"source": "group"})
ood_metrics = aggregate(ood_raw, ["layer", "config", "ood_group"]).rename(columns={"ood_group": "group"})

configs    = sorted(iid_metrics["config"].unique())
iid_groups = sorted(iid_metrics["group"].unique())
ood_groups = sorted(ood_metrics["group"].unique())
all_layers = sorted(iid_metrics["layer"].unique())

print("IID groups:", iid_groups)
print("OOD groups:", ood_groups)

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

all_groups    = sorted(set(iid_groups) | set(ood_groups))
colors        = plt.rcParams["axes.prop_cycle"].by_key()["color"]
group_color   = {g: colors[i % len(colors)] for i, g in enumerate(all_groups)}
config_color  = {c: colors[i % len(colors)] for i, c in enumerate(configs)}
metric_labels = {"qwk": "Quadratic Weighted Kappa", "mae": "Mean Absolute Error"}
metric_ylim   = {"qwk": (0, 1), "mae": None}

# ---------------------------------------------------------------------------
# Per-config condition plots (IID or OOD only)
# ---------------------------------------------------------------------------

def plot_condition(metrics_df, groups, condition_label, config, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, met in zip(axes, ["qwk", "mae"]):
        for group in groups:
            subset = metrics_df[metrics_df["group"] == group].sort_values("layer")
            if subset.empty:
                continue
            ax.plot(subset["layer"], subset[met],
                    marker="o", color=group_color[group],
                    label=group, alpha=0.85, markersize=4)
        ax.set_title(metric_labels[met])
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_labels[met])
        ax.set_xticks(all_layers)
        if metric_ylim[met] is not None:
            ax.set_ylim(*metric_ylim[met])
        ax.legend(fontsize="x-small", ncol=2, loc="best")
        ax.grid(alpha=0.4)
    fig.suptitle(f"{condition_label} — {config}", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

for config in configs:
    plot_condition(
        iid_metrics[iid_metrics["config"] == config], iid_groups,
        "IID (source seen in training)", config,
        f"./src/results/plot_iid_{config}_{model_name.replace('/', '_')}_{pooling}.png",
    )
    plot_condition(
        ood_metrics[ood_metrics["config"] == config], ood_groups,
        "OOD (source held out from training)", config,
        f"./src/results/plot_ood_{config}_{model_name.replace('/', '_')}_{pooling}.png",
    )



# ---------------------------------------------------------------------------
# Delta plot: IID − OOD per shared group, per config
# x = layer, lines = group, subplots = (QWK delta, MAE delta)
# ---------------------------------------------------------------------------

# For LOO: OOD ood_group corresponds to the same source name as IID source
shared_groups = sorted(set(iid_groups) & set(ood_groups))

for config in configs:
    iid_cfg = iid_metrics[(iid_metrics["config"] == config) & (iid_metrics["group"].isin(shared_groups))]
    ood_cfg = ood_metrics[(ood_metrics["config"] == config) & (ood_metrics["group"].isin(shared_groups))]

    # Merge on (layer, group) to compute deltas
    merged = pd.merge(
        iid_cfg[["layer", "group", "qwk", "mae"]],
        ood_cfg[["layer", "group", "qwk", "mae"]],
        on=["layer", "group"],
        suffixes=("_iid", "_ood"),
    )
    merged["delta_qwk"] = merged["qwk_iid"] - merged["qwk_ood"]  # positive = IID better
    merged["delta_mae"] = merged["mae_iid"] - merged["mae_ood"]   # positive = IID worse (higher error)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for ax, (delta_col, title, ylabel) in zip(axes, [
        ("delta_qwk", "ΔQWK (IID − OOD)", "QWK difference"),
        ("delta_mae", "ΔMAE (IID − OOD)", "MAE difference"),
    ]):
        for group in shared_groups:
            subset = merged[merged["group"] == group].sort_values("layer")
            if subset.empty:
                continue
            ax.plot(
                subset["layer"], subset[delta_col],
                marker="o", color=group_color[group],
                label=group, alpha=0.85, markersize=4,
            )

        ax.axhline(0, color="black", lw=1.0, linestyle="--", alpha=0.5, zorder=3)
        ax.set_title(title)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_xticks(all_layers)
        ax.legend(fontsize="x-small", ncol=2, loc="best")
        ax.grid(alpha=0.4)

    fig.suptitle(
        f"IID − OOD gap per dataset — {config}\n"
        f"(ΔQWK > 0 → IID better;  ΔMAE > 0 → IID worse)",
        fontsize=13,
    )
    plt.tight_layout()
    plt.savefig(
        f"./src/results/plot_delta_{config}_{model_name.replace('/', '_')}_{pooling}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.show()


# ---------------------------------------------------------------------------
# Summary: mean across groups — IID and OOD on the same plot, per config
# CI: bootstrap std over all individual predictions (n_boot resamples)
# linestyle: solid = IID, dashed = OOD  |  shaded band = ±1 bootstrap std
# ---------------------------------------------------------------------------

N_BOOT = 500

def bootstrap_std(series, n_boot=N_BOOT, rng=None):
    """Return bootstrap std of the mean via resampling individual predictions."""
    rng   = np.random.default_rng(rng)
    means = [rng.choice(series, size=len(series), replace=True).mean()
             for _ in range(n_boot)]
    return float(np.std(means))

def build_mean_ci(metrics_df, group_col, condition_label):
    """
    For each (layer, config) compute mean QWK/MAE and bootstrap std
    over ALL individual raw predictions pooled across groups.
    metrics_df must have columns: layer, config, group_col, qwk, mae.
    """
    records = []
    for (layer, config), g in metrics_df.groupby(["layer", "config"]):
        for met in ["qwk", "mae"]:
            vals = g[met].dropna().values
            if len(vals) == 0:
                continue
            records.append({
                "layer":     layer,
                "config":    config,
                "condition": condition_label,
                "metric":    met,
                "mean":      vals.mean(),
                "std":       bootstrap_std(vals),
            })
    return pd.DataFrame(records)

iid_ci  = build_mean_ci(iid_metrics, "group", "IID")
ood_ci  = build_mean_ci(ood_metrics, "group", "OOD")
ci_df   = pd.concat([iid_ci, ood_ci], ignore_index=True)

# color  = condition (IID vs OOD)
# linestyle = config
condition_color = {"IID": colors[0], "OOD": colors[1]}
config_ls       = {c: ls for c, ls in zip(configs, ["-", "--", ":", "-."])}
markers         = {"IID": "o", "OOD": "s"}

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
for ax, met in zip(axes, ["qwk", "mae"]):
    for config in configs:
        for condition in ["IID", "OOD"]:
            sub = (
                ci_df[(ci_df["config"] == config) &
                      (ci_df["condition"] == condition) &
                      (ci_df["metric"] == met)]
                .sort_values("layer")
            )
            if sub.empty:
                continue
            color = condition_color[condition]
            ax.plot(sub["layer"], sub["mean"],
                    linestyle=config_ls[config],
                    marker=markers[condition],
                    color=color,
                    label=f"{config} ({condition})",
                    alpha=0.85, markersize=4)
            ax.fill_between(sub["layer"],
                            sub["mean"] - sub["std"],
                            sub["mean"] + sub["std"],
                            color=color, alpha=0.12)

    ax.set_title(metric_labels[met])
    ax.set_xlabel("Layer")
    ax.set_ylabel(metric_labels[met])
    ax.set_xticks(all_layers)
    if metric_ylim[met] is not None:
        ax.set_ylim(*metric_ylim[met])
    ax.legend(fontsize="x-small", ncol=2)
    ax.grid(alpha=0.4)

fig.suptitle(
    f"Mean across groups ± bootstrap std — IID (solid) vs OOD (dashed)",
    fontsize=13,
)
plt.tight_layout()
plt.savefig(
    f"./src/results/plot_mean_{model_name.replace('/', '_')}_{pooling}.png",
    dpi=150, bbox_inches="tight",
)
plt.show()

# ---------------------------------------------------------------------------
# Merlin-only: IID (individual merlin sources) vs OOD (merlin-all group)
# solid = IID per source, dashed = OOD, color = source (black for merlin-all)
# ---------------------------------------------------------------------------

merlin_iid_groups = [g for g in iid_groups if g.startswith("merlin-")]
merlin_ood_groups = [g for g in ood_groups if g.startswith("merlin-")]

merlin_iid = iid_metrics[iid_metrics["group"].isin(merlin_iid_groups)]
merlin_ood = ood_metrics[ood_metrics["group"].isin(merlin_ood_groups)]

for config in configs:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    for ax, met in zip(axes, ["qwk", "mae"]):

        # IID — solid line per individual merlin source
        for group in merlin_iid_groups:
            subset = (
                merlin_iid[(merlin_iid["config"] == config) & (merlin_iid["group"] == group)]
                .sort_values("layer")
            )
            if subset.empty:
                continue
            ax.plot(subset["layer"], subset[met],
                    linestyle="-", marker="o",
                    color=group_color[group],
                    label=f"{group} (IID)",
                    alpha=0.85, markersize=4)

        # OOD — dashed line per merlin ood_group (individual + merlin-all)
        for group in merlin_ood_groups:
            subset = (
                merlin_ood[(merlin_ood["config"] == config) & (merlin_ood["group"] == group)]
                .sort_values("layer")
            )
            if subset.empty:
                continue
            color = "black" if group == "merlin-all" else group_color[group]
            ax.plot(subset["layer"], subset[met],
                    linestyle="--", marker="s",
                    color=color,
                    label=f"{group} (OOD)",
                    alpha=0.85, markersize=4)

        ax.set_title(metric_labels[met])
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric_labels[met])
        ax.set_xticks(all_layers)
        if metric_ylim[met] is not None:
            ax.set_ylim(*metric_ylim[met])
        ax.legend(fontsize="x-small", ncol=2, loc="best")
        ax.grid(alpha=0.4)

    fig.suptitle(f"Merlin — IID (solid) vs OOD (dashed) — {config}", fontsize=13)
    plt.tight_layout()
    plt.savefig(
        f"./src/results/plot_merlin_{config}_{model_name.replace('/', '_')}_{pooling}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.show()

# ---------------------------------------------------------------------------
# Ridgeplot 2x2 — per config:
#   top-left:  IID  residuals   top-right: OOD residuals
#   bot-left:  IID  y_pred      bot-right: OOD y_pred
# ---------------------------------------------------------------------------

import numpy as np
from scipy.stats import gaussian_kde

iid_raw["residual"] = iid_raw["y_pred"] - iid_raw["y_true"]
ood_raw["residual"] = ood_raw["y_pred"] - ood_raw["y_true"]

# Individual OOD runs only (exclude merlin-all)
ood_individual = ood_raw[ood_raw["ood_group"] == ood_raw["source"]].copy()

RIDGE_SCALE = 0.6
cmap        = plt.cm.viridis
n_layers    = len(all_layers)

# Shared x-grids
resid_min   = min(iid_raw["residual"].min(), ood_individual["residual"].min()) - 0.5
resid_max   = max(iid_raw["residual"].max(), ood_individual["residual"].max()) + 0.5
x_resid     = np.linspace(resid_min, resid_max, 300)
x_pred      = np.linspace(0, NUM_CLASSES - 1, 300)
cefr_labels = ["A1", "A2", "B1", "B2", "C+"]

def draw_ridge(ax, series_by_layer, x_grid, vline=None, xticks=None, xticklabels=None, xlabel=""):
    """Draw stacked KDE ridges onto ax. series_by_layer: list of (layer_label, pd.Series)."""
    for i, (layer_label, vals) in enumerate(series_by_layer):
        vals = vals.dropna()
        if len(vals) < 5:
            continue
        kde   = gaussian_kde(vals, bw_method=0.03)
        y_kde = kde(x_grid)
        y_kde = y_kde / y_kde.max() * RIDGE_SCALE
        color = cmap(i / max(len(series_by_layer) - 1, 1))
        ax.fill_between(x_grid, i, i + y_kde, alpha=0.55, color=color)
        ax.plot(x_grid, i + y_kde, color=color, lw=1.2)
        ax.axhline(i, color="white", lw=0.5, zorder=3)
    if vline is not None:
        ax.axvline(vline, color="black", lw=1.2, linestyle="--", alpha=0.6, zorder=5)
    ax.set_yticks(range(len(series_by_layer)))
    ax.set_yticklabels([str(l) for l, _ in series_by_layer], fontsize=7)
    ax.set_xlim(x_grid[0], x_grid[-1])
    if xticks is not None:
        ax.set_xticks(xticks)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    ax.spines[["top", "right", "left"]].set_visible(False)

for config in configs:
    iid_cfg = iid_raw[iid_raw["config"] == config]
    ood_cfg = ood_individual[ood_individual["config"] == config]

    # Build (layer_label, series) lists
    iid_resid_by_layer = [(f"Layer {l}", iid_cfg[iid_cfg["layer"] == l]["residual"]) for l in all_layers]
    ood_resid_by_layer = [(f"Layer {l}", ood_cfg[ood_cfg["layer"] == l]["residual"]) for l in all_layers]
    iid_pred_by_layer  = [(f"Layer {l}", iid_cfg[iid_cfg["layer"] == l]["y_pred"])   for l in all_layers]
    ood_pred_by_layer  = [(f"Layer {l}", ood_cfg[ood_cfg["layer"] == l]["y_pred"])   for l in all_layers]

    fig, axes = plt.subplots(2, 2, figsize=(16, n_layers * 0.55 + 2))

    draw_ridge(axes[0, 0], iid_resid_by_layer, x_resid,
               vline=0, xlabel="Residual (y_pred − y_true)")
    axes[0, 0].set_title("IID — Residuals", fontsize=11)

    draw_ridge(axes[0, 1], ood_resid_by_layer, x_resid,
               vline=0, xlabel="Residual (y_pred − y_true)")
    axes[0, 1].set_title("OOD — Residuals", fontsize=11)

    draw_ridge(axes[1, 0], iid_pred_by_layer, x_pred,
               xticks=list(range(NUM_CLASSES)), xticklabels=cefr_labels,
               xlabel="Predicted CEFR level")
    axes[1, 0].set_title("IID — Predicted values", fontsize=11)

    draw_ridge(axes[1, 1], ood_pred_by_layer, x_pred,
               xticks=list(range(NUM_CLASSES)), xticklabels=cefr_labels,
               xlabel="Predicted CEFR level")
    axes[1, 1].set_title("OOD — Predicted values", fontsize=11)

    fig.suptitle(f"Ridge distributions by Layer — {config}", fontsize=13)
    plt.tight_layout()
    plt.savefig(
        f"./src/results/plot_ridge_2x2_{config}_{model_name.replace('/', '_')}_{pooling}.png",
        dpi=150, bbox_inches="tight",
    )
    plt.show()
# %%
