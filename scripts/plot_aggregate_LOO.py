# %%

"""
LOO probe results — summary table + per-model QWK plots.
  - Summary table: best (mean) format, F1/QWK/MAE/R², mean-across-configs row per model
  - Combined plot: QWK by layer, all models, IID vs OOD (no grand mean)
  - Per-model plots: one figure per Qwen model, all configs shown individually
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    cohen_kappa_score,
    f1_score,
    r2_score,
)
# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_CLASSES = 5
CLIP_MIN    = 0
CLIP_MAX    = 4   # labels are 0-indexed: A1=0 … C+=4

MODELS = [
    {"name": "Qwen/Qwen3-Embedding-0.6B", "pooling": "last"},
    {"name": "Qwen/Qwen3-Embedding-4B",   "pooling": "last"},
    {"name": "Qwen/Qwen3-Embedding-8B",   "pooling": "last"},
]

# Configs whose y_pred is continuous and must be clipped before rounding
LINEAR_CONFIGS = {"linear_regression", "nn_linear_regression"}

XGB_IID_PATH = "./src/results/xgb_surface_iid_predictions.csv"
XGB_OOD_PATH = "./src/results/xgb_surface_ood_predictions.csv"

N_BOOT = 500

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def clip_if_linear(df, config_col="config"):
    df = df.copy()
    mask = df[config_col].str.lower().isin(LINEAR_CONFIGS)
    df.loc[mask, "y_pred"] = df.loc[mask, "y_pred"].clip(CLIP_MIN, CLIP_MAX)
    return df


def compute_metrics(y_true, y_pred):
    y_pred_r = np.array(y_pred).round().astype(int).clip(CLIP_MIN, CLIP_MAX)
    y_true   = np.array(y_true)
    out = {}
    out["qwk"] = (
        cohen_kappa_score(y_true, y_pred_r, weights="quadratic")
        if len(np.unique(y_true)) >= 2 else float("nan")
    )
    out["f1"]  = f1_score(y_true, y_pred_r, average="macro", zero_division=0)
    out["mae"] = mean_absolute_error(y_true, y_pred)
    out["r2"]  = r2_score(y_true, y_pred)
    return out


def aggregate(df, group_cols):
    records = []
    for keys, g in df.groupby(group_cols):
        row = dict(zip(group_cols, keys if isinstance(keys, tuple) else [keys]))
        row.update(compute_metrics(g["y_true"].values, g["y_pred"].values))
        records.append(row)
    return pd.DataFrame(records)


def load_model_data(model_name, pooling):
    slug    = model_name.replace("/", "_")
    base    = f"./src/results/loo_limited_{slug}_{pooling}"
    iid_raw = pd.read_csv(f"{base}_iid_predictions.csv")
    ood_raw = pd.read_csv(f"{base}_ood_predictions.csv")

    iid_raw = iid_raw[iid_raw["layer"] != 0]
    ood_raw = ood_raw[ood_raw["layer"] != 0]
    ood_raw = ood_raw[ood_raw["ood_group"] != "merlin-all"]

    iid_raw = clip_if_linear(iid_raw)
    ood_raw = clip_if_linear(ood_raw)

    iid_metrics = aggregate(iid_raw, ["layer", "config", "source"]).rename(columns={"source": "group"})
    ood_metrics = aggregate(ood_raw, ["layer", "config", "ood_group"]).rename(columns={"ood_group": "group"})
    return iid_metrics, ood_metrics


def bootstrap_std(vals, n_boot=N_BOOT, rng=None):
    rng   = np.random.default_rng(rng)
    means = [rng.choice(vals, size=len(vals), replace=True).mean()
             for _ in range(n_boot)]
    return float(np.std(means))


def build_qwk_ci(metrics_df, condition_label):
    """Mean QWK ± bootstrap std per (layer, config), averaged across groups."""
    records = []
    for (layer, config), g in metrics_df.groupby(["layer", "config"]):
        vals = g["qwk"].dropna().values
        if len(vals) == 0:
            continue
        records.append({
            "layer":     layer,
            "config":    config,
            "condition": condition_label,
            "mean":      vals.mean(),
            "std":       bootstrap_std(vals),
        })
    return pd.DataFrame(records)


def remove_borders(ax):
    """Remove all four spines from an axes."""
    for spine in ax.spines.values():
        spine.set_visible(False)


# ---------------------------------------------------------------------------
# Summary table: best (mean) per (model, config, condition)
# ---------------------------------------------------------------------------

METRICS_COLS = ["qwk", "f1", "mae", "r2"]
METRIC_LABEL = {"qwk": "QWK", "f1": "F1", "mae": "MAE", "r2": "R²"}

def fmt(best, mean):
    return f"{best:.3f} ({mean:.3f})"

summary_rows = []

for m in MODELS:
    model_name, pooling = m["name"], m["pooling"]
    model_label = model_name.split("-")[-1]

    try:
        iid_metrics, ood_metrics = load_model_data(model_name, pooling)
    except FileNotFoundError as e:
        print(f"[SKIP] {model_label}: {e}")
        continue

    configs = sorted(iid_metrics["config"].unique())

    for config in configs:
        row = {"Model": model_label, "Config": config}

        for condition, mdf in [("IID", iid_metrics), ("OOD", ood_metrics)]:
            cfg_df    = mdf[mdf["config"] == config]
            layer_qwk = cfg_df.groupby("layer")["qwk"].mean()
            if layer_qwk.empty:
                continue
            best_layer = layer_qwk.idxmax()
            best_df    = cfg_df[cfg_df["layer"] == best_layer]

            for met in METRICS_COLS:
                row[f"{condition}_{met}_best"] = best_df[met].mean()
                row[f"{condition}_{met}_mean"] = cfg_df[met].mean()

        summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)

# Build display table with "best (mean)" cells
display_rows = []
for _, r in summary_df.iterrows():
    drow = {"Model": r["Model"], "Config": r["Config"]}
    for condition in ["IID", "OOD"]:
        for met in METRICS_COLS:
            bk, mk = f"{condition}_{met}_best", f"{condition}_{met}_mean"
            drow[f"{condition} {METRIC_LABEL[met]}"] = (
                fmt(r[bk], r[mk]) if bk in r.index and not pd.isna(r.get(bk, float("nan"))) else "—"
            )
    display_rows.append(drow)

display_df = pd.DataFrame(display_rows)

# Mean-across-configs row per model
mean_rows = []
for model_label, grp in summary_df.groupby("Model"):
    mrow = {"Model": model_label, "Config": "MEAN"}
    for condition in ["IID", "OOD"]:
        for met in METRICS_COLS:
            bk, mk = f"{condition}_{met}_best", f"{condition}_{met}_mean"
            mrow[f"{condition} {METRIC_LABEL[met]}"] = (
                fmt(grp[bk].mean(), grp[mk].mean()) if bk in grp.columns else "—"
            )
    mean_rows.append(mrow)

display_df = pd.concat([display_df, pd.DataFrame(mean_rows)], ignore_index=True)
display_df["_is_mean"] = display_df["Config"] == "MEAN"
display_df = (
    display_df
    .sort_values(["Model", "_is_mean", "Config"])
    .drop(columns="_is_mean")
    .reset_index(drop=True)
)

col_order = ["Model", "Config"] + [
    f"{cond} {METRIC_LABEL[met]}" for met in METRICS_COLS for cond in ["IID", "OOD"]
]
display_df = display_df[[c for c in col_order if c in display_df.columns]]

# ---------------------------------------------------------------------------
# Also compute XGB all-metrics for a summary row
# ---------------------------------------------------------------------------

def xgb_all_metrics(path, group_col="source"):
    """Return mean-across-groups for all four metrics for the XGB baseline."""
    df = pd.read_csv(path)
    df["y_pred_c"] = df["y_pred"].clip(CLIP_MIN, CLIP_MAX)
    per_group = []
    for _, g in df.groupby(group_col):
        if g["y_true"].nunique() >= 2:
            per_group.append(compute_metrics(g["y_true"].values, g["y_pred_c"].values))
    if not per_group:
        return {m: float("nan") for m in METRICS_COLS}
    return {m: float(np.mean([r[m] for r in per_group])) for m in METRICS_COLS}

xgb_iid_metrics = xgb_all_metrics(XGB_IID_PATH)
xgb_ood_metrics = xgb_all_metrics(XGB_OOD_PATH)
xgb_iid_qwk     = xgb_iid_metrics["qwk"]
xgb_ood_qwk     = xgb_ood_metrics["qwk"]


# Append XGB row to the display table (replaces the earlier scalar-only version)
xgb_row = {"Model": "XGB-surface", "Config": "—"}
for condition, met_dict in [("IID", xgb_iid_metrics), ("OOD", xgb_ood_metrics)]:
    for met in METRICS_COLS:
        v = met_dict[met]
        xgb_row[f"{condition} {METRIC_LABEL[met]}"] = f"{v:.3f} (—)" if not np.isnan(v) else "—"

display_df = pd.concat(
    [pd.DataFrame([xgb_row]), display_df],
    ignore_index=True,
)

print("\n=== Summary Table (with XGB baseline) ===")
print(display_df.to_string(index=False))
display_df.to_csv("./src/results/summary_table.csv", index=False)

from matplotlib.lines import Line2D

# ---------------------------------------------------------------------------
# Combined QWK plot — all 3 models in one figure (no grand mean)
#
# Layout:
#   color     = model size  (0.6B / 4B / 8B)
#   marker    = model size  (circle = 0.6B, square = 4B, triangle = 8B)
#   linestyle = condition   (solid = IID, dashed = OOD)
#   black solid hlines = XGB IID / OOD baselines
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    "0.6B": "#f05d0d",
    "4B":   "#1773cf",
    "8B":   "#119e6a",
}
MODEL_MARKERS = {
    "0.6B": "o",
    "4B":   "^",
    "8B":   "*",
}
COND_LS = {"IID": "-", "OOD": "--"}

fig, ax = plt.subplots(figsize=(12, 6))
remove_borders(ax)

for m in MODELS:
    model_name, pooling = m["name"], m["pooling"]
    model_label = model_name.split("-")[-1]

    try:
        iid_metrics, ood_metrics = load_model_data(model_name, pooling)
    except FileNotFoundError:
        continue

    color  = MODEL_COLORS[model_label]
    marker = MODEL_MARKERS[model_label]
    iid_ci = build_qwk_ci(iid_metrics, "IID")
    ood_ci = build_qwk_ci(ood_metrics, "OOD")
    ci_df  = pd.concat([iid_ci, ood_ci], ignore_index=True)

    for condition in ["IID", "OOD"]:
        sub = (
            ci_df[ci_df["condition"] == condition]
            .groupby("layer", as_index=False)
            .agg(mean=("mean", "mean"), std=("std", "mean"))
            .sort_values("layer")
        )
        if sub.empty:
            continue

        ax.plot(
            sub["layer"], sub["mean"],
            linestyle=COND_LS[condition],
            marker=marker,
            color=color,
            alpha=0.6, markersize=10, lw=1.8,
        )
        ax.fill_between(
            sub["layer"],
            sub["mean"] - sub["std"],
            sub["mean"] + sub["std"],
            color=color, alpha=0.08,
        )

ax.axhline(xgb_iid_qwk, color="black", linestyle="-",  lw=1.4, alpha=0.25, zorder=0)
ax.axhline(xgb_ood_qwk, color="black", linestyle="--", lw=1.4, alpha=0.25, zorder=0)

x_label = ax.get_xlim()[1]
ax.text(x_label, xgb_iid_qwk + 0.013, f"XGBoost IID ({xgb_iid_qwk:.3f})",
        ha="right", va="bottom", fontsize=8.5, color="black", alpha=0.60)
ax.text(x_label, xgb_ood_qwk + 0.013, f"XGBoost OOD ({xgb_ood_qwk:.3f})",
        ha="right", va="bottom", fontsize=8.5, color="black", alpha=0.60)

ax.set_xlabel("Hidden Layer", fontsize=12)
ax.set_ylabel("Mean QWK (across groups & configs)", fontsize=12)
ax.set_ylim(0, 1)
ax.grid(alpha=0.2)

# Minor grid — y every 0.1, x every 5
ax.set_yticks([i * 0.1 for i in range(11)], minor=True)
ax.tick_params(axis="y", which="minor", length=0)
ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, 5), minor=True)
ax.grid(which="minor", alpha=0.15, linestyle="--")

# --- two-section legend ---
blank        = Line2D([0], [0], linestyle="none", label="")
header_model = Line2D([0], [0], linestyle="none", label=r"$\bf{Model\ size}$")
header_cond  = Line2D([0], [0], linestyle="none", label=r"$\bf{Condition}$")

model_handles = [
    Line2D([0], [0], color=MODEL_COLORS["0.6B"], marker="o", linestyle=" ", markersize=8,  label="Qwen3-0.6B"),
    Line2D([0], [0], color=MODEL_COLORS["4B"],   marker="^", linestyle=" ", markersize=8,  label="Qwen3-4B"),
    Line2D([0], [0], color=MODEL_COLORS["8B"],   marker="*", linestyle=" ", markersize=10, label="Qwen3-8B"),
]
cond_handles = [
    Line2D([0], [0], color="grey", linestyle="-",  lw=1.8, label="Identically Distributed (IID)"),
    Line2D([0], [0], color="grey", linestyle="--", lw=1.8, label="Out-of-Distribution (OOD)"),
]

ax.legend(
    handles=[header_model] + model_handles + [blank, header_cond] + cond_handles,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
    frameon=True,
    handlelength=2,
)

fig.suptitle(
    "QWK by layer — all models\n"
    "color/marker = model size  |  solid = IID, dashed = OOD  |  black = XGB surface baseline",
    fontsize=11,
)
plt.tight_layout()
plt.savefig("./src/results/plot_qwk_all_models.pdf", dpi=150, bbox_inches="tight")
plt.show()
# ---------------------------------------------------------------------------
# Per-model plots — one figure per Qwen model, all configs shown individually
# ---------------------------------------------------------------------------

# ── Define your condition markers here (line type) ──────────────────────────────
COND_MARKER = {"IID": "o", "OOD": "s"}

# ── Define your config colors and markers here (color and marker) ───────────────
CONFIG_COLORS = {
    # MLP family — cool blues/purples
    "MLP_linear_regression":   "#5B8FF9",   # cornflower blue
    "MLP_logistic_regression": "#7B5EA7",   # muted purple

    # linear/logistic family — warm coral/amber
    "linear_regression":       "#F4845F",   # soft coral
    "logistic_regression":     "#E8B84B",   # warm amber

    # ordinal — neutral teal, stands alone
    "ordinal_cumlink":         "#3DBDA7",   # teal
}
CONFIG_MARKERS = {
    "MLP_linear_regression":   "v",   # triangle down
    "MLP_logistic_regression": "s",   # square
    "linear_regression":       "^",   # triangle up
    "logistic_regression":     "o",   # circle
    "ordinal_cumlink":         "*",   # star
}
# ─────────────────────────────────────────────────────────────────────────────────
for m in MODELS:
    model_name, pooling = m["name"], m["pooling"]
    model_label = model_name.split("-")[-1]

    try:
        iid_metrics, ood_metrics = load_model_data(model_name, pooling)
    except FileNotFoundError:
        print(f"[SKIP per-model plot] {model_label}")
        continue

    configs = sorted(iid_metrics["config"].unique())

    iid_ci = build_qwk_ci(iid_metrics, "IID")
    ood_ci = build_qwk_ci(ood_metrics, "OOD")
    ci_df  = pd.concat([iid_ci, ood_ci], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    remove_borders(ax)

    for config in configs:
        color = CONFIG_COLORS.get(config, "#333333")   # fallback to dark grey
        for condition in ["IID", "OOD"]:
            sub = (
                ci_df[(ci_df["config"] == config) & (ci_df["condition"] == condition)]
                .sort_values("layer")
            )
            if sub.empty:
                continue

            ax.plot(
                sub["layer"], sub["mean"],
                linestyle=COND_LS[condition],
                marker=CONFIG_MARKERS.get(config, "x"),
                color=color,
                alpha=0.6, markersize=8, lw=1.8,
            )
            ax.fill_between(
                sub["layer"],
                sub["mean"] - sub["std"],
                sub["mean"] + sub["std"],
                color=color, alpha=0.08,
            )

    ax.set_xlabel("Hidden Layer", fontsize=12)
    ax.set_ylabel("Mean QWK (across groups)", fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)

    # Minor grid — y every 0.1, x every 5
    ax.set_yticks([i * 0.1 for i in range(11)], minor=True)
    ax.tick_params(axis="y", which="minor", length=0)
    ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, 5), minor=True)
    ax.grid(which="minor", alpha=0.15, linestyle="--")

    # --- two-section legend (matches combined plot style) ---
    blank         = Line2D([0], [0], linestyle="none", label="")
    header_config = Line2D([0], [0], linestyle="none", label=r"$\bf{Config}$")
    header_cond   = Line2D([0], [0], linestyle="none", label=r"$\bf{Condition}$")

    config_handles = [
        Line2D([0], [0],
               color=CONFIG_COLORS.get(cfg, "#333333"),
               marker=CONFIG_MARKERS.get(cfg, "x"),
               linestyle=" ", markersize=8, label=cfg)
        for cfg in configs
    ]
    cond_handles = [
        Line2D([0], [0], color="grey", linestyle="-",  lw=1.8, label="Identically Distributed (IID)"),
        Line2D([0], [0], color="grey", linestyle="--", lw=1.8, label="Out-of-Distribution (OOD)"),
    ]

    ax.legend(
        handles=[header_config] + config_handles + [blank, header_cond] + cond_handles,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=True,
        handlelength=2,
    )

    fig.suptitle(
        f"QWK by layer — Qwen3-Embedding-{model_label} — all configs\n"
        "color/marker = config  |  solid = IID, dashed = OOD  |  black = XGB surface baseline",
        fontsize=11,
    )
    plt.tight_layout()
    save_path = f"./src/results/plot_qwk_{model_label}_all_configs.pdf"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {save_path}")

# %%
# ---------------------------------------------------------------------------
# Per-dataset plots — Qwen3-4B, MLP_linear_regression
# One figure per condition (IID / OOD), one line per dataset/group
# ---------------------------------------------------------------------------

# Assign a distinct color to each group
GROUP_PALETTE = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3",
]

_4b_name    = "Qwen/Qwen3-Embedding-4B"
_4b_pooling = "last"
_4b_config  = "MLP_linear_regression"

try:
    iid_raw_4b = pd.read_csv(
        f"./src/results/loo_limited_{_4b_name.replace('/', '_')}_{_4b_pooling}_iid_predictions.csv"
    )
    ood_raw_4b = pd.read_csv(
        f"./src/results/loo_limited_{_4b_name.replace('/', '_')}_{_4b_pooling}_ood_predictions.csv"
    )

    iid_raw_4b = iid_raw_4b[(iid_raw_4b["layer"] != 0) & (iid_raw_4b["config"] == _4b_config)]
    ood_raw_4b = ood_raw_4b[
        (ood_raw_4b["layer"] != 0) &
        (ood_raw_4b["config"] == _4b_config) &
        (ood_raw_4b["ood_group"] != "merlin-all")
    ]
    iid_raw_4b = clip_if_linear(iid_raw_4b)
    ood_raw_4b = clip_if_linear(ood_raw_4b)

    for condition, raw_df, group_col, xgb_qwk, cond_label in [
        ("IID", iid_raw_4b, "source",    xgb_iid_qwk, "Identically Distributed"),
        ("OOD", ood_raw_4b, "ood_group", xgb_ood_qwk, "Out-of-Distribution"),
    ]:
        groups = sorted(raw_df[group_col].unique())
        color_map = {g: GROUP_PALETTE[i % len(GROUP_PALETTE)] for i, g in enumerate(groups)}

        # Compute QWK per (layer, group)
        per_group_metrics = aggregate(raw_df, ["layer", group_col])

        fig, ax = plt.subplots(figsize=(12, 6))
        remove_borders(ax)

        for group in groups:
            sub = (
                per_group_metrics[per_group_metrics[group_col] == group]
                .sort_values("layer")
            )
            if sub.empty:
                continue
            ax.plot(
                sub["layer"], sub["qwk"],
                marker="o", linestyle="-",
                color=color_map[group],
                label=group, alpha=0.75, markersize=6, lw=1.8,
            )


        ax.set_xlabel("Hidden Layer", fontsize=12)
        ax.set_ylabel("QWK", fontsize=12)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.2)

        ax.set_yticks([i * 0.1 for i in range(11)], minor=True)
        ax.tick_params(axis="y", which="minor", length=0)
        ax.set_xticks(range(0, int(ax.get_xlim()[1]) + 1, 5), minor=True)
        ax.grid(which="minor", alpha=0.15, linestyle="--")

        # Legend — one entry per dataset
        blank          = Line2D([0], [0], linestyle="none", label="")
        header_dataset = Line2D([0], [0], linestyle="none", label=r"$\bf{Dataset}$")
        group_handles  = [
            Line2D([0], [0], color=color_map[g], marker="o", linestyle="-",
                   markersize=6, lw=1.8, label=g)
            for g in groups
        ]
        ax.legend(
            handles=[header_dataset] + group_handles,
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            frameon=True,
            handlelength=2,
        )

        fig.suptitle(
            f"QWK by layer — Qwen3-4B · MLP Regression · {cond_label}\n"
            "one line per dataset  |  dashed black = XGBoost baseline",
            fontsize=11,
        )
        plt.tight_layout()
        save_path = f"./src/results/plot_qwk_4B_MLP_regression_per_dataset_{condition}.pdf"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"Saved: {save_path}")

except FileNotFoundError as e:
    print(f"[SKIP per-dataset plot] {e}")


# %%
# ---------------------------------------------------------------------------
# Boxplot — IID vs OOD, all models pooled
# ---------------------------------------------------------------------------

aggregate_records = []

for m in MODELS:
    model_name, pooling = m["name"], m["pooling"]

    try:
        iid_metrics, ood_metrics = load_model_data(model_name, pooling)
    except FileNotFoundError:
        continue

    for condition, mdf in [("IID", iid_metrics), ("OOD", ood_metrics)]:
        for (layer, config), g in mdf.groupby(["layer", "config"]):
            vals = g["qwk"].dropna().values
            if len(vals) == 0:
                continue

            aggregate_records.append({
                "condition": condition,
                "model": model_name,
                "layer": layer,
                "config": config,
                "mean_qwk":  vals.mean(),
            })

aggregate_df = pd.DataFrame(aggregate_records)

conditions = ["IID", "OOD"]
data = [aggregate_df[aggregate_df["condition"] == c]["mean_qwk"].dropna().values for c in conditions]

CONDITION_COLORS = {
    "IID": "#349a1d",
    "OOD": "#ff5e0e",
}


fig, ax = plt.subplots(figsize=(4, 4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

ax.boxplot(
    data,
    labels=["IID", "OOD"],
    patch_artist=True,
    boxprops=dict(alpha=0.4, facecolor="white", color="black", linewidth=1.2),
    medianprops=dict(linewidth=2, color="black"),
    whiskerprops=dict(linewidth=1.2, color="black"),
    capprops=dict(linewidth=1.2, color="black"),
    flierprops=dict(marker="o", markersize=4, alpha=0.4, linestyle="none"),
)

for patch, condition in zip(ax.patches, conditions):
    color = CONDITION_COLORS[condition]
    patch.set_facecolor(color)
    patch.set_edgecolor("black")


ax.set_ylabel("")
ax.set_xlabel("")
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis="both", labelsize=11)
ax.grid(axis="y", alpha=0.15)
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(axis="both", which="both", length=0)

plt.tight_layout()
plt.savefig("./src/results/plot_qwk_boxplot_pooled.pdf", dpi=150, bbox_inches="tight",
            facecolor="white")
plt.show()
print("Saved: ./src/results/plot_qwk_boxplot_pooled.pdf")

fig, ax = plt.subplots(figsize=(4, 4))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

parts = ax.violinplot(data, positions=[1, 2], showmedians=False, showextrema=False)

for i, (pc, condition) in enumerate(zip(parts["bodies"], conditions)):
    color = CONDITION_COLORS[condition]
    pc.set_facecolor(color)
    pc.set_edgecolor("black")
    pc.set_alpha(0.4)

# Overlay a slim boxplot for IQR reference
ax.boxplot(
    data,
    positions=[1, 2],
    widths=0.08,
    patch_artist=True,
    boxprops=dict(facecolor="white", color="black", linewidth=1.2),
    medianprops=dict(linewidth=2, color="black"),
    whiskerprops=dict(linewidth=1.2, color="black"),
    capprops=dict(linewidth=1.2, color="black"),
    flierprops=dict(visible=False),
)

ax.set_xticks([1, 2])
ax.set_xticklabels(["IID", "OOD"])
ax.set_ylim(0, 1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.tick_params(axis="both", labelsize=11, length=0)
ax.grid(axis="y", alpha=0.15)
for spine in ax.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("./src/results/plot_qwk_violin_pooled.pdf", dpi=150, bbox_inches="tight", facecolor="white")

# %%
# ---------------------------------------------------------------------------
# Violin plot — distribution of mean QWK per (model, layer, config)
#
# Each data point = mean QWK across groups for one (model, layer, config) combo.
# X-axis: model size (0.6B, 4B, 8B)
# Color/split: IID (solid fill) vs OOD (hatched)
# Overlaid strip plot shows individual (layer, config) data points.
# XGB IID / OOD baselines drawn as horizontal reference lines.
# ---------------------------------------------------------------------------
 
# Collect one mean-QWK value per (model, layer, config, condition)
violin_records = []
 
for m in MODELS:
    model_name, pooling = m["name"], m["pooling"]
    model_label = model_name.split("-")[-1]
 
    try:
        iid_metrics, ood_metrics = load_model_data(model_name, pooling)
    except FileNotFoundError:
        continue
 
    for condition, mdf in [("IID", iid_metrics), ("OOD", ood_metrics)]:
        # Mean QWK across groups for each (layer, config) pair
        for (layer, config), g in mdf.groupby(["layer", "config"]):
            vals = g["qwk"].dropna().values
            if len(vals) == 0:
                continue
            violin_records.append({
                "model":     model_label,
                "layer":     layer,
                "config":    config,
                "condition": condition,
                "mean_qwk":  vals.mean(),
            })
 
violin_df = pd.DataFrame(violin_records)
 
model_order = ["0.6B", "4B", "8B"]
conditions  = ["IID", "OOD"]
 
# Colors matching the rest of the script — one per model
VIOLIN_COLORS = {
    "0.6B": "#f05d0d",
    "4B":   "#1773cf",
    "8B":   "#119e6a",
}
 
# Horizontal positions: each condition gets a cluster of 3 violins (one per model)
CLUSTER_WIDTH = 0.28   # half-width of each violin body
CLUSTER_GAP   = 0.32   # spacing between model violins within a condition cluster
COND_SPACING  = 2.0    # distance between IID and OOD cluster centres
 
positions = {}   # (condition, model) -> x centre
for ci, condition in enumerate(conditions):
    base = ci * COND_SPACING
    for mi, model in enumerate(model_order):
        offset = (mi - 1) * CLUSTER_GAP   # centre the trio around base
        positions[(condition, model)] = base + offset
 
fig, ax = plt.subplots(figsize=(10, 6))
remove_borders(ax)
 
rng = np.random.default_rng(42)
 
from scipy.stats import gaussian_kde
 
for condition in conditions:
    for model in model_order:
        vals = violin_df[
            (violin_df["model"] == model) &
            (violin_df["condition"] == condition)
        ]["mean_qwk"].dropna().values
 
        if len(vals) < 3:
            continue
 
        xc    = positions[(condition, model)]
        color = VIOLIN_COLORS[model]
 
        # --- violin body via kernel density estimate ---
        kde  = gaussian_kde(vals, bw_method="scott")
        ymin = max(vals.min() - 0.05, 0)
        ymax = min(vals.max() + 0.05, 1)
        ygrid = np.linspace(ymin, ymax, 300)
        dens  = kde(ygrid)
        dens  = dens / dens.max() * CLUSTER_WIDTH
 
        ax.fill_betweenx(
            ygrid, xc - dens, xc + dens,
            color=color, alpha=0.45, linewidth=0,
        )
        ax.plot(xc - dens, ygrid, color=color, lw=0.8, alpha=0.7)
        ax.plot(xc + dens, ygrid, color=color, lw=0.8, alpha=0.7)
 
        # --- median line ---
        med = np.median(vals)
        half_w = float(kde([med])[0]) / kde(ygrid).max() * CLUSTER_WIDTH
        ax.hlines(med, xc - half_w, xc + half_w,
                  color="black", lw=2.2, alpha=0.3, zorder=5)
 
        # --- IQR box ---
        #q1, q3 = np.percentile(vals, [25, 75])
        #ax.vlines(xc, q1, q3, color="black", lw=4, alpha=0.25, zorder=3)
 
        # --- strip / jitter ---
        jitter = rng.uniform(-CLUSTER_WIDTH * 0.45, CLUSTER_WIDTH * 0.45, size=len(vals))
        ax.scatter(
            xc + jitter, vals,
            color=color, s=18, alpha=0.55,
            edgecolors="none", zorder=4,
        )
 
# --- XGB baselines — IID line near IID cluster, OOD near OOD cluster ---
for condition, xgb_qwk, ls in [("IID", xgb_iid_qwk, "-"), ("OOD", xgb_ood_qwk, "--")]:
    x_left  = positions[(condition, "0.6B")] - CLUSTER_WIDTH * 2
    x_right = positions[(condition, "8B")]   + CLUSTER_WIDTH * 2
    ax.hlines(xgb_qwk, x_left, x_right, color="black", linestyle=ls, lw=1.3, alpha=0.35, zorder=0)

 
# --- axes formatting ---
xtick_positions = [ci * COND_SPACING for ci in range(len(conditions))]
ax.set_xticks(xtick_positions)
ax.set_xticklabels(
    ["Identically Distributed (IID)", "Out-of-Distribution (OOD)"],
    fontsize=12,
)
ax.set_ylabel("Mean QWK (across groups)", fontsize=12)
ax.set_ylim(0, 1)
ax.set_yticks([i * 0.1 for i in range(11)], minor=True)
ax.tick_params(axis="y", which="minor", length=0)
ax.grid(axis="y", alpha=0.0)
ax.grid(axis="y", which="minor", alpha=0.0, linestyle="--")
ax.set_xlim(
    positions[("IID", "0.6B")] - CLUSTER_WIDTH * 3,
    positions[("OOD", "8B")]   + CLUSTER_WIDTH * 6,
)
 
# --- legend ---
legend_handles = [
    plt.matplotlib.patches.Patch(facecolor=VIOLIN_COLORS[m], alpha=0.6, label=f"Qwen3-{m}")
    for m in model_order
]
blank       = Line2D([0], [0], linestyle="none", label="")
header_note = Line2D([0], [0], linestyle="none", label=r"$\bf{Each\ point}$: one (layer, config)")
 
ax.legend(
    handles=legend_handles + [blank, header_note],
    bbox_to_anchor=(1.01, 1),
    loc="upper left",
    frameon=True,
    fontsize=9,
    handlelength=1.5,
)
 
fig.suptitle(
    "Distribution of mean QWK — each point is one (layer × config) combination\n"
    "color = model size  |  horizontal bar = median  |  thick line = IQR",
    fontsize=11,
)
plt.tight_layout()
plt.savefig("./src/results/plot_qwk_violin.pdf", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: ./src/results/plot_qwk_violin.pdf")
# %%