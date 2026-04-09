# %%
# ---------------------------------------------------------------------------
# Ridge plots — residuals and predictions
# IID and OOD in separate figures (4 plots total)
# Colors cycle through a colormap as layers increase (bottom = dark, top = light)
# Two configs overlaid per ridge row (solid vs dashed)
# All layers shown (no sampling) — square / quadratic figure shape
# ---------------------------------------------------------------------------

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap

# ── Pick your model & config here ───────────────────────────────────────────
RIDGE_MODEL_NAME = "Qwen/Qwen3-Embedding-4B"   # change to 0.6B or 8B as needed
RIDGE_POOLING    = "last"

# Both configs to overlay
RIDGE_CONFIGS = ["linear_regression", "MLP_linear_regression"]
CONFIG_STYLES = {
    "linear_regression":     {"ls": "-",  "label": "Linear Regression"},
    "MLP_linear_regression": {"ls": "--", "label": "MLP Linear Regression"},
}

# Tuning knobs
RIDGE_BANDWIDTH  = 0.05     # KDE bandwidth in label units
RIDGE_OVERLAP    = 0.85     # fraction of row height each ridge may fill
COLORMAP         = "viridis" # any matplotlib colormap, e.g. "plasma", "cividis", "turbo"

CLIP_MIN = 0
CLIP_MAX = 4

LINEAR_CONFIGS = {"linear_regression", "nn_linear_regression"}

# ── Helpers ──────────────────────────────────────────────────────────────────

def clip_if_linear(df, config_col="config"):
    df = df.copy()
    mask = df[config_col].str.lower().isin(LINEAR_CONFIGS)
    df.loc[mask, "y_pred"] = df.loc[mask, "y_pred"].clip(CLIP_MIN, CLIP_MAX)
    return df


def remove_borders(ax):
    for spine in ax.spines.values():
        spine.set_visible(False)


def kde1d(vals, x_grid, bw=0.25):
    """Gaussian KDE evaluated on x_grid."""
    vals  = np.asarray(vals, dtype=float)
    diffs = (x_grid[:, None] - vals[None, :]) / bw
    return np.exp(-0.5 * diffs**2).sum(axis=1) / (len(vals) * bw * np.sqrt(2 * np.pi))


def ridge_figure_multi(layer_data_by_config, layers, configs,
                       x_min, x_max, title, xlabel, fname,
                       cmap_name=COLORMAP, xtick_override=None):
    """
    Draw a ridge / joy-plot overlaying multiple configs per layer row.

    layer_data_by_config : dict  config -> dict { layer -> 1-d array of values }
    layers               : ordered list of layers (bottom = first, top = last)
    configs              : ordered list of config names to overlay
    """
    n      = len(layers)
    x_grid = np.linspace(x_min, x_max, 600)
    cmap   = get_cmap(cmap_name)

    row_h = 1.0
    fig_h = max(6, n * row_h * 0.55)
    fig_w = fig_h                           # square canvas
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    remove_borders(ax)

    y_step = row_h

    for i, layer in enumerate(layers):
        y_base = i * y_step
        color  = cmap(i / max(n - 1, 1))

        ax.text(x_min - 0.04 * (x_max - x_min), y_base,
                f"Layer {layer}", va="center", ha="right",
                fontsize=7.5, color="#444444")

        for cfg in configs:
            vals = layer_data_by_config[cfg].get(layer, np.array([]))
            if len(vals) < 5:
                continue

            ls    = CONFIG_STYLES[cfg]["ls"]
            alpha_fill = 0.30 if ls == "--" else 0.45

            density = kde1d(vals, x_grid, bw=RIDGE_BANDWIDTH)
            scale   = (RIDGE_OVERLAP * y_step) / (density.max() + 1e-12)
            y_ridge = y_base + density * scale

            ax.fill_between(x_grid, y_base, y_ridge,
                            color=color, alpha=alpha_fill)
            ax.plot(x_grid, y_ridge,
                    color=color, lw=0.9, alpha=0.90, ls=ls)

        ax.axhline(y_base, color="white", lw=0.4, alpha=0.6, zorder=3)

    if "residual" in fname.lower():
        ax.axvline(0, color="#333333", lw=1.0, ls="--", alpha=0.45, zorder=4)
    else:
        for tick in range(int(x_min), int(x_max) + 1):
            ax.axvline(tick, color="#bbbbbb", lw=0.6, ls=":", alpha=0.6, zorder=0)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(-y_step * 0.5, n * y_step + y_step * 0.6)
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_yticks([])
    if xtick_override is not None:
        ticks, labels = zip(*xtick_override)
        ax.set_xticks(list(ticks))
        ax.set_xticklabels(list(labels))

    # Remove legend
    legend_handles = [
        mpatches.Patch(
            facecolor="grey",
            alpha=0.55,
            label=CONFIG_STYLES[cfg]["label"],
            linestyle=CONFIG_STYLES[cfg]["ls"],
            linewidth=1.5,
        )
        for cfg in configs
    ]
    # Use Line2D for linestyle distinction in legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0],
               color="grey",
               lw=1.5,
               ls=CONFIG_STYLES[cfg]["ls"],
               label=CONFIG_STYLES[cfg]["label"])
        for cfg in configs
    ]
    # Place legend outside the plot area on the right
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8,
              bbox_to_anchor=(1.15, 1.15), frameon=False)

    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved: {fname}")


# ── Load & filter ────────────────────────────────────────────────────────────
_slug  = RIDGE_MODEL_NAME.replace("/", "_")
_base  = f"./src/results/loo_limited_{_slug}_{RIDGE_POOLING}"
_label = RIDGE_MODEL_NAME.split("-")[-1]

iid_raw = pd.read_csv(f"{_base}_iid_predictions.csv")
ood_raw = pd.read_csv(f"{_base}_ood_predictions.csv")

available_configs = iid_raw["config"].unique().tolist()
for cfg in RIDGE_CONFIGS:
    if cfg not in available_configs:
        raise ValueError(
            f"Config '{cfg}' not found.\n"
            f"Available configs: {available_configs}"
        )

# Filter to only the two configs we want
iid_raw = iid_raw[(iid_raw["layer"] != 0) &
                  (iid_raw["config"].isin(RIDGE_CONFIGS))].copy()
ood_raw = ood_raw[(ood_raw["layer"] != 0) &
                  (ood_raw["config"].isin(RIDGE_CONFIGS)) &
                  (ood_raw["ood_group"] != "merlin-all")].copy()

iid_raw = clip_if_linear(iid_raw)
ood_raw = clip_if_linear(ood_raw)

iid_raw["residual"] = iid_raw["y_pred"] - iid_raw["y_true"]
ood_raw["residual"] = ood_raw["y_pred"] - ood_raw["y_true"]

all_layers = sorted(set(iid_raw["layer"].unique()) |
                    set(ood_raw["layer"].unique()))

# Build per-config, per-layer dicts
def build_layer_dict(df, value_col):
    return {
        cfg: {
            L: df.loc[(df["config"] == cfg) & (df["layer"] == L), value_col].values
            for L in all_layers
        }
        for cfg in RIDGE_CONFIGS
    }

iid_res_by_cfg  = build_layer_dict(iid_raw, "residual")
ood_res_by_cfg  = build_layer_dict(ood_raw, "residual")
iid_pred_by_cfg = build_layer_dict(iid_raw, "y_pred")
ood_pred_by_cfg = build_layer_dict(ood_raw, "y_pred")

_safe_configs = "_vs_".join(c.replace("/", "_") for c in RIDGE_CONFIGS)

# ── (1a) Residuals — IID ─────────────────────────────────────────────────────
ridge_figure_multi(
    layer_data_by_config=iid_res_by_cfg,
    layers=all_layers,
    configs=RIDGE_CONFIGS,
    x_min=-4.5, x_max=4.5,
    title=f"Ridge distributions by Layer\nIID — Residuals  ({' vs '.join(CONFIG_STYLES[c]['label'] for c in RIDGE_CONFIGS)})",
    xlabel="Residual  (y_pred − y_true)",
    fname=f"./src/results/plot_ridge_residuals_{_label}_{_safe_configs}_IID.pdf",
)

# ── (1b) Residuals — OOD ─────────────────────────────────────────────────────
ridge_figure_multi(
    layer_data_by_config=ood_res_by_cfg,
    layers=all_layers,
    configs=RIDGE_CONFIGS,
    x_min=-4.5, x_max=4.5,
    title=f"Ridge distributions by Layer\nOOD — Residuals  ({' vs '.join(CONFIG_STYLES[c]['label'] for c in RIDGE_CONFIGS)})",
    xlabel="Residual  (y_pred − y_true)",
    fname=f"./src/results/plot_ridge_residuals_{_label}_{_safe_configs}_OOD.pdf",
)

cefr_labels   = ["A1", "A2", "B1", "B2", "C+"]
cefr_ticks    = list(range(CLIP_MIN, CLIP_MAX + 1))
cefr_override = list(zip(cefr_ticks, cefr_labels))

# ── (2a) Predictions — IID ───────────────────────────────────────────────────
ridge_figure_multi(
    layer_data_by_config=iid_pred_by_cfg,
    layers=all_layers,
    configs=RIDGE_CONFIGS,
    x_min=CLIP_MIN - 0.5, x_max=CLIP_MAX + 0.5,
    title=f"Ridge distributions by Layer\nIID — Predicted values  ({' vs '.join(CONFIG_STYLES[c]['label'] for c in RIDGE_CONFIGS)})",
    xlabel="Predicted CEFR level",
    fname=f"./src/results/plot_ridge_predictions_{_label}_{_safe_configs}_IID.pdf",
    xtick_override=cefr_override,
)

# ── (2b) Predictions — OOD ───────────────────────────────────────────────────
ridge_figure_multi(
    layer_data_by_config=ood_pred_by_cfg,
    layers=all_layers,
    configs=RIDGE_CONFIGS,
    x_min=CLIP_MIN - 0.5, x_max=CLIP_MAX + 0.5,
    title=f"Ridge distributions by Layer\nOOD — Predicted values  ({' vs '.join(CONFIG_STYLES[c]['label'] for c in RIDGE_CONFIGS)})",
    xlabel="Predicted CEFR level",
    fname=f"./src/results/plot_ridge_predictions_{_label}_{_safe_configs}_OOD.pdf",
    xtick_override=cefr_override,
)

# %%