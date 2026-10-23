#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Publication-Quality Figure 1: Performance Scatter Plots (Nature Style)
-------------------------------------------------------------------------------
  - Font fixed to Arial.
  - 1x3 Layout for PC1, PC2, PC3 Model Performance.
  - UPDATED: Removed gridlines for a cleaner look.
  - UPDATED: Tightened horizontal spacing between subplots.
  - UPDATED: Moved colorbar significantly closer to the plots.
"""

import os
import re
from pathlib import Path
import string
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize, LinearSegmentedColormap
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "/data/global_mapping/models_V100/drive/final_shap_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_TIF = os.path.join(OUTPUT_DIR, "Fig_Performance_Scatter.tif")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Fig_Performance_Scatter.pdf")

PCS = [1_NC, 2_NC, 3_NC]
DPI = 600

# ============================================================================
# FONT & NATURE STYLE CONFIGURATION
# ============================================================================
font_family_name = "Arial"

mpl.rcParams.update({
    "font.family":       font_family_name,
    "font.sans-serif":   [font_family_name, "DejaVu Sans"],
    "font.size":         12,
    "text.color":        "black",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    
    "axes.linewidth":    0.8,
    "axes.spines.top":    True,  
    "axes.spines.right":  True,  
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size":  3.5,     
    "ytick.major.size":  3.5,
    "xtick.direction":   "out",   
    "ytick.direction":   "out",
    
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "mathtext.default":  "regular",   
})

CUSTOM_COLORS = ["#1E316A", "#00C3C9", "#FBE344", "#AD002A"] 
GLOBAL_CMAP = LinearSegmentedColormap.from_list("my_custom_cmap", CUSTOM_COLORS)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================
def get_fast_kde_density(x: np.ndarray, y: np.ndarray, max_samples=5000):
    xy = np.vstack([x, y])
    if xy.shape[1] > max_samples:
        idx = np.random.choice(xy.shape[1], max_samples, replace=False)
        kde = gaussian_kde(xy[:, idx])
    else:
        kde = gaussian_kde(xy)
    return kde(xy)

def polish_ax(ax, add_grid=False):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color("black")
    if add_grid:
        pass  # Force turning off gridlines

def parse_metric(val):
    if pd.isna(val):
        return ""
    val_str = str(val)
    floats = re.findall(r"[-+]?(?:\d*\.\d+|\d+)", val_str)
    if len(floats) >= 2:
        return f"{float(floats[0]):.2f} $\\pm$ {float(floats[1]):.2f}"
    elif len(floats) == 1:
        return f"{float(floats[0]):.2f}"
    return val_str

# ============================================================================
# MAIN PLOTTING LOOP
# ============================================================================
def main():
    print(f"Using font: {font_family_name}")
    print("Initializing Performance Scatter figure (Tight Layout)...")
    
    # Adjust to a wider aspect ratio to compress spacing
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(11, 4), dpi=150)
    text_bbox = dict(boxstyle="square,pad=0.2", facecolor="white", edgecolor="none", alpha=0.75)

    # Recalculate coordinates: increase W, reduce spacing in x_starts
    W = 0.26  
    H = 0.70  
    x_starts = [0.07, 0.38, 0.69]  
    y_starts = [0.22]  

    for row_idx, PC in enumerate(PCS):
        print(f"\nProcessing PC: {PC} Scatter...")

        base_dir = f"/data/global_mapping/models_V100/drive/PC{PC}/XGB_shap/XGB_shap"
        pred_csv = f"{base_dir}/block_1500km_buf_1000km/1b_Block_CV_Per_Point_Predictions.csv"
        summary_csv = f"{base_dir}/Block_CV/Summary_Across_Buffers_paper.csv"

        ax_perf = axes[row_idx]
        ax_perf.set_position([x_starts[row_idx], y_starts[0], W, H])

        # Extract R2
        global_r2_text = ""
        if os.path.exists(summary_csv):
            try:
                df_sum = pd.read_csv(summary_csv)
                if "Multivar_R2" in df_sum.columns:
                    global_r2_text = parse_metric(df_sum["Multivar_R2"].values[0])
            except Exception:
                pass

        # Plot Scatter
        df_pred = pd.read_csv(pred_csv)
        y_obs = df_pred["y_true"].values
        y_prd = df_pred["y_pred_cv"].values

        if not global_r2_text:
            global_r2_text = f"{r2_score(y_obs, y_prd):.2f}"
        
        z = get_fast_kde_density(y_obs, y_prd)
        sort_idx = z.argsort()
        y_obs_s, y_prd_s, z_s = y_obs[sort_idx], y_prd[sort_idx], z[sort_idx]

        ax_perf.scatter(y_obs_s, y_prd_s, c=z_s, cmap=GLOBAL_CMAP, s=1.5, alpha=0.6, edgecolors="none")

        min_val = min(y_obs.min(), y_prd.min())
        max_val = max(y_obs.max(), y_prd.max())
        ax_perf.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.2, zorder=3)

        ax_perf.set_xlabel(f"PC{PC} Observed Value", fontsize=12, fontweight="normal")
        if row_idx == 0:
            ax_perf.set_ylabel("Predicted Value", fontsize=12, fontweight="normal")
        
        ax_perf.text(0.05, 0.95, f"$R^2$ = {global_r2_text}", transform=ax_perf.transAxes, 
                     fontsize=12, fontweight="normal", va="top", bbox=text_bbox)
                     
        polish_ax(ax_perf, add_grid=False)

    # Standardize panel letter positions (a, b, c)
    panel_letters = list(string.ascii_lowercase)
    for i, ax in enumerate(axes):
        ax.text(-0.14, 1.03, panel_letters[i], transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="right")

    # Move bottom colorbar significantly upwards to cling to subplots
    fig.canvas.draw()
    cax_w = 0.35 
    cax_h = 0.025
    cax_x = 0.5 - (cax_w / 2)
    # Lift from -0.05 to 0.04 to close the gap with subplots
    cax_y = 0.04 
    
    cax = fig.add_axes([cax_x, cax_y, cax_w, cax_h])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=GLOBAL_CMAP, norm=Normalize(0, 1)),
                      cax=cax, orientation="horizontal", ticks=[0, 1])
    cb.ax.set_xticklabels(["Low Density", "High Density"], fontsize=12, fontweight="normal")
    cb.outline.set_linewidth(0.8)

    print(f"\nSaving Scatter TIF figure to:\n{OUTPUT_TIF}")
    fig.savefig(OUTPUT_TIF, dpi=DPI, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")
    plt.close()
    print("Scatter generation complete!")

if __name__ == "__main__":
    main()