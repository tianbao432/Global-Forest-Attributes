#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Publication-Quality Composite Figure for Global Mapping Models (Nature Style)
-------------------------------------------------------------------------------
  - Font fixed to Arial.
  - Custom Hex colormaps for high-density spatial visualization.
  - FIXED: Calculates R2 as "1000km buffer R2 +/- cross-buffer SD".
  - FIXED: Removed bounding box and bold font from R2 text for a cleaner look.
  - FIXED: Eliminates abstract 1e8 scientific notation for SSR.
  - FIXED: Implements MaxNLocator for extremely clean, uniform tick intervals.
-------------------------------------------------------------------------------
"""

import os
import re
from pathlib import Path
import math
import string
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from sklearn.metrics import r2_score, mean_squared_error
import statsmodels.api as sm
import shap

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================
OUTPUT_DIR = "/data/global_mapping/models_V100/drive_fina/final_shap_visuals"
os.makedirs(OUTPUT_DIR, exist_ok=True)

OUTPUT_TIF = os.path.join(OUTPUT_DIR, "Fig_Composite_Global_SHAP.tif")
OUTPUT_PDF = os.path.join(OUTPUT_DIR, "Fig_Composite_Global_SHAP.pdf")

PCS = [1, 2, 3]
DPI = 600

# Trend line (dependence panels) styling
TREND_COLOR = "#e74c3c"   
TREND_LW    = 1.5         

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
    
    # Nature Style: Delicate lines
    "axes.linewidth":    0.8,
    
    # Nature Style: Full borders
    "axes.spines.top":    True,  
    "axes.spines.right":  True,  
    "axes.spines.left":   True,
    "axes.spines.bottom": True,
    
    # Nature Style: Ticks formatting
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
def get_fast_kde_density(x: np.ndarray, y: np.ndarray, max_samples=50000):
    xy = np.vstack([x, y])
    if xy.shape[1] > max_samples:
        idx = np.random.choice(xy.shape[1], max_samples, replace=False)
        kde = gaussian_kde(xy[:, idx])
    else:
        kde = gaussian_kde(xy)
    return kde(xy)

def long_to_explanation(df_long: pd.DataFrame, feature_order: list[str]) -> shap.Explanation:
    if "row_idx" not in df_long.columns:
        df_long = df_long.copy()
        df_long["row_idx"] = df_long.groupby("feature").cumcount()

    sv_wide = df_long.pivot_table(index="row_idx", columns="feature", values="shap_value", aggfunc="first").reindex(columns=feature_order)
    fv_wide = df_long.pivot_table(index="row_idx", columns="feature", values="feature_value", aggfunc="first").reindex(columns=feature_order)

    valid = ~(sv_wide.isna().any(axis=1) | fv_wide.isna().any(axis=1))
    sv_wide = sv_wide.loc[valid]
    fv_wide = fv_wide.loc[valid]

    return shap.Explanation(
        values        = sv_wide.to_numpy(dtype=np.float64),
        data          = fv_wide.to_numpy(dtype=np.float64),
        feature_names = list(sv_wide.columns),
        base_values   = np.zeros(len(sv_wide), dtype=np.float64)
    )

def polish_ax(ax, add_grid=False):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(True)
        ax.spines[s].set_linewidth(0.8)
        ax.spines[s].set_color("black")
    if add_grid:
        ax.grid(True, linestyle=':', color='#e5e5e5', linewidth=0.8, zorder=0)

def pull_in_yticklabels(ax):
    ax.tick_params(axis="y", labelsize=11, colors="black", pad=5, length=0)
    for lbl in ax.get_yticklabels():
        lbl.set_horizontalalignment("right")

# ============================================================================
# MAIN PLOTTING LOOP
# ============================================================================
def main():
    print(f"Using font: {font_family_name}")
    print("Initializing composite figure with Custom palette & perfect symmetry...")
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 10), dpi=150)

    # ========================================================================
    # PERFECTLY BALANCED LAYOUT COORDINATES
    # ========================================================================
    W = 0.24  
    H = 0.24  
    x_starts = [0.06, 0.41, 0.71]  
    y_starts = [0.72, 0.42, 0.12]  

    for row_idx, PC in enumerate(PCS):
        print(f"\nProcessing PC: {PC}...")

        if PC == 1:
            pred_csv = "/data/global_mapping/models_V100/drive_fina/PC1/XGB_shap/XGB_shap/block_1500km_buf_1000km/1b_Block_CV_Per_Point_Predictions.csv"
            bee_csv  = "/data/global_mapping/models_V100/drive_fina/PC1/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/10_SHAP_Beeswarm_Global_Top10.csv"
            dep_csv  = "/data/global_mapping/models_V100/drive_fina/PC1/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/4_SHAP_Dependence_Nonlinear_Top10.csv"
            importance_csv = "/data/global_mapping/models_V100/drive_fina/PC1/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/3_SHAP_Global_Summary_Importance.csv"
            numeric_csv = "/data/global_mapping/models_V100/drive_fina/PC1/XGB_shap/XGB_shap/Block_CV/Summary_Across_Buffers_numeric.csv"
            unit_str = " (mm)"
            
        elif PC == 2:
            pred_csv = "/data/global_mapping/models_V100/drive_fina/PC2/XGB_shap/XGB_shap/block_1500km_buf_1000km/1b_Block_CV_Per_Point_Predictions.csv"
            bee_csv  = "/data/global_mapping/models_V100/drive_fina/PC2/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/10_SHAP_Beeswarm_Global_Top10.csv"
            dep_csv  = "/data/global_mapping/models_V100/drive_fina/PC2/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/4_SHAP_Dependence_Nonlinear_Top10.csv"
            importance_csv = "/data/global_mapping/models_V100/drive_fina/PC2/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/3_SHAP_Global_Summary_Importance.csv"
            numeric_csv = "/data/global_mapping/models_V100/drive_fina/PC2/XGB_shap/XGB_shap/Block_CV/Summary_Across_Buffers_numeric.csv"
            unit_str = " (\u00B0C)"
            
        elif PC == 3:
            pred_csv = "/data/global_mapping/models_V100/drive_fina/PC3/XGB_shap/XGB_shap/block_1500km_buf_1000km/1b_Block_CV_Per_Point_Predictions.csv"
            bee_csv  = "/data/global_mapping/models_V100/drive_fina/PC3/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/10_SHAP_Beeswarm_Global_Top10.csv"
            dep_csv  = "/data/global_mapping/models_V100/drive_fina/PC3/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/4_SHAP_Dependence_Nonlinear_Top10.csv"
            importance_csv = "/data/global_mapping/models_V100/drive_fina/PC3/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/3_SHAP_Global_Summary_Importance.csv"
            numeric_csv = "/data/global_mapping/models_V100/drive_fina/PC3/XGB_shap/XGB_shap/Block_CV/Summary_Across_Buffers_numeric.csv"
            unit_str = " (J m$^{-2}$)"

        ax_perf = axes[row_idx, 0]
        ax_bee  = axes[row_idx, 1]
        ax_dep  = axes[row_idx, 2]

        ax_perf.set_position([x_starts[0], y_starts[row_idx], W, H])
        ax_bee.set_position([x_starts[1], y_starts[row_idx], W, H])
        ax_dep.set_position([x_starts[2], y_starts[row_idx], W, H])

        # -----------------------------------------------------------------
        # Data Extraction
        # -----------------------------------------------------------------
        df_bee = pd.read_csv(bee_csv)
        df_imp = pd.read_csv(importance_csv)
        
        contrib_map = df_imp.set_index("feature")["contribution_pct"].to_dict()
        all_ordered_feats = df_bee[["feature", "rank"]].drop_duplicates().sort_values("rank")["feature"].tolist()
        feat_order = all_ordered_feats[:6]
        
        if not feat_order:
            feat_order = all_ordered_feats[:1]
            
        top1_feat = feat_order[0]

        # -----------------------------------------------------------------
        # Col 1: Performance Scatter (Global R2)
        # -----------------------------------------------------------------
        df_pred = pd.read_csv(pred_csv)
        y_obs = df_pred["y_true"].values
        y_prd = df_pred["y_pred_cv"].values

        # Dynamic R2 extraction: 1000km baseline +/- SD across buffers
        global_r2_text = ""
        if os.path.exists(numeric_csv):
            try:
                df_num = pd.read_csv(numeric_csv)
                if "Global_R2" in df_num.columns and 1000 in df_num["Buffer_km"].values:
                    base_r2 = df_num.loc[df_num["Buffer_km"] == 1000, "Global_R2"].iloc[0]
                    sd_r2 = df_num["Global_R2"].std(ddof=1)
                    if pd.isna(sd_r2):
                        sd_r2 = 0.0
                    global_r2_text = f"$R^2$ = {base_r2:.2f} $\\pm$ {sd_r2:.2f}"
            except Exception:
                pass
        
        # Fallback if numeric csv is missing
        if not global_r2_text:
            actual_global_r2 = r2_score(y_obs, y_prd)
            global_r2_text = f"$R^2$ = {actual_global_r2:.2f}"
        
        z = get_fast_kde_density(y_obs, y_prd)
        sort_idx = z.argsort()
        y_obs_s, y_prd_s, z_s = y_obs[sort_idx], y_prd[sort_idx], z[sort_idx]

        ax_perf.scatter(y_obs_s, y_prd_s, c=z_s, cmap=GLOBAL_CMAP, s=1.5, alpha=0.6, edgecolors="none")

        min_val = min(y_obs.min(), y_prd.min())
        max_val = max(y_obs.max(), y_prd.max())
        ax_perf.plot([min_val, max_val], [min_val, max_val], 'k--', lw=1.2, zorder=3)

        ax_perf.set_xlabel("Observed Value", fontsize=12, fontweight="normal")
        ax_perf.set_ylabel("Predicted Value", fontsize=12, fontweight="normal")
        
        # Clean R2 Text: No bbox, normal font weight
        ax_perf.text(0.05, 0.95, global_r2_text, transform=ax_perf.transAxes, 
                     fontsize=12, fontweight="normal", va="top")
        
        polish_ax(ax_perf, add_grid=True)
        ax_perf.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_perf.yaxis.set_major_locator(MaxNLocator(nbins=5))

        # -----------------------------------------------------------------
        # Col 2: SHAP Beeswarm
        # -----------------------------------------------------------------
        new_features = [f"{f} ({contrib_map.get(f, 0.0):.1f}%)" for f in feat_order]

        expl = long_to_explanation(df_bee, feat_order)
        expl.feature_names = new_features

        plt.sca(ax_bee)
        shap.plots.beeswarm(
            expl, max_display=len(feat_order), show=False, color=GLOBAL_CMAP,
            s=8, alpha=0.8, color_bar=False, plot_size=None
        )
        ax_bee.set_xlabel("SHAP Value", fontsize=12, fontweight="normal")
        ax_bee.axvline(0.0, color="#888888", linewidth=0.8, zorder=0, linestyle="--")
        
        polish_ax(ax_bee)
        pull_in_yticklabels(ax_bee)
        ax_bee.xaxis.set_major_locator(MaxNLocator(nbins=5))

        # -----------------------------------------------------------------
        # Col 3: Dependence / Non-linear Response
        # -----------------------------------------------------------------
        df_dep = pd.read_csv(dep_csv)
        df_top1 = df_dep[df_dep["feature"] == top1_feat]
        x_dep = df_top1["feature_value"].values
        y_dep = df_top1["shap_value"].values

        # Eliminate abstract scientific notation for SSR
        if top1_feat == "SSR":
            x_dep = x_dep / 1e8
            xlabel_text = "SSR ($\\times 10^8$ J m$^{-2}$)"
        else:
            xlabel_text = top1_feat + (unit_str if unit_str.strip() not in top1_feat else "")

        z_dep = get_fast_kde_density(x_dep, y_dep)
        sort_idx_dep = z_dep.argsort()
        x_dep_s, y_dep_s, z_dep_s = x_dep[sort_idx_dep], y_dep[sort_idx_dep], z_dep[sort_idx_dep]

        ax_dep.scatter(x_dep_s, y_dep_s, c=z_dep_s, cmap=GLOBAL_CMAP, s=2.0, alpha=0.6, edgecolors="none")

        subset_idx = np.random.choice(len(x_dep), min(20000, len(x_dep)), replace=False)
        lowess = sm.nonparametric.lowess(y_dep[subset_idx], x_dep[subset_idx], frac=0.15)
        
        ax_dep.plot(lowess[:, 0], lowess[:, 1], color="white", linewidth=TREND_LW + 2.0,
                    solid_capstyle="round", zorder=4, alpha=0.85)
        ax_dep.plot(lowess[:, 0], lowess[:, 1], color=TREND_COLOR, linewidth=TREND_LW,
                    solid_capstyle="round", zorder=5)

        ax_dep.axhline(0.0, color="#888888", linewidth=0.8, zorder=0, linestyle="--")
        
        ax_dep.set_xlabel(xlabel_text, fontsize=12, fontweight="normal")
        ax_dep.set_ylabel("SHAP Value", fontsize=12, fontweight="normal")

        polish_ax(ax_dep, add_grid=True)
        ax_dep.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax_dep.yaxis.set_major_locator(MaxNLocator(nbins=5))

    # ========================================================================
    # GLOBAL FORMATTING & ANNOTATIONS (a, b, c...)
    # ========================================================================
    print("Applying precise panel letters...")
    panel_letters = list(string.ascii_lowercase)
    for i, ax in enumerate(axes.flatten()):
        col_idx = i % 3
        x_offset = -0.32 if col_idx == 1 else -0.18
        ax.text(x_offset, 1.05, panel_letters[i], transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="bottom", ha="right")

    # ========================================================================
    # FIXED BOTTOM COLORBAR
    # ========================================================================
    fig.canvas.draw()
    cax_w = 0.30 
    cax_h = 0.012
    cax_x = 0.5 - (cax_w / 2)
    cax_y = 0.040 
    
    cax = fig.add_axes([cax_x, cax_y, cax_w, cax_h])
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=GLOBAL_CMAP, norm=Normalize(0, 1)),
                      cax=cax, orientation="horizontal", ticks=[0, 1])
    
    cb.ax.set_xticklabels(["Low", "High"], fontsize=12, fontweight="normal")
    cb.outline.set_linewidth(0.8)
    fig.text(0.5, cax_y - 0.035, "", 
             ha="center", fontsize=13, fontweight="normal", color="#222222")

    print(f"\nSaving composite TIF figure to:\n{OUTPUT_TIF}")
    fig.savefig(OUTPUT_TIF, dpi=DPI, bbox_inches="tight", facecolor="white", pil_kwargs={"compression": "tiff_lzw"})

    print(f"Saving vector PDF figure to:\n{OUTPUT_PDF}")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", facecolor="white")

    plt.close()
    print("All done!")

if __name__ == "__main__":
    main()