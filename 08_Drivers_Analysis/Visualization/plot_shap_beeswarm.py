#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 SHAP-native beeswarm visualisation (publication-quality)
--------------------------------------------------------------------------------
 Processes and exports a combined Biome-scale SHAP beeswarm plot for 3 buffers.
 
 Features:
   - Font unified to Arial (with custom font path loader).
   - Annotations (a, b, c) placed strictly in the top-left margin.
   - Distinct colormaps for buffers a, b, and c.
   - Three independent colorbars rendered at the bottom.
   - Enforced strict black font color.
   - NEW: Retains exactly the top 6 variables based on their rank.
================================================================================
"""

from __future__ import annotations
from pathlib import Path
import math
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as fm
import shap

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIG ！ paths
# ============================================================================
BUFFER_FILES = {
    "a": Path("/data/global_mapping/models_V100/drive_fina/PC1/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/11_SHAP_Beeswarm_Biome_Top10.csv"),
    "b": Path("/data/global_mapping/models_V100/drive_fina/PC2/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/11_SHAP_Beeswarm_Biome_Top10.csv"),
    "c": Path("/data/global_mapping/models_V100/drive_fina/PC3/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/11_SHAP_Beeswarm_Biome_Top10.csv")
}

OUTPUT_DIR = Path(
    "/data/global_mapping/models_V100/drive_fina/final_shap_visuals/"
)

# ============================================================================
# CONFIG ！ custom font path
# ============================================================================
CUSTOM_FONT_PATH = "" 

# ============================================================================
# CONFIG ！ figure style & distinct colormaps
# ============================================================================
DPI                = 600
BIOME_MIN_SAMPLES  =1200      
N_COLS_BIOME       = 3            
DOT_SIZE_BIOME     = 10

# Assign distinct, colorblind-friendly, publication-ready colormaps
CMAPS = {
    "a": shap.plots.colors.red_blue,  # Native SHAP (Blue to Red-Pink)
    "b": plt.cm.viridis,              # Purple to Yellow
    "c": plt.cm.plasma                # Purple to Orange
}

# ============================================================================
# Font Registration and Base Styling
# ============================================================================
if CUSTOM_FONT_PATH and Path(CUSTOM_FONT_PATH).exists():
    fm.fontManager.addfont(CUSTOM_FONT_PATH)
    custom_font_prop = fm.FontProperties(fname=CUSTOM_FONT_PATH)
    font_family_name = custom_font_prop.get_name()
else:
    font_family_name = "Arial"

mpl.rcParams.update({
    "font.family":       font_family_name,
    "font.size":         12,
    "text.color":        "black",
    "axes.labelcolor":   "black",
    "xtick.color":       "black",
    "ytick.color":       "black",
    "axes.linewidth":    1.0,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size":  5,
    "ytick.major.size":  5,
    "pdf.fonttype":      42, 
    "ps.fonttype":       42,
    "savefig.facecolor": "white",
})


# ============================================================================
# Long-format CSV  ->  shap.Explanation
# ============================================================================
def long_to_explanation(df_long: pd.DataFrame,
                        feature_order: list[str]) -> shap.Explanation:
    if "row_idx" not in df_long.columns:
        df_long = df_long.copy()
        df_long["row_idx"] = (df_long.groupby("feature").cumcount())

    sv_wide = (df_long
               .pivot_table(index="row_idx", columns="feature",
                            values="shap_value", aggfunc="first")
               .reindex(columns=feature_order))
    fv_wide = (df_long
               .pivot_table(index="row_idx", columns="feature",
                            values="feature_value", aggfunc="first")
               .reindex(columns=feature_order))

    valid = ~(sv_wide.isna().any(axis=1) | fv_wide.isna().any(axis=1))
    sv_wide = sv_wide.loc[valid]
    fv_wide = fv_wide.loc[valid]

    return shap.Explanation(
        values        = sv_wide.to_numpy(dtype=np.float64),
        data          = fv_wide.to_numpy(dtype=np.float64),
        feature_names = list(sv_wide.columns),
        base_values   = np.zeros(len(sv_wide), dtype=np.float64),
    )

def get_feature_order_biome(df_one_biome: pd.DataFrame) -> list[str]:
    return (df_one_biome[["feature", "rank_in_biome"]].drop_duplicates()
            .sort_values("rank_in_biome")["feature"].tolist())


# ============================================================================
# Cosmetic polishing applied AFTER shap.plots.beeswarm()
# ============================================================================
def polish_beeswarm_axes(ax, *, xlabel: str = "SHAP value",
                         title: str | None = None):
    ax.axvline(0.0, color="#666666", linewidth=1.0, zorder=0, linestyle="--")
    
    ax.tick_params(axis="x", labelsize=11, colors="black")
    ax.tick_params(axis="y", labelsize=12, colors="black")
    ax.set_xlabel(xlabel, fontsize=12, labelpad=8, color="black")
    
    if title is not None:
        ax.set_title(title, fontsize=14, fontweight="normal",
                     loc="left", pad=14, color="black")
                     
    for s in ("top", "right"):
        if s in ax.spines:
            ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        if s in ax.spines:
            ax.spines[s].set_linewidth(1.0)
            ax.spines[s].set_color("black")


def save_fig(fig, stem: str):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tif = OUTPUT_DIR / f"{stem}.tif"
    pdf = OUTPUT_DIR / f"{stem}.png"
    
    fig.savefig(tif, dpi=DPI, bbox_inches="tight",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(pdf, bbox_inches="tight")
    print(f"  [OK] {tif}")
    print(f"  [OK] {pdf}")


# ============================================================================
# FIGURE 2 ！ Combined Biome beeswarms (grid)
# ============================================================================
def make_combined_biome_figure(dfs: dict[str, dict[str, pd.DataFrame]]):
    print("\n[Processing] Combined Per-biome beeswarms with distinct colormaps")
    
    buffer_info = []
    total_rows = 0
    
    for label, payload in dfs.items():
        df = payload["bee"]
        df_imp = payload["imp"]
        
        biome_sizes = (df.drop_duplicates(["BIOME_NAME"])
                       .set_index("BIOME_NAME")["n_samples_in_biome"])
        keep_biomes = biome_sizes[biome_sizes >= BIOME_MIN_SAMPLES].sort_values(
            ascending=False).index.tolist()
        
        rows_needed = math.ceil(len(keep_biomes) / N_COLS_BIOME)
        buffer_info.append({
            "label": label,
            "biomes": keep_biomes,
            "df": df,
            "df_imp": df_imp,
            "rows": rows_needed,
            "cmap": CMAPS[label]
        })
        total_rows += rows_needed

    if total_rows == 0:
        print("  no biomes qualify; nothing to plot.")
        return

    # Panel dimensions
    panel_w, panel_h = 3.2, 2.5
    fig_w = panel_w * N_COLS_BIOME
    fig_h = panel_h * total_rows

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150)

    current_row_offset = 0
    
    for buf in buffer_info:
        label = buf["label"]
        keep_biomes = buf["biomes"]
        df_biome = buf["df"]
        df_imp = buf["df_imp"]
        n_rows_for_buf = buf["rows"]
        current_cmap = buf["cmap"]

        for i, b in enumerate(keep_biomes):
            sub = df_biome[df_biome["BIOME_NAME"] == b].copy()
            sub_imp = df_imp[df_imp["BIOME_NAME"] == b]
            
            # Map importance percentages for the current biome
            contrib_map = sub_imp.set_index("feature")["importance_pct"].to_dict()
            
            # Ensure proper row index exists before grouping
            sub["row_idx"] = sub.groupby("feature").cumcount()
            all_ordered_feats = get_feature_order_biome(sub)
            
            # Filter logic: Retain exactly the top 6 features based on rank
            keep_feats = all_ordered_feats[:6]
            
            # Fallback in case no feature is found
            if not keep_feats:
                keep_feats = all_ordered_feats[:1]

            # Drop everything that didn't make the cut
            sub = sub[sub["feature"].isin(keep_feats)].copy()
            feat_order_b = keep_feats

            expl_b = long_to_explanation(sub, feat_order_b)
            
            new_feature_names = []
            for feat in feat_order_b:
                val = contrib_map.get(feat, 0.0)
                new_feature_names.append(f"{feat} ({val:.1f}%)")
            
            expl_b.feature_names = new_feature_names

            # Actual grid position
            local_row_idx = i // N_COLS_BIOME
            col_idx = i % N_COLS_BIOME
            
            global_row_idx = current_row_offset + local_row_idx
            ax_idx = global_row_idx * N_COLS_BIOME + col_idx + 1
            
            ax = fig.add_subplot(total_rows, N_COLS_BIOME, ax_idx)
            plt.sca(ax)

            shap.plots.beeswarm(
                expl_b,
                max_display = len(feat_order_b),
                show        = False,
                color       = current_cmap,
                s           = DOT_SIZE_BIOME,
                color_bar   = False,
                plot_size   = None,
            )
            
            polish_beeswarm_axes(
                ax,
                xlabel="SHAP value",
                title=f"{b}", 
            )
            
            # Add annotation (a, b, c) specifically to the top-left margin
            if i == 0:
                ax.text(-0.30, 1.25, f"{label}", transform=ax.transAxes,
                        fontsize=26, fontweight="bold", va="bottom", ha="right", color="black")

        current_row_offset += n_rows_for_buf

    # ---- Layout formatting ------------------------------------------------
    # Increased bottom margin and hspace to fit 3 distinct colorbars
    fig.subplots_adjust(left=0.08, right=0.96, top=0.96,
                        bottom=0.10, hspace=0.75, wspace=0.90)
    
    # ---- Render 3 distinct colorbars side-by-side -------------------------
    y_cbar = 0.03
    h_cbar = 0.012
    w_cbar = 0.20
    spacing = 0.06
    
    # Calculate starting X to ensure all 3 bars are centered collectively
    total_cbar_width = (3 * w_cbar) + (2 * spacing)
    start_x = (1.0 - total_cbar_width) / 2.0

    for idx, (lbl, colormap) in enumerate(CMAPS.items()):
        cax_x = start_x + idx * (w_cbar + spacing)
        cax = fig.add_axes([cax_x, y_cbar, w_cbar, h_cbar])
        
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=mpl.colors.Normalize(0, 1))
        sm.set_array([])
        
        cbar = fig.colorbar(sm, cax=cax, orientation="horizontal", ticks=[0, 1])
        cbar.ax.set_xticklabels(["Low", "High"], fontsize=11, color="black")
        cbar.outline.set_linewidth(1.0)
        cbar.outline.set_edgecolor("black")
        pc_labels = {"a": "PC1", "b": "PC2", "c": "PC3"}
        cbar.set_label(pc_labels[lbl], fontsize=12, labelpad=6, color="black")

    save_fig(fig, "Fig_Combined_Buffers_SHAP_Beeswarm_Colored")
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
def main():
    print(f"SHAP version:        {shap.__version__}")
    print(f"matplotlib version: {mpl.__version__}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dfs = {}
    for label, path in BUFFER_FILES.items():
        imp_path = path.parent / "8_SHAP_Biome_Importance_Pct.csv"
        print(f"Reading {label}: {path.name} and {imp_path.name}")
        
        dfs[label] = {
            "bee": pd.read_csv(path),
            "imp": pd.read_csv(imp_path)
        }
        print(f"  shape: {dfs[label]['bee'].shape}")

    make_combined_biome_figure(dfs)

    print("\nDone.")


if __name__ == "__main__":
    main()