#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Directional Dominant Driver - global 3-panel Robinson map (Nature style)
-------------------------------------------------------------------------------
  - Strictly uses exactly 7 colors for the entire figure (6 top drivers + 1 Other).
  - Colors are assigned dynamically PER PANEL based on rank (1st to 6th + Other).
  - Uses the official Nature Publishing Group (NPG) color palette.
  - Antarctica included (global extent).
  - Latitude and Longitude labels added on all edges (draw_labels=True).
  - Per-panel top-6 + per-panel right-side legend.
  - NEW: Retains exactly the top 6 variables, clustering the rest into 'Other'.
-------------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================================================
# 0. Config
# ============================================================
OUTDIR = "/data/global_mapping/models_V100/drive_fina/final_shap_visuals"
os.makedirs(OUTDIR, exist_ok=True)
INPUTS = [
    ("/data/global_mapping/models_V100/drive_fina/PC1/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC1"),
    ("/data/global_mapping/models_V100/drive_fina/PC2/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC2"),
    ("/data/global_mapping/models_V100/drive_fina/PC3/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC3"),
]

COL_LON, COL_LAT, COL_DRIVER = "lon", "lat", "Dominant_Driver"
TOP_N = 6    

mpl.rcParams.update({
    "font.family":       "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.linewidth":    0.8,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
})

# ============================================================
# 2. Strict 7-Color Palette (Nature Publishing Group Style)
# ============================================================
PALETTE = [
    "#00A087", # Rank 1: NPG Teal/Green
    "#4DBBD5", # Rank 2: NPG Cyan
    "#E64B35", # Rank 3: NPG Red
    "#3C5488", # Rank 4: NPG Dark Blue
    "#FFC000", # Rank 5: NPG Slate/Yellow
    "#F39B7F", # Rank 6: NPG Peach/Orange
]
COLOR_OTHER = "#CCCCCC" # Rank 7: NPG Light Gray for 'Other'

# ============================================================
# 1. Read data & Identify Top N per panel (Top 6 overall)
# ============================================================
panels = []   
panel_tops = {}

SAFE_TOP_N = min(TOP_N, len(PALETTE))

for path, label in INPUTS:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
        
    d = pd.read_csv(path)[[COL_LON, COL_LAT, COL_DRIVER]].dropna(subset=[COL_DRIVER]).copy()
    panels.append((label, d))
    
    # Load corresponding Importance CSV to ensure consistency with other figures
    imp_path = path.replace("7_SHAP_Directional_Dominant_Map.csv", "3_SHAP_Global_Summary_Importance.csv")
    if not os.path.exists(imp_path):
        raise FileNotFoundError(f"Importance file not found: {imp_path}")
        
    df_imp = pd.read_csv(imp_path)
    
    # Check column name (handles both variations)
    pct_col = "contribution_pct" if "contribution_pct" in df_imp.columns else "importance_pct"
    
    # Identify the strict top 6 drivers globally
    valid_drivers = df_imp.sort_values(by=pct_col, ascending=False)["feature"].tolist()[:SAFE_TOP_N]
    
    # Identify top drivers for this panel based on pixel count
    freq_drivers = d[COL_DRIVER].value_counts().index.tolist()
    
    # Keep only those frequent drivers that are within the globally valid top 6
    filtered_tops = [drv for drv in freq_drivers if drv in valid_drivers]
    
    # Restrict to maximum SAFE_TOP_N
    panel_tops[label] = filtered_tops[:SAFE_TOP_N]

# Export frequencies
rows = []
for label, d in panels:
    f = d[COL_DRIVER].value_counts()
    for rank, (drv, cnt) in enumerate(f.items(), 1):
        rows.append({"panel": label, "rank": rank, "driver": drv,
                     "pixel_count": int(cnt), "in_top_set": drv in panel_tops[label]})
pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "driver_frequency_per_panel.csv"), index=False)


# ============================================================
# 3. Plot: 3 vertical panels + Robinson + custom extent
# ============================================================
proj = ccrs.Robinson(central_longitude=0)

fig = plt.figure(figsize=(8.4, 9.0)) 
gs = fig.add_gridspec(nrows=3, ncols=1,
                      left=0.06, right=0.86, top=0.985, bottom=0.015, hspace=0.20)
panel_tags = ["a", "b", "c"]

def draw_panel(ax, label, d, tag):
    tops = panel_tops[label]
    
    local_color_map = {drv: PALETTE[i] for i, drv in enumerate(tops)}

    def map_driver(x):
        return x if x in tops else "Other"

    # Set global extent to include Antarctica
    ax.set_global()

    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor="#bbbbbb", zorder=2)
    
    # Gridlines and perimeter labels
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="#aaaaaa", alpha=0.5,
                      linestyle="--", xlocs=np.arange(-180, 181, 60),
                      ylocs=np.arange(-90, 91, 30), zorder=3)
    
    gl.xlabel_style = {'size': 8, 'color': '#555555'}
    gl.ylabel_style = {'size': 8, 'color': '#555555'}
    
    ax.spines["geo"].set_edgecolor("black")
    ax.spines["geo"].set_linewidth(0.8)

    dd = d.copy()
    dd["cat"] = dd[COL_DRIVER].apply(map_driver)
    
    # Plot 'Other' first (zorder=4), then the dominant drivers (zorder=5)
    for cat in ["Other"] + tops:
        c = dd[dd["cat"] == cat]
        if len(c) == 0:
            continue
        col = COLOR_OTHER if cat == "Other" else local_color_map[cat]
        ax.scatter(c[COL_LON].values, c[COL_LAT].values, s=0.5, marker="s",
                   linewidths=0, c=col, transform=ccrs.PlateCarree(),
                   rasterized=True, zorder=5 if cat != "Other" else 4)

    # Panel tag (a, b, c)
    ax.text(-0.06, 1.05, tag, transform=ax.transAxes, fontsize=18,
            fontweight="bold", va="bottom", ha="right")

    # Local Panel Legend
    cats = tops + ["Other"]
    handles = [Patch(facecolor=(COLOR_OTHER if c == "Other" else local_color_map[c]),
                     edgecolor="none", label=c) for c in cats]
                     
    leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.05, 0.5),
                    frameon=False, fontsize=9, title="Dominant Driver",
                    title_fontsize=10, handlelength=1.2, handleheight=1.2,
                    handletextpad=0.6, borderaxespad=0.0)
    leg.get_title().set_fontweight("normal")

for i, (label, d) in enumerate(panels):
    ax = fig.add_subplot(gs[i, 0], projection=proj)
    draw_panel(ax, label, d, panel_tags[i])

# ============================================================
# 4. Export
# ============================================================
out = os.path.join(OUTDIR, "7_SHAP_Directional_Dominant_3panel_Robinson_Nature")
fig.savefig(out + ".pdf",  bbox_inches="tight", dpi=600)
fig.savefig(out + ".png",  bbox_inches="tight", dpi=600)
fig.savefig(out + ".tiff", bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)