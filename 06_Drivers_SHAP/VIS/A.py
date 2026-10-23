#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Directional Dominant Driver - High-End Academic Poster Layout
-------------------------------------------------------------------------------
  - Displays all 10 environmental variables.
  - Elegant vertical legend placed closely on the right side.
  - Premium cartographic styling (subtle ocean tint, crisp coastlines, solid thin graticules).
  - Removed longitude labels and enlarged latitude labels for a cleaner look.
  - Adjusted panel tags (A, B, C) to align perfectly with the map's top edge.
  - VPD strictly assigned Purple, PH strictly assigned Pink.
  - No Chinese characters in the script.
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
# 0. Config & Formatting
# ============================================================
OUTDIR = "/data/global_mapping/models_V100/drive/final_shap_visuals"
os.makedirs(OUTDIR, exist_ok=True)
INPUTS = [
    ("/data/global_mapping/models_V100/drive/PC1/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC1"),
    ("/data/global_mapping/models_V100/drive/PC2/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC2"),
    ("/data/global_mapping/models_V100/drive/PC3/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC3"),
]

COL_LON, COL_LAT, COL_DRIVER = "lon", "lat", "Dominant_Driver"

mpl.rcParams.update({
    "font.family":       "Arial",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
    "svg.fonttype":      "none",
})

# ============================================================
# 1. Global 10-Color Palette Mapping (Premium High-Contrast)
# ============================================================
ALL_DRIVERS = ["MAT", "MAP", "VPD", "SSR", "SAND", "CLAY", "PH", "CEC", "ROUGH", "HMI"]

GLOBAL_COLOR_MAP = {
    "MAT":   "#007F5F",  # Vermilion/Rust - Deep, sophisticated heat
    "MAP":   "#2196F3",  # Deep Ocean Blue - Classic scientific water
    "VPD":   "#8A2BE2",  # Vivid Purple - Strict requirement
    "SSR":   "#E69F00",  # Golden Amber - Soft, rich solar radiation
    "PH":    "#E377C2",  # Carnation Pink - Strict requirement, pops beautifully
    "CEC":   "#D55E00",  # Emerald Green - Natural and distinct
    "CLAY":  "#8C564B",  # Clay Brown - Grounded earth tone
    "SAND":  "#BCBD22",  # Olive Sand - Distinct from SSR gold
    "ROUGH": "#1A237E",  # Dark Navy - Deep shadow for topography
    "HMI":   "#E74C3C",  # Slate Grey - Clean urban/disturbance neutral
}

# ============================================================
# 2. Read data & Calculate Frequencies
# ============================================================
panels = []

for path, label in INPUTS:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    
    d = pd.read_csv(path)[[COL_LON, COL_LAT, COL_DRIVER]].dropna(subset=[COL_DRIVER]).copy()
    d = d[d[COL_DRIVER].isin(ALL_DRIVERS)]
    panels.append((label, d))

# Export frequency statistics table
rows = []
for label, d in panels:
    f = d[COL_DRIVER].value_counts()
    for rank, (drv, cnt) in enumerate(f.items(), 1):
        rows.append({
            "panel": label, 
            "rank": rank, 
            "driver": drv,
            "pixel_count": int(cnt)
        })
pd.DataFrame(rows).to_csv(os.path.join(OUTDIR, "driver_frequency_per_panel.csv"), index=False)

# ============================================================
# 3. Plot: 3 vertical panels + Robinson + Right Side Legend
# ============================================================
proj = ccrs.Robinson(central_longitude=0)

# Adjusted figure size to perfectly balance the 3 stacked Robinson maps and the side legend
fig = plt.figure(figsize=(8.5, 10.5)) 

# GridSpec tuned for compact layout
gs = fig.add_gridspec(nrows=3, ncols=1,
                      left=0.02, right=0.80, top=0.97, bottom=0.03, hspace=0.05)
panel_tags = ["a", "b", "c"]

def draw_panel(ax, d, tag):
    ax.set_global()

    # Premium Cartographic Styling"#FFFFFF"
    ax.add_feature(cfeature.OCEAN, facecolor="#F4F7F9", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#FDFDFD", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#34495E", zorder=2)
    
    # Ultra-thin solid graticules look more professional than thick dashed lines
    gl = ax.gridlines(draw_labels=True, linewidth=0.1, color="#AAB7B8", alpha=0.8,
                      linestyle="-", xlocs=np.arange(-180, 181, 60),
                      ylocs=np.arange(-90, 91, 30), zorder=3)
    
    # Hide top, right, and bottom (longitude) labels completely.
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    
    # Enlarge latitude labels for better readability
    gl.ylabel_style = {'size': 12, 'color': '#7F8C8D'}
    
    # Map boundary
    ax.spines["geo"].set_edgecolor("#555555")
    ax.spines["geo"].set_linewidth(0.7)

    # Plot data - Decreased scatter size (s=0.15) for finer resolution
    for drv in ALL_DRIVERS:
        c = d[d[COL_DRIVER] == drv]
        if len(c) == 0:
            continue
        col = GLOBAL_COLOR_MAP[drv]
        ax.scatter(c[COL_LON].values, c[COL_LAT].values, s=0.15, marker="s",
                   linewidths=0, c=col, transform=ccrs.PlateCarree(),
                   rasterized=True, zorder=5)

    # Bold, clean panel tags - lowered y-coordinate to 1.0 and anchored to "top"
    ax.text(-0.02, 1.0, tag, transform=ax.transAxes, fontsize=20,
            fontweight="bold", color="#2C3E50", va="top", ha="right")

for i, (label, d) in enumerate(panels):
    ax = fig.add_subplot(gs[i, 0], projection=proj)
    draw_panel(ax, d, panel_tags[i])

# ------------------------------------------------------------
# Elegant Vertical Legend (Right Side - Pulled In Closer)
# ------------------------------------------------------------
handles = [Patch(facecolor=GLOBAL_COLOR_MAP[drv], edgecolor="#D5D8DC", linewidth=0.6, label=drv) for drv in ALL_DRIVERS]
                      
# Positioned cleanly in the vertical white space on the right
leg = fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.82, 0.5),
                 ncol=1, frameon=False, fontsize=11, 
                 title="Drivers",
                 title_fontsize=13, handlelength=1.6, handleheight=1.6,
                 handletextpad=1.0, borderaxespad=0.0, labelspacing=1.3)

leg.get_title().set_position((0, 15)) 

# ============================================================
# 4. Export
# ============================================================
out = os.path.join(OUTDIR, "7_SHAP_Directional_Dominant_3panel_Robinson_Nature")
fig.savefig(out + ".pdf",  bbox_inches="tight", dpi=600)
fig.savefig(out + ".png",  bbox_inches="tight", dpi=600, facecolor="white")
fig.savefig(out + ".tiff", bbox_inches="tight", dpi=600, facecolor="white", pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig);