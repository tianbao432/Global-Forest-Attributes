#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================================================
# 0. Configuration
# ============================================================
OUTDIR = "/data/global_mapping/models_V100/drive_fina/final_shap_visuals"
os.makedirs(OUTDIR, exist_ok=True)

INPUTS = [
    ("/data/global_mapping/models_V100/drive_fina/PC1/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC1"),
    ("/data/global_mapping/models_V100/drive_fina/PC2/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC2"),
    ("/data/global_mapping/models_V100/drive_fina/PC3/XGB_shap/XGB_shap/block_1500km_buf_1000km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC3"),
]

# Manually specified target drivers for each panel
TARGET_DRIVERS = {
    "PC1": "PCQ",
    "PC2": "MTWeQ",
    "PC3": "SSR",
}

COL_LON    = "lon"
COL_LAT    = "lat"
COL_DRIVER = "Dominant_Driver"
COL_SHAP   = "Dominant_SHAP"

PIXEL_SIZE = 0.5

# High-quality publication diverging colormap (Smooth ColorBrewer RdBu)
CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "nature_diverging",
    [
        "#001146", # Extremely dark blue
        "#003CB3", # Deep blue
        "#4A90E2", # Medium vivid blue
        "#B3D4FF", # Light blue
        "#F6F6F6", # Neutral near-white center
        "#FFC2C2", # Light red/pink
        "#E24A4A", # Medium vivid red
        "#B30000", # Deep red
        "#460000"  # Extremely dark red
    ],
    N=256
)
# ============================================================
# 1. Font / style
# ============================================================
mpl.rcParams.update({
    "font.family":       "Arial",
    "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":         9,
    "axes.linewidth":    0.8,
    "pdf.fonttype":      42,
    "ps.fonttype":       42,
})

# ============================================================
# 2. Read data
# ============================================================
panels = []
for path, label in INPUTS:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")
    d = pd.read_csv(path)
    for col in [COL_LON, COL_LAT, COL_DRIVER, COL_SHAP]:
        if col not in d.columns:
            raise KeyError(f"Column '{col}' missing in {path}")
    d = d[[COL_LON, COL_LAT, COL_DRIVER, COL_SHAP]].dropna(
        subset=[COL_DRIVER, COL_SHAP]
    ).copy()
    panels.append((label, d))

# ============================================================
# 3. Plot setup
# ============================================================
proj = ccrs.Robinson(central_longitude=0)
fig = plt.figure(figsize=(8.5, 9.0))
# Adjusted right margin to provide more breathing room for colorbars
gs = fig.add_gridspec(
    nrows=3, ncols=1,
    left=0.06, right=0.84, top=0.96, bottom=0.04, hspace=0.25
)
panel_tags = ["a", "b", "c"]

def draw_panel(ax, label, d, tag):
    """Plot points for the specified target driver, colored by SHAP value."""
    target_driver = TARGET_DRIVERS[label]

    # Filter data
    sub = d[d[COL_DRIVER] == target_driver].copy()
    if len(sub) == 0:
        ax.text(0.5, 0.5, f"No data for {target_driver}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=10, color="gray")
        ax.set_global()
        ax.spines["geo"].set_visible(True)
        return

    shap_vals = sub[COL_SHAP].values.astype(float)
    abs_max = np.max(np.abs(shap_vals)) if len(shap_vals) > 0 else 0.01
    if abs_max == 0:
        abs_max = 0.01
    norm = Normalize(vmin=-abs_max, vmax=abs_max)

    # Basemap
    ax.set_global()
    ax.add_feature(cfeature.OCEAN, facecolor="white", edgecolor="none", zorder=0)
    ax.add_feature(cfeature.LAND, facecolor="#f0f0f0", edgecolor="none", zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.3, edgecolor="#aaaaaa", zorder=2)

    gl = ax.gridlines(
        draw_labels=True, linewidth=0.4, color="#cccccc", alpha=0.6,
        linestyle="--", xlocs=np.arange(-180, 181, 60),
        ylocs=np.arange(-90, 91, 30), zorder=3
    )
    
    # Disable right-sided labels to completely prevent overlap with the colorbar
    gl.right_labels = False 
    
    gl.xlabel_style = {'size': 7, 'color': '#555555'}
    gl.ylabel_style = {'size': 7, 'color': '#555555'}
    ax.spines["geo"].set_edgecolor("#333333")
    ax.spines["geo"].set_linewidth(0.8)

    # Plot points (Sort so most extreme values are plotted on top)
    sort_idx = np.argsort(np.abs(shap_vals))
    lons = sub[COL_LON].values[sort_idx]
    lats = sub[COL_LAT].values[sort_idx]
    sv   = shap_vals[sort_idx]

    ax.scatter(
        lons, lats,
        s=PIXEL_SIZE, marker="s", linewidths=0,
        c=sv, cmap=CMAP, norm=norm,
        transform=ccrs.PlateCarree(),
        rasterized=True, zorder=5
    )

    # Panel letter moved completely outside the map for clean formatting
    ax.text(-0.06, 1.05, tag, transform=ax.transAxes, fontsize=16,
            fontweight="bold", va="bottom", ha="right")

    # Colorbar pushed slightly further right and made more elegant
    cax = ax.inset_axes([1.06, 0.1, 0.025, 0.8])
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                      orientation="vertical", extend="both")
    cb.ax.tick_params(labelsize=7, width=0.6, length=3)
    cb.outline.set_linewidth(0.6)
    cb.outline.set_edgecolor("#333333")
    
    # Integrate variable name into colorbar label
    cb.set_label(f"SHAP value\n({target_driver})", fontsize=9, labelpad=8,
                 linespacing=1.3)

for i, (label, d) in enumerate(panels):
    ax = fig.add_subplot(gs[i, 0], projection=proj)
    draw_panel(ax, label, d, panel_tags[i])

# ============================================================
# 4. Export
# ============================================================
out_base = os.path.join(OUTDIR, "7_SHAP_Dominant_T1")
fig.savefig(out_base + ".pdf",  bbox_inches="tight", dpi=600)
fig.savefig(out_base + ".png",  bbox_inches="tight", dpi=600)
fig.savefig(out_base + ".tiff", bbox_inches="tight", dpi=600,
            pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)