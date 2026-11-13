#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Dominant Driver - Combined 6-panel Robinson map
Optimized for Nature Journal Publication Standards (Double-column width: ~183mm)
Ultra-tight layout with high data-ink ratio.
-------------------------------------------------------------------------------
"""

import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

# ============================================================
# 0. Configuration & Nature-standard Parameters
# ============================================================
OUTDIR = "/data/global_mapping/models_V100/drive_fina/final_shap_visuals"
os.makedirs(OUTDIR, exist_ok=True)

INPUTS = [
    ("/data/global_mapping/models_V100/drive_fina/PC1/XGB_shap/XGB_shap/block_1500km_buf_800km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC1"),
    ("/data/global_mapping/models_V100/drive_fina/PC2/XGB_shap/XGB_shap/block_1500km_buf_800km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC2"),
    ("/data/global_mapping/models_V100/drive_fina/PC3/XGB_shap/XGB_shap/block_1500km_buf_800km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC3"),
]

COL_LON, COL_LAT = "lon", "lat"
COL_DRIVER, COL_SHAP = "Dominant_Driver", "Dominant_SHAP"
TOP_N = 6
PIXEL_SIZE = 0.1 # Ensures dense points do not overly blend

TARGET_DRIVERS = {"PC1": "PCQ", "PC2": "MTWeQ", "PC3": "SSR"}

# Nature Publishing Group (NPG) Palette
PALETTE = [
    "#00A087",  # Rank 1: Teal
    "#4DBBD5",  # Rank 2: Cyan
    "#E64B35",  # Rank 3: Red
    "#3C5488",  # Rank 4: Dark Blue
    "#FFC000",  # Rank 5: Slate Yellow
    "#F39B7F",  # Rank 6: Peach
]
COLOR_OTHER = "#E0E0E0"  # Extremely soft gray to push 'Other' to background

# Ultra-smooth Nature-style Diverging Colormap (Deep Blue -> White -> Deep Red)
CMAP = mpl.colors.LinearSegmentedColormap.from_list(
    "nature_diverging",
    [ "#001146", "#003CB3", "#4A90E2", "#B3D4FF",  
     "#ffffff", 
     "#FFC2C2", "#E24A4A", "#B30000", "#460000"],
    N=512
)

# ============================================================
# 1. Typography & Global Style (Nature Standards)
# ============================================================
FIG_WIDTH = 7.2  # Nature standard double column width (183 mm)
FIG_HEIGHT = 5.0 # Slightly tightened vertical height

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,                # Standard text size 5-7 pt
    "axes.titlesize": 7,
    "axes.labelsize": 7,
    "xtick.labelsize": 6,
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.title_fontsize": 7,
    "axes.linewidth": 0.5,         # Ultra-thin and clean axes
    "grid.linewidth": 0.2,
    "pdf.fonttype": 42,            # TrueType text for post-editing in Illustrator
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300
})

# High-resolution (50m) cartopy features for publication quality
LAND_50M = cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor='#F8F8F8', edgecolor='none')
OCEAN_50M = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor='#FFFFFF', edgecolor='none')
COAST_50M = cfeature.NaturalEarthFeature('physical', 'coastline', '50m', facecolor='none', edgecolor='#404040')

# ============================================================
# 2. Data Loading & Preprocessing
# ============================================================
panels = []
panel_tops = {}
SAFE_TOP_N = min(TOP_N, len(PALETTE))

for path, label in INPUTS:
    if not os.path.exists(path):
        print(f"[Warning] File not found: {path} - Skipping {label}.")
        continue
        
    d = pd.read_csv(path, usecols=[COL_LON, COL_LAT, COL_DRIVER, COL_SHAP]).dropna()
    panels.append((label, d))
    
    imp_path = path.replace("7_SHAP_Directional_Dominant_Map.csv", "3_SHAP_Global_Summary_Importance.csv")
    if os.path.exists(imp_path):
        df_imp = pd.read_csv(imp_path)
        pct_col = "contribution_pct" if "contribution_pct" in df_imp.columns else "importance_pct"
        valid_drivers = df_imp.sort_values(by=pct_col, ascending=False)["feature"].tolist()[:SAFE_TOP_N]
        
        freq_drivers = d[COL_DRIVER].value_counts().index.tolist()
        filtered_tops = [drv for drv in freq_drivers if drv in valid_drivers]
        panel_tops[label] = filtered_tops[:SAFE_TOP_N]
    else:
        # Fallback if importance file is missing
        panel_tops[label] = d[COL_DRIVER].value_counts().index.tolist()[:SAFE_TOP_N]

# ============================================================
# 3. Core Plotting Functions
# ============================================================
def add_map_features(ax, is_continuous=False):
    """Uniformly apply publication-grade map styling."""
    ax.set_global()
    ax.add_feature(OCEAN_50M, zorder=0)
    
    # Continuous panels use a slightly contrasting land to highlight bright dots
    land_color = '#EFEFEF' if is_continuous else '#F9F9F9'
    ax.add_feature(cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor=land_color), zorder=1)
    
    # Ultra-thin coastline
    ax.add_feature(COAST_50M, linewidth=0.25, zorder=2)
    
    # Extremely subtle gridlines
    gl = ax.gridlines(draw_labels=False, linewidth=0.2, color="#000000", alpha=0.1,
                      linestyle="--", xlocs=np.arange(-180, 181, 60),
                      ylocs=np.arange(-90, 91, 30), zorder=3)
    
    ax.spines["geo"].set_edgecolor("#333333")
    ax.spines["geo"].set_linewidth(0.5)
    return gl

def add_panel_label(ax, tag, label):
    """Add Nature-style bold panel letters (a, b, c) tightly into the top left and row labels."""
    ax.text(-0.01, 1.05, tag, transform=ax.transAxes, fontsize=12,
            fontweight="bold", va="bottom", ha="right", fontfamily="Arial")
    
    if tag in ["a", "c", "e"]:
        ax.text(-0.08, 0.5, label, transform=ax.transAxes, fontsize=10,
                fontweight="normal", va="center", ha="right", rotation=90)

def draw_categorical_panel(ax, label, d, tag):
    tops = panel_tops[label]
    local_color_map = {drv: PALETTE[i] for i, drv in enumerate(tops)}

    add_map_features(ax, is_continuous=False)

    dd = d.copy()
    dd["cat"] = np.where(dd[COL_DRIVER].isin(tops), dd[COL_DRIVER], "Other")
    
    # Plot 'Other' first (zorder 4), then important ones (zorder 5)
    for cat in ["Other"] + tops:
        c = dd[dd["cat"] == cat]
        if c.empty: continue
        col = COLOR_OTHER if cat == "Other" else local_color_map[cat]
        ax.scatter(c[COL_LON].values, c[COL_LAT].values, s=PIXEL_SIZE, marker="s",
                   edgecolors="none", facecolors=col, transform=ccrs.PlateCarree(),
                   rasterized=True, zorder=5 if cat != "Other" else 4)

    add_panel_label(ax, tag, label)

    # Clean, borderless legend, closely hugged to the right edge of the map
    cats = tops + ["Other"]
    handles = [Patch(facecolor=(COLOR_OTHER if c == "Other" else local_color_map[c]),
                     edgecolor="none", label=c) for c in cats]
                     
    leg = ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5),
                    frameon=False, fontsize=6, title="Driver",
                    title_fontproperties={'weight':'normal', 'size':7}, handlelength=0.9, handleheight=0.9,
                    handletextpad=0.4, borderaxespad=0.0, labelspacing=0.6)
    leg.get_title()


def draw_continuous_panel(ax, label, d, tag):
    target_driver = TARGET_DRIVERS[label]
    sub = d[d[COL_DRIVER] == target_driver].copy()
    
    add_map_features(ax, is_continuous=True)
    add_panel_label(ax, tag, label)
    
    if sub.empty:
        ax.text(0.5, 0.5, f"No significant data for {target_driver}",
                transform=ax.transAxes, ha="center", va="center", fontsize=7, color="#888888")
        return

    shap_vals = sub[COL_SHAP].values.astype(float)
    abs_max = np.max(np.abs(shap_vals)) if len(shap_vals) > 0 else 0.01
    abs_max = abs_max if abs_max > 0 else 0.01
    # Symmetric normalization
    norm = Normalize(vmin=-abs_max, vmax=abs_max)

    # Sort values by absolute SHAP so highest impact drivers plot on top (vital for visual fidelity)
    sort_idx = np.argsort(np.abs(shap_vals))
    lons, lats, sv = sub[COL_LON].values[sort_idx], sub[COL_LAT].values[sort_idx], shap_vals[sort_idx]

    ax.scatter(lons, lats, s=PIXEL_SIZE, marker="s", edgecolors="none",
               c=sv, cmap=CMAP, norm=norm, transform=ccrs.PlateCarree(),
               rasterized=True, zorder=5)

    # Sleek colorbar, inset exact coordinates (x=1.01) closely hugged to map
    cax = ax.inset_axes([1.01, 0.15, 0.02, 0.7])
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                      orientation="vertical", extend="both")
    cb.ax.tick_params(labelsize=6, width=0.4, length=2.5, pad=2)
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor("#333333")
    
    # Title adjusted to be normal and compact
    cb.ax.set_title(f"{target_driver}\nSHAP", fontsize=7, fontweight="normal", pad=5, ha='center')

# ============================================================
# 4. Master Layout construction (GridSpec)
# ============================================================
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
proj = ccrs.Robinson(central_longitude=0)

# Tightened wspace (0.22) and right margin (0.90) to ensure map uses max canvas
# Left margin adjusted to 0.05 to accommodate row labels
gs = fig.add_gridspec(nrows=3, ncols=2, left=0.05, right=0.90, 
                      top=0.95, bottom=0.02, hspace=0.15, wspace=0.22)

left_tags = ["a", "c", "e"]
right_tags = ["b", "d", "f"]

for i, (label, d) in enumerate(panels):
    # Left panel: Categorical Dominant Drivers
    ax_left = fig.add_subplot(gs[i, 0], projection=proj)
    draw_categorical_panel(ax_left, label, d, left_tags[i])
    
    # Right panel: Continuous SHAP values
    ax_right = fig.add_subplot(gs[i, 1], projection=proj)
    draw_continuous_panel(ax_right, label, d, right_tags[i])

# ============================================================
# 5. Export (Vector & Raster formats)
# ============================================================
out_base = os.path.join(OUTDIR, "7_SHAP_Combined_6panel")

# PDF is strictly required by Nature for final typesetting. 
fig.savefig(out_base + ".pdf", bbox_inches="tight", dpi=600, transparent=False)
fig.savefig(out_base + ".png", bbox_inches="tight", dpi=600, transparent=False)
fig.savefig(out_base + ".tiff", bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"})

print(f"Plot generation complete! Files saved to: {out_base}.[pdf|png|tiff]")
plt.close(fig)