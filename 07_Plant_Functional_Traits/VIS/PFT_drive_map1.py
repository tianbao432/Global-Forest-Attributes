#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
===============================================================================
SHAP Dominant Driver - Combined 6-panel Robinson map
Optimized for Nature Journal Publication Standards (Double-column width: ~183mm)
Features:
  - Ultra-tight layout with reduced hspace/wspace for cohesive grouping.
  - Neutralized background (White ocean, ultra-light grey land).
  - Z-order optimization: Highest importance traits plotted on top.
  - Perceptually uniform, colorblind-safe diverging colormap (Zero = White).
  - Publication-grade typography: panel letters bold & enlarged, others normal.
  - Fixed 5 variables with a shared global legend at the bottom.
===============================================================================
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

# ============================================================
# 0. Configuration & Input Paths
# ============================================================
OUTDIR = "/data/global_mapping/models_V100/Plant_Functional_Traits/final_shap_visuals"
os.makedirs(OUTDIR, exist_ok=True)

INPUTS = [
    ("/data/global_mapping/models_V100/Plant_Functional_Traits/PC1/XGB_shap/XGB_shap/block_1500km_buf_800km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC1"),
    ("/data/global_mapping/models_V100/Plant_Functional_Traits/PC2/XGB_shap/XGB_shap/block_1500km_buf_800km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC2"),
    ("/data/global_mapping/models_V100/Plant_Functional_Traits/PC3/XGB_shap/XGB_shap/block_1500km_buf_800km/final_shap/7_SHAP_Directional_Dominant_Map.csv", "PC3"),
]

COL_LON, COL_LAT = "lon", "lat"
COL_DRIVER, COL_SHAP = "Dominant_Driver", "Dominant_SHAP"
PIXEL_SIZE = 0.1  

TARGET_DRIVERS = {"PC1": "LPC", "PC2": "WD", "PC3": "WD"}

# Global configuration for the 5 fixed variables and their distinct colors
GLOBAL_CATEGORIES = ["LPC", "WD", "LDMC", "LNC", "SLA"]
GLOBAL_COLORS = [
    "#F39B7F",  # Deep Red
    "#00A087",  # Cyan
    "#3C5488",  # Teal
    "#4DBBD5",  # Dark Blue
    "#E64B35",  # Peach
]
DRIVER_COLOR_MAP = dict(zip(GLOBAL_CATEGORIES, GLOBAL_COLORS))

# Publication-grade Diverging Colormap (Deep Blue -> White -> Deep Red)
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
FIG_WIDTH = 7.2  
FIG_HEIGHT = 4.8 # Slightly taller to fit bottom legend

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 7,                
    "axes.titlesize": 7,           
    "axes.labelsize": 7,
    "xtick.labelsize": 6,          
    "ytick.labelsize": 6,
    "legend.fontsize": 6,
    "legend.title_fontsize": 7,
    "axes.linewidth": 0.4,         
    "grid.linewidth": 0.15,        
    "pdf.fonttype": 42,            
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.dpi": 300,
    "axes.unicode_minus": True
})

# High-resolution map features
LAND_COLOR = '#F6F6F6'     
OCEAN_COLOR = '#FFFFFF'    
COAST_COLOR = '#444444'    

LAND_50M = cfeature.NaturalEarthFeature('physical', 'land', '50m', facecolor=LAND_COLOR, edgecolor='none')
OCEAN_50M = cfeature.NaturalEarthFeature('physical', 'ocean', '50m', facecolor=OCEAN_COLOR, edgecolor='none')
COAST_50M = cfeature.NaturalEarthFeature('physical', 'coastline', '50m', facecolor='none', edgecolor=COAST_COLOR)

# ============================================================
# 2. Data Loading & Preprocessing
# ============================================================
panels = []
panel_tops = {}

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
        
        valid_drivers = df_imp.sort_values(by=pct_col, ascending=False)["feature"].tolist()
        existing_drivers = set(d[COL_DRIVER].unique())
        
        # Keep only the 5 valid global categories, sorted by importance
        filtered_tops = [drv for drv in valid_drivers if drv in existing_drivers and drv in GLOBAL_CATEGORIES]
        panel_tops[label] = filtered_tops
    else:
        raw_tops = d[COL_DRIVER].value_counts().index.tolist()
        panel_tops[label] = [drv for drv in raw_tops if drv in GLOBAL_CATEGORIES]

# ============================================================
# 3. Core Plotting Functions
# ============================================================
def add_map_features(ax):
    ax.set_global()
    ax.add_feature(OCEAN_50M, zorder=0)
    ax.add_feature(LAND_50M, zorder=1)
    ax.add_feature(COAST_50M, linewidth=0.25, zorder=2)
    
    gl = ax.gridlines(draw_labels=False, linewidth=0.1, color="#000000", alpha=0.1,
                      linestyle="--", xlocs=np.arange(-180, 181, 60),
                      ylocs=np.arange(-90, 91, 30), zorder=3)
    
    ax.spines["geo"].set_edgecolor("#555555")
    ax.spines["geo"].set_linewidth(0.4)
    return gl

def add_panel_label(ax, tag, label):
    ax.text(-0.02, 1.02, tag, transform=ax.transAxes, fontsize=11,
            fontweight="bold", va="bottom", ha="right")
    
    if tag in ["a", "c", "e"]:
        ax.text(-0.05, 0.5, label, transform=ax.transAxes, fontsize=8,
                fontweight="normal", va="center", ha="right", rotation=90)

def draw_categorical_panel(ax, label, d, tag):
    tops = panel_tops[label]
    
    add_map_features(ax)
    add_panel_label(ax, tag, label)

    # Plotted in reverse so top rank stays physically on top
    for cat in reversed(tops):
        c = d[d[COL_DRIVER] == cat]
        if c.empty: continue
        col = DRIVER_COLOR_MAP[cat]
        ax.scatter(c[COL_LON].values, c[COL_LAT].values, s=PIXEL_SIZE, marker="s",
                   edgecolors="none", facecolors=col, transform=ccrs.PlateCarree(),
                   rasterized=True, zorder=5)

def draw_continuous_panel(ax, label, d, tag):
    target_driver = TARGET_DRIVERS[label]
    sub = d[d[COL_DRIVER] == target_driver].copy()
    
    add_map_features(ax)
    add_panel_label(ax, tag, label)
    
    if sub.empty:
        ax.text(0.5, 0.5, f"No data: {target_driver}", transform=ax.transAxes, 
                ha="center", va="center", fontsize=7, color="#888888")
        return

    shap_vals = sub[COL_SHAP].values.astype(float)
    abs_max = np.max(np.abs(shap_vals)) if len(shap_vals) > 0 else 0.01
    abs_max = abs_max if abs_max > 0 else 0.01
    
    norm = Normalize(vmin=-abs_max, vmax=abs_max)

    sort_idx = np.argsort(np.abs(shap_vals))
    lons, lats, sv = sub[COL_LON].values[sort_idx], sub[COL_LAT].values[sort_idx], shap_vals[sort_idx]

    ax.scatter(lons, lats, s=PIXEL_SIZE, marker="s", edgecolors="none",
               c=sv, cmap=CMAP, norm=norm, transform=ccrs.PlateCarree(),
               rasterized=True, zorder=5)

    cax = ax.inset_axes([1.02, 0.15, 0.025, 0.7])
    cb = plt.colorbar(ScalarMappable(norm=norm, cmap=CMAP), cax=cax,
                      orientation="vertical", extend="both")
    
    cb.ax.tick_params(labelsize=6, width=0.4, length=3, pad=3, direction='out')
    cb.outline.set_linewidth(0.4)
    cb.outline.set_edgecolor("#555555")
    cb.ax.set_title(f"{target_driver}\nSHAP", fontsize=7, fontweight="normal", pad=5, ha='center')

# ============================================================
# 4. Master Layout construction
# ============================================================
fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))
proj = ccrs.Robinson(central_longitude=0)

# Reduced wspace from 0.12 to 0.05 to pull the left and right columns closer
gs = fig.add_gridspec(nrows=3, ncols=2, left=0.05, right=0.89, 
                      top=0.97, bottom=0.09, hspace=0.05, wspace=0.02)

left_tags = ["a", "c", "e"]
right_tags = ["b", "d", "f"]

for i, (label, d) in enumerate(panels):
    # Left Map (Categorical)
    ax_left = fig.add_subplot(gs[i, 0], projection=proj)
    draw_categorical_panel(ax_left, label, d, left_tags[i])
    
    # Right Map (Continuous)
    ax_right = fig.add_subplot(gs[i, 1], projection=proj)
    draw_continuous_panel(ax_right, label, d, right_tags[i])

# ============================================================
# 5. Shared Global Legend Setup
# ============================================================
# Create patches for the 5 fixed categories
legend_handles = [Patch(facecolor=DRIVER_COLOR_MAP[cat], edgecolor="none", label=cat) 
                  for cat in GLOBAL_CATEGORIES]

# Place legend centered at the bottom of the figure
fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.47, 0.0),
           ncol=5, frameon=False, fontsize=7, title="Driver",
           title_fontproperties={'weight': 'normal', 'size': 7},
           handlelength=1.2, handleheight=1.2, handletextpad=0.5, columnspacing=2.5)

# ============================================================
# 6. Export (Vector & Raster)
# ============================================================
out_base = os.path.join(OUTDIR, "7_SHAP_Combined_6panel_Nature")

fig.savefig(out_base + ".pdf", bbox_inches="tight", dpi=600, transparent=False)
fig.savefig(out_base + ".tiff", bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(out_base + ".png", bbox_inches="tight", dpi=600, facecolor="white")

print(f"[OK] Plot successfully generated! Saved to {out_base}.[pdf|tiff|png]")
plt.close(fig)