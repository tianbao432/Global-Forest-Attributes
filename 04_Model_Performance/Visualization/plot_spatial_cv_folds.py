#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Publication-quality spatial visualization for Block CV
-------------------------------------------------------------
Features:
1. Integrated Cartopy for robust Equal Earth projection handling.
2. Uses local fused land boundary shapefile.
3. Aesthetic color palette for Train/Buffer/Test splits.
4. Ultra-compact layout with Top-Tier journal formatting.
5. Fixed legend positioning and added padding to prevent border clipping.
-------------------------------------------------------------
"""

import string
import warnings
from pathlib import Path

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cartopy.crs as ccrs

# Ignore warnings for clean execution output
warnings.filterwarnings("ignore")

# =====================================================
# Configuration & Paths
# =====================================================
INPUT_FILE = (
    "/data/global_mapping/models_V100/drive_fina/PC3/XGB_shap/XGB_shap/block_1500km_buf_1000km/1b_Block_CV_Per_Point_Predictions.csv"
)

LAND_SHAPEFILE = (
    "/data/global_mapping/models_V100/land_bound/world_land_fused.shp"
)

OUTPUT_DIR = Path(
    "/data/global_mapping/models_V100/drive_fina/PC3/XGB_shap/"
    "XGB_shap/fold_cv_visuals/"
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_TIF = OUTPUT_DIR / "BlockCV_10Fold_Publication.tif"
OUTPUT_PDF = OUTPUT_DIR / "BlockCV_10Fold_Publication.pdf"

# =====================================================
# Publication-Ready Styling
# =====================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 10,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42
})

# Aesthetic Color Palette (Colorblind friendly & high contrast)
ROLE_COLORS = {
    "Train": "#457B9D",   # Muted Blue
    "Buffer": "#E9C46A",  # Sandy Yellow
    "Test": "#E63946"     # Vibrant Red
}

# =====================================================
# Data Loading
# =====================================================
print("Reading CSV file ...")
df = pd.read_csv(INPUT_FILE)

print("Creating GeoDataFrame ...")
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
    crs="EPSG:4326"
)

print(f"Loading local land basemap from:\n{LAND_SHAPEFILE}")
world = gpd.read_file(LAND_SHAPEFILE)

# =====================================================
# Figure Setup
# =====================================================
print("Initializing figure ...")
proj = ccrs.EqualEarth()        
data_crs = ccrs.PlateCarree()  

# Optimized figsize for an 4x2 grid of EqualEarth maps to minimize white space
fig, axes = plt.subplots(
    nrows=5,
    ncols=2,
    figsize=(12, 14), 
    dpi=300,
    subplot_kw={'projection': proj}
)
axes = axes.flatten()

# =====================================================
# Plotting Loop
# =====================================================
for fold in range(1, 11):
    print(f"  Rendering Fold {fold} ...")
    ax = axes[fold - 1]
    
    ax.set_global()

    role_col = f"Fold_{fold}_Role"
    if role_col not in gdf.columns:
        print(f"  Warning: Column {role_col} not found. Skipping.")
        continue

    fold_df = gdf.dropna(subset=[role_col]).copy()

    train_df = fold_df[fold_df[role_col] == "Train"]
    buffer_df = fold_df[fold_df[role_col] == "Buffer"]
    test_df = fold_df[fold_df[role_col] == "Test"]

    # 1. Plot Base Map
    ax.add_geometries(
        world.geometry,
        crs=data_crs,
        facecolor="#F0F3F4",   
        edgecolor="#B2BABB",   
        linewidth=0.5,
        zorder=0
    )

    # 2. Plot Data Points
    ax.scatter(train_df["lon"], train_df["lat"], transform=data_crs,
               s=0.3, color=ROLE_COLORS["Train"], alpha=0.5, zorder=1)
    
    ax.scatter(buffer_df["lon"], buffer_df["lat"], transform=data_crs,
               s=1.5, color=ROLE_COLORS["Buffer"], alpha=0.8, zorder=2)
    
    ax.scatter(test_df["lon"], test_df["lat"], transform=data_crs,
               s=3.0, color=ROLE_COLORS["Test"], alpha=1.0, zorder=3)

    # 3. Add Lat/Lon Graticules
    gl = ax.gridlines(
        draw_labels=True, 
        linewidth=0.5, 
        color='#999999', 
        alpha=0.5, 
        linestyle=':'
    )
    
    row = (fold - 1) // 2
    col = (fold - 1) % 2
    gl.top_labels = (row == 0)       
    gl.bottom_labels = (row == 3)    
    gl.left_labels = (col == 0)      
    gl.right_labels = (col == 1)    
    
    gl.x_inline = False
    gl.y_inline = False
    gl.xlabel_style = {'size': 9, 'color': '#555555'}
    gl.ylabel_style = {'size': 9, 'color': '#555555'}

    # 4. Add Panel Letter
    panel_letter = string.ascii_lowercase[fold - 1]
    ax.text(
        0.01, 0.99, panel_letter, 
        transform=ax.transAxes, 
        fontsize=16, fontweight='bold', 
        va='top', ha='left'
    )
    
    # 5. Overlay Stats Box
    stats_text = (
        f"Fold {fold}\n"
        f"Train: {len(train_df)}\n"
        f"Buffer: {len(buffer_df)}\n"
        f"Test: {len(test_df)}"
    )
    
    ax.text(
        0.09, 0.32, stats_text,
        transform=ax.transAxes,
        fontsize=9, linespacing=1.4, color="#333333",
        va='center', ha='left',
        bbox=dict(
            facecolor='white', 
            alpha=0.9, 
            edgecolor='#CCCCCC', 
            boxstyle='round,pad=0.5',
            linewidth=0.5
        ) 
    )

# =====================================================
# Global Legend
# =====================================================
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Train',
           markerfacecolor=ROLE_COLORS["Train"], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Buffer',
           markerfacecolor=ROLE_COLORS["Buffer"], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Test',
           markerfacecolor=ROLE_COLORS["Test"], markersize=10)
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=3,
    fontsize=13,
    frameon=False,
    # Shifted Y-coordinate up further to clear the map projection edges
    bbox_to_anchor=(0.5, 0.07) 
)

# =====================================================
# Ultra-Compact Layout Adjustments
# =====================================================
plt.subplots_adjust(
    left=0.05,  
    right=0.95, 
    top=0.96,   
    bottom=0.12, 
    wspace=0.02, 
    hspace=0.02  
)

# =====================================================
# Export
# =====================================================
print(f"\nSaving publication-ready TIF figure to:\n{OUTPUT_TIF}")
fig.savefig(
    OUTPUT_TIF,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0.15,  # ADDED: This guarantees a clean white border around the final image
    facecolor="white",
    pil_kwargs={"compression": "tiff_lzw"} 
)

print(f"Saving vector PDF figure to:\n{OUTPUT_PDF}")
fig.savefig(
    OUTPUT_PDF,
    bbox_inches="tight",
    pad_inches=0.15,  # ADDED: Prevents the PDF from cropping right to the legend text
    facecolor="white"
)

plt.close()
print("All done. Visualizations saved successfully.")