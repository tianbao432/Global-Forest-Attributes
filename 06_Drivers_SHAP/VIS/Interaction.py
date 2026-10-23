#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script to generate a publication-quality SHAP interaction plot for PC2.
Only the middle panel (PC2) is retained; panel labels are removed.
Improved colormap: emerald green gradient for better aesthetics.
Uses a smooth GAM (Generalized Additive Model) trendline (no legend).
Exports both PNG and TIF formats.
Fixed Z-order masking issue by sorting points by color value before scattering.
All code and comments are strictly in English.
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Import pygam for the trendline
try:
    from pygam import LinearGAM, s
except ImportError:
    raise ImportError("Please install pygam to use GAM fitting: pip install pygam")

def process_and_plot_axis(csv_path: Path, ax: plt.Axes):
    """
    Reads the interaction CSV and plots it on the provided matplotlib axis.
    No panel label and no trendline legend are added.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        ax.text(0.5, 0.5, f"Error reading data", ha='center', va='center')
        print(f"Error reading {csv_path}: {e}")
        return

    # Dynamically identify f1 and f2 from column names
    interaction_col = None
    for col in df.columns:
        if col.endswith("_interaction_SHAP"):
            interaction_col = col
            break

    if not interaction_col:
        ax.text(0.5, 0.5, f"No interaction column found", ha='center', va='center')
        return

    # Extract feature names
    parts = interaction_col.replace("_interaction_SHAP", "").split("_")
    if len(parts) < 2:
        return

    f1 = parts[0]
    f2 = parts[1]

    col_f1 = f"{f1}_value"
    col_f2 = f"{f2}_value"

    if col_f1 not in df.columns or col_f2 not in df.columns:
        return

    x_val = df[col_f1]
    y_val = df[interaction_col]
    c_val = df[col_f2]

    # --- KEY VISUAL FIX: Sort data by color value (c_val) ---
    # By sorting ascending, we guarantee that high-value points (dark green)
    # are plotted LAST, sitting on top of low-value points (light green).
    sort_idx_color = np.argsort(c_val.values)
    x_scatter = x_val.iloc[sort_idx_color]
    y_scatter = y_val.iloc[sort_idx_color]
    c_scatter = c_val.iloc[sort_idx_color]

    # Emerald green colormap: light mint -> deep emerald
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "emerald_cmap", ["#1E88E5", "#FF0D57"]
    )

    # Scatter plot with optimized aesthetics
    scatter = ax.scatter(
        x_scatter, y_scatter,
        c=c_scatter,
        cmap=cmap,
        s=8,
        alpha=0.75,
        edgecolors="none",
        zorder=2
    )

    # Generate a perfectly smooth GAM (Generalized Additive Model) trendline
    try:
        gam = LinearGAM(s(0)).fit(x_val, y_val)
        x_smooth = np.linspace(x_val.min(), x_val.max(), 500)
        y_smooth = gam.predict(x_smooth)
        
        # Beautified trendline: thicker black line with white stroke, no legend entry
        trendline_effect = [PathEffects.withStroke(linewidth=5.0, foreground="white")]
        ax.plot(
            x_smooth, y_smooth,
            color='#111111', linestyle='-', linewidth=2.5,
            zorder=5,
            path_effects=trendline_effect
            # No label, therefore no legend needed
        )
    except Exception as e:
        print(f"GAM fitting failed: {e}")

    # Light background grid
    ax.grid(True, linestyle=':', alpha=0.5, zorder=0)

    # Axis parameters
    ax.tick_params(axis='both', which='major', labelsize=13, length=5)

    # Axis labels
    ax.set_xlabel(f"{f1} Value", fontsize=15)
    ax.set_ylabel(f"SHAP Interaction Value\n({f1} x {f2})", fontsize=15)

    # Zero-line reference
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.2, zorder=1)

    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.ax.tick_params(labelsize=11)
    cbar.set_label(f"{f2} Value", fontsize=13)
    cbar.outline.set_linewidth(1.2)


def main():
    # Keep only the PC2 (middle) path
    input_path = Path("/data/global_mapping/models_V100/drive_fina/PC2/XGB_shap/XGB_shap")

    target_directory = Path("/data/global_mapping/models_V100/drive_fina/final_shap_visuals")
    target_directory.mkdir(parents=True, exist_ok=True)

    # Publication-quality figure settings
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    })

    # Single plot without panel labels
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    if not input_path.exists():
        print(f"Warning: Path does not exist -> {input_path}")
        ax.text(0.5, 0.5, f"Path Not Found", ha='center', va='center')
        ax.axis('off')
    else:
        # Scan for the 800km CSV
        csv_files = list(input_path.rglob("6_SHAP_Interaction_Top2_Pointwise.csv"))
        csv_files = [f for f in csv_files if "buf_1000km" in str(f)]

        if not csv_files:
            print(f"Warning: No 1000km CSV found in -> {input_path}")
            ax.text(0.5, 0.5, f"No Data Found", ha='center', va='center')
            ax.axis('off')
        else:
            target_csv = csv_files[0]
            print(f"Processing: {target_csv.name} from PC2")
            process_and_plot_axis(target_csv, ax)

    plt.tight_layout()

    # Save in both PNG and TIF formats
    base_filename = "SHAP_Interaction_PC2_1000km"
    output_png = target_directory / f"{base_filename}.png"
    output_tif = target_directory / f"{base_filename}.tif"

    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_tif, dpi=300, bbox_inches='tight', facecolor='white', format='tif')
    plt.close()

    print(f"\nSuccessfully generated plots:")
    print(f"  {output_png}")
    print(f"  {output_tif}")

if __name__ == "__main__":
    main()