#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Standalone script to generate a publication-quality SHAP interaction plot.
Combines PC1 and PC2 side-by-side in a single figure.
Features:
- Robust pairwise feature extraction.
- Random index shuffling to prevent Z-order visual bias.
- GAM (Generalized Additive Model) for non-linear trendline fitting.
- Beautiful "plasma" colormap.
- Independent axes with bold panel labels (a, b).
"""

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Ensure pygam is installed: pip install pygam
try:
    from pygam import LinearGAM, s
except ImportError:
    raise ImportError("Please install pygam to use GAM fitting: pip install pygam")


def extract_feature_names(df: pd.DataFrame):
    """
    Safely extracts f1 and f2 names from dataframe columns by rigorously
    matching combinations of candidate features to the interaction column prefix.
    """
    interaction_cols = [c for c in df.columns if c.endswith("_interaction_SHAP")]
    if not interaction_cols:
        raise ValueError("No interaction column found in the dataframe.")
    
    interaction_col = interaction_cols[0]
    prefix = interaction_col.replace("_interaction_SHAP", "")
    
    candidate_features = [c.replace("_value", "") for c in df.columns if c.endswith("_value")]
    
    if len(candidate_features) < 2:
        raise ValueError("Not enough '_value' columns to form an interaction pair.")

    f1_match = None
    f2_match = None
    
    for feat1 in candidate_features:
        for feat2 in candidate_features:
            if f"{feat1}_{feat2}" == prefix:
                f1_match = feat1
                f2_match = feat2
                break
        if f1_match:
            break
            
    if not f1_match or not f2_match:
        raise ValueError(f"Could not map interaction prefix '{prefix}' to any two _value columns.")
        
    return f1_match, f2_match, interaction_col


def process_and_plot_axis(csv_path: Path, ax: plt.Axes, panel_label: str):
    """
    Reads the interaction CSV and plots it on the provided matplotlib axis.
    Adds a bold panel label (e.g., 'a', 'b') to the top left.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        ax.text(0.5, 0.5, "Error reading data", ha='center', va='center')
        print(f"Error reading {csv_path}: {e}")
        return

    try:
        f1, f2, interaction_col = extract_feature_names(df)
    except ValueError as ve:
        ax.text(0.5, 0.5, "Feature parsing error", ha='center', va='center')
        print(f"Parsing error in {csv_path.name}: {ve}")
        return

    col_f1 = f"{f1}_value"
    col_f2 = f"{f2}_value"

    x_val = df[col_f1]
    y_val = df[interaction_col]
    c_val = df[col_f2]

    # Randomly shuffle data to prevent Z-order visual bias
    np.random.seed(42)
    shuffle_idx = np.random.permutation(len(df))
    x_scatter = x_val.iloc[shuffle_idx]
    y_scatter = y_val.iloc[shuffle_idx]
    c_scatter = c_val.iloc[shuffle_idx]

    # AESTHETICS: Publication-ready 'plasma' colormap
    cmap = plt.get_cmap("plasma")

    # Scatter plot
    scatter = ax.scatter(
        x_scatter, y_scatter,
        c=c_scatter,
        cmap=cmap,
        s=12,
        alpha=0.6,
        edgecolors="none",
        zorder=2
    )

    # GAM (Generalized Additive Model) Trendline
    try:
        gam = LinearGAM(s(0)).fit(x_val, y_val)
        x_smooth = np.linspace(x_val.min(), x_val.max(), 500)
        y_smooth = gam.predict(x_smooth)
        
        trendline_effect = [PathEffects.withStroke(linewidth=4.5, foreground="white")]
        ax.plot(
            x_smooth, y_smooth,
            color='#111111', linestyle='-', linewidth=2.0,
            zorder=5,
            path_effects=trendline_effect
        )
    except Exception as e:
        print(f"GAM fitting failed for {panel_label}: {e}")

    # Formatting and aesthetics (No bold fonts except panel label)
    ax.grid(True, linestyle=':', alpha=0.6, zorder=0)
    ax.tick_params(axis='both', which='major', labelsize=12, length=5)
    
    # Bold Panel Label (a, b) at the top left
    ax.text(-0.12, 1.05, panel_label, transform=ax.transAxes, 
            fontsize=18, fontweight='bold', va='bottom', ha='left')
    
    # Independent axes labels for both subplots
    ax.set_xlabel(f"{f1} Value", fontsize=14, fontweight='normal')
    ax.set_ylabel(f"SHAP Interaction Value\n({f1} x {f2})", fontsize=14, fontweight='normal')

    # Zero-line reference
    ax.axhline(0, color='gray', linestyle='--', linewidth=1.2, zorder=1)

    # Colorbar appending
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.15)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label(f"{f2} Value", fontsize=12, fontweight='normal')
    cbar.outline.set_linewidth(1.2)


def main():
    target_directory = Path("/data/global_mapping/models_V100/Plant_Functional_Traits/final_shap_visuals")
    target_directory.mkdir(parents=True, exist_ok=True)

    # Global Plot Settings (Normal font weight)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica"],
        "axes.linewidth": 1.2,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "font.weight": "normal",
        "axes.labelweight": "normal",
        "axes.titleweight": "normal"
    })

    # Create a 1x2 grid for PC1 and PC2
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    targets = ["PC1", "PC2"]
    panel_labels = ["a", "b"]
    
    for ax, pc, p_label in zip(axes, targets, panel_labels):
        input_path = Path(f"/data/global_mapping/models_V100/Plant_Functional_Traits/{pc}/XGB_shap/XGB_shap")
        
        if not input_path.exists():
            print(f"Warning: Directory not found -> {input_path}")
            ax.text(0.5, 0.5, f"Directory Not Found\n{pc}", ha='center', va='center')
            ax.axis('off')
            continue

        # Scan for the correct CSV
        csv_files = list(input_path.rglob("*6_SHAP_Interaction*.csv"))
        # Filter for 1000km buffer explicitly
        csv_files = [f for f in csv_files if "buf_1000km" in str(f)]

        if not csv_files:
            print(f"Warning: No 1000km target CSV found in -> {input_path}")
            ax.text(0.5, 0.5, f"No Data Found\n{pc}", ha='center', va='center')
            ax.axis('off')
        else:
            target_csv = csv_files[0]
            print(f"Processing: {target_csv.name} for {pc}")
            process_and_plot_axis(target_csv, ax, panel_label=p_label)

    plt.tight_layout()

    # Save outputs
    base_filename = "SHAP_Interaction_Combined_1000km"
    output_png = target_directory / f"{base_filename}.png"
    output_tif = target_directory / f"{base_filename}.tif"

    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_tif, dpi=300, bbox_inches='tight', facecolor='white', format='tif')
    plt.close()

    print(f"\nSuccessfully generated combined plots:")
    print(f"  {output_png}")
    print(f"  {output_tif}")

if __name__ == "__main__":
    main()