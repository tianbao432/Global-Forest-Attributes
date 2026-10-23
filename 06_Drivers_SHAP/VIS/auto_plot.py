#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import string
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

DATA_PATHS = {
    "PC1": "/data/global_mapping/models_V100/drive/PC1/Spatial_Autocorrelation/auto/AC_MoranI_along_distance.csv",
    "PC2": "/data/global_mapping/models_V100/drive/PC2/Spatial_Autocorrelation/auto/AC_MoranI_along_distance.csv",
    "PC3": "/data/global_mapping/models_V100/drive/PC3/Spatial_Autocorrelation/auto/AC_MoranI_along_distance.csv"
}

OUTDIR = "/data/global_mapping/models_V100/drive/final_shap_visuals/"
os.makedirs(OUTDIR, exist_ok=True)

COL_DIST  = "distance_km"
COL_MORAN = "MoranI"
COL_LOW   = "ci_lower"
COL_HIGH  = "ci_upper"
THRESHOLD = 0.05

mpl.rcParams.update({
    "font.family":     "Arial",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size":       10,
    "axes.linewidth":  0.9,
    "pdf.fonttype":    42,
    "ps.fonttype":     42,
    "svg.fonttype":    "none",
})

C_LINE   = "#1f4e9c"
C_BAND   = "#1f4e9c"
C_REF    = "#c0392b"
C_ZERO   = "#7f8c8d"

fig, axes = plt.subplots(3, 1, figsize=(5.0, 8.5))
panel_labels = string.ascii_lowercase

for i, (pc_name, csv_path) in enumerate(DATA_PATHS.items()):
    ax = axes[i]
    
    df = pd.read_csv(csv_path).sort_values(COL_DIST).reset_index(drop=True)
    x  = df[COL_DIST].values
    y  = df[COL_MORAN].values
    lo = df[COL_LOW].values
    hi = df[COL_HIGH].values
    
    ax.fill_between(x, lo, hi, color=C_BAND, alpha=0.18, linewidth=0, label="95% CI")
    ax.plot(x, y, color=C_LINE, linewidth=2.0, solid_capstyle="round", label=f"Moran's I")
    
    ax.axhline(0.0, color=C_ZERO, linewidth=0.8, linestyle="-", alpha=0.6, zorder=0)
    ax.axhline(THRESHOLD, color=C_REF, linewidth=1.1, linestyle="--", dashes=(5, 3), alpha=0.85, zorder=1)
    
    ax.text(x.max(), THRESHOLD + 0.02, f"I = {THRESHOLD:g}", color=C_REF, 
            fontsize=9, va="bottom", ha="right", fontweight="normal")
    
    cross = np.where(np.diff(np.sign(y - THRESHOLD)))[0]
    for idx in cross:
        x0, x1 = x[idx], x[idx + 1]
        y0, y1 = y[idx], y[idx + 1]
        xc = x0 + (THRESHOLD - y0) * (x1 - x0) / (y1 - y0)
        ax.plot([xc], [THRESHOLD], "o", color=C_REF, ms=5, zorder=5)
        ax.annotate(f"{xc:.0f} km", (xc, THRESHOLD),
                    textcoords="offset points", xytext=(4, 8),
                    fontsize=8.5, color=C_REF, fontweight="normal")
                    
    ax.set_ylabel("Moran's I", fontweight="normal")
    if i == 2:
        ax.set_xlabel("Distance (km)", fontweight="normal")
    
    ax.set_xlim(x.min(), x.max())
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(direction="out", length=4, width=0.9)
    ax.set_title(pc_name, fontsize=11, fontweight="normal", pad=8)
    
    ax.text(-0.12, 1.05, panel_labels[i], transform=ax.transAxes, 
            fontsize=12, fontweight="bold", va="top", ha="right", family="Arial")

fig.tight_layout()

out = os.path.join(OUTDIR, "AC_MoranI_along_distance_All_PCs_Vertical")
fig.savefig(out + ".pdf",  bbox_inches="tight", dpi=600)
fig.savefig(out + ".png",  bbox_inches="tight", dpi=600)
fig.savefig(out + ".tiff", bbox_inches="tight", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
plt.close(fig)

print(f"Saved merged PDF / PNG / TIFF to: {OUTDIR}")