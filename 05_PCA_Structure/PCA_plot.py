# -*- coding: utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import rasterio
from rasterio.windows import from_bounds
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# Optimized Global Forest PC Spatial Distribution and Latitudinal Profile
# Nature-style: white background, land boundaries, clean layout
# =============================================================================

# 1. Global Parameters & Plot Styling -----------------------------------------
LAT_MIN, LAT_MAX = -60, 85
LON_MIN, LON_MAX = -180, 180

# Land boundary shapefile (user-provided)
LAND_SHP = "/data/global_mapping/models_V100/land_bound/world_land_fused.shp"

# Fonts (Nature standard)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Scientific diverging color palette (Spectral-like)
nature_colors = [
    "#5e4fa2", "#3287bd", "#65c2a5", "#abdca4", "#e6f598", "#ffffbf",
    "#fedf8c", "#fdae61", "#f56d43", "#d53e4f", "#9f0042"
]
nature_cmap = LinearSegmentedColormap.from_list("nature_div", nature_colors)

# Pre-load land boundary feature (loaded once, reused for all 3 panels)
try:
    land_geoms = list(Reader(LAND_SHP).geometries())
    land_feature = ShapelyFeature(
        land_geoms, ccrs.PlateCarree(),
        facecolor='none', edgecolor='black', linewidth=0.35
    )
    print(f"Loaded {len(land_geoms)} land geometries from shapefile.")
except Exception as e:
    print(f"Warning: Could not load shapefile ({e}). Falling back to Natural Earth.")
    land_feature = cfeature.NaturalEarthFeature(
        category='physical', name='land', scale='110m',
        edgecolor='black', facecolor='none', linewidth=0.35
    )


# 2. Core Processing and Plotting Function ------------------------------------
def process_and_plot(tif_path, pc_name, tag_label, ax_map, ax_lat, is_bottom_row=False):
    print(f"Processing {pc_name}...")

    # ----- Data Processing ---------------------------------------------------
    with rasterio.open(tif_path) as src:
        window = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, src.transform)
        window = window.round_offsets().round_lengths()
        data = src.read(1, window=window)
        nodata = src.nodata
        transform = src.window_transform(window)

    # Mask invalid/nodata
    if nodata is not None:
        data_ma = np.ma.masked_equal(data, nodata)
    elif np.isnan(data).any():
        data_ma = np.ma.masked_invalid(data)
    else:
        data_ma = np.ma.masked_array(data)

    # Symmetric color limits via 1st / 99th percentiles
    valid_data = data_ma.compressed()
    if valid_data.size > 0:
        vmin_q, vmax_q = np.percentile(valid_data, [1, 99])
        v_limit = max(abs(vmin_q), abs(vmax_q))
    else:
        v_limit = 1.0

    # ----- Latitudinal Profile (2-deg bins) ----------------------------------
    rows, cols = data_ma.shape
    y_coords = np.array([transform * (0, r + 0.5) for r in range(rows)])[:, 1]
    rounded_y = np.round(y_coords / 2.0) * 2.0
    unique_y = np.unique(rounded_y)

    lats, mean_vals, sd_vals = [], [], []
    for uy in unique_y:
        if LAT_MIN <= uy <= LAT_MAX:
            row_indices = np.where(rounded_y == uy)[0]
            row_data = data_ma[row_indices, :].compressed()
            if row_data.size > 0:
                mean_vals.append(np.mean(row_data))
                sd_vals.append(np.std(row_data))
                lats.append(uy)

    mean_vals = np.array(mean_vals)
    sd_vals = np.array(sd_vals)
    lats = np.array(lats)

    img_extent = [
        transform.c, transform.c + transform.a * cols,
        transform.f + transform.e * rows, transform.f
    ]

    # ----- Spatial Map -------------------------------------------------------
    # White background (ocean = white)
    ax_map.set_facecolor("white")
    ax_map.set_extent([LON_MIN, LAT_MAX_lon := LON_MAX, LAT_MIN, LAT_MAX],
                      crs=ccrs.PlateCarree())

    # Raster layer
    im = ax_map.imshow(
        data_ma, cmap=nature_cmap, vmin=-v_limit, vmax=v_limit,
        extent=img_extent, transform=ccrs.PlateCarree(),
        origin='upper', interpolation='none', zorder=2
    )

    # Land boundary overlay (thin black outline)
    ax_map.add_feature(land_feature, zorder=3)

    # Graticules
    gl = ax_map.gridlines(
        draw_labels=False, crs=ccrs.PlateCarree(),
        color="grey", alpha=0.35, linestyle=(0, (3, 3)),
        linewidth=0.4, zorder=1
    )
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 60))
    gl.ylocator = mticker.FixedLocator(np.arange(-60, 91, 30))

    # Outer frame
    ax_map.spines['geo'].set_edgecolor('black')
    ax_map.spines['geo'].set_linewidth(0.7)
    ax_map.spines['geo'].set_zorder(5)

    # Colorbar ¡ª placed BELOW map, smaller & centered to avoid overlap
    cax = inset_axes(
        ax_map, width="35%", height="3.5%", loc='lower center',
        bbox_to_anchor=(0, -0.18, 1, 1),
        bbox_transform=ax_map.transAxes, borderpad=0
    )
    cb = plt.colorbar(im, cax=cax, orientation='horizontal', extend='both')
    cb.set_label(f"{pc_name}", fontsize=10.5, labelpad=3, fontweight='bold')
    cb.ax.tick_params(labelsize=9, length=2.5, width=0.5,
                      color='black', pad=2)
    cb.outline.set_linewidth(0.5)
    # Use 3 ticks: -v, 0, +v (rounded)
    tick_v = np.round(v_limit * 0.8, 1)
    cb.set_ticks([-tick_v, 0, tick_v])

    # Panel label (a, b, c)
    ax_map.text(
        -0.04, 1.02, tag_label, transform=ax_map.transAxes,
        fontsize=15, fontweight='bold', va='bottom', ha='left', zorder=6
    )

    # ----- Latitudinal Profile -----------------------------------------------
    ax_lat.axvline(0, color="gray", linestyle="dashed",
                   linewidth=0.7, zorder=1)

    ax_lat.fill_betweenx(
        lats, mean_vals - sd_vals, mean_vals + sd_vals,
        color="#92C5DE", alpha=0.45, edgecolor='none', zorder=2
    )
    ax_lat.plot(mean_vals, lats, color="#B2182B",
                linewidth=1.3, zorder=3)

    ax_lat.set_ylim([LAT_MIN, LAT_MAX])
    y_ticks = np.arange(-60, 81, 20)
    ax_lat.set_yticks(y_ticks)
    ax_lat.set_yticklabels([f"{y}\u00B0" for y in y_ticks])

    ax_lat.set_ylabel("Latitude", fontsize=10.5, labelpad=3)

    # Auto-symmetric x limits for the profile
    x_pad = max(abs(np.nanmin(mean_vals - sd_vals)),
                abs(np.nanmax(mean_vals + sd_vals))) * 1.1
    ax_lat.set_xlim([-x_pad, x_pad])

    # Always show x ticks/labels (fixes overlap issue from original)
    ax_lat.tick_params(axis='x', which='major', labelsize=9,
                       length=3, width=0.7)
    ax_lat.tick_params(axis='y', which='major', labelsize=9,
                       length=3, width=0.7)

    if is_bottom_row:
        ax_lat.set_xlabel("Score", fontsize=10.5, labelpad=3)
    else:
        ax_lat.set_xlabel("")

    # Clean spines (theme_classic style)
    ax_lat.spines['top'].set_visible(False)
    ax_lat.spines['right'].set_visible(False)
    ax_lat.spines['bottom'].set_linewidth(0.7)
    ax_lat.spines['left'].set_linewidth(0.7)


# 3. Execution & Layout -------------------------------------------------------
def main():
    tif_paths = [
        "/data/global_mapping/models_V100/Principal_Components/Global_Forest_PC1.tif",
        "/data/global_mapping/models_V100/Principal_Components/Global_Forest_PC2.tif",
        "/data/global_mapping/models_V100/Principal_Components/Global_Forest_PC3.tif"
    ]

    labels = ["a", "b", "c"]
    pc_names = ["PC1", "PC2", "PC3"]

    # Larger canvas + more vertical space between rows to fit colorbar cleanly
    fig = plt.figure(figsize=(9, 11), dpi=600, facecolor='white')

    # hspace increased so colorbars don't overlap next row's panel label
    gs = gridspec.GridSpec(
        3, 2,
        width_ratios=[4, 1.1],
        wspace=0.18,
        hspace=0.55,
        left=0.06, right=0.97, top=0.97, bottom=0.05
    )

    for i in range(3):
        ax_map = fig.add_subplot(gs[i, 0], projection=ccrs.Robinson())
        ax_lat = fig.add_subplot(gs[i, 1])
        is_bottom = (i == 2)
        process_and_plot(tif_paths[i], pc_names[i], labels[i],
                         ax_map, ax_lat, is_bottom_row=is_bottom)

    # 4. Output ---------------------------------------------------------------
    out_dir = "/data/global_mapping/models_V100/Principal_Components/PCA_plot/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "Global_Forest_PCs_Nature.tif")

    plt.savefig(
        out_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.15,
        facecolor="white",
        pil_kwargs={"compression": "tiff_lzw"}
    )
    plt.close()
    print(f"Done! Nature-level figure saved at: {out_path}")


if __name__ == "__main__":
    main()