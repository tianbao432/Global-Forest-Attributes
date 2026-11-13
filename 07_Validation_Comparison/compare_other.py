# -*- coding: utf-8 -*-
import os
import warnings
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import stats

# Silence unharmful runtime warnings for clean server logs
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Strict Nature-style typography configurations (Arial)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Arial'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'

output_dir = "/data/global_mapping/models_V100/compare_other_study/result/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Coordinated Top-Journal Palette Definitions
COLOR_OURS = '#A60F2B'       # Deep Crimson / Madder Red
COLOR_REF = '#1A4263'        # Prussian Blue / Deep Ocean
COLOR_TREND = '#2B6CB0'      # Sophisticated Cobalt Blue for Scatter Trendline
COLOR_TEXT_MAIN = '#1A202C'  # Charcoal Black for clean text
CMAP_DENSITY = 'magma'       # Perceptually uniform high-end sequential map

# Managed configurations for all 4 variables (Ordered sequentially for rows)
configs = {
    'AGB': {
        'path_ours': "/data/global_mapping/models_V100/fina_proditionA/Masked/AGB_MEAN_Masked.tif",
        'path_ref': "/data/global_mapping/models_V100/compare_other_study/CC_AGB.tif",
        'xlabel': "Reference AGB (Mg ha$^{-1}$)",
        'ylabel': "Ours AGB (Mg ha$^{-1}$)",
        'proflabel': "AGB (Mg ha$^{-1}$)"
    }
   
    ,
    'Age': {
        'path_ours': "/data/global_mapping/models_V100/fina_proditionA/Masked/AGE_MEAN_300.tif",
        'path_ref': "/data/global_mapping/models_V100/compare_other_study/ForestAge_TC000.tif",
        'xlabel': "Reference Age (years)",
        'ylabel': "Ours Age (years)",
        'proflabel': "Age (years)"
    },
    'Height': {
        'path_ours': "/data/global_mapping/models_V100/fina_proditionA/Masked/H_MEAN_Masked.tif",
        'path_ref': "/data/global_mapping/models_V100/compare_other_study/ETH_GlobalCanopyHeight_2020_1km_Bilinear.tif",
        'xlabel': "Reference Height (m)",
        'ylabel': "Ours Height (m)",
        'proflabel': "Height (m)"
    } ,'Density': {
        'path_ours': "/data/global_mapping/models_V100/fina_proditionA/Masked/Trees_N_MEAN_Masked.tif",
        'path_ref': "/data/global_mapping/models_V100/compare_other_study/Crowther_Density_Stems_per_Ha.tif",
        'xlabel': "Reference Density (stems ha$^{-1}$)",
        'ylabel': "Ours Density (stems ha$^{-1}$)",
        'proflabel': "Density (stems ha$^{-1}$)"
    }
}

def apply_nature_style(ax):
    # Clean framework layout adhering to standard graphic rules
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.spines['left'].set_color(COLOR_TEXT_MAIN)
    ax.spines['bottom'].set_color(COLOR_TEXT_MAIN)
    ax.xaxis.set_tick_params(width=0.8, length=4.5, direction='out', labelsize=10.5, labelcolor=COLOR_TEXT_MAIN)
    ax.yaxis.set_tick_params(width=0.8, length=4.5, direction='out', labelsize=10.5, labelcolor=COLOR_TEXT_MAIN)
    ax.title.set_fontweight('normal')
    ax.xaxis.label.set_fontweight('normal')
    ax.xaxis.label.set_fontsize(11.5)
    ax.xaxis.label.set_color(COLOR_TEXT_MAIN)
    ax.yaxis.label.set_fontweight('normal')
    ax.yaxis.label.set_fontsize(11.5)
    ax.yaxis.label.set_color(COLOR_TEXT_MAIN)

# Initialize a large single canvas for 4 rows x 2 columns grid
fig, axs = plt.subplots(4, 2, figsize=(9, 12))

for row_idx, (var_name, cfg) in enumerate(configs.items()):
    print(f">>> Processing row {row_idx + 1}/4: [{var_name}]")
    
    # Define active axes for the current product row
    ax_scatter = axs[row_idx, 0]
    ax_profile = axs[row_idx, 1]
    
    # Load Geospatial Layers
    with rasterio.open(cfg['path_ref']) as src_ref:
        ref_arr = src_ref.read(1).astype(np.float32)
        ref_meta = src_ref.meta.copy()
        ref_transform = src_ref.transform
        ref_nodata = src_ref.nodata
        
    with rasterio.open(cfg['path_ours']) as src_ours:
        ours_arr_raw = src_ours.read(1).astype(np.float32)
        ours_transform = src_ours.transform
        ours_crs = src_ours.crs
        ours_nodata = src_ours.nodata

    # On-the-fly Bilinear Resampling if geometries mismatch
    if ref_arr.shape != ours_arr_raw.shape:
        print(f"    -> Resampling 'Ours' layer to match 'Reference' geometry...")
        ours_arr = np.full(ref_arr.shape, np.nan, dtype=np.float32)
        reproject(
            source=ours_arr_raw,
            destination=ours_arr,
            src_transform=ours_transform,
            src_crs=ours_crs,
            dst_transform=ref_transform,
            dst_crs=ref_meta['crs'],
            resampling=Resampling.bilinear,
            src_nodata=ours_nodata,
            dst_nodata=np.nan
        )
    else:
        ours_arr = ours_arr_raw
        if ours_nodata is not None:
            ours_arr[ours_arr == ours_nodata] = np.nan

    if ref_nodata is not None:
        ref_arr[ref_arr == ref_nodata] = np.nan

    # Create solid unified mask for 100% data coverage
    valid_mask = np.isfinite(ref_arr) & np.isfinite(ours_arr) & (ref_arr >= 0) & (ours_arr >= 0)
    
    ref_valid = ref_arr[valid_mask]
    ours_valid = ours_arr[valid_mask]
    diff_valid = ours_valid - ref_valid

    print(f"    -> Valid pixels extracted: {len(ref_valid)}")

    # -------------------------------------------------------------------------
    # Column 1: 2D Density Scatter Plot
    # -------------------------------------------------------------------------
    print("    -> Plotting Scatter Column...")
    min_val = min(ref_valid.min(), ours_valid.min())
    max_val = max(ref_valid.max(), ours_valid.max())
    
    counts, xedges, yedges, im = ax_scatter.hist2d(
        ref_valid, ours_valid, bins=260, cmap=CMAP_DENSITY, norm=LogNorm(), cmin=1
    )
    ax_scatter.plot([min_val, max_val], [min_val, max_val], color='#718096', linestyle='--', linewidth=0.8)
    
    slope, intercept, r_value, p_value, std_err = stats.linregress(ref_valid, ours_valid)
    x_line = np.array([ref_valid.min(), ref_valid.max()])
    y_line = slope * x_line + intercept
    ax_scatter.plot(x_line, y_line, color=COLOR_TREND, linewidth=1.4)
    
    # Combined Metrics Output (Both R and Mean Bias inside white bbox)
    mean_bias = np.mean(diff_valid)
    stats_text = f"$R$ = {r_value:.2f}\nBias = {mean_bias:.2f}"
    bbox_props = dict(boxstyle="round,pad=0.35", fc="#ffffff", ec="#e2e8f0", lw=0.6, alpha=0.9)
    ax_scatter.text(0.06, 0.94, stats_text, transform=ax_scatter.transAxes, fontsize=11.5,
                    verticalalignment='top', fontweight='normal', bbox=bbox_props, color=COLOR_TEXT_MAIN)
             
    ax_scatter.set_xlabel(cfg['xlabel'])
    ax_scatter.set_ylabel(cfg['ylabel'])
    apply_nature_style(ax_scatter)

    # -------------------------------------------------------------------------
    # Column 2: Latitudinal Profile with Confidence / Variability Bands
    # -------------------------------------------------------------------------
    print("    -> Plotting Latitudinal Profile Column with Error Bands...")
    ref_masked = np.where(valid_mask, ref_arr, np.nan)
    ours_masked = np.where(valid_mask, ours_arr, np.nan)
    
    # Calculate means and standard deviations along the longitude axis (axis=1)
    with np.errstate(all='ignore'):
        lat_mean_ref = np.nanmean(ref_masked, axis=1)
        lat_mean_ours = np.nanmean(ours_masked, axis=1)
        lat_std_ref = np.nanstd(ref_masked, axis=1)
        lat_std_ours = np.nanstd(ours_masked, axis=1)
        
    rows = np.arange(ref_arr.shape[0])
    lats = ref_transform[5] + rows * ref_transform[4]
    
    # Clip lower bounds to 0 to prevent unphysical negative values in the shadow bands
    lower_bound_ours = np.clip(lat_mean_ours - lat_std_ours, 0, None)
    upper_bound_ours = lat_mean_ours + lat_std_ours
    lower_bound_ref = np.clip(lat_mean_ref - lat_std_ref, 0, None)
    upper_bound_ref = lat_mean_ref + lat_std_ref
    
    # Plot shaded areas for spatial variability (zorder=1 to keep them in background)
    ax_profile.fill_betweenx(lats, lower_bound_ours, upper_bound_ours, 
                             color=COLOR_OURS, alpha=0.15, edgecolor='none', zorder=1)
    ax_profile.fill_betweenx(lats, lower_bound_ref, upper_bound_ref, 
                             color=COLOR_REF, alpha=0.15, edgecolor='none', zorder=1)
    
    # Plot central mean lines (zorder=2 to overlay clearly on top of shadows)
    ax_profile.plot(lat_mean_ours, lats, color=COLOR_OURS, linewidth=1.3, label='Ours', zorder=2)
    ax_profile.plot(lat_mean_ref, lats, color=COLOR_REF, linewidth=1.3, label='Reference', zorder=2)
    
    ax_profile.set_ylabel("Latitude ($^\\circ$)")
    ax_profile.set_xlabel(cfg['proflabel'])
    ax_profile.legend(loc='upper right', frameon=False, fontsize=10, labelcolor=COLOR_TEXT_MAIN)
    apply_nature_style(ax_profile)

    # -------------------------------------------------------------------------
    # Sequential Panel Tags (a through h across the grid) - Strict Bold
    # -------------------------------------------------------------------------
    tag_left = chr(97 + row_idx * 2)       # Generates: a, c, e, g
    tag_right = chr(97 + row_idx * 2 + 1)  # Generates: b, d, f, h
    
    ax_scatter.text(-0.14, 1.05, tag_left, transform=ax_scatter.transAxes, fontsize=17, fontweight='bold', va='top', ha='right', color='#000000')
    ax_profile.text(-0.14, 1.05, tag_right, transform=ax_profile.transAxes, fontsize=17, fontweight='bold', va='top', ha='right', color='#000000')
    
    print(f">>> Finished subplots [{tag_left}] and [{tag_right}] for [{var_name}]\n")

# Global tight layout adjustment before saving
plt.tight_layout()

# Save the consolidated multi-panel large plate
final_out_pref = os.path.join(output_dir, "All_Products_Combined_Plate")
print(f">>> Exporting ultimate combined figures to: {final_out_pref}")
plt.savefig(f"{final_out_pref}.pdf", dpi=600, bbox_inches='tight')
plt.savefig(f"{final_out_pref}.tif", dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})

plt.close()
print("\n======================================================")
print("ALL RUNS GENERATED AND CONSOLIDATED SUCCESSFULLY!")
print("======================================================")