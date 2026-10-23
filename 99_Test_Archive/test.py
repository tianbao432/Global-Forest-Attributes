import os
import glob
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # For progress bar display

# ==============================================================================
# 1. Path Configuration
# ==============================================================================
input_dir = "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/"
output_dir = os.path.join(input_dir, "plot")

# Create output directory if it does not exist
os.makedirs(output_dir, exist_ok=True)

# Get all tif files (compatible with .tif and .tiff)
tif_files = glob.glob(os.path.join(input_dir, "*.tif*"))
print(f"-> Found {len(tif_files)} raster files to process.")

# ==============================================================================
# 2. Core Plotting Parameters Configuration
# ==============================================================================
cmap_style = 'viridis'  # Common scientific colormap (Nature/Science style, colorblind friendly)
dpi_setting = 300       # Output image resolution
decimation_factor = 10  # Decimation factor. 10 means taking 1 pixel per 10x10 window to prevent OOM

# ==============================================================================
# 3. Batch Loop Processing
# ==============================================================================
for tif_path in tqdm(tif_files, desc="Plotting in progress"):
    filename = os.path.basename(tif_path)
    base_name = os.path.splitext(filename)[0]
    out_png = os.path.join(output_dir, f"{base_name}.png")
    
    # Skip if image already exists (allows resuming after interruption)
    if os.path.exists(out_png):
        continue

    try:
        # Read raster using rasterio
        with rasterio.open(tif_path) as src:
            # Read decimated data to save memory
            data = src.read(1, out_shape=(
                1,
                int(src.height / decimation_factor),
                int(src.width / decimation_factor)
            ))
            nodata_val = src.nodata
            
        # Mask NoData regions
        if nodata_val is not None:
            # Handle float precision issues using np.isclose or range
            if np.isnan(nodata_val):
                data_masked = np.ma.masked_invalid(data)
            else:
                data_masked = np.ma.masked_equal(data, nodata_val)
        else:
            # If NoData is not explicitly defined, mask NaN and common extreme values
            data_masked = np.ma.masked_invalid(data)
            data_masked = np.ma.masked_less(data_masked, -9000)

        # Skip if the entire raster is empty (all masked)
        if data_masked.count() == 0:
            print(f"\n[Warning] {filename} contains only NoData, skipped.")
            continue

        # Calculate 2nd and 98th percentiles for robust contrast stretching
        # Note: np.percentile requires compressed array (excluding masked values)
        valid_data = data_masked.compressed()
        vmin = np.percentile(valid_data, 2)
        vmax = np.percentile(valid_data, 98)
        
        # Add a small margin if vmin == vmax (solid color image)
        if vmin == vmax:
            vmin -= 0.1
            vmax += 0.1

        # ======================================================================
        # 4. Matplotlib Plotting
        # ======================================================================
        # Global maps typically have a ~2:1 aspect ratio
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot raster, dynamically applying current vmin and vmax
        im = ax.imshow(data_masked, cmap=cmap_style, vmin=vmin, vmax=vmax, interpolation='none')
        
        # Add colorbar, pad tightly to the plot
        cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        
        # Aesthetics: remove axis ticks, add filename as title
        ax.axis('off')
        ax.set_title(base_name, fontsize=14, fontweight='bold', pad=15)
        
        # Save and close figure to release memory
        plt.savefig(out_png, dpi=dpi_setting, bbox_inches='tight', transparent=False, facecolor='white')
        plt.close(fig)
        
    except Exception as e:
        print(f"\n[Error] Exception occurred while processing {filename}: {str(e)}")
        plt.close('all') # Ensure figure resources are released on exception

print("\n-> All raster visualization tasks completed! Please check the plot folder for results.")