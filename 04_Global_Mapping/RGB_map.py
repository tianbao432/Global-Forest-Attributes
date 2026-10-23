# =============================================================================
# Script 1: Principal Component Data Processing & RGB Synthesis
# Features: PC1-PC3 Robinson projection, 2-98% linear stretch, raster export.
# =============================================================================

import os
import gc
import numpy as np
import xarray as xr
import rioxarray
from rasterio.enums import Resampling
import cartopy.crs as ccrs

# --- 1. Global Parameters Setup ----------------------------------------------
input_dir = "/data/global_mapping/models_V100/Principal_Components/"
output_dir = os.path.join(input_dir, "RGB")
os.makedirs(output_dir, exist_ok=True)

output_data_path = os.path.join(output_dir, "Global_Forest_PCA_RGB_Composite_Data.tif")

# Extent and target CRS
lat_min, lat_max = -60, 85
lon_min, lon_max = -180, 180
target_crs = ccrs.Robinson().proj4_init 

# --- 2. 2%-98% Linear Stretch Function (Using ALL pixels) --------------------
def stretch_band(da_band, band_name):
    print(f"Processing {band_name} Channel...")
    
    vals = da_band.values.ravel()
    nodata_val = da_band.rio.nodata
    
    if nodata_val is not None:
        valid_mask = (vals != nodata_val) & (~np.isnan(vals))
    else:
        valid_mask = ~np.isnan(vals)
        
    valid_vals = vals[valid_mask]
    
    p2, p98 = np.percentile(valid_vals, [2, 98])
    print(f"   > Stretching range: {p2:.4f} to {p98:.4f}")
    
    clamped = np.clip(da_band.values, p2, p98)
    normalized = ((clamped - p2) / (p98 - p2) * 255.0)
    
    mask_2d = valid_mask.reshape(da_band.shape)
    normalized[~mask_2d] = 0
    
    return normalized.astype(np.uint8)

# --- 3. Data Reading, Projection, and Stretching -----------------------------
print("Starting RGB Composite creation...")

pc1 = rioxarray.open_rasterio(os.path.join(input_dir, "Global_Forest_PC1.tif"), masked=True)
pc2 = rioxarray.open_rasterio(os.path.join(input_dir, "Global_Forest_PC2.tif"), masked=True)
pc3 = rioxarray.open_rasterio(os.path.join(input_dir, "Global_Forest_PC3.tif"), masked=True)

for pc in [pc1, pc2, pc3]:
    if pc.rio.crs is None:
        pc.rio.write_crs("EPSG:4326", inplace=True)

print("Cropping data...")
pc1_crop = pc1.rio.clip_box(minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max)
pc2_crop = pc2.rio.clip_box(minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max)
pc3_crop = pc3.rio.clip_box(minx=lon_min, miny=lat_min, maxx=lon_max, maxy=lat_max)

print("Projecting rasters to Robinson coordinate system...")
pc1_rob = pc1_crop.rio.reproject(target_crs, resampling=Resampling.bilinear)
pc2_rob = pc2_crop.rio.reproject(target_crs, resampling=Resampling.bilinear)
pc3_rob = pc3_crop.rio.reproject(target_crs, resampling=Resampling.bilinear)

r_red = stretch_band(pc1_rob, "Red (PC1)")
r_green = stretch_band(pc2_rob, "Green (PC2)")
r_blue = stretch_band(pc3_rob, "Blue (PC3)")

print("Saving Robinson RGB Raster dataset...")
rgb_stack = np.vstack([r_red, r_green, r_blue])

rgb_da = xr.DataArray(
    rgb_stack,
    coords={"band": [1, 2, 3], "y": pc1_rob.y, "x": pc1_rob.x},
    dims=("band", "y", "x")
)
rgb_da.rio.write_crs(target_crs, inplace=True)
rgb_da.rio.write_nodata(0, inplace=True)
rgb_da.rio.to_raster(output_data_path, dtype="uint8", compress="lzw")

print(f"Data saved successfully to: {output_data_path}")