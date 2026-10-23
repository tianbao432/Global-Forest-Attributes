import xarray as xr
import rioxarray
from rasterio.enums import Resampling

# ====================================================
# Part 1: Process MEAN image
# ====================================================
path_mean ="/data/global_mapping/models_V100/fina_proditionA/AGE_MEAN_GapFilled.tif"
path_ref  = "/data/global_mapping/models_V100/AGE_300/ForestAge_TC000.tif"
output_path = "/data/global_mapping/models_V100/fina_proditionA/AGE_MEAN_GapFilled300.tif"
# 1. Read raster data
# (Note: If global data is too large and causes memory issues, add chunks="auto" to the open_rasterio function to enable Dask out-of-core computation)
r_mean = rioxarray.open_rasterio(path_mean)
r_ref  = rioxarray.open_rasterio(path_ref)

# 2. Resample (Equivalent to R's resample(..., method = "near"))
r_ref_aligned = r_ref.rio.reproject_match(r_mean, resampling=Resampling.nearest)

# 3. Conditional replacement (Equivalent to R's ifel(r_ref_aligned == 300, 300, r_mean))
r_updated = xr.where(r_ref_aligned == 300, 300, r_mean)

# 4. Restore attributes 
# (xr.where drops spatial metadata, so we must manually inherit the CRS and transform from r_mean)
r_updated.rio.write_crs(r_mean.rio.crs, inplace=True)
r_updated.rio.write_transform(r_mean.rio.transform(), inplace=True)
r_updated.rio.update_attrs(r_mean.attrs, inplace=True)

# 5. Export with LZW compression 
# (Equivalent to writeRaster. Note: rioxarray overwrites existing files by default)
r_updated.rio.to_raster(output_path, compress="LZW")

print(f"MEAN image processing complete. Saved to: {output_path}")

# ====================================================
# Part 2: Process CV image
# ====================================================
path_cv ="/data/global_mapping/models_V100/fina_proditionA/AGE_CV_GapFilled.tif"
# path_ref remains unchanged
output_path1 ="/data/global_mapping/models_V100/fina_proditionA/AGE_CV_GapFilled300.tif"

# 1. Read CV image
r_cv = rioxarray.open_rasterio(path_cv)

# 2. Resample Reference image to match the pixel grid of the CV image
r_ref_aligned1 = r_ref.rio.reproject_match(r_cv, resampling=Resampling.nearest)

# 3. Conditional replacement 
# (Equivalent to R's ifel(r_ref_aligned == 300, 0, r_cv). Replaces value with 0 based on original logic)
r_updated1 = xr.where(r_ref_aligned1 == 300, 0, r_cv)

# 4. Restore attributes
r_updated1.rio.write_crs(r_cv.rio.crs, inplace=True)
r_updated1.rio.write_transform(r_cv.rio.transform(), inplace=True)
r_updated1.rio.update_attrs(r_cv.attrs, inplace=True)

# 5. Export and apply LZW compression
r_updated1.rio.to_raster(output_path1, compress="LZW")

print(f"CV image processing complete. Saved to: {output_path1}")