import os
import numpy as np
from osgeo import gdal

# Input paths
boundary_path = "/data/global_mapping/models_V100/foerst_bound/consensus_forest_bound.tif"
param_paths = [
    "/data/global_mapping/models_V100/Tradition_prodictionA/V/Global_V_MEAN_Mosaic.tif",
    "/data/global_mapping/models_V100/Tradition_prodictionA/Trees_N/Global_Trees_N_MEAN_Mosaic.tif",
    "/data/global_mapping/models_V100/Tradition_prodictionA/shannon/Global_shannon_MEAN_Mosaic.tif",
    "/data/global_mapping/models_V100/Tradition_prodictionA/H/Global_H_MEAN_Mosaic.tif",
    "/data/global_mapping/models_V100/Tradition_prodictionA/DBH/Global_DBH_MEAN_Mosaic.tif",
    "/data/global_mapping/models_V100/Tradition_prodictionA/CV_DBH/Global_CV_DBH_MEAN_Mosaic.tif",
    "/data/global_mapping/models_V100/Tradition_prodictionA/BA/Global_BA_MEAN_Mosaic.tif",
    "/data/global_mapping/models_V100/Tradition_prodictionA/AGE/Global_AGE_MEAN_300.tif",
    "/data/global_mapping/models_V100/Tradition_prodictionA/AGB/Global_AGB_MEAN_Mosaic.tif",
]
out_dir = "/data/global_mapping/models_V100/test_pca/forest/"

os.makedirs(out_dir, exist_ok=True)

# Tile size for block-wise processing
BLOCK_SIZE = 2048

for param_path in param_paths:
    print(f"Processing: {param_path}")

    # Open parameter dataset
    param_ds = gdal.Open(param_path, gdal.GA_ReadOnly)
    if param_ds is None:
        print(f"  ERROR: Cannot open {param_path}")
        continue

    param_proj = param_ds.GetProjection()
    param_gt = param_ds.GetGeoTransform()
    param_xsize = param_ds.RasterXSize
    param_ysize = param_ds.RasterYSize
    param_band = param_ds.GetRasterBand(1)
    param_nodata = param_band.GetNoDataValue()
    param_dtype = param_band.DataType

    # Define output nodata value if the parameter file has no nodata set
    if param_nodata is None:
        param_nodata = -9999.0 if "Float" in gdal.GetDataTypeName(param_dtype) else -9999

    # Compute output bounds from geotransform (assumes north-up, rotation ignored)
    min_x = param_gt[0]
    max_x = param_gt[0] + param_xsize * param_gt[1] + param_ysize * param_gt[2]
    max_y = param_gt[3]
    min_y = param_gt[3] + param_xsize * param_gt[4] + param_ysize * param_gt[5]

    # Warp forest boundary to exactly match the parameter grid (in-memory)
    boundary_ds = gdal.Open(boundary_path, gdal.GA_ReadOnly)
    if boundary_ds is None:
        print(f"  ERROR: Cannot open boundary file")
        param_ds = None
        continue

    warp_opts = gdal.WarpOptions(
        format='MEM',
        dstSRS=param_proj,
        outputBounds=[min_x, min_y, max_x, max_y],
        width=param_xsize,
        height=param_ysize,
        resampleAlg='near',
        dstNodata=0
    )
    boundary_warped = gdal.Warp('', boundary_ds, options=warp_opts)
    if boundary_warped is None:
        print("  ERROR: Warping boundary failed")
        param_ds = None
        boundary_ds = None
        continue
    boundary_band = boundary_warped.GetRasterBand(1)

    # Create output file (GeoTIFF, tiled, compressed)
    out_name = os.path.basename(param_path)
    out_path = os.path.join(out_dir, out_name)
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(
        out_path,
        param_xsize, param_ysize, 1,
        param_dtype,
        options=['COMPRESS=LZW', 'TILED=YES', 'BIGTIFF=IF_SAFER']
    )
    out_ds.SetProjection(param_proj)
    out_ds.SetGeoTransform(param_gt)
    out_band = out_ds.GetRasterBand(1)
    out_band.SetNoDataValue(param_nodata)

    # Process in blocks
    for y_off in range(0, param_ysize, BLOCK_SIZE):
        y_size = min(BLOCK_SIZE, param_ysize - y_off)
        for x_off in range(0, param_xsize, BLOCK_SIZE):
            x_size = min(BLOCK_SIZE, param_xsize - x_off)

            # Read parameter data
            param_data = param_band.ReadAsArray(x_off, y_off, x_size, y_size)
            # Read aligned boundary data
            bound_data = boundary_band.ReadAsArray(x_off, y_off, x_size, y_size)

            if param_data is None or bound_data is None:
                continue

            # Build combined mask: keep original valid pixels that are within forest boundary (value == 1)
            valid_param = (param_data != param_nodata) if param_nodata is not None else np.ones_like(param_data, dtype=bool)
            forest_mask = (bound_data == 1)
            final_mask = valid_param & forest_mask

            masked = np.where(final_mask, param_data, param_nodata)

            # Write block
            out_band.WriteArray(masked, x_off, y_off)

    # Cleanup
    out_band.FlushCache()
    out_ds = None
    boundary_warped = None
    param_ds = None
    boundary_ds = None

    print(f"  -> saved to {out_path}")

print("All files processed.")