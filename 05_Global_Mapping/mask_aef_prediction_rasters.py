import os
import rasterio
import numpy as np
from rasterio.warp import reproject, Resampling

def mask_forest_variables():
    # 1. Define Paths
    mask_path = "/data/global_mapping/models_V100/foerst_bound/consensus_forest_bound.tif"
    output_dir = "/data/global_mapping/models_V100/AEF_prodiction/forest_slip/CV"
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # 2. List of target files (Absolute paths)
    target_files = [
        "/data/global_mapping/models_V100/AEF_prodiction/AGB/Global_AGB_CV_Mosaic.tif",
        "/data/global_mapping/models_V100/AEF_prodiction/AGE/Global_AGE_CV_300_Mosaic.tif",
        "/data/global_mapping/models_V100/AEF_prodiction/BA/Global_BA_CV_Mosaic.tif",
        "/data/global_mapping/models_V100/AEF_prodiction/CV_DBH/Global_CV_DBH_CV_Mosaic.tif",
        "/data/global_mapping/models_V100/AEF_prodiction/DBH/Global_DBH_CV_Mosaic.tif",
        "/data/global_mapping/models_V100/AEF_prodiction/H/Global_H_CV_Mosaic.tif",
        "/data/global_mapping/models_V100/AEF_prodiction/shannon/Global_shannon_CV_Mosaic.tif",
        "/data/global_mapping/models_V100/AEF_prodiction/Trees_N/Global_Trees_N_CV_Mosaic.tif",
        "/data/global_mapping/models_V100/AEF_prodiction/V/Global_V_CV_Mosaic.tif"
    ]

    # 3. Open the Forest Boundary mask source
    with rasterio.open(mask_path) as mask_src:
        
        # 4. Process each variable
        for input_path in target_files:
            
            if not os.path.exists(input_path):
                print(f"Warning: {input_path} not found on disk. Skipping.")
                continue
            
            # --- BUG FIX HERE ---
            # Extract just the file name (e.g., 'Global_AGB_MEAN_Mosaic.tif')
            file_name = os.path.basename(input_path)
            # Safely join it with the output directory
            output_path = os.path.join(output_dir, file_name)

            print(f"Processing and aligning: {file_name}...")
            
            with rasterio.open(input_path) as src:
                # Read the variable data
                data = src.read(1)
                meta = src.meta.copy()
                nodata_val = src.nodata if src.nodata is not None else -9999
                
                # Create an empty array to hold the mask reprojected to the variable's grid
                ftc_mask_aligned = np.zeros((src.height, src.width), dtype=mask_src.dtypes[0])
                
                # --- ALIGNMENT LOGIC ---
                reproject(
                    source=rasterio.band(mask_src, 1),
                    destination=ftc_mask_aligned,
                    src_transform=mask_src.transform,
                    src_crs=mask_src.crs,
                    dst_transform=src.transform,
                    dst_crs=src.crs,
                    resampling=Resampling.nearest  # nearest to maintain binary (1/0)
                )
                
                # Ensure data type is float to handle NoData
                data = data.astype(np.float32)
                
                # --- MASKING LOGIC ---
                # Set values to NoData where Forest Boundary is NOT 1
                data[ftc_mask_aligned != 1] = nodata_val
                
                # Update metadata for the new file
                meta.update({
                    "driver": "GTiff",
                    "nodata": nodata_val,
                    "dtype": 'float32',
                    "compress": "lzw"
                })

                # Write the masked raster to the output folder
                with rasterio.open(output_path, "w", **meta) as dst:
                    dst.write(data, 1)
                    
            print(f"Successfully exported to: {output_path}")

    print("\nAll tasks completed.")

if __name__ == "__main__":
    mask_forest_variables()