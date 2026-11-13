import os
import glob
import numpy as np
import rasterio
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

forest_x = None
forest_y = None
total_forest_pixels = 0

def process_single_raster(file):
    try:
        with rasterio.open(file) as src:
            data = src.read(1)
            nodata_val = src.nodata
            
            cols, rows = ~src.transform * (forest_x, forest_y)
            cols = np.floor(cols).astype(int)
            rows = np.floor(rows).astype(int)
            
            valid_mask = (cols >= 0) & (cols < src.width) & (rows >= 0) & (rows < src.height)
            
            sampled_values = np.full(total_forest_pixels, np.nan)
            sampled_values[valid_mask] = data[rows[valid_mask], cols[valid_mask]]
            
            if nodata_val is not None:
                if np.isnan(nodata_val):
                    missing_count = np.isnan(sampled_values).sum()
                else:
                    missing_count = (sampled_values == nodata_val).sum() + np.isnan(sampled_values).sum()
            else:
                missing_count = np.isnan(sampled_values).sum()
                
            missing_rate = missing_count / total_forest_pixels
            return os.path.basename(file), missing_count, missing_rate, None
            
    except Exception as e:
        return os.path.basename(file), None, None, str(e)

if __name__ == '__main__':
    raster_folder = "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/"
    mask_path = "/data/global_mapping/models_V100/foerst_bound/consensus_forest_bound.tif"
    
    raster_files = glob.glob(os.path.join(raster_folder, "*.tif"))
    if not raster_files:
        raise ValueError(f"No .tif files found in {raster_folder}")

    print("Loading forest mask and extracting coordinates...")
    with rasterio.open(mask_path) as src_mask:
        mask_data = src_mask.read(1)
        mask_transform = src_mask.transform

    forest_rows, forest_cols = np.where(mask_data == 1)
    total_forest_pixels = len(forest_rows)
    print(f"Total forest pixels identified: {total_forest_pixels:,}")

    forest_x, forest_y = mask_transform * (forest_cols, forest_rows)
    
    del mask_data, forest_rows, forest_cols

    results = []
    max_workers = min(16, (os.cpu_count() or 1) * 2)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_raster, file): file for file in raster_files}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing Rasters"):
            fname, m_count, m_rate, err = future.result()
            if err is None:
                results.append((fname, m_count, m_rate))
            else:
                print(f"\nError reading {fname}: {err}")

    df = pd.DataFrame(results, columns=["Raster_Name", "Missing_Pixel_Count", "Missing_Rate"])
    df = df.sort_values(by="Missing_Rate", ascending=False)

    print("\n" + "="*60)
    print("Top 10 Rasters with Highest Missing Rates in Forest Areas:")
    print("="*60)
    print(df.head(10).to_string(formatters={'Missing_Rate': '{:.4%}'.format}, index=False))

    output_csv = os.path.join(raster_folder, "forest_missing_rates.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nFull results explicitly saved to: {output_csv}")