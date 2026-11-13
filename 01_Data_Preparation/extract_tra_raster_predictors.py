# ==============================================================================
# Extract Raster Predictor Variables to Plot Locations (Multiprocessing Version)
# ==============================================================================
import os
import glob
import re
import gc
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import warnings

# Ignore np.nanmean warnings for all-NaN slices
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")

# ==============================================================================
# 1. Settings & Paths
# ==============================================================================
raster_dir       = "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/"
csv_path         = "/data/global_mapping/models_V100/Complete/fina_plot.csv"

# Output paths
output_path      = "/data/global_mapping/plotdata_CV_tr/fina_plot_with_RS_traditional.csv"
biome_out_path   = "/data/global_mapping/models_V100/Complete/fina_biome_count_traditional.csv"
log_path         = "/data/global_mapping/models_V100/Complete/extract_log.txt"
summary_path     = "/data/global_mapping/models_V100/Complete/fina_plot_with_tradition_summary.txt"

# Buffer pixel radius for NA filling (equivalent to BUFFER_FACTOR = 3)
# Radius of 3 means extending 3 pixels outward from the center, i.e., a 7x7 window
BUFFER_PIXELS = 3  

# Multiprocessing setup: Use available CPU cores (leave 1-2 free to prevent system freeze)
NUM_WORKERS = max(1, os.cpu_count() - 2)

# ==============================================================================
# 2. Worker Function (For Multiprocessing)
# ==============================================================================
def process_single_raster(args):
    """
    Worker function to extract data from a single raster and handle NA buffer filling.
    """
    file_path, coords_list = args
    results = {}
    
    # Generate valid column names (remove non-alphanumeric characters)
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    base_name = re.sub(r'\W|^(?=\d)', '_', base_name)
    
    try:
        with rasterio.open(file_path) as src:
            nodata = src.nodata
            
            # Use rasterio.sample to directly extract raw values for all points
            # Returns shape (n_points, n_bands)
            samples = np.array(list(src.sample(coords_list)))
            
            # Convert defined NoData to np.nan
            if nodata is not None:
                samples = np.where(samples == nodata, np.nan, samples)
                
            n_bands = src.count
            for b in range(n_bands):
                band_idx = b + 1
                
                # Naming logic
                if n_bands == 1:
                    col_name = base_name
                else:
                    col_name = f"{base_name}_b{band_idx:02d}"
                    
                vals = samples[:, b].astype(np.float64)
                
                # Handle NA values (Buffered mean)
                na_indices = np.where(np.isnan(vals))[0]
                if len(na_indices) > 0:
                    for idx in na_indices:
                        lon, lat = coords_list[idx]
                        
                        try:
                            # Convert coordinates to matrix indices
                            row, col = src.index(lon, lat)
                            
                            # Construct a 7x7 reading window, boundless=True handles edge overflow
                            window = Window(col - BUFFER_PIXELS, row - BUFFER_PIXELS, 
                                            BUFFER_PIXELS * 2 + 1, BUFFER_PIXELS * 2 + 1)
                            
                            w_data = src.read(band_idx, window=window, boundless=True)
                            
                            if nodata is not None:
                                w_data = np.where(w_data == nodata, np.nan, w_data)
                            
                            # Calculate mean of valid values in the 7x7 window, ignoring nans
                            mean_val = np.nanmean(w_data)
                            vals[idx] = mean_val
                        except Exception:
                            # Keep nan if index is out of bounds or an exception occurs
                            pass
                            
                results[col_name] = vals

        return {"status": "success", "file": file_path, "data": results}
        
    except Exception as e:
        return {"status": "error", "file": file_path, "error": str(e)}

# ==============================================================================
# 3. Main Execution Block
# ==============================================================================
if __name__ == '__main__':
    print(f">>> Loading plot data from {csv_path}...")
    plot_df = pd.read_csv(csv_path, encoding='utf-8', encoding_errors='ignore')
    print(f"    {len(plot_df)} plots loaded.")

    # Prepare coordinate list [(lon1, lat1), (lon2, lat2), ...]
    coords_list = list(zip(plot_df['Longitude'], plot_df['Latitude']))

    # Get all tif files
    tif_files = glob.glob(os.path.join(raster_dir, "**/*.tif"), recursive=True)
    # Also support uppercase .TIF
    tif_files += glob.glob(os.path.join(raster_dir, "**/*.TIF"), recursive=True)
    tif_files = list(set(tif_files))
    
    print(f">>> Found {len(tif_files)} raster files in {raster_dir}")
    if len(tif_files) == 0:
        raise ValueError("No .tif files found. Please check the raster_dir path.")

    # Build multiprocessing task arguments
    tasks = [(f, coords_list) for f in tif_files]
    
    extracted_data = {}
    error_log = []
    
    print(f"\n>>> Starting parallel extraction with {NUM_WORKERS} workers...")
    
    # Start process pool
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Wrap with tqdm for progress bar
        futures = {executor.submit(process_single_raster, task): task[0] for task in tasks}
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            res = future.result()
            if res["status"] == "success":
                for col, vals in res["data"].items():
                    # Handle duplicate column names
                    final_col = col
                    suffix = 1
                    while final_col in extracted_data:
                        final_col = f"{col}_{suffix}"
                        suffix += 1
                    extracted_data[final_col] = vals
            else:
                err_msg = f"[ERROR] {res['file']}: {res['error']}"
                error_log.append(err_msg)

    # Free large memory variables like coordinates
    del coords_list
    gc.collect()

    # Output error log
    if error_log:
        with open(log_path, 'w') as f:
            for line in error_log:
                f.write(line + "\n")
        print(f"\n[WARNING] {len(error_log)} errors logged to {log_path}")

    # ==============================================================================
    # 4. Merge and Clean Data
    # ==============================================================================
    print("\n>>> Merging extracted data into dataframe...")
    extracted_df = pd.DataFrame(extracted_data)
    predictor_cols = list(extracted_df.columns)
    
    # Merge original DataFrame and extracted DataFrame
    final_df = pd.concat([plot_df, extracted_df], axis=1)
    
    # Free temporary DF to save memory
    del extracted_df, extracted_data
    gc.collect()

    print(f">>> {len(predictor_cols)} predictor columns extracted from {len(tif_files)} rasters.")

    # Remove rows with any NA/NaN predictors
    print(">>> Filtering rows with any NA/NaN predictors...")
    initial_rows = len(final_df)
    final_df_clean = final_df.dropna(subset=predictor_cols)
    removed_rows = initial_rows - len(final_df_clean)
    print(f"    Removed {removed_rows} rows ({initial_rows} -> {len(final_df_clean)})")

    # Ensure output directories exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    os.makedirs(os.path.dirname(biome_out_path), exist_ok=True)

    # Export complete CSV
    final_df_clean.to_csv(output_path, index=False)
    print(f">>> Saved cleaned data to {output_path}")

    # Write Summary to text file
    with open(summary_path, 'w') as f:
        # describe() output is wide, convert to string for saving
        f.write(final_df_clean.describe().to_string())
    print(f">>> Summary written to {summary_path}")

    # ==============================================================================
    # 5. Biome-level Sample Count
    # ==============================================================================
    target_vars = ["DBH", "H", "AGE", "BA", "N", "AGB", "V", "Shannon.Index", "CV_DBH"]
    valid_vars = [v for v in target_vars if v in final_df_clean.columns]

    if "BIOME" in final_df_clean.columns and valid_vars:
        # Calculate total samples for each BIOME and non-NA counts for valid variables
        agg_funcs = {'BIOME': 'size'}  # Total_Samples
        agg_funcs.update({v: 'count' for v in valid_vars}) # count only tallies non-NA values
        
        biome_counts = final_df_clean.groupby('BIOME').agg(agg_funcs).rename(columns={'BIOME': 'Total_Samples'})
        biome_counts = biome_counts.rename(columns={v: f"Count_{v}" for v in valid_vars})
        biome_counts.reset_index(inplace=True)
        
        biome_counts.to_csv(biome_out_path, index=False)
        print(f">>> Biome counts saved to {biome_out_path}")
    else:
        print(">>> Skipped biome counting (BIOME column or target vars missing)")

    print("\n>>> Done. All processes completed successfully.")