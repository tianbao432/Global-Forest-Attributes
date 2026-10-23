import os
import pandas as pd
import rasterio
from rasterio.windows import from_bounds
import numpy as np
from osgeo import gdal

# ================= 1. Configuration =================
# Source files root directories
DIR_TR_ROOT = "/data/global_mapping/models_V100/Tradition_mappingA/"
DIR_AEF_ROOT = "/data/global_mapping/models_V100/AEF_mappingA/"

# Path to the comparison report
COMPARISON_CSV = "/data/global_mapping/models_V100/fina_proditionA/all_variables_comparison_rmse.csv"

# Global Forest Consensus Boundary (Mask, Value=1 is forest)
FOREST_MASK = "/data/global_mapping/models_V100/foerst_bound/consensus_forest_bound.tif"

# Output Directories
OUTPUT_DIR = "/data/global_mapping/models_V100/fina_proditionA/"
MASKED_DIR = os.path.join(OUTPUT_DIR, "Masked") 

# Variables to process
TARGET_VARS = ["AGB","AGE", "V", "Trees_N", "shannon", "CV_DBH", "BA",  "DBH", "H"]

# ================= 2. Helper Functions =================

def get_file_lists(variable_name):
    """
    Reads the CSV and returns paths for: 
    1. The 'Best' (Winner) models
    2. The AEF models (used strictly for gap-filling TR missing pixels)
    """
    print(f"Reading comparison report: {COMPARISON_CSV}")
    df = pd.read_csv(COMPARISON_CSV)
    df_var = df[df['Variable'] == variable_name]
    
    best_mean_files, best_cv_files = [], []
    aef_mean_files, aef_cv_files = [], []

    for i in range(1, 13):
        biome_name = f"biome{i}"
        row = df_var[df_var['Biome'] == biome_name]
        if row.empty:
            continue
            
        winner = row['Winner'].values[0]
        root_dir = DIR_AEF_ROOT if winner == "AEF" else DIR_TR_ROOT
        
        # 1. Best Files (Winner)
        best_mean_files.append(os.path.join(root_dir, biome_name, variable_name, "Final_Mosaic", f"Biome{i}_{variable_name}_MEAN_Mosaic.tif"))
        best_cv_files.append(os.path.join(root_dir, biome_name, variable_name, "Final_Mosaic", f"Biome{i}_{variable_name}_CV_Mosaic.tif"))
        
        # 2. AEF Files (Always grab AEF for fallback/gap-filling)
        aef_mean_files.append(os.path.join(DIR_AEF_ROOT, biome_name, variable_name, "Final_Mosaic", f"Biome{i}_{variable_name}_MEAN_Mosaic.tif"))
        aef_cv_files.append(os.path.join(DIR_AEF_ROOT, biome_name, variable_name, "Final_Mosaic", f"Biome{i}_{variable_name}_CV_Mosaic.tif"))

    return best_mean_files, best_cv_files, aef_mean_files, aef_cv_files


def build_vrt(file_list, output_path, description):
    """
    Builds a Virtual Raster (VRT) using GDAL instead of a physical mosaic. 
    """
    print(f"\n[{description}] Building VRT for {len(file_list)} files...")
    valid_files = [fp for fp in file_list if os.path.exists(fp)]
    
    if not valid_files:
        raise FileNotFoundError(f"No valid files found for {description}!")

    vrt_options = gdal.BuildVRTOptions(resampleAlg='nearest')
    vrt = gdal.BuildVRT(output_path, valid_files, options=vrt_options)
    vrt = None # Flush and save to disk
    
    print(f"[{description}] Temp VRT saved to: {output_path}")


def process_gapfill_and_mask(best_path, aef_path, mask_path, out_unmasked_path, out_masked_path):
    """
    FAST full-memory version: Loads 1km global data entirely into RAM 
    for vectorized operations, eliminating I/O bottlenecks.
    """
    print(f"\n[Processing] Starting FAST full-memory gap-filling and masking...")
    print(f"  -> Reference Mask: {mask_path}")

    # 1. Load BEST data entirely into memory
    with rasterio.open(best_path) as src_best:
        print("  -> Loading BEST data into RAM...")
        best_data = src_best.read(1)
        meta = src_best.meta.copy()
        best_bounds = src_best.bounds
        
        nodata = meta.get('nodata', -9999.0)
        is_nodata_nan = np.isnan(nodata) if nodata is not None else False

    # Force the output driver to be GeoTIFF instead of VRT
    meta.update({
        "driver": "GTiff",
        "compress": "lzw",
        "tiled": True,
        "bigtiff": "IF_NEEDED"
    })

    # 2. Load AEF data entirely into memory, aligning to BEST shape
    with rasterio.open(aef_path) as src_aef:
        print("  -> Loading AEF data into RAM (Aligning shapes)...")
        aef_window = from_bounds(*best_bounds, transform=src_aef.transform)
        aef_data = src_aef.read(1, window=aef_window, boundless=True, fill_value=nodata, out_shape=best_data.shape)

    # 3. Load Mask data entirely into memory, aligning to BEST shape
    with rasterio.open(mask_path) as src_mask:
        print("  -> Loading Global Mask into RAM...")
        mask_window = from_bounds(*best_bounds, transform=src_mask.transform)
        mask_data = src_mask.read(1, window=mask_window, boundless=True, fill_value=0, out_shape=best_data.shape)

    print("  -> Performing lightning-fast matrix calculations...")
    
    # Identify missing pixels
    if is_nodata_nan:
        missing_mask = np.isnan(best_data)
    else:
        missing_mask = (best_data == nodata)

    # RULE 1: GAP-FILLING
    fill_condition = (mask_data == 1) & missing_mask
    gap_filled_data = np.where(fill_condition, aef_data, best_data)

    # RULE 2: MASKING
    final_masked_data = np.where(mask_data == 1, gap_filled_data, nodata)

    # Free memory of unused large arrays immediately
    del best_data, aef_data, mask_data, missing_mask, fill_condition

    # 4. Write results to disk
    print("  -> Writing outputs to disk (This may take a minute depending on disk speed)...")
    with rasterio.open(out_unmasked_path, 'w', **meta) as dest_unmasked:
        dest_unmasked.write(gap_filled_data, 1)

    with rasterio.open(out_masked_path, 'w', **meta) as dest_masked:
        dest_masked.write(final_masked_data, 1)

    print(f"  -> [Success] Gap-filled (Unmasked) saved to: {out_unmasked_path}")
    print(f"  -> [Success] Gap-filled (Masked) saved to: {out_masked_path}")


# ================= 3. Main Execution =================

if __name__ == "__main__":
    print("================ Process Start ================")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MASKED_DIR, exist_ok=True)

    for target_var in TARGET_VARS:
        print(f"\n{'='*50}")
        print(f"========== PROCESSING VARIABLE: {target_var} ==========")
        print(f"{'='*50}")

        # 1. Fetch file lists for current variable
        best_mean_files, best_cv_files, aef_mean_files, aef_cv_files = get_file_lists(target_var)

        if not best_mean_files:
            print(f"No files selected for {target_var}. Skipping to next variable.")
            continue

        print("\n-----------------------------------------------")
        
        # 2. Define Output Paths for current variable
        temp_best_mean = os.path.join(OUTPUT_DIR, f"TEMP_BEST_{target_var}_MEAN.vrt")
        temp_aef_mean  = os.path.join(OUTPUT_DIR, f"TEMP_AEF_{target_var}_MEAN.vrt")
        temp_best_cv   = os.path.join(OUTPUT_DIR, f"TEMP_BEST_{target_var}_CV.vrt")
        temp_aef_cv    = os.path.join(OUTPUT_DIR, f"TEMP_AEF_{target_var}_CV.vrt")

        final_mean_unmasked = os.path.join(OUTPUT_DIR, f"{target_var}_MEAN_GapFilled.tif")
        final_mean_masked   = os.path.join(MASKED_DIR, f"{target_var}_MEAN_Masked.tif")
        
        final_cv_unmasked = os.path.join(OUTPUT_DIR, f"{target_var}_CV_GapFilled.tif")
        final_cv_masked   = os.path.join(MASKED_DIR, f"{target_var}_CV_Masked.tif")

        try:
            # ---------------- PROCESSING MEAN ----------------
            print(f"\n>>> PROCESSING MEAN FOR {target_var} <<<")
            build_vrt(best_mean_files, temp_best_mean, f"MEAN: Best Models [{target_var}]")
            build_vrt(aef_mean_files, temp_aef_mean, f"MEAN: AEF Fallback [{target_var}]")
            process_gapfill_and_mask(temp_best_mean, temp_aef_mean, FOREST_MASK, final_mean_unmasked, final_mean_masked)

            # ---------------- PROCESSING CV ----------------
            print(f"\n>>> PROCESSING CV FOR {target_var} <<<")
            build_vrt(best_cv_files, temp_best_cv, f"CV: Best Models [{target_var}]")
            build_vrt(aef_cv_files, temp_aef_cv, f"CV: AEF Fallback [{target_var}]")
            process_gapfill_and_mask(temp_best_cv, temp_aef_cv, FOREST_MASK, final_cv_unmasked, final_cv_masked)

        except Exception as e:
            print(f"\n[ERROR] An error occurred while processing {target_var}: {e}")
            
        finally:
            # 3. Clean up temporary VRT files before moving to the next variable
            print(f"\n[Cleanup] Removing temporary VRT files for {target_var}...")
            for temp_file in [temp_best_mean, temp_aef_mean, temp_best_cv, temp_aef_cv]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)

    print("\n================ All Processes Finished ================")