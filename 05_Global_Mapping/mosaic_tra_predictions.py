import os
import rasterio
from rasterio.merge import merge

# 1. Define variables and statistics to iterate over
variables = ["H", "DBH", "AGE", "BA", "Trees_N", "AGB", "V", "shannon", "CV_DBH"]
stats = ["MEAN", "CV"]
biomes = range(1, 13)

base_input_dir ="/data/global_mapping/models_V100/Tradition_mappingA/"
base_output_dir ="/data/global_mapping/models_V100/Tradition_prodictionA/"

# 2. Loop through each variable and each stat (MEAN and CV)
for var in variables:
    for stat in stats:
        print(f"\nProcessing Global Mosaic for [{var}] - [{stat}]...")
        
        # Build the list of 12 biome input files for the current var/stat
        input_files = [
            os.path.join(base_input_dir, f"biome{i}", var, "Final_Mosaic", f"Biome{i}_{var}_{stat}_Mosaic.tif")
            for i in biomes
        ]
        
        # Create output directory for the current variable
        output_dir = os.path.join(base_output_dir, var)
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"Global_{var}_{stat}_Mosaic.tif")
        
        # Skip if the global mosaic already exists (resume protection)
        if os.path.exists(output_file):
            print(f"  -> Already exists: {os.path.basename(output_file)}. Skipping.")
            continue
            
        # 3. Open valid source files
        src_files_to_mosaic = []
        for fp in input_files:
            if os.path.exists(fp):
                src = rasterio.open(fp)
                src_files_to_mosaic.append(src)
            else:
                print(f"  -> Warning: File not found: {fp}")
                
        # 4. Merge and save
        if src_files_to_mosaic:
            try:
                # Use the nodata value from the first file to ensure empty spaces are handled correctly
                nodata_val = src_files_to_mosaic[0].nodata
                
                # Perform the merge
                mosaic, out_trans = merge(src_files_to_mosaic, nodata=nodata_val)
                
                # Copy and update metadata
                out_meta = src_files_to_mosaic[0].meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans,
                    "compress": "lzw",
                    "nodata": nodata_val,
                    "bigtiff": "YES",    # Crucial for huge global files
                    "tiled": True,       # Highly recommended for fast reading in GEE or QGIS
                    "blockxsize": 256,
                    "blockysize": 256
                })
                
                # Write to disk
                with rasterio.open(output_file, "w", **out_meta) as dest:
                    dest.write(mosaic)
                    
                print(f"  -> Mosaic successfully saved to: {output_file}")
                
            except Exception as e:
                print(f"  -> Error occurred while mosaicking {var} {stat}: {e}")
                
            finally:
                # Ensure all opened files are safely closed
                for src in src_files_to_mosaic:
                    src.close()
        else:
            print(f"  -> No input files were loaded for {var} {stat}. Process aborted.")

print("\nAll mosaicking tasks completed successfully.")