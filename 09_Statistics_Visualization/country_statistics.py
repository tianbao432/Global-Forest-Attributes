import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.warp import reproject, Resampling
from rasterstats import zonal_stats
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ================= Configuration =================

CONFIG = {
    # Input Raster Files
    'tif_files': [
        "/data/global_mapping/models_V100/fina_proditionA/Masked/V_MEAN_Masked.tif",  # Index 0 will be the Reference Grid
        "/data/global_mapping/models_V100/fina_proditionA/Masked/shannon_MEAN_Masked.tif",
        "/data/global_mapping/models_V100/fina_proditionA/Masked/Trees_N_MEAN_Masked.tif",
        "/data/global_mapping/models_V100/fina_proditionA/Masked/H_MEAN_Masked.tif",
        "/data/global_mapping/models_V100/fina_proditionA/Masked/DBH_MEAN_Masked.tif",
        "/data/global_mapping/models_V100/fina_proditionA/Masked/CV_DBH_MEAN_Masked.tif",
        "/data/global_mapping/models_V100/fina_proditionA/Masked/BA_MEAN_Masked.tif",
        "/data/global_mapping/models_V100/fina_proditionA/Masked/AGE_MEAN_300.tif",
        "/data/global_mapping/models_V100/fina_proditionA/Masked/AGB_MEAN_Masked.tif"
    ],

    # Variable Names
    'var_names': [
        "Vol", "Shannon", "N_Density", "Height", 
        "DBH", "CV_DBH", "BA", "Age", "AGB"
    ],

    # Variables requiring "Total" calculation (Stock/Quantity metrics)
    'vars_to_sum': ["Vol", "N_Density", "BA", "AGB"],

    # WWF Biome ID to Name Mapping
    'BIOME_ID_MAP': {
        1: "Tropical Moist",
        2: "Tropical Dry",
        3: "Tropical Conifer",
        4: "Temperate Broadleaf",
        5: "Temperate Conifer",
        6: "Boreal Forests",
        7: "Tropical Grasslands",
        8: "Temperate Grasslands",
        9: "Flooded Grasslands",
        10: "Montane Grasslands",
        11: "Tundra",
        12: "Mediterranean Forests",
        13: "Deserts",
        14: "Mangroves"
    },

    # Shapefile Configurations (Zonal Stats)
    'shapefiles': [
        {
            'path':"/data/global_mapping/union_bio/BIOME.shp", 
            'id_column': "BIOME", 
            'tag': "Biome"
        },
        {
            'path':"/data/global_mapping/drive_forest/global_shp/GADM_global_shp.shp", 
            'id_column': "COUNTRY", 
            'tag': "Country"
        }
    ],

    # Output Directory
    'output_dir': "/data/global_mapping/models_V100/Partition_statistics/"
}

# Ensure output directory exists
if not os.path.exists(CONFIG['output_dir']):
    os.makedirs(CONFIG['output_dir'])

# ================= Helper Function: Raster Alignment =================

def align_rasters(input_tifs, out_dir):
    """
    Aligns all input rasters to the exact spatial grid (transform, shape, crs) 
    of the first raster in the list. This prevents NA mismatches due to micro-shifts.
    """
    print("\n>>> Phase 0: Spatial Raster Alignment...")
    
    aligned_dir = os.path.join(out_dir, "Aligned_Rasters")
    os.makedirs(aligned_dir, exist_ok=True)
    
    aligned_tifs = []
    
    # Define the reference raster (the first one in the list)
    ref_path = input_tifs[0]
    print(f"   [Reference Grid]: {os.path.basename(ref_path)}")
    
    with rasterio.open(ref_path) as ref_src:
        ref_transform = ref_src.transform
        ref_crs = ref_src.crs
        ref_width = ref_src.width
        ref_height = ref_src.height
        
    for tif in input_tifs:
        basename = os.path.basename(tif)
        out_path = os.path.join(aligned_dir, f"Aligned_{basename}")
        aligned_tifs.append(out_path)
        
        # Skip if already aligned
        if os.path.exists(out_path):
            print(f"   [Skipping] Already aligned: {basename}")
            continue
            
        print(f"   [Aligning] Processing: {basename}...")
        
        with rasterio.open(tif) as src:
            # Copy metadata from source, but update with reference spatial parameters
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': ref_crs,
                'transform': ref_transform,
                'width': ref_width,
                'height': ref_height
            })
            
            with rasterio.open(out_path, 'w', **kwargs) as dst:
                # Nearest neighbor resampling is used to strictly preserve NoData boundaries
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=ref_transform,
                    dst_crs=ref_crs,
                    resampling=Resampling.nearest
                )
                
    print("   [Success] All rasters are perfectly aligned.")
    return aligned_tifs

# ================= Helper Function: Calculate Pixel Area =================

def get_pixel_area_ha(src):
    """
    Calculates the real area (in hectares) of pixels for each row based on latitude.
    Solves the distortion problem in WGS84 where high-latitude pixels are smaller.
    Returns: A numpy array of shape (height, 1) for broadcasting.
    """
    transform = src.transform
    height = src.height
    
    # Resolution in degrees
    res_x = abs(transform.a) 
    res_y = abs(transform.e)
    
    # 1. Calculate Latitudes of row centers
    row_indices = np.arange(height)
    latitudes = transform.f + transform.e * (row_indices + 0.5)
    
    # 2. Calculate Area
    # Earth Radius ~ 6371 km
    deg_to_km_lat = 111.320
    deg_to_km_lon = 111.320 * np.cos(np.radians(latitudes))
    
    # Area in km2 = (Height_km) * (Width_km)
    area_km2 = (res_y * deg_to_km_lat) * (res_x * deg_to_km_lon)
    
    # Convert to Hectares (1 km2 = 100 ha)
    area_ha = area_km2 * 100.0
    
    # Reshape to (height, 1) so we can multiply it with the raster data (height, width)
    return area_ha.reshape(-1, 1)

# ================= Function 1: Zonal Statistics (Shapefile based) =================

def process_zonal_stats(shp_info, tif_files, var_names, vars_to_sum, out_dir):
    
    tag = shp_info['tag']
    shp_path = shp_info['path']
    id_col = shp_info['id_column']
    
    print(f"\n>>> Phase 1: Processing Zonal Stats - {tag}")
    
    # 1. Load Shapefile
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as e:
        print(f"Error loading shapefile: {e}")
        return

    # Initialize results DataFrame with just the ID column
    results_df = gdf[[id_col]].copy()

    # 2. Iterate through Aligned Rasters
    for tif_path, var_name in zip(tif_files, var_names):
        print(f"   Calculating: {var_name}...")
        
        try:
            with rasterio.open(tif_path) as src:
                nodata = src.nodata
                affine = src.transform
                
                # Check if we need to calculate Total (Stock)
                is_sum_var = var_name in vars_to_sum
                
                # Reprojection Logic
                if gdf.crs != src.crs:
                    gdf_proj = gdf.to_crs(src.crs)
                else:
                    gdf_proj = gdf

                # --- 1. Calculate MEAN (Density) ---
                # all_touched=True ensures small polygons/countries capture border pixels
                stats_mean = zonal_stats(
                    vectors=gdf_proj, 
                    raster=tif_path, 
                    stats=['mean'], 
                    nodata=nodata,
                    all_touched=True 
                )
                results_df[f"{var_name}_Mean"] = [x['mean'] if x['mean'] is not None else np.nan for x in stats_mean]
                
                # --- 2. Calculate TOTAL (Stock) ---
                if is_sum_var:
                    data_arr = src.read(1, masked=True)
                    area_grid = get_pixel_area_ha(src)
                    total_stock_arr = data_arr * area_grid
                    
                    stats_sum = zonal_stats(
                        vectors=gdf_proj,
                        raster=total_stock_arr.filled(np.nan), 
                        affine=affine,
                        stats=['sum'],
                        nodata=np.nan,
                        all_touched=True 
                    )
                    results_df[f"{var_name}_Total"] = [x['sum'] if x['sum'] is not None else np.nan for x in stats_sum]

        except Exception as e:
            print(f"      Error processing {var_name}: {e}")
            results_df[f"{var_name}_Mean"] = -9999
            if var_name in vars_to_sum:
                results_df[f"{var_name}_Total"] = -9999

    # 3. Handle NAs
    results_df = results_df.fillna(-9999)

    # 4. Map Biome Names and Reorder Columns
    if tag == "Biome" and id_col in results_df.columns:
        results_df['BIOME_NAME'] = results_df[id_col].map(CONFIG['BIOME_ID_MAP'])
        
        cols = list(results_df.columns)
        cols.insert(1, cols.pop(cols.index('BIOME_NAME')))
        results_df = results_df[cols]

    # 5. Save to CSV
    output_filename = f"Statistics_Py_{tag}.csv"
    output_path = os.path.join(out_dir, output_filename)
    
    results_df.to_csv(output_path, index=False)
    print(f"   Saved: {output_path}")

# ================= Function 2: Global Statistics (Whole Raster) =================

def process_global_stats(tif_files, var_names, vars_to_sum, out_dir):
    
    print("\n>>> Phase 2: Processing Global Statistics...")
    
    global_data = {}
    
    for tif_path, var_name in zip(tif_files, var_names):
        print(f"   Calculating Global: {var_name}")
        
        try:
            with rasterio.open(tif_path) as src:
                arr = src.read(1, masked=True)
                area_grid = get_pixel_area_ha(src)
                
                # Get valid mask (exclude NoData)
                valid_mask = ~arr.mask 
                valid_data = arr.data[valid_mask]
                
                # Broadcast area matrix to match array shape and extract valid pixel areas
                area_grid_2d = np.broadcast_to(area_grid, arr.shape)
                valid_area = area_grid_2d[valid_mask]
                
                # --- True Spatial Area-Weighted Mean ---
                if np.sum(valid_area) > 0:
                    g_mean_weighted = np.sum(valid_data * valid_area) / np.sum(valid_area)
                else:
                    g_mean_weighted = np.nan
                    
                global_data[f"{var_name}_Mean"] = float(g_mean_weighted) if not np.isnan(g_mean_weighted) else -9999
                
                # --- Calculate Global Total (Stock) ---
                if var_name in vars_to_sum:
                    g_sum = np.sum(valid_data * valid_area)
                    global_data[f"{var_name}_Total"] = float(g_sum)
                    
        except Exception as e:
            print(f"      Error processing Global {var_name}: {e}")
            global_data[f"{var_name}_Mean"] = -9999
            if var_name in vars_to_sum:
                global_data[f"{var_name}_Total"] = -9999
    
    results_df = pd.DataFrame([global_data])
    results_df = results_df.fillna(-9999)
    
    output_path = os.path.join(out_dir, "Statistics_Py_Global.csv")
    results_df.to_csv(output_path, index=False)
    print(f"   Saved Global Stats: {output_path}")

# ================= Main Execution =================

if __name__ == "__main__":
    
    # 0. Align Rasters first
    aligned_tif_files = align_rasters(
        input_tifs=CONFIG['tif_files'], 
        out_dir=CONFIG['output_dir']
    )
    
    # 1. Run Zonal Statistics using ALIGNED rasters
    for shp_config in CONFIG['shapefiles']:
        process_zonal_stats(
            shp_info=shp_config,
            tif_files=aligned_tif_files,
            var_names=CONFIG['var_names'],
            vars_to_sum=CONFIG['vars_to_sum'],
            out_dir=CONFIG['output_dir']
        )
        
    # 2. Run Global Statistics using ALIGNED rasters
    process_global_stats(
        tif_files=aligned_tif_files,
        var_names=CONFIG['var_names'],
        vars_to_sum=CONFIG['vars_to_sum'],
        out_dir=CONFIG['output_dir']
    )

    print("\n[Success] All processing complete.")