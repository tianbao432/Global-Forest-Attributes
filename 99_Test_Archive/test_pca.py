import os
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.features import rasterize
import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ================= Configuration =================
# Input raster files
tif_files = [
    "/data/global_mapping/models_V100/fina_prodition/Masked/V_MEAN_Masked.tif",
    "/data/global_mapping/models_V100/fina_prodition/Masked/shannon_MEAN_Masked.tif",
    "/data/global_mapping/models_V100/fina_prodition/Masked/Trees_N_MEAN_Masked.tif",
    "/data/global_mapping/models_V100/fina_prodition/Masked/H_MEAN_Masked.tif",
    "/data/global_mapping/models_V100/fina_prodition/Masked/DBH_MEAN_Masked.tif",
    "/data/global_mapping/models_V100/fina_prodition/Masked/CV_DBH_MEAN_Masked.tif",
    "/data/global_mapping/models_V100/fina_prodition/Masked/BA_MEAN_Masked.tif",
    "/data/global_mapping/models_V100/fina_prodition/Masked/AGE_MEAN_300.tif",
    "/data/global_mapping/models_V100/fina_prodition/Masked/AGB_MEAN_Masked.tif",
]

# Variable names corresponding to the files
var_names = ["GSV", "Shannon", "Density", "Height", "DBH", "CV_DBH", "BA", "Age", "AGB"]

# Biome Configuration
biome_shp_path = "/data/global_mapping/union_bio/BIOME.shp"
biome_col = "BIOME"  # <<< IMPORTANT: Change this to the exact column name in your shapefile that holds the Biome ID (1-14)
BIOME_ID_MAP = {
    1: "Tropical Moist", 2: "Tropical Dry", 3: "Tropical Conifer",
    4: "Temperate Broadleaf", 5: "Temperate Conifer", 6: "Boreal Forests",
    7: "Tropical Grasslands", 8: "Temperate Grasslands", 9: "Flooded Grasslands",
    10: "Montane Grasslands", 11: "Tundra", 12: "Mediterranean Forests",
    13: "Deserts", 14: "Mangroves",
}

# Output directory
output_dir = "/data/global_mapping/models_V100/test_pca/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Initializing pipeline (Dynamic Read & Align Mode)...")

# ================= 1. Data Loading & Alignment =================
print(f"Reading reference info from: {os.path.basename(tif_files[0])}")

with rasterio.open(tif_files[0]) as src:
    ref_profile = src.profile.copy()
    ref_shape = src.shape
    ref_crs = src.crs
    ref_transform = src.transform
    ref_width = src.width
    ref_height = src.height
    
    first_layer = src.read(1)
    
    if src.nodata is not None:
        valid_mask = (first_layer != src.nodata) & (~np.isnan(first_layer))
    else:
        valid_mask = ~np.isnan(first_layer)

print(f"Reference shape: {ref_shape}. Loading stacked data...")

stacked_data = np.zeros((len(tif_files), ref_shape[0], ref_shape[1]), dtype=np.float32)
stacked_data[0] = first_layer 

for i, f in enumerate(tif_files):
    if i == 0: continue 
    
    print(f"Reading and aligning: {os.path.basename(f)}")
    with rasterio.open(f) as src:
        vrt_options = {
            'crs': ref_crs,
            'transform': ref_transform,
            'width': ref_width,
            'height': ref_height,
            'resampling': Resampling.nearest
        }
        
        with WarpedVRT(src, **vrt_options) as vrt:
            arr = vrt.read(1)
            
            if vrt.nodata is not None:
                current_mask = (arr != vrt.nodata) & (~np.isnan(arr))
            else:
                current_mask = ~np.isnan(arr)
            
            valid_mask = valid_mask & current_mask
            stacked_data[i] = arr

first_layer = None

# Extract valid pixels
X_valid = stacked_data[:, valid_mask].T
print(f"Total globally valid pixels: {X_valid.shape[0]}")

if X_valid.shape[0] == 0: 
    raise ValueError("Error: No valid pixels found in intersection. Check data overlap.")

# ================= 2. Biome Rasterization =================
print(f"Loading and rasterizing Biome shapefile: {biome_shp_path}")
biomes_gdf = gpd.read_file(biome_shp_path)

# Ensure CRS match
if biomes_gdf.crs != ref_crs:
    print("Reprojecting shapefile to match raster CRS...")
    biomes_gdf = biomes_gdf.to_crs(ref_crs)

# Rasterize geometry to array
shapes = ((geom, value) for geom, value in zip(biomes_gdf.geometry, biomes_gdf[biome_col]))
biome_raster = rasterize(
    shapes=shapes,
    out_shape=ref_shape,
    transform=ref_transform,
    fill=0, # Use 0 for outside/NoData
    dtype=np.int16
)

# Extract biome IDs matching the valid pixel mask
valid_biome_ids = biome_raster[valid_mask]

# ================= 3. Zonal PCA Computation =================
print("Starting Biome-specific PCA...")

# Arrays to hold the global output scores for top 3 PCs
global_pc_scores = {
    0: np.full(X_valid.shape[0], np.nan, dtype=np.float32),
    1: np.full(X_valid.shape[0], np.nan, dtype=np.float32),
    2: np.full(X_valid.shape[0], np.nan, dtype=np.float32)
}

# Iterate over each Biome
for biome_id, biome_name in BIOME_ID_MAP.items():
    print(f"\n--- Processing Biome {biome_id}: {biome_name} ---")
    
    # Filter valid pixels belonging to current biome
    biome_idx = (valid_biome_ids == biome_id)
    X_biome = X_valid[biome_idx]
    
    pixel_count = X_biome.shape[0]
    print(f"Valid pixels in {biome_name}: {pixel_count}")
    
    if pixel_count <= len(tif_files):
        print(f"Warning: Not enough pixels to perform PCA in {biome_name}. Skipping...")
        continue
        
    # Create output directory for this biome's tables
    biome_safe_name = biome_name.replace(" ", "_")
    biome_out_dir = os.path.join(output_dir, f"Biome_{biome_id}_{biome_safe_name}")
    os.makedirs(biome_out_dir, exist_ok=True)
    
    # Run PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_biome)
    
    pca = PCA(n_components=len(tif_files))
    X_pca = pca.fit_transform(X_scaled)
    
    # Map PCA scores back to the global array indices
    for i in range(3):
        global_pc_scores[i][biome_idx] = X_pca[:, i]
        
    # --- Calculate Metrics ---
    eigenvalues = pca.explained_variance_ 
    eigenvectors = pca.components_.T
    loadings = eigenvectors * np.sqrt(eigenvalues)
    exp_var_ratio = pca.explained_variance_ratio_
    cum_var_ratio = np.cumsum(exp_var_ratio)
    contribution = (loadings ** 2) / eigenvalues * 100
    
    # Scale factor for Biplot
    max_score_pc1 = np.percentile(np.abs(X_pca[:, 0]), 99)
    max_score_pc2 = np.percentile(np.abs(X_pca[:, 1]), 99)
    max_score_pc3 = np.percentile(np.abs(X_pca[:, 2]), 99)
    scale_factor = max(max_score_pc1, max_score_pc2, max_score_pc3) 
    scale_factor = 1 if scale_factor == 0 else scale_factor * 0.85
    
    # --- Export Data Tables for this Biome ---
    # 1. Variance Table
    df_var = pd.DataFrame({
        'PC': [f"PC{i+1}" for i in range(len(eigenvalues))],
        'Eigenvalue': eigenvalues,
        'Explained_Variance_Ratio': exp_var_ratio,
        'Cumulative_Variance': cum_var_ratio,
        'Variance_Percent': exp_var_ratio * 100
    })
    df_var.to_csv(os.path.join(biome_out_dir, "PCA_Eigenvalues_Variance.csv"), index=False)
    
    # 2. Loadings Matrix
    df_loadings = pd.DataFrame(loadings, columns=[f"PC{i+1}" for i in range(len(eigenvalues))], index=var_names)
    df_loadings.index.name = 'Variable'
    df_loadings.to_csv(os.path.join(biome_out_dir, "PCA_Loadings_Matrix_Correlations.csv"))
    
    # 3. Variable Contributions
    df_contrib = pd.DataFrame(contribution, columns=[f"PC{i+1}" for i in range(len(eigenvalues))], index=var_names)
    df_contrib.index.name = 'Variable'
    df_contrib.to_csv(os.path.join(biome_out_dir, "PCA_Variable_Contributions.csv"))
    
    # 4. Biplot Data
    biplot_data = []
    for i, var in enumerate(var_names):
        row = {'Variable': var}
        for pc_idx in range(3): 
            pc_name = f"PC{pc_idx+1}"
            row[f'{pc_name}_Loading'] = loadings[i, pc_idx]
            row[f'{pc_name}_Coord_Scaled'] = loadings[i, pc_idx] * scale_factor
            row[f'{pc_name}_Contribution'] = contribution[i, pc_idx]
        biplot_data.append(row)
    pd.DataFrame(biplot_data).to_csv(os.path.join(biome_out_dir, "PCA_Variable_Biplot_Data.csv"), index=False)
    
    # 5. Summary Text Report
    with open(os.path.join(biome_out_dir, "PCA_Summary_Report.txt"), "w") as f:
        f.write(f"PCA Analysis Summary - Biome {biome_id}: {biome_name}\n")
        f.write("==================================================\n\n")
        f.write("1. Eigenvalues & Variance:\n")
        f.write(df_var[['PC', 'Eigenvalue', 'Variance_Percent', 'Cumulative_Variance']].round(4).to_string(index=False))
        f.write("\n\n2. Top Contributing Variables:\n")
        for i in range(3):
            pc = f"PC{i+1}"
            top_vars = pd.DataFrame(biplot_data).nlargest(3, f'{pc}_Contribution')[['Variable', f'{pc}_Loading', f'{pc}_Contribution']]
            f.write(f"\n--- {pc} (Explains {df_var.loc[i, 'Variance_Percent']:.2f}%) ---\n")
            f.write(top_vars.round(4).to_string(index=False))
            f.write("\n")

print(f"\nAll biome tables saved to subdirectories in: {output_dir}")

# ================= 4. Export Rasters =================
print("Exporting top 3 Principal Components (Global merged Rasters)...")

ref_profile.update({
    'dtype': 'float32',
    'count': 1,
    'nodata': -9999,
    'compress': 'lzw' 
})

for pc_idx in range(3):
    pc_name = f"Global_Forest_PC{pc_idx+1}_Zonal"
    output_path = os.path.join(output_dir, f"{pc_name}.tif")
    
    out_img = np.full(ref_shape, -9999, dtype=np.float32)
    
    # Retrieve the scores for this PC. Replace NaNs (from skipped biomes) with -9999
    valid_scores = global_pc_scores[pc_idx]
    valid_scores[np.isnan(valid_scores)] = -9999
    
    out_img[valid_mask] = valid_scores
    
    with rasterio.open(output_path, 'w', **ref_profile) as dst:
        dst.write(out_img, 1)
    
    print(f"Saved merged zonal raster: {output_path}")

print("Processing complete.")