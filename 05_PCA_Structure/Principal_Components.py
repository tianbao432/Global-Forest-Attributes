import os
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ================= Configuration =================
# Input raster files
tif_files = [
    "/data/global_mapping/models_V100/fina_proditionA/Masked/V_MEAN_Masked.tif",
    "/data/global_mapping/models_V100/fina_proditionA/Masked/shannon_MEAN_Masked.tif",
    "/data/global_mapping/models_V100/fina_proditionA/Masked/Trees_N_MEAN_Masked.tif",
    "/data/global_mapping/models_V100/fina_proditionA/Masked/H_MEAN_Masked.tif",
    "/data/global_mapping/models_V100/fina_proditionA/Masked/DBH_MEAN_Masked.tif",
    "/data/global_mapping/models_V100/fina_proditionA/Masked/CV_DBH_MEAN_Masked.tif",
    "/data/global_mapping/models_V100/fina_proditionA/Masked/BA_MEAN_Masked.tif",
    "/data/global_mapping/models_V100/fina_proditionA/Masked/AGE_MEAN_300.tif",
    "/data/global_mapping/models_V100/fina_proditionA/Masked/AGB_MEAN_Masked.tif",
]

# Variable names corresponding to the files
var_names = ["GSV", "Shannon", "Density", "Height", "DBH", "CV_DBH", "BA", "Age", "AGB"]

# Output directory
output_dir ="/data/global_mapping/models_V100/Principal_Components/"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("Initializing pipeline (Dynamic Read & Align Mode)...")

# ================= 1. Data Loading =================
# --- Step 1: Establish Reference & Pre-allocate Memory ---
print(f"Reading reference info from: {os.path.basename(tif_files[0])}")

with rasterio.open(tif_files[0]) as src:
    ref_profile = src.profile.copy()
    ref_shape = src.shape
    ref_crs = src.crs             # Capture CRS for alignment
    ref_transform = src.transform # Capture Transform for alignment
    ref_width = src.width         # Capture Width
    ref_height = src.height       # Capture Height
    
    # Read first layer to initialize valid mask
    first_layer = src.read(1)
    
    # Create initial valid mask (Not NoData and Not NaN)
    if src.nodata is not None:
        valid_mask = (first_layer != src.nodata) & (~np.isnan(first_layer))
    else:
        valid_mask = ~np.isnan(first_layer)

print(f"Reference shape: {ref_shape}. Loading stacked data...")

# Pre-allocate memory (Bands, Rows, Cols) to prevent memory fragmentation
stacked_data = np.zeros((len(tif_files), ref_shape[0], ref_shape[1]), dtype=np.float32)
stacked_data[0] = first_layer 

# --- Step 2: Read Remaining Files ---
for i, f in enumerate(tif_files):
    if i == 0: continue # Skip first file (already loaded)
    
    print(f"Reading and aligning: {os.path.basename(f)}")
    with rasterio.open(f) as src:
        # Define how the data should be aligned to match the reference
        vrt_options = {
            'crs': ref_crs,
            'transform': ref_transform,
            'width': ref_width,
            'height': ref_height,
            'resampling': Resampling.nearest # Nearest neighbor resampling
        }
        
        # Direct read with on-the-fly alignment
        with WarpedVRT(src, **vrt_options) as vrt:
            arr = vrt.read(1)
            
            # Update Mask (Intersection of valid pixels across all layers)
            if vrt.nodata is not None:
                current_mask = (arr != vrt.nodata) & (~np.isnan(arr))
            else:
                current_mask = ~np.isnan(arr)
            
            valid_mask = valid_mask & current_mask
            stacked_data[i] = arr

# Release the temporary first_layer to free memory
first_layer = None

# Extract valid pixels for PCA -> Shape: (Samples, Features)
X_valid = stacked_data[:, valid_mask].T
print(f"Total valid pixels for PCA: {X_valid.shape[0]}")

if X_valid.shape[0] == 0: 
    raise ValueError("Error: No valid pixels found in intersection. Check data overlap.")

# ================= 2. PCA Computation =================
print("Running PCA...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_valid)

pca = PCA(n_components=len(tif_files))
X_pca = pca.fit_transform(X_scaled)

# ================= 3. Metrics Calculation =================
print("Calculating Standard PCA Metrics...")

# A. Eigenvalues (Variance explained by each PC)
eigenvalues = pca.explained_variance_ 

# B. Eigenvectors (Unit vectors)
eigenvectors = pca.components_.T # Shape: (n_vars, n_pcs)

# C. Standard Loadings (Correlation Coefficients)
# Formula: Eigenvector * sqrt(Eigenvalue)
# This represents the correlation between the original variable and the PC
loadings = eigenvectors * np.sqrt(eigenvalues)

# D. Variance Ratios
exp_var_ratio = pca.explained_variance_ratio_
cum_var_ratio = np.cumsum(exp_var_ratio)

# E. Variable Contributions (%)
# Formula: (Loading^2 / Eigenvalue) * 100
contribution = (loadings ** 2) / eigenvalues * 100

# F. Scaling Factor for Biplot Visualization
# Calculates a scaling factor to map scores to the same space as loadings
max_score_pc1 = np.percentile(np.abs(X_pca[:, 0]), 99)
max_score_pc2 = np.percentile(np.abs(X_pca[:, 1]), 99)
max_score_pc3 = np.percentile(np.abs(X_pca[:, 2]), 99)

scale_factor = max(max_score_pc1, max_score_pc2, max_score_pc3) 
if scale_factor == 0: scale_factor = 1
scale_factor = scale_factor * 0.85

# ================= 4. Export Data Tables =================
print("Exporting Analysis Tables...")

# --- A. Variance Table ---
df_var = pd.DataFrame({
    'PC': [f"PC{i+1}" for i in range(len(eigenvalues))],
    'Eigenvalue': eigenvalues,
    'Explained_Variance_Ratio': exp_var_ratio,
    'Cumulative_Variance': cum_var_ratio,
    'Variance_Percent': exp_var_ratio * 100
})
df_var.to_csv(os.path.join(output_dir, "PCA_Eigenvalues_Variance.csv"), index=False)

# --- B. Loadings Matrix (Correlations) ---
df_loadings = pd.DataFrame(loadings, columns=[f"PC{i+1}" for i in range(len(eigenvalues))], index=var_names)
df_loadings.index.name = 'Variable'
df_loadings.to_csv(os.path.join(output_dir, "PCA_Loadings_Matrix_Correlations.csv"))

# --- C. Variable Contributions ---
df_contrib = pd.DataFrame(contribution, columns=[f"PC{i+1}" for i in range(len(eigenvalues))], index=var_names)
df_contrib.index.name = 'Variable'
df_contrib.to_csv(os.path.join(output_dir, "PCA_Variable_Contributions.csv"))

# --- D. Biplot Data (Merged for plotting) ---
biplot_data = []
for i, var in enumerate(var_names):
    row = {'Variable': var}
    # For Top 3 PCs
    for pc_idx in range(3): 
        pc_name = f"PC{pc_idx+1}"
        row[f'{pc_name}_Loading'] = loadings[i, pc_idx]
        row[f'{pc_name}_Coord_Scaled'] = loadings[i, pc_idx] * scale_factor
        row[f'{pc_name}_Contribution'] = contribution[i, pc_idx]
    biplot_data.append(row)

df_biplot = pd.DataFrame(biplot_data)
df_biplot.to_csv(os.path.join(output_dir, "PCA_Variable_Biplot_Data.csv"), index=False)

# --- E. Summary Text Report ---
report_path = os.path.join(output_dir, "PCA_Summary_Report.txt")
with open(report_path, "w") as f:
    f.write("PCA Analysis Summary Report\n")
    f.write("===========================\n\n")
    f.write("1. Eigenvalues & Variance:\n")
    f.write(df_var[['PC', 'Eigenvalue', 'Variance_Percent', 'Cumulative_Variance']].round(4).to_string(index=False))
    f.write("\n\n2. Top Contributing Variables:\n")
    
    for i in range(3):
        pc = f"PC{i+1}"
        top_vars = df_biplot.nlargest(3, f'{pc}_Contribution')[['Variable', f'{pc}_Loading', f'{pc}_Contribution']]
        f.write(f"\n--- {pc} (Explains {df_var.loc[i, 'Variance_Percent']:.2f}%) ---\n")
        f.write(top_vars.round(4).to_string(index=False))
        f.write("\n")

print(f"Tables saved to: {output_dir}")

# ================= 5. Export Rasters =================
print("Exporting top 3 Principal Components (Rasters)...")

# Update profile for output
# Use LZW compression to save disk space
ref_profile.update({
    'dtype': 'float32',
    'count': 1,
    'nodata': -9999,
    'compress': 'lzw' 
})

for i in range(3):
    pc_name = f"Global_Forest_PC{i+1}"
    output_path = os.path.join(output_dir, f"{pc_name}.tif")
    
    # Initialize array with NoData
    out_img = np.full(ref_shape, -9999, dtype=np.float32)
    # Map PCA scores back to spatial grid
    out_img[valid_mask] = X_pca[:, i]
    
    with rasterio.open(output_path, 'w', **ref_profile) as dst:
        dst.write(out_img, 1)
    
    print(f"Saved raster: {output_path}")

print("Processing complete.")