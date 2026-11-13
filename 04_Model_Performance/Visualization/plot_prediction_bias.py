import os
import pandas as pd
import numpy as np
import rasterio

# 1. Define paths and files
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

csv_path = "/data/global_mapping/plotdata_CV/fina_plot_with_RS.csv"
out_dir = "/data/global_mapping/models_V100/bais/"

# Ensure output directory exists
os.makedirs(out_dir, exist_ok=True)

# 2. Map raster filenames to EXACT CSV observation columns
# Updated based on the exact raw column names printed
col_mapping = {
    "V_MEAN_Masked.tif": "V",
    "shannon_MEAN_Masked.tif": "shannon",      # Fixed from Shannon.Index
    "Trees_N_MEAN_Masked.tif": "Trees_N",      # Fixed from N
    "H_MEAN_Masked.tif": "H",
    "DBH_MEAN_Masked.tif": "DBH",
    "CV_DBH_MEAN_Masked.tif": "CV_DBH",
    "BA_MEAN_Masked.tif": "BA",
    "AGE_MEAN_300.tif": "AGE",
    "AGB_MEAN_Masked.tif": "AGB"
}

# 3. Read observation data
print(f"Reading observation data: {csv_path}")
df_plot = pd.read_csv(csv_path)

# Strip hidden whitespaces from column names just in case
df_plot.columns = df_plot.columns.str.strip()

# Extract coordinates as a list of tuples for rasterio
coords = list(zip(df_plot['Longitude'], df_plot['Latitude']))

# List to store summary metrics
summary_metrics = []

# Copy dataframe to store extracted values
df_results = df_plot.copy()

# 4. Loop through each raster file
for tif_path in tif_files:
    filename = os.path.basename(tif_path)
    if filename not in col_mapping:
        print(f"Warning: File {filename} not in mapping dictionary, skipping.")
        continue
        
    obs_col = col_mapping[filename]
    pred_col = f"Pred_{obs_col}"
    
    print(f"Processing: {filename} (Mapped to column: {obs_col})")
    
    # Extract raster values
    sampled_values = []
    with rasterio.open(tif_path) as src:
        nodata_val = src.nodata
        for val in src.sample(coords):
            sampled_values.append(val[0])
            
    # Add predictions to dataframe
    df_results[pred_col] = sampled_values
    
    # Handle NoData values
    if nodata_val is not None:
        df_results[pred_col] = df_results[pred_col].replace(nodata_val, np.nan)
        
    # Calculate bias: filter out rows with NaN in either observed or predicted
    valid_mask = df_results[obs_col].notna() & df_results[pred_col].notna()
    
    y_obs = df_results.loc[valid_mask, obs_col]
    y_pred = df_results.loc[valid_mask, pred_col]
    
    if len(y_obs) == 0:
        print(f"  -> Warning: No valid data pairs found for {filename}!")
        continue
        
    # Bias = Mean(Predicted - Observed)
    bias = np.mean(y_pred - y_obs)
    
    # Relative Bias (%) = (Bias / Mean(Observed)) * 100
    mean_obs = np.mean(y_obs)
    if mean_obs != 0:
        rel_bias = (bias / mean_obs) * 100
    else:
        rel_bias = np.nan
        
    # Record statistics
    summary_metrics.append({
        "Product": filename,
        "Variable": obs_col,
        "Valid_N": len(y_obs),
        "Mean_Observed": mean_obs,
        "Mean_Predicted": np.mean(y_pred),
        "Bias": bias,
        "Relative_Bias_(%)": rel_bias
    })
    
    print(f"  -> Samples: {len(y_obs)} | Bias: {bias:.4f} | Rel Bias: {rel_bias:.2f}%")

# 5. Save results
# Save summary metrics
df_summary = pd.DataFrame(summary_metrics)
summary_out_path = os.path.join(out_dir, "product_bias_summary.csv")
df_summary.to_csv(summary_out_path, index=False)
print(f"\nSummary saved to: {summary_out_path}")

# Save detailed point predictions
details_out_path = os.path.join(out_dir, "plot_predictions_details.csv")
df_results.to_csv(details_out_path, index=False)
print(f"Details saved to: {details_out_path}")