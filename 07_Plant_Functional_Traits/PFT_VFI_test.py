import os
import numpy as np
import pandas as pd
import rasterio
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
import warnings

# Suppress runtime warnings from statsmodels during high-collinearity inversion
warnings.filterwarnings("ignore", category=RuntimeWarning)

CONFIG = {
    "target_tif_path": "/data/global_mapping/models_V100/Principal_Components/Global_Forest_PC1.tif",
    "biome_shp_path":  "/data/global_mapping/union_bio/BIOME.shp",
    "output_root":     "/data/global_mapping/models_V100/Plant_Functional_Traits/VIF_test",
    
    # 0.25 degree sampling grid
    "grid_resolution_deg": 0.25,
    
    # VIF threshold (Strictly drop > 10)
    "vif_threshold": 10,
    
    # Target column name in the output CSV
    "target_name": "PC_target",
    
    # Random seed
    "seed": 42,
    
    "BIOME_ID_MAP": {
        1: "Tropical Moist", 2: "Tropical Dry", 3: "Tropical Conifer",
        4: "Temperate Broadleaf", 5: "Temperate Conifer", 6: "Boreal Forests",
        7: "Tropical Grasslands", 8: "Temperate Grasslands", 9: "Flooded Grasslands",
        10: "Montane Grasslands", 11: "Tundra", 12: "Mediterranean Forests",
        13: "Deserts", 14: "Mangroves",
    },
    
    "driver_mapping": {
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/WD.tif": "WD",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SLA.tif": "SLA",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/LDMC.tif": "LDMC",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/LNC.tif": "LNC",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/LPC.tif": "LPC",

    }
}

def extract_data_from_grid(config):
    """
    Extracts data from the target and driver rasters using a regular point grid.
    This avoids loading massive global 1km rasters entirely into memory.
    """
    print("--- Starting Data Extraction ---")
    target_path = config["target_tif_path"]
    res = config["grid_resolution_deg"]
    drivers = config["driver_mapping"]

    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Target file not found: {target_path}")

    # 1. Create a spatial grid based on the target raster properties
    print(f"Generating sample grid at {res} degree resolution...")
    with rasterio.open(target_path) as src:
        transform = src.transform
        width = src.width
        height = src.height
        nodata = src.nodata

        # Calculate step size in pixels based on the desired geographic resolution
        pixel_size_x = abs(transform[0])
        pixel_size_y = abs(transform[4])
        step_x = max(1, int(res / pixel_size_x))
        step_y = max(1, int(res / pixel_size_y))

        cols = np.arange(0, width, step_x)
        rows = np.arange(0, height, step_y)
        cc, rr = np.meshgrid(cols, rows)

        # Convert pixel coordinates to geographic coordinates
        x_coords, y_coords = rasterio.transform.xy(transform, rr.flatten(), cc.flatten())
        all_coords = list(zip(x_coords, y_coords))

        # Sample the target raster first to identify valid pixels (non-NoData)
        print("Sampling target raster to filter valid locations...")
        target_samples = list(src.sample(all_coords))
        target_values = np.array([v[0] for v in target_samples])

        if nodata is not None:
            if np.isnan(nodata):
                valid_mask = ~np.isnan(target_values)
            else:
                valid_mask = (target_values != nodata)
        else:
            valid_mask = ~np.isnan(target_values)

        valid_coords = [all_coords[i] for i in range(len(all_coords)) if valid_mask[i]]
        valid_targets = target_values[valid_mask]

    print(f"Retained {len(valid_coords)} valid points. Extracting driver variables...")

    # 2. Extract driver values at valid coordinates
    df = pd.DataFrame({
        "Lon": [c[0] for c in valid_coords],
        "Lat": [c[1] for c in valid_coords],
        config["target_name"]: valid_targets
    })

    for tif_path, col_name in drivers.items():
        if not os.path.exists(tif_path):
            print(f"  [Warning] Driver not found, skipping {col_name}: {tif_path}")
            continue
        
        print(f"  Extracting {col_name}...")
        with rasterio.open(tif_path) as src:
            samples = list(src.sample(valid_coords))
            driver_nodata = src.nodata
            
            vals = []
            for v in samples:
                val = v[0]
                if driver_nodata is not None and val == driver_nodata:
                    vals.append(np.nan)
                else:
                    vals.append(val)
            df[col_name] = vals

    # Drop any row containing NaN values to ensure a clean matrix for VIF calculation
    initial_rows = len(df)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    print(f"Dropped {initial_rows - len(df)} points containing NoData in drivers.")
    print(f"Final dataset shape for VIF analysis: {df.shape}")

    return df

def iterative_vif_selection(df, features, threshold):
    """
    Iteratively calculates VIF and drops the feature with the highest VIF 
    until all remaining features are below the strict threshold.
    """
    print(f"\n--- Starting Iterative VIF Selection (Threshold = {threshold}) ---")
    selected_features = list(features)

    while True:
        # add_constant is required by statsmodels to calculate VIF correctly
        X = df[selected_features]
        X_const = add_constant(X)
        
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X_const.columns
        
        # Calculate VIF for each feature
        vif_data["VIF"] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]
        
        # Exclude the constant term from our threshold logic
        vif_features = vif_data[vif_data["Variable"] != "const"]
        max_vif = vif_features["VIF"].max()
        
        if max_vif > threshold:
            # Drop the worst offender
            max_feature = vif_features.loc[vif_features["VIF"] == max_vif, "Variable"].iloc[0]
            print(f"[-] Dropping '{max_feature}' (VIF = {max_vif:.2f})")
            selected_features.remove(max_feature)
        else:
            print("[+] All remaining variables are below the VIF threshold.")
            break

    final_vif = vif_data[vif_data["Variable"] != "const"].reset_index(drop=True)
    
    # Sort for cleaner output
    final_vif = final_vif.sort_values(by="VIF", ascending=False).reset_index(drop=True)
    return final_vif, selected_features

def main():
    np.random.seed(CONFIG["seed"])
    os.makedirs(CONFIG["output_root"], exist_ok=True)

    # 1. Extract the aligned dataset
    df = extract_data_from_grid(CONFIG)
    raw_csv_path = os.path.join(CONFIG["output_root"], "sampled_raw_data.csv")
    df.to_csv(raw_csv_path, index=False)
    print(f"Saved raw extraction data to: {raw_csv_path}")

    # 2. Run Iterative VIF Diagnosis
    driver_cols = [v for k, v in CONFIG["driver_mapping"].items() if v in df.columns]
    final_vif_df, selected_features = iterative_vif_selection(
        df=df,
        features=driver_cols,
        threshold=CONFIG["vif_threshold"]
    )

    # 3. Export the Final Result
    output_csv = os.path.join(CONFIG["output_root"], "final_selected_variables_vif.csv")
    final_vif_df.to_csv(output_csv, index=False)

    print(f"\n--- Final Selected Variables ({len(selected_features)}) ---")
    print(final_vif_df.to_string())
    print(f"\nSuccessfully saved the final variable list and VIF scores to: {output_csv}")

if __name__ == "__main__":
    main()