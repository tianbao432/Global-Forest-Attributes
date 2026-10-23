# -*- coding: utf-8 -*-
"""
AEF vs TRA (Tradition2) Validation Results Aggregator
This script automatically extracts and aggregates Metrics.csv from AEF and TRA models,
compiling R, RMSE, MAE, and comparison results into a single CSV.
"""

import pandas as pd
import os

# ================= 1. Configuration =================
# Base directories for model results
DIR_AEF = "/data/global_mapping/models_V100/AEF_A/"
DIR_TRA = "/data/global_mapping/models_V100/TraditionC/"

# Output directory for the aggregated results
OUTPUT_DIR = "/data/global_mapping/models_V100/fina_proditionA/"

# Target variables list
VARIABLES_LIST = [
    "AGE", "BA", "Trees_N", "AGB", "V", 
    "shannon", "CV_DBH", "DBH", "H"
]

# Biome IDs 1 to 12
BIOMES = list(range(1, 13))

# Create output directory if it does not exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= 2. Main Execution =================
print("Starting aggregation for AEF vs TRA metrics...")
print("-" * 80)

all_records = []

# Define validation strategies and types to extract
strategies = ["Random", "KMeans"]
model_types = ["Single", "Ensemble"]

for var_name in VARIABLES_LIST:
    print(f"Processing Variable: {var_name}...")
    for b_id in BIOMES:
        # Construct file paths for Metrics.csv
        aef_csv = os.path.join(DIR_AEF, f"biome{b_id}", var_name, f"B{b_id}_Metrics.csv")
        tra_csv = os.path.join(DIR_TRA, f"biome{b_id}", var_name, f"B{b_id}_Metrics.csv")

        # Read CSV if it exists, else set to None
        df_aef = pd.read_csv(aef_csv) if os.path.exists(aef_csv) else None
        df_tra = pd.read_csv(tra_csv) if os.path.exists(tra_csv) else None

        # Iterate through all combinations (e.g., Random-Single, KMeans-Ensemble)
        for strat in strategies:
            for typ in model_types:
                record = {
                    "Variable": var_name,
                    "Biome": f"Biome_{b_id}",
                    "Strategy": strat,
                    "Type": typ,
                    "Status": "Success"
                }

                # --- Extract AEF Metrics ---
                if df_aef is not None and not df_aef.empty:
                    aef_row = df_aef[(df_aef['Strategy'] == strat) & (df_aef['Type'] == typ)]
                    if not aef_row.empty:
                        record["R_AEF"] = aef_row["R"].values[0]
                        record["RMSE_AEF"] = aef_row["RMSE"].values[0]
                        record["MAE_AEF"] = aef_row["MAE"].values[0]
                    else:
                        record["R_AEF"] = record["RMSE_AEF"] = record["MAE_AEF"] = None
                else:
                    record["R_AEF"] = record["RMSE_AEF"] = record["MAE_AEF"] = None

                # --- Extract TRA Metrics ---
                if df_tra is not None and not df_tra.empty:
                    tra_row = df_tra[(df_tra['Strategy'] == strat) & (df_tra['Type'] == typ)]
                    if not tra_row.empty:
                        record["R_TRA"] = tra_row["R"].values[0]
                        record["RMSE_TRA"] = tra_row["RMSE"].values[0]
                        record["MAE_TRA"] = tra_row["MAE"].values[0]
                    else:
                        record["R_TRA"] = record["RMSE_TRA"] = record["MAE_TRA"] = None
                else:
                    record["R_TRA"] = record["RMSE_TRA"] = record["MAE_TRA"] = None

                # --- Calculate comparison and winner (based on RMSE, lower is better) ---
                if pd.notna(record["RMSE_AEF"]) and pd.notna(record["RMSE_TRA"]):
                    # Difference: If > 0, AEF has a lower error (improvement)
                    diff = record["RMSE_TRA"] - record["RMSE_AEF"]
                    record["RMSE_Improvement (TRA-AEF)"] = diff
                    
                    if diff > 0:
                        record["Winner"] = "AEF"
                    elif diff < 0:
                        record["Winner"] = "TRA"
                    else:
                        record["Winner"] = "Tie"
                else:
                    record["RMSE_Improvement (TRA-AEF)"] = None
                    record["Winner"] = "Missing Data"
                    record["Status"] = "Missing Data"

                all_records.append(record)

# ================= 3. Export & Summary =================
# Convert to DataFrame
df_final = pd.DataFrame(all_records)

# Reorder columns for better readability
columns_order = [
    "Variable", "Biome", "Strategy", "Type", 
    "RMSE_AEF", "RMSE_TRA", "RMSE_Improvement (TRA-AEF)", "Winner",
    "R_AEF", "R_TRA", "MAE_AEF", "MAE_TRA", "Status"
]
df_final = df_final[columns_order]

# Save to CSV
output_csv_path = os.path.join(OUTPUT_DIR, "AEF_vs_TRA_Full_Metrics_Comparison.csv")
df_final.to_csv(output_csv_path, index=False)

print("-" * 80)
print("Aggregation complete!")
print(f"Detailed report saved to: {output_csv_path}")
print("\n=== Global Summary by Validation Type (Win Counts based on RMSE) ===")

# Simple statistical summary
summary_subset = df_final[df_final["Status"] == "Success"]
if not summary_subset.empty:
    pivot = pd.crosstab(
        index=[summary_subset['Strategy'], summary_subset['Type']], 
        columns=summary_subset['Winner'],
        margins=True
    )
    print(pivot)
else:
    print("No valid comparison data found. Please check your input paths.")