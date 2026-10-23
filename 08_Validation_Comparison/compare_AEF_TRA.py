import pandas as pd
import os

# ================= 1. Configuration =================
# Root directory for Traditional Remote Sensing (TR) results
DIR_TR = "/data/global_mapping/models_V100/TraditionC/"

# Root directory for AEF (V5) results
DIR_AEF = "/data/global_mapping/models_V100/AEF_A/"

# Output directory for the CSV report
OUTPUT_DIR = "/data/global_mapping/models_V100/fina_proditionA/"

# List of variables to compare
# Note: Ensure these names match the folder names exactly (case-sensitive)
VARIABLES_LIST = [
    "V", "Trees_N", "shannon", 
    "CV_DBH", "BA", "AGE", "AGB", "DBH", "H"
]
#"H", "DBH", 
# Create output directory if it does not exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# ================= 2. Helper Function =================
def get_kmeans_ensemble_rmse(csv_path):
    """
    Extracts the RMSE value for the 'KMeans' strategy with 'Ensemble' type 
    from the specified Metrics.csv file.
    """
    # Check if file exists
    if not os.path.exists(csv_path):
        return None, "File Not Found"
    
    try:
        # Read the CSV file
        df = pd.read_csv(csv_path)
        
        # Filter condition: Strategy="KMeans" AND Type="Ensemble"
        target_row = df[
            (df['Strategy'] == 'KMeans') & 
            (df['Type'] == 'Ensemble')
        ]
        
        # Check if the row exists
        if not target_row.empty:
            # Extract the 'RMSE' value
            return float(target_row['RMSE'].values[0]), "Success"
        else:
            return None, "Row Not Found"
            
    except Exception as e:
        return None, str(e)

# ================= 3. Main Execution Loop =================

print("Starting comparison based on RMSE (Lower is Better)...")
print("-" * 80)

all_results = []

# Loop through each variable
for var_name in VARIABLES_LIST:
    print(f"Processing Variable: {var_name}...")
    
    # Loop through Biome 1 to Biome 12
    for i in range(1, 13):
        biome_folder = f"biome{i}"
        csv_filename = f"B{i}_Metrics.csv" 
        
        # Construct full file paths
        # Format: /root_dir/biomeX/VARIABLE_NAME/BX_Metrics.csv
        path_tr = os.path.join(DIR_TR, biome_folder, var_name, csv_filename)
        path_aef = os.path.join(DIR_AEF, biome_folder, var_name, csv_filename)
        
        # Extract RMSE values
        rmse_tr, msg_tr = get_kmeans_ensemble_rmse(path_tr)
        rmse_aef, msg_aef = get_kmeans_ensemble_rmse(path_aef)
        
        # Comparison Logic (Lower RMSE is better)
        winner = "N/A"
        diff = 0.0
        
        if rmse_tr is not None and rmse_aef is not None:
            if rmse_aef < rmse_tr:
                winner = "AEF"
                diff = rmse_tr - rmse_aef # Positive improvement
            elif rmse_tr < rmse_aef:
                winner = "TRA"
                diff = rmse_aef - rmse_tr # Negative indicates regression
            else:
                winner = "Tie"
                diff = 0.0
                
            # formatting for list
            rmse_tr_val = rmse_tr
            rmse_aef_val = rmse_aef
            status = "Success"
        else:
            rmse_tr_val = -9999 # Placeholder for missing
            rmse_aef_val = -9999 # Placeholder for missing
            status = "Missing Data"

        # Append to results list
        all_results.append({
            "Variable": var_name,
            "Biome": biome_folder,
            "RMSE_TR": rmse_tr_val,
            "RMSE_AEF": rmse_aef_val,
            "Diff_RMSE": diff,
            "Winner": winner,
            "Status": status
        })

# ================= 4. Export to CSV =================

# Convert to DataFrame
df_results = pd.DataFrame(all_results)

# Define output path
output_csv_path = os.path.join(OUTPUT_DIR, "all_variables_comparison_rmse.csv")

# Save to CSV
df_results.to_csv(output_csv_path, index=False)

print("-" * 80)
print(f"Comparison complete!")
print(f"Results saved to: {output_csv_path}")

# ================= 5. Simple Summary Report =================
print("\n=== Summary by Variable (Win Counts) ===")
print(f"{'Variable':<10} | {'AEF Wins':<10} | {'TR Wins':<10} | {'Ties':<5}")
print("-" * 45)

for var_name in VARIABLES_LIST:
    subset = df_results[df_results['Variable'] == var_name]
    aef_wins = len(subset[subset['Winner'] == "AEF"])
    tr_wins = len(subset[subset['Winner'] == "TRA"])
    ties = len(subset[subset['Winner'] == "Tie"])
    
    print(f"{var_name:<10} | {aef_wins:<10} | {tr_wins:<10} | {ties:<5}")

print("-" * 45)