#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce

def main():
    # ==========================================
    # 0. Global Plot Settings (Font: Arial)
    # ==========================================
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial']
    # Ensure minus signs are displayed correctly if any
    plt.rcParams['axes.unicode_minus'] = False 

    # ==========================================
    # 1. Directory and Parameter Setup
    # ==========================================
    base_dir = "/data/global_mapping/models_V100/AEF/"
    out_dir = "/data/global_mapping/models_V100/AEF/cv_plot/va_important"
    
    # Create output directory
    os.makedirs(out_dir, exist_ok=True)
    
    # Define the 9 target parameters and 12 biomes
    target_vars = ["AGE", "BA", "Trees_N", "AGB", "V", "shannon", "CV_DBH", "DBH", "H"]
    biomes = list(range(1, 13))
    
    # Use ExcelWriter to combine 9 parameters into 9 sheets
    excel_out_path = os.path.join(out_dir, "All_Targets_SHAP_Importance.xlsx")
    
    print(f"Output directory ready: {out_dir}")
    print("-" * 50)
    
    # ==========================================
    # 2. Process Data and Plot
    # ==========================================
    with pd.ExcelWriter(excel_out_path, engine='openpyxl') as writer:
        for target in target_vars:
            print(f"Processing Target Parameter: {target} ...")
            
            target_data_list = []
            
            # Read SHAP data for all 12 biomes
            for b_id in biomes:
                shap_csv_path = os.path.join(base_dir, f"biome{b_id}", target, "SHAP.csv")
                
                if os.path.exists(shap_csv_path):
                    df = pd.read_csv(shap_csv_path)
                    # Use shorter names (e.g., B1, B2) for cleaner x-axis visualization
                    df = df.rename(columns={"Importance": f"B{b_id}"})
                    target_data_list.append(df)
                else:
                    print(f"  [Warning] Missing data for Biome {b_id}: {shap_csv_path}")
            
            # Skip if no data is found
            if not target_data_list:
                print(f"  [Skip] No SHAP data found for {target}. Moving to next.\n")
                continue
                
            # Merge all biome data
            merged_df = reduce(lambda left, right: pd.merge(left, right, on="Feature", how="outer"), target_data_list)
            
            # Set index and fill missing values with 0
            merged_df.set_index("Feature", inplace=True)
            merged_df.fillna(0, inplace=True)
            
            # Calculate mean importance for the Excel report
            merged_df['Mean_Importance'] = merged_df.mean(axis=1)
            
            # Sort the index alphabetically so variables go from A00 to A63 sequentially
            merged_df = merged_df.sort_index(ascending=True)
            
            # Write to a specific sheet in the Excel file
            merged_df.to_excel(writer, sheet_name=target)
            
            # ==========================================
            # 3. Generate Aesthetic Heatmap (PDF & TIF)
            # ==========================================
            # Remove Mean_Importance for the actual plot matrix
            plot_df = merged_df.drop(columns=['Mean_Importance'])
            
            # Set up the figure
            plt.figure(figsize=(10, 12))
            
            # Plot heatmap 
            # You can change "magma" to "viridis", "plasma", "Blues", "YlGnBu", etc.
            ax = sns.heatmap(
                plot_df, 
                cmap="YlGnBu",         
                annot=False,            
                linewidths=0.5,         
                cbar_kws={'label': 'Mean |SHAP| Value', 'shrink': 0.82}
            )
            
            # Titles and labels
            plt.title(target, fontsize=18, fontweight='bold', pad=15)
            plt.xlabel("Biomes", fontsize=14, labelpad=10)
            plt.ylabel("Variables", fontsize=14, labelpad=10)
            
            # X and Y axis tick formatting
            plt.xticks(rotation=0, fontsize=11)
            plt.yticks(rotation=0, fontsize=9)
            
            # Auto-adjust layout
            plt.tight_layout()
            
            # Export as PDF (Vector format, great for publications)
            pdf_out_path = os.path.join(out_dir, f"{target}_SHAP_Heatmap_AEF.pdf")
            plt.savefig(pdf_out_path, format='pdf', bbox_inches='tight')
            
            # Export as TIF (High resolution raster, requested by many journals)
            tif_out_path = os.path.join(out_dir, f"{target}_SHAP_Heatmap_AEF.tif")
            plt.savefig(tif_out_path, format='tiff', dpi=300, bbox_inches='tight')
            
            plt.close()
            
            print(f"  [Save] Plots exported -> .pdf and .tif for {target}")
            print("-" * 50)

    print(f"All target datasets successfully combined into Excel -> {os.path.basename(excel_out_path)}")
    print("All tasks completed successfully!")

if __name__ == "__main__":
    main()