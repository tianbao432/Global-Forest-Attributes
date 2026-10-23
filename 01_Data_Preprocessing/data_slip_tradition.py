import pandas as pd
import numpy as np
import os
import chardet

def process_data_no_split():
    # ================= CONFIGURATION =================
    input_path = "/data/global_mapping/plotdata_CV_tr/fina_plot_with_RS_traditional.csv"
    output_root = "/data/global_mapping/plotdata_CV_tr/"

    # Target variables (using the new renamed names)
    target_columns = ["DBH", "H", "AGE", "BA", "Trees_N", "AGB", "V", "shannon", "CV_DBH"]

    # Grouping variable (categorical)
    group_col = "BIOME"

    # ================= 1. LOAD DATA =================
    print(f"Loading raw data: {input_path}")

    # Automatically detect file encoding to prevent read errors
    with open(input_path, "rb") as f:
        raw = f.read(1024 * 1024)
        enc = chardet.detect(raw)["encoding"] or "utf-8"

    try:
        df = pd.read_csv(input_path, encoding=enc, low_memory=False)
    except Exception as e:
        print(f"Failed with detected encoding, trying latin1: {e}")
        df = pd.read_csv(input_path, encoding="latin1", low_memory=False)

    print(f"Data loaded. Total rows: {len(df)}")

    # ================= 2. RENAME VARIABLES =================
    # Renaming specific columns as requested:
    # Shannon.Index -> shannon
    # N -> Trees_N
    rename_dict = {
        "Shannon.Index": "shannon",
        "N": "Trees_N",
    }
    df.rename(columns=rename_dict, inplace=True)
    print("Variables renamed: 'Shannon.Index' -> 'shannon', 'N' -> 'Trees_N'")

    # ================= 3. EXPLICIT FEATURE LIST =================
    # Predictors (environment + remote sensing) listed explicitly,
    # grouped by category for readability. Order does not matter.
    env_features = [
        "A00", "A01", "A02", "A03", "A04", "A05", "A06", "A07", "A08", "A09",
        "A10", "A11", "A12", "A13", "A14", "A15", "A16", "A17", "A18", "A19",
        "A20", "A21", "A22", "A23", "A24", "A25", "A26", "A27", "A28", "A29",
        "A30", "A31", "A32", "A33", "A34", "A35", "A36", "A37", "A38", "A39",
        "A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49",
        "A50", "A51", "A52", "A53", "A54", "A55", "A56", "A57", "A58", "A59",
        "A60", "A61", "A62", "A63",

        "ALOS_ASM", "ALOS_CON", "ALOS_COR", "ALOS_DIS", "ALOS_ENT", "ALOS_HOM", "ALOS_MEAN", "ALOS_VAR",

        "LS_ASM", "LS_CON", "LS_COR", "LS_DIS", "LS_DVI", "LS_ENT", "LS_EVI2",
        "LS_HOM", "LS_MEAN", "LS_MSAVI2", "LS_MSR", "LS_NBR", "LS_NBR2", "LS_NDMI",
        "LS_NDVI", "LS_NIR", "LS_RED", "LS_SAVI", "LS_SR", "LS_SWIR1", "LS_SWIR2", "LS_VAR",

        "MOD_ARVI", "MOD_ASM", "MOD_BLUE", "MOD_BNDVI", "MOD_CON", "MOD_COR", "MOD_DIS",
        "MOD_DVI", "MOD_ENT", "MOD_EVI", "MOD_HOM", "MOD_MEAN", "MOD_MIR", "MOD_MSAVI2",
        "MOD_MSR", "MOD_NBR", "MOD_NDVI", "MOD_NIR", "MOD_OSAVI", "MOD_RED", "MOD_RVI",
        "MOD_SAVI", "MOD_SRWI", "MOD_VAR",

        "MTCM", "MTCQ", "MTDQ", "MTWaQ", "MTWeQ", "MTWM",

        "AET", "ASP", "CEC", "CF", "CLAY", "CLD_IAV",  "CPR", "CWD",
        "ELEV", "ENT_GEO", "EOS", "EWT", "FF", "FPAR", "GPP", "GR", "HH",
        "HMI", "HV", "IEVI2", "ISO", "LAI", "LDMC", "LNC", "LOS", "LPC", "MAP",
        "MAT", "MDR", "NDEP", "NPP", "NTL", "PA", "PCQ", "PDM", "PDQ", "PDSI",
        "PET", "PEVI2", "PH", "POPD", "PS", "PWaQ", "PWeQ", "PWM", "RO", "ROUGH",
        "SAND", "SHA_GEO", "SILT", "SKT", "SLA", "SLHF", "SLP", "SOC", "SOILM",
        "SOS", "SRAD", "SSHF", "SSR", "STL1", "STL3", "SWE", "SWVL1", "SWVL3",
        "TAR", "TCURV", "TMAX", "TMIN", "TN", "TRI", "TS", "TTC", "TWI", "VAP",
        "VPD", "VRM", "WD", "WS", "WS10", "XCO2"
    ]
    assert len(env_features) == 205

    # Coordinates are kept alongside the predictors
    feature_columns = env_features + ["Latitude", "Longitude"]

    # Sanity check: warn about any predictor missing from the CSV
    missing = [c for c in feature_columns if c not in df.columns]
    if missing:
        print(f"[WARN] {len(missing)} predictor(s) not found in CSV and will be skipped: {missing}")
    feature_columns = [c for c in feature_columns if c in df.columns]

    print(f"Using {len(feature_columns)} predictors (including Latitude/Longitude).")

    # ================= 4. PROCESS LOOP =================
    for target in target_columns:
        # Skip if target variable is not found in the dataframe
        if target not in df.columns:
            print(f"\n[SKIP] Target '{target}' not found in CSV columns.")
            continue

        print(f"\n>>> Processing Target: {target}")

        # Identify unique BIOMEs (dropping NAs to avoid errors)
        unique_biomes = df[group_col].dropna().unique()

        for biome_val in unique_biomes:
            # --- Format Folder Name (standardize integer IDs) ---
            try:
                if float(biome_val).is_integer():
                    biome_str = str(int(biome_val))
                else:
                    biome_str = str(biome_val)
            except (ValueError, TypeError):
                biome_str = str(biome_val)

            biome_folder_name = f"BIOME_{biome_str}"

            # 1. Filter data for the specific current BIOME
            biome_raw_df = df[df[group_col] == biome_val]

            # 2. Select required columns (Target + All Predictors)
            cols_to_keep = [target] + feature_columns

            # Ensure only columns that actually exist in the dataframe are used
            available_cols = [c for c in cols_to_keep if c in biome_raw_df.columns]
            subset_df = biome_raw_df[available_cols]

            # 3. Drop NA rows
            # Rows must have valid data for the target AND all environmental predictors
            clean_df = subset_df.dropna()

            # Logging statistics for the current BIOME
            original_count = len(subset_df)
            clean_count = len(clean_df)
            dropped_count = original_count - clean_count

            # Skip output generation if no valid data rows exist
            if clean_count < 1:
                continue

            # 4. Save Data (Organized by Target and BIOME)
            # Path structure: root / TargetName / BIOME_X / data.csv
            save_dir = os.path.join(output_root, target, biome_folder_name)
            os.makedirs(save_dir, exist_ok=True)

            output_file = os.path.join(save_dir, "data.csv")
            clean_df.to_csv(output_file, index=False)

            print(f"   - {biome_folder_name:<10}: Raw {original_count} -> Clean {clean_count} (Dropped {dropped_count})")

    print("\n" + "=" * 50)
    print("DONE! Process completed.")
    print(f"Output Path: {output_root}")
    print("=" * 50)


if __name__ == "__main__":
    process_data_no_split()