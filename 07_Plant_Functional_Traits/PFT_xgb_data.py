#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 Part 3 / Pipeline-B (XGB-SHAP) - DIRECT EXPORT VERSION (27 Variables)
   - 0.25-degree regular grid sampling (Shared across PC1, PC2, PC3)
   - Extracted exactly 27 finalized independent covariates
   - Removed VIF collinearity diagnosis
   - [Modified] Set 0 values in independent covariates to NA (np.nan)
--------------------------------------------------------------------------------
 Outputs go to:
   - /data/global_mapping/models_V100/drive_fina/PC1/XGB_shap/xgb_data/XGB_Dataset_27Vars.csv
   - /data/global_mapping/models_V100/drive_fina/PC2/XGB_shap/xgb_data/XGB_Dataset_27Vars.csv
   - /data/global_mapping/models_V100/drive_fina/PC3/XGB_shap/xgb_data/XGB_Dataset_27Vars.csv
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import logging
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as rio_transform
import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIG -- edit paths here ONLY
# -----------------------------------------------------------------------------
CONFIG = {
    # Define the 3 Principal Components targets and their respective output directories
    "targets": {
        "PC1": {
            "tif_path": "/data/global_mapping/models_V100/Principal_Components/Global_Forest_PC1.tif",
            "out_dir": "/data/global_mapping/models_V100/Plant_Functional_Traits/PC1/XGB_shap/xgb_data"
        },
        "PC2": {
            "tif_path": "/data/global_mapping/models_V100/Principal_Components/Global_Forest_PC2.tif",
            "out_dir": "/data/global_mapping/models_V100/Plant_Functional_Traits/PC2/XGB_shap/xgb_data"
        },
        "PC3": {
            "tif_path": "/data/global_mapping/models_V100/Principal_Components/Global_Forest_PC3.tif",
            "out_dir": "/data/global_mapping/models_V100/Plant_Functional_Traits/PC3/XGB_shap/xgb_data"
        }
    },
    
    "biome_shp_path":  "/data/global_mapping/union_bio/BIOME.shp",

    # 0.25 degree shared sampling grid
    "grid_resolution_deg": 0.25,
    "seed": 42,

    "BIOME_ID_MAP": {
        1: "Tropical Moist", 2: "Tropical Dry", 3: "Tropical Conifer",
        4: "Temperate Broadleaf", 5: "Temperate Conifer", 6: "Boreal Forests",
        7: "Tropical Grasslands", 8: "Temperate Grasslands", 9: "Flooded Grasslands",
        10: "Montane Grasslands", 11: "Tundra", 12: "Mediterranean Forests",
        13: "Deserts", 14: "Mangroves",
    },

    #
    "driver_mapping": {
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/WD.tif": "WD",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SLA.tif": "SLA",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/LDMC.tif": "LDMC",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/LNC.tif": "LNC",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/LPC.tif": "LPC"
    }
}


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logger() -> logging.Logger:
    log_dir = Path("/data/global_mapping/models_V100/drive_fina")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"XGB_Extraction_log_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logger = logging.getLogger("XGB_Extraction")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%H:%M:%S")

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_file}")
    return logger


# -----------------------------------------------------------------------------
# Step 1. Build Base Grid & Extract PC targets
# -----------------------------------------------------------------------------
def build_grid_and_extract_targets(targets_dict: dict, res_deg: float, logger: logging.Logger) -> pd.DataFrame:
    """
    Build a shared 0.25-degree grid based on PC1, then extract PC1, PC2, PC3 values.
    """
    # Use PC1 as the spatial reference
    ref_tif = targets_dict["PC1"]["tif_path"]
    logger.info(f"Building base grid using reference raster: {ref_tif}")
    
    with rasterio.open(ref_tif) as src:
        bounds = src.bounds
        ref_crs = src.crs

    # Determine grid extent (snap to multiples of res_deg)
    x_min = max(-180.0, np.floor(bounds.left   / res_deg) * res_deg)
    x_max = min( 180.0, np.ceil (bounds.right  / res_deg) * res_deg)
    y_min = max( -90.0, np.floor(bounds.bottom / res_deg) * res_deg)
    y_max = min(  90.0, np.ceil (bounds.top    / res_deg) * res_deg)

    xs = np.arange(x_min + res_deg / 2.0, x_max, res_deg)
    ys = np.arange(y_min + res_deg / 2.0, y_max, res_deg)
    xx, yy = np.meshgrid(xs, ys)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    logger.info(f"Initial global grid points: {coords.shape[0]:,}")

    df = pd.DataFrame({"lon": coords[:, 0], "lat": coords[:, 1]})
    
    # Extract each PC target
    for pc_name, pc_info in targets_dict.items():
        tif_path = pc_info["tif_path"]
        logger.info(f"Extracting target: {pc_name} from {tif_path}")
        
        with rasterio.open(tif_path) as src:
            nodata = src.nodata
            vals = np.fromiter(
                (v[0] for v in src.sample(coords.tolist(), masked=False)),
                dtype=np.float64,
                count=coords.shape[0],
            )
            if nodata is not None:
                vals = np.where(vals == nodata, np.nan, vals)
            df[pc_name] = vals

    # Drop rows where ANY of the PC targets are missing (ensure identical spatial locations)
    df = df.dropna(subset=list(targets_dict.keys())).reset_index(drop=True)
    logger.info(f"Valid shared grid points (after masking by PC targets): {len(df):,}")
    return df


# -----------------------------------------------------------------------------
# Step 2. Extract 27 driver rasters
# -----------------------------------------------------------------------------
def extract_drivers(df: pd.DataFrame, driver_mapping: dict, logger: logging.Logger) -> pd.DataFrame:
    coords = list(zip(df["lon"].values, df["lat"].values))
    for tif_path, var_name in driver_mapping.items():
        if not os.path.exists(tif_path):
            logger.warning(f"  [SKIP] missing raster -> {tif_path}")
            df[var_name] = np.nan
            continue
        try:
            with rasterio.open(tif_path) as src:
                nodata = src.nodata
                vals = np.fromiter(
                    (v[0] for v in src.sample(coords, masked=False)),
                    dtype=np.float64,
                    count=len(coords),
                )
                if nodata is not None:
                    vals = np.where(vals == nodata, np.nan, vals)
                
                # 将提取出的数据中所有为 0.0 的值设置为 np.nan (NA)
                vals = np.where(vals == 0.0, np.nan, vals)
                
                df[var_name] = vals
            logger.info(f"  [OK]   {var_name:<8s} extracted (0 values replaced with NA).")
        except Exception as e:
            logger.error(f"  [FAIL] {var_name} -- {e}")
            df[var_name] = np.nan
            
    return df


# -----------------------------------------------------------------------------
# Step 3. Attach BIOME ID
# -----------------------------------------------------------------------------
def attach_biome(df: pd.DataFrame, biome_shp: str, biome_id_map: dict, logger: logging.Logger) -> pd.DataFrame:
    logger.info(f"Reading BIOME shapefile: {biome_shp}")
    if not os.path.exists(biome_shp):
        df["BIOME_ID"] = np.nan
        df["BIOME_NAME"] = "Unknown"
        return df

    biomes = gpd.read_file(biome_shp)
    if biomes.crs is None or biomes.crs.to_epsg() != 4326:
        biomes = biomes.to_crs("EPSG:4326")

    biome_col = next((c for c in ["BIOME", "BIOME_ID", "BIOME_NUM", "biome", "biome_id"] if c in biomes.columns), None)
    if biome_col is None:
        biome_col = biomes.select_dtypes(include=[np.number]).columns[0]

    pts = gpd.GeoDataFrame(df, geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])], crs="EPSG:4326")
    joined = gpd.sjoin(pts, biomes[[biome_col, "geometry"]], how="left", predicate="within")
    
    joined["BIOME_ID"] = pd.to_numeric(joined[biome_col], errors="coerce")
    joined["BIOME_NAME"] = joined["BIOME_ID"].map(biome_id_map).fillna("Unknown")
    
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right", biome_col], errors="ignore"))


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    logger = setup_logger()
    np.random.seed(CONFIG["seed"])
    
    logger.info("=" * 78)
    logger.info(" XGB Extraction -- 27 Final Variables | Shared Grid | No VIF")
    logger.info("=" * 78)

    # Step 1: Build shared grid and extract PC1, PC2, PC3
    df = build_grid_and_extract_targets(CONFIG["targets"], CONFIG["grid_resolution_deg"], logger)

    # Step 2: Extract the 27 drivers
    logger.info("-" * 40)
    logger.info(f"Extracting {len(CONFIG['driver_mapping'])} environmental drivers...")
    df = extract_drivers(df, CONFIG["driver_mapping"], logger)

    # Step 3: Attach Biome
    logger.info("-" * 40)
    df = attach_biome(df, CONFIG["biome_shp_path"], CONFIG["BIOME_ID_MAP"], logger)

    # Step 4: Drop any row that has NaN in the 27 drivers
    drivers = list(CONFIG["driver_mapping"].values())
    df_clean = df.dropna(subset=drivers, how="any").reset_index(drop=True)
    logger.info(f"Final valid shared dataset (No NaNs): {len(df_clean):,} rows.")

    # Step 5: Split and export separately for PC1, PC2, PC3
    logger.info("-" * 40)
    logger.info("Exporting specific datasets for PC1, PC2, PC3...")
    
    for pc_name, pc_info in CONFIG["targets"].items():
        out_dir = Path(pc_info["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # Select columns: lon, lat, [specific PC target], biome info, and 27 drivers
        keep_cols = ["lon", "lat", pc_name, "BIOME_ID", "BIOME_NAME"] + drivers
        df_export = df_clean[keep_cols].copy()
        
        # Rename the target column to 'PC_target' to standardize downstream XGBoost scripts
        df_export = df_export.rename(columns={pc_name: "PC_target"})
        
        out_csv = out_dir / "XGB_Dataset_27Vars.csv"
        df_export.to_csv(out_csv, index=False)
        logger.info(f"  [Saved] {pc_name} -> {out_csv} (Shape: {df_export.shape})")

    logger.info("=" * 78)
    logger.info("All data extracted and exported successfully.")


if __name__ == "__main__":
    main()