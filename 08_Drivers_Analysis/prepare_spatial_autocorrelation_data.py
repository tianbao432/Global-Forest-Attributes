# -*- coding: utf-8 -*-

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
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

warnings.filterwarnings("ignore")

# -----------------------------------------------------------------------------
# CONFIG -- edit paths here ONLY
# -----------------------------------------------------------------------------
CONFIG = {
    "target_tif_path": "/data/global_mapping/models_V100/Principal_Components/Global_Forest_PC3.tif",
    "biome_shp_path":  "/data/global_mapping/union_bio/BIOME.shp",
    "output_root":     "/data/global_mapping/models_V100/drive/PC3/Spatial_Autocorrelation/",

    # Initial random candidate pool size.
    # Whatever valid points survive the masks from this pool will form the
    # final dataset.
    "random_candidate_pool": 5_000_000,

    # VIF threshold
    "vif_threshold": 5.0,

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
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/HMI.tif": "HMI",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/NTL.tif": "NTL",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/POPD.tif": "POPD",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/TTC.tif": "TTC",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/MAT.tif": "MAT",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/MDR.tif": "MDR",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/ISO.tif": "ISO",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/TS.tif": "TS",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/MTWM.tif": "MTWM",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/MTCM.tif": "MTCM",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/TAR.tif": "TAR",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/MTWeQ.tif": "MTWeQ",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/MTDQ.tif": "MTDQ",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/MTWaQ.tif": "MTWaQ",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/MTCQ.tif": "MTCQ",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/MAP.tif": "MAP",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/PWM.tif": "PWM",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/PDM.tif": "PDM",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/PS.tif": "PS",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/PWeQ.tif": "PWeQ",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/PDQ.tif": "PDQ",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/PWaQ.tif": "PWaQ",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/PCQ.tif": "PCQ",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/CEC.tif": "CEC",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/CF.tif": "CF",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/CLAY.tif": "CLAY",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/TN.tif": "TN",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/PH.tif": "PH",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SAND.tif": "SAND",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SILT.tif": "SILT",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SOC.tif": "SOC",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/PET.tif": "PET",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SRAD.tif": "SRAD",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/VAP.tif": "VAP",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/VPD.tif": "VPD",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/WS.tif": "WS",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/RO.tif": "RO",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SWE.tif": "SWE",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/AET.tif": "AET",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/CWD.tif": "CWD",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/PDSI.tif": "PDSI",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/PA.tif": "PA",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SOILM.tif": "SOILM",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/TMIN.tif": "TMIN",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/TMAX.tif": "TMAX",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/XCO2.tif": "XCO2",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/FF.tif": "FF",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/ROUGH.tif": "ROUGH",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/ASP.tif": "ASP",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/ELEV.tif": "ELEV",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SLP.tif": "SLP",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/TRI.tif": "TRI",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/VRM.tif": "VRM",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/TCURV.tif": "TCURV",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SHA_GEO.tif": "SHA_GEO",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/ENT_GEO.tif": "ENT_GEO",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/CLD_IAV.tif": "CLD_IAV",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SKT.tif": "SKT",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/STL1.tif": "STL1",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/STL3.tif": "STL3",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SWVL1.tif": "SWVL1",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SWVL3.tif": "SWVL3",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SSHF.tif": "SSHF",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SLHF.tif": "SLHF",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SSR.tif": "SSR",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/WS10.tif": "WS10",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/EWT.tif": "EWT",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/TWI.tif": "TWI",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/WD.tif": "WD",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/LDMC.tif": "LDMC",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/LNC.tif": "LNC",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/LPC.tif": "LPC",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/SLA.tif": "SLA",
        "/data/global_mapping/models_V100/1km_traditional_bilinear_no_forestslip/NDEP.tif": "NDEP"
    },
}


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def setup_logger(out_dir: Path) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"AC_Part1_log_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger("AC_Part1_Autocorrelation")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.info(f"Log file: {log_file}")
    return logger


# -----------------------------------------------------------------------------
# Step 1. Generate random candidate points (AREA-UNIFORM on the sphere)
# -----------------------------------------------------------------------------
def generate_random_candidates(target_tif: str, n_candidates: int,
                               seed: int, logger: logging.Logger):
    """
    Generate random (lon, lat) points UNIFORMLY by surface area on the sphere
    (i.e. sample sin(lat) uniformly, then arcsin back). This avoids the
    high-latitude over-sampling bias of naive uniform-lat sampling, which would
    artificially up-weight polar regions in any subsequent Moran's I estimate.
    Points are sampled within the target raster's bounding box, then masked by
    the target raster's nodata mask.
    """
    logger.info(f"Generating {n_candidates:,} random candidate points "
                f"(area-uniform on sphere) ...")
    rng = np.random.default_rng(seed)

    with rasterio.open(target_tif) as src:
        bounds = src.bounds
        nodata = src.nodata
        target_crs = src.crs
        logger.info(f"Target CRS: {target_crs}")
        logger.info(f"Target bounds: {bounds}")
        logger.info(f"Target nodata: {nodata}")

    lon_min = max(-180.0, bounds.left)
    lon_max = min( 180.0, bounds.right)
    lat_min = max( -90.0, bounds.bottom)
    lat_max = min(  90.0, bounds.top)

    # Longitude: uniform is fine.
    xs = rng.uniform(lon_min, lon_max, n_candidates)
    # Latitude: uniform on sin(lat) -> area-uniform on the sphere.
    sin_lo = np.sin(np.deg2rad(lat_min))
    sin_hi = np.sin(np.deg2rad(lat_max))
    ys = np.rad2deg(np.arcsin(rng.uniform(sin_lo, sin_hi, n_candidates)))

    coords = np.column_stack([xs, ys])

    logger.info("Extracting target values to filter out nodata (oceans / non-forest) ...")

    pts_for_sample = coords
    if target_crs and target_crs.to_epsg() != 4326:
        xs_proj, ys_proj = rio_transform(
            "EPSG:4326", target_crs,
            coords[:, 0].tolist(), coords[:, 1].tolist()
        )
        pts_for_sample = np.column_stack([xs_proj, ys_proj])

    with rasterio.open(target_tif) as src:
        vals = np.fromiter(
            (v[0] for v in src.sample(pts_for_sample.tolist(), masked=False)),
            dtype=np.float64,
            count=pts_for_sample.shape[0],
        )

    if nodata is not None:
        valid = np.isfinite(vals) & (vals != nodata)
    else:
        valid = np.isfinite(vals)

    coords = coords[valid]
    vals = vals[valid]
    logger.info(f"Valid points after target raster mask (hits on forest): "
                f"{coords.shape[0]:,}")

    df = pd.DataFrame({
        "lon": coords[:, 0],
        "lat": coords[:, 1],
        CONFIG["target_name"]: vals,
    })
    return df


# -----------------------------------------------------------------------------
# Step 2. Attach BIOME ID via point-in-polygon
# -----------------------------------------------------------------------------
def attach_biome(df: pd.DataFrame, biome_shp: str, biome_id_map: dict,
                 logger: logging.Logger) -> pd.DataFrame:
    if not os.path.exists(biome_shp):
        logger.warning(f"BIOME shapefile not found at {biome_shp}. "
                       f"Setting BIOME_ID=NaN.")
        df["BIOME_ID"] = np.nan
        df["BIOME_NAME"] = "Unknown"
        return df

    logger.info(f"Reading BIOME shapefile: {biome_shp}")
    biomes = gpd.read_file(biome_shp)
    if biomes.crs is None:
        biomes = biomes.set_crs("EPSG:4326")
    elif biomes.crs.to_epsg() != 4326:
        biomes = biomes.to_crs("EPSG:4326")

    # Heuristically detect the biome ID column
    candidate_cols = ["BIOME", "BIOME_ID", "BIOME_NUM", "biome", "biome_id",
                      "BIOME_NAME"]
    biome_col = next((c for c in candidate_cols if c in biomes.columns), None)
    if biome_col is None:
        biome_col = biomes.select_dtypes(include=[np.number]).columns[0]
        logger.warning(f"Could not detect BIOME id column; falling back to "
                       f"'{biome_col}'.")

    pts = gpd.GeoDataFrame(
        df.copy(),
        geometry=[Point(xy) for xy in zip(df["lon"], df["lat"])],
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, biomes[[biome_col, "geometry"]],
                       how="left", predicate="within")
    joined["BIOME_ID"] = pd.to_numeric(joined[biome_col], errors="coerce")
    joined["BIOME_NAME"] = joined["BIOME_ID"].map(biome_id_map).fillna("Unknown")
    out = pd.DataFrame(joined.drop(
        columns=["geometry", "index_right", biome_col], errors="ignore"
    ))
    return out


# -----------------------------------------------------------------------------
# Step 3. Extract each driver raster at the candidate points
# -----------------------------------------------------------------------------
def extract_drivers(df: pd.DataFrame, driver_mapping: dict,
                    logger: logging.Logger) -> pd.DataFrame:
    coords = list(zip(df["lon"].values, df["lat"].values))
    for tif_path, var_name in driver_mapping.items():
        if not os.path.exists(tif_path):
            logger.warning(f"  [SKIP] missing raster -> {tif_path}")
            df[var_name] = np.nan
            continue
        try:
            with rasterio.open(tif_path) as src:
                nodata = src.nodata
                src_crs = src.crs
                # Sanity-check: warn if this raster uses [0, 360] longitude
                if src.bounds.right > 180.5:
                    logger.warning(f"  [WARN] {os.path.basename(tif_path)} "
                                   f"appears to span [0, 360] longitude; "
                                   f"check whether reprojection is needed.")
                if src_crs and src_crs.to_epsg() != 4326:
                    xs_proj, ys_proj = rio_transform(
                        "EPSG:4326", src_crs,
                        df["lon"].tolist(), df["lat"].tolist()
                    )
                    sample_coords = list(zip(xs_proj, ys_proj))
                else:
                    sample_coords = coords

                vals = np.fromiter(
                    (v[0] for v in src.sample(sample_coords, masked=False)),
                    dtype=np.float64,
                    count=len(sample_coords),
                )
                if nodata is not None:
                    vals = np.where(vals == nodata, np.nan, vals)
                df[var_name] = vals
            logger.info(f"  [OK]   {var_name:<8s} extracted from "
                        f"{os.path.basename(tif_path)}")
        except Exception as e:
            logger.error(f"  [FAIL] {var_name} -- {e}")
            df[var_name] = np.nan
    return df


# -----------------------------------------------------------------------------
# Step 4. Iterative VIF filtering  (UNCHANGED -- as requested)
# -----------------------------------------------------------------------------
def iterative_vif(df: pd.DataFrame, predictors: list[str], threshold: float,
                  logger: logging.Logger) -> tuple[list[str], pd.DataFrame]:
    X = df[predictors].copy()
    # Drop zero-variance columns first
    nzv = [c for c in X.columns if X[c].std(skipna=True) > 0]
    dropped_const = [c for c in X.columns if c not in nzv]
    for c in dropped_const:
        logger.warning(f"  dropping constant variable: {c}")
    X = X[nzv]

    history = []
    iteration = 0
    while True:
        iteration += 1
        Xc = add_constant(X, has_constant="add")
        vifs = {}
        for i, col in enumerate(Xc.columns):
            if col == "const":
                continue
            try:
                v = variance_inflation_factor(Xc.values, i)
            except Exception:
                v = np.inf
            vifs[col] = v

        vif_series = pd.Series(vifs).sort_values(ascending=False)
        max_var, max_val = vif_series.index[0], vif_series.iloc[0]
        logger.info(f"  iter {iteration:>2d}: max VIF = {max_val:8.2f}  "
                    f"({max_var}); n_vars = {len(vif_series)}")
        history.append({
            "iter": iteration,
            "dropped": (max_var if max_val >= threshold else None),
            "max_vif": max_val,
            "n_remaining": len(vif_series),
        })

        if max_val < threshold or len(vif_series) <= 1:
            final_vifs = vif_series.reset_index().rename(
                columns={"index": "variable", 0: "VIF"})
            final_vifs.columns = ["variable", "VIF"]
            return list(vif_series.index), final_vifs

        X = X.drop(columns=[max_var])


# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    out_root = Path(CONFIG["output_root"])
    out_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_root)

    np.random.seed(CONFIG["seed"])
    logger.info("=" * 78)
    logger.info("Part 1 [AC]  --  Random Monte-Carlo sampling + Covariates + VIF")
    logger.info("            (Pipeline: Spatial Autocorrelation analysis)")
    logger.info("=" * 78)
    logger.info("Config:\n" + json.dumps(
        {k: v for k, v in CONFIG.items() if k != "driver_mapping"}, indent=2))

    # Step 1: Generate random candidate points (area-uniform)
    df = generate_random_candidates(CONFIG["target_tif_path"],
                                    CONFIG["random_candidate_pool"],
                                    CONFIG["seed"],
                                    logger)

    # Step 2: attach biome
    logger.info("Attaching biome ID ...")
    df = attach_biome(df, CONFIG["biome_shp_path"], CONFIG["BIOME_ID_MAP"], logger)

    # Drop points that fell outside target mask or biome polygons
    df = df.dropna(subset=[CONFIG["target_name"], "BIOME_ID"])
    logger.info(f"Points remaining after target + biome filters: {len(df):,}")

    # Step 3: extract driver covariates on all surviving points
    logger.info("Extracting environmental drivers ...")
    df_full = extract_drivers(df, CONFIG["driver_mapping"], logger)

    # Clean NAs across all required columns
    drivers = list(CONFIG["driver_mapping"].values())
    keep_cols = ["lon", "lat", CONFIG["target_name"],
                 "BIOME_ID", "BIOME_NAME"] + drivers
    df_full = df_full[keep_cols].dropna(how="any").reset_index(drop=True)

    logger.info(f"Final valid sampled dataset (NA dropped): "
                f"{len(df_full):,} rows  x  {len(df_full.columns)} cols.")

    # Save the full sampled table (A_AC_*.csv)
    a_csv = out_root / "A_AC_Random_Sampled_with_Covariates.csv"
    df_full.to_csv(a_csv, index=False)
    logger.info(f"Saved A.csv -> {a_csv}")

    # Step 4: iterative VIF
    logger.info("=" * 60)
    logger.info(f"Iterative VIF filtering (threshold = {CONFIG['vif_threshold']})")
    logger.info("=" * 60)
    retained, final_vifs = iterative_vif(df_full, drivers,
                                         CONFIG["vif_threshold"], logger)
    logger.info(f"\nRetained {len(retained)} variables:\n{retained}")

    # Save final VIF table (audit trail)
    vif_audit = out_root / "VIF_AC_Final_Report.csv"
    final_vifs.to_csv(vif_audit, index=False)
    logger.info(f"Saved VIF report -> {vif_audit}")

    # B_AC_*.csv -- only the retained variables (+ keys)
    keep_b = ["lon", "lat", CONFIG["target_name"],
              "BIOME_ID", "BIOME_NAME"] + retained
    df_b = df_full[keep_b].reset_index(drop=True)
    b_csv = out_root / "B_AC_VIF_Filtered_Dataset.csv"
    df_b.to_csv(b_csv, index=False)
    logger.info(f"Saved B.csv -> {b_csv}  (shape = {df_b.shape})")

    # Also save the retained variable list as a sidecar
    retained_json = out_root / "B_AC_retained_variables.json"
    with open(retained_json, "w", encoding="utf-8") as f:
        json.dump({
            "pipeline": "AC_spatial_autocorrelation",
            "retained": retained,
            "vif_threshold": CONFIG["vif_threshold"],
        }, f, indent=2)
    logger.info(f"Saved retained variable list -> {retained_json}")

    logger.info("Part 1 [AC] finished successfully.")


if __name__ == "__main__":
    main()