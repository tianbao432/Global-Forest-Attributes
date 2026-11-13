# -*- coding: utf-8 -*-
# =============================================================================
# Nature-Standard Spatial Machine Learning Pipeline
# =============================================================================

import os
import sys
import time
import json
import platform
import subprocess
import warnings
import random
import logging
import importlib
import gc
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio import features
from rasterio.enums import MergeAlg

import xgboost as xgb
import shap

import libpysal
from esda.moran import Moran

from scipy.spatial import cKDTree
from sklearn.model_selection import GroupKFold, GridSearchCV, cross_val_predict
from sklearn.cluster import KMeans
from sklearn.base import clone
from sklearn.metrics import r2_score, mean_squared_error

from pyproj import CRS, Transformer
from tqdm import tqdm

import matplotlib as mpl

warnings.filterwarnings("ignore")

# ============================================================================
# Publication-Standard Plotting Configurations (Nature Style)
# ============================================================================
mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = ['Arial']
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
# NPG (Nature Publishing Group) Color Palette
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=[
    '#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', '#8491B4', '#91D1C2', '#DC0000'
])

# ============================================================================
# Logging
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("spatial_ml")


# ============================================================================
# 0. Global seeding
# ============================================================================
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    log.info(f"Global seed set to: {seed}")


# ============================================================================
# Environment snapshot methods (Tracked Packages & Hardware)
# ============================================================================
TRACKED_PACKAGES = [
    ("gdal",         None),
    ("rasterio",     "rasterio"),
    ("geopandas",    "geopandas"),
    ("pyproj",       "pyproj"),
    ("shapely",      "shapely"),
    ("fiona",        "fiona"),
    ("libpysal",     "libpysal"),
    ("esda",         "esda"),
    ("numpy",        "numpy"),
    ("pandas",       "pandas"),
    ("scipy",        "scipy"),
    ("scikit-learn", "sklearn"),
    ("xgboost",      "xgboost"),
    ("shap",         "shap"),
    ("tqdm",         "tqdm"),
    ("matplotlib",   "matplotlib"),
]

def _detect_install_source(distribution_name: str) -> str:
    try:
        from importlib.metadata import distribution
        dist = distribution(distribution_name)
        try:
            if dist.read_text("direct_url.json"): return "pip (direct_url)"
        except Exception: pass
        conda_prefix = os.environ.get("CONDA_PREFIX", "")
        if conda_prefix:
            meta_dir = os.path.join(conda_prefix, "conda-meta")
            if os.path.isdir(meta_dir):
                prefix = f"{distribution_name.lower()}-"
                for fname in os.listdir(meta_dir):
                    if fname.lower().startswith(prefix) and fname.endswith(".json"):
                        return "conda"
        return "pip"
    except Exception:
        return "unknown"

def _get_package_version(import_name: str | None, distribution_name: str) -> str:
    try:
        from importlib.metadata import version as _v
        return _v(distribution_name)
    except Exception: pass
    if import_name:
        try:
            mod = importlib.import_module(import_name)
            return getattr(mod, "__version__", "unknown")
        except Exception: pass
    return "not installed"

def _gdal_version() -> str:
    try:
        from osgeo import gdal
        return gdal.__version__
    except Exception:
        try:
            out = subprocess.run(["gdalinfo", "--version"], capture_output=True, text=True, timeout=5)
            if out.returncode == 0:
                return out.stdout.split(",")[0].replace("GDAL", "").strip()
        except Exception: pass
    return "not installed"

def _gpu_info() -> dict:
    info = {"cuda_available": False, "cuda_runtime_version": "n/a", "gpu_devices": [], "nvidia_driver_version": "n/a"}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode == 0 and out.stdout.strip():
            info["cuda_available"] = True
            for line in out.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    info["gpu_devices"].append({"name": parts[0], "memory_total": parts[2]})
                    info["nvidia_driver_version"] = parts[1]
    except Exception: pass
    return info

def export_environment_snapshot(output_dir: str, run_config: dict) -> None:
    log.info(">>> Exporting environment snapshot...")
    rows = []
    rows.append({"Package": "python", "Version": platform.python_version(), "Source": "system", "Notes": f"exec: {sys.executable}"})
    rows.append({"Package": "platform", "Version": platform.platform(), "Source": "system", "Notes": f"{platform.machine()}"})

    for dist_name, import_name in TRACKED_PACKAGES:
        ver = _gdal_version() if dist_name == "gdal" else _get_package_version(import_name, dist_name)
        src = _detect_install_source(dist_name) if ver != "not installed" else "n/a"
        rows.append({"Package": dist_name, "Version": ver, "Source": src, "Notes": ""})

    gpu = _gpu_info()
    rows.append({"Package": "cuda_runtime", "Version": gpu["cuda_runtime_version"], "Source": "system", "Notes": f"driver: {gpu['nvidia_driver_version']}"})
    for i, dev in enumerate(gpu["gpu_devices"]):
        rows.append({"Package": f"gpu_device_{i}", "Version": dev["name"], "Source": "hardware", "Notes": f"memory: {dev['memory_total']}"})

    snap_csv = os.path.join(output_dir, "00_environment_snapshot.csv")
    pd.DataFrame(rows).to_csv(snap_csv, index=False)
    
    metadata = {
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "gpu_info": gpu,
        "key_config": {k: v for k, v in run_config.items() if not isinstance(v, dict)}
    }
    meta_json = os.path.join(output_dir, "00_run_metadata.json")
    with open(meta_json, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2, default=str)


# ============================================================================
# 1. Configuration
# ============================================================================
CONFIG = {
    # --- Paths ---
    "target_tif_path": "/data/global_mapping/models_V100/Principal_Components/Global_Forest_PC2.tif",
    "biome_shp_path":  "/data/global_mapping/union_bio/BIOME.shp",
    "output_root":     "/data/global_mapping/models_V100/drive/PC22",

    "BIOME_ID_MAP": {
        1: "Tropical Moist", 2: "Tropical Dry", 3: "Tropical Conifer",
        4: "Temperate Broadleaf", 5: "Temperate Conifer", 6: "Boreal Forests",
        7: "Tropical Grasslands", 8: "Temperate Grasslands", 9: "Flooded Grasslands",
        10: "Montane Grasslands", 11: "Tundra", 12: "Mediterranean Forests",
        13: "Deserts", 14: "Mangroves",
    },

    # --- Variables (Abridged for brevity in script, maintain your full list here) ---
    "driver_mapping": {
        "/data/global_mapping/drive_factor/drive_variables/AET.tif": "AET",
        "/data/global_mapping/drive_factor/drive_variables/Aspect.tif": "Aspect",
        "/data/global_mapping/drive_factor/drive_variables/DEM.tif": "DEM",
        # ... Include the full 50+ variables ...
    },

    "grid_size_km": 100,
    "vif_threshold": 5.0,

    "n_outer_folds": 5,
    "n_inner_folds": 3,
    "spatial_cv_clusters": 20,
    "bootstrap_reps": 30,

    "buffer_search_start_km": 50,
    "buffer_search_step_km": 50,
    "buffer_search_max_km": 1000,
    "target_moran_i": 0.05,

    # Tuned for high RAM environments
    "shap_interaction_max_n": 10000, 

    # Expanded to perform a genuine grid search
    "param_grid": {
        "n_estimators":  [300, 500],
        "max_depth":     [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample":     [0.8],
    },

    "univariate_params": {
        "n_estimators":  200,
        "max_depth":     3,
        "learning_rate": 0.1,
        "subsample":     0.8,
    },

    "random_seed": 42,
    "use_gpu": True,
    "gpu_id": 0,
}

os.makedirs(CONFIG["output_root"], exist_ok=True)


# ============================================================================
# Helper: build an XGBRegressor with the right device API
# ============================================================================
def make_xgb_regressor(use_gpu: bool, gpu_id: int, seed: int, **kwargs) -> xgb.XGBRegressor:
    base_params = dict(
        tree_method="hist",
        random_state=seed,
        n_jobs=1,
        verbosity=0,
    )
    base_params.update(kwargs)
    if use_gpu:
        base_params["device"] = f"cuda:{gpu_id}"
    return xgb.XGBRegressor(**base_params)


# ============================================================================
# 2. Custom Buffered CV
# ============================================================================
class BufferedGroupKFold:
    def __init__(self, groups, coords, buffer_distance_meters, n_splits=5):
        self.groups = np.asarray(groups)
        self.coords = np.asarray(coords)
        self.buffer = float(buffer_distance_meters)
        self.n_splits = int(n_splits)

    def split(self, X, y=None, groups=None):
        gkf = GroupKFold(n_splits=self.n_splits)
        tree = cKDTree(self.coords)
        for train_idx_raw, test_idx in gkf.split(X, y, self.groups):
            test_coords = self.coords[test_idx]
            indices_in_buffer = tree.query_ball_point(test_coords, r=self.buffer)
            if len(indices_in_buffer) == 0:
                indices_to_ban = np.array([], dtype=int)
            else:
                indices_to_ban = np.unique(np.concatenate(indices_in_buffer)).astype(int)

            train_mask = np.zeros(len(self.coords), dtype=bool)
            train_mask[train_idx_raw] = True
            train_mask[indices_to_ban] = False
            yield np.where(train_mask)[0], test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# ============================================================================
# 3. Data loading & spatial thinning
# ============================================================================
def project_coordinates(lon, lat):
    lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
    if (lon_max - lon_min) > 10:
        target_crs_str = "+proj=moll +lon_0=0 +datum=WGS84 +units=m +no_defs"
    else:
        zone = int(np.floor((np.mean(lon) + 180) / 6) + 1)
        target_crs_str = f"+proj=utm +zone={zone} +datum=WGS84 +units=m +no_defs"
    transformer = Transformer.from_crs(CRS.from_epsg(4326), CRS.from_string(target_crs_str), always_xy=True)
    return transformer.transform(lon, lat)


def load_data_optimized(target_path, biome_path, driver_dict):
    log.info(f">>> Phase 1: Data Loading & {CONFIG['grid_size_km']}km Spatial Thinning...")

    with rasterio.open(target_path) as src:
        target_data = src.read(1)
        transform = src.transform
        nodata = src.nodata
        target_crs = src.crs

    gdf = gpd.read_file(biome_path)
    if gdf.crs is None: gdf.set_crs(epsg=4326, inplace=True)
    if gdf.crs != target_crs: gdf = gdf.to_crs(target_crs)

    gdf = gdf.copy()
    gdf["__area"] = gdf.geometry.area
    gdf = gdf.sort_values("__area", ascending=False).reset_index(drop=True)

    shapes = ((geom, i + 1) for i, geom in enumerate(gdf.geometry))
    biome_raster = features.rasterize(
        shapes, out_shape=target_data.shape, transform=transform,
        fill=0, all_touched=True, dtype="uint16", merge_alg=MergeAlg.replace,
    )

    valid_mask = (~np.isnan(target_data)) & (biome_raster > 0)
    if nodata is not None: valid_mask &= (target_data != nodata)

    valid_indices = np.argwhere(valid_mask)
    rows, cols = valid_indices[:, 0], valid_indices[:, 1]
    xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")

    df_temp = pd.DataFrame({
        "Target":   target_data[rows, cols],
        "X_raster": xs,
        "Y_raster": ys,
        "Biome_ID": biome_raster[rows, cols],
    })
    
    del target_data, biome_raster, valid_mask
    gc.collect()

    if target_crs is not None and target_crs.to_epsg() != 4326:
        to_wgs = Transformer.from_crs(target_crs, CRS.from_epsg(4326), always_xy=True)
        lon_arr, lat_arr = to_wgs.transform(df_temp["X_raster"].values, df_temp["Y_raster"].values)
        df_temp["Lon"], df_temp["Lat"] = lon_arr, lat_arr
    else:
        df_temp["Lon"], df_temp["Lat"] = df_temp["X_raster"], df_temp["Y_raster"]

    xx, yy = project_coordinates(df_temp["Lon"].values, df_temp["Lat"].values)
    df_temp["X_proj"], df_temp["Y_proj"] = xx, yy

    grid_m = CONFIG["grid_size_km"] * 1000
    df_temp["Grid_ID"] = (
        (df_temp["X_proj"] // grid_m).astype(int).astype(str) + "_" +
        (df_temp["Y_proj"] // grid_m).astype(int).astype(str)
    )

    df_out = (df_temp.groupby("Grid_ID")
                     .sample(n=1, random_state=CONFIG["random_seed"])
                     .copy()
                     .drop(columns=["Grid_ID"]))
    log.info(f"    Points retained after thinning: {len(df_out)}")
    del df_temp
    gc.collect()

    pts_lon, pts_lat = df_out["Lon"].values, df_out["Lat"].values
    extracted, missing_paths = 0, []
    
    for path, clean_name in tqdm(driver_dict.items(), desc="Extracting Drivers"):
        if not os.path.exists(path):
            missing_paths.append(path)
            continue
        try:
            with rasterio.open(path) as src:
                if src.crs is not None and src.crs != CRS.from_epsg(4326):
                    tr = Transformer.from_crs(CRS.from_epsg(4326), src.crs, always_xy=True)
                    sx, sy = tr.transform(pts_lon, pts_lat)
                    sample_coords = list(zip(sx, sy))
                else:
                    sample_coords = list(zip(pts_lon, pts_lat))

                vals = np.fromiter((v[0] for v in src.sample(sample_coords)), dtype="float64", count=len(sample_coords))
                if src.nodata is not None:
                    vals = np.where(vals == src.nodata, np.nan, vals)
                df_out[clean_name] = vals
                extracted += 1
        except Exception as e:
            log.warning(f"    Failed to read {path}: {e}")

    if missing_paths:
        log.warning(f"    {len(missing_paths)} driver raster(s) not found and skipped.")
        with open(os.path.join(CONFIG["output_root"], "00_missing_rasters.log"), "w") as fh:
            fh.write("\n".join(missing_paths))

    if extracted == 0:
        raise FileNotFoundError("[CRITICAL] Failed to load any driver features.")

    df_final = df_out.dropna().reset_index(drop=True)
    df_final.to_csv(os.path.join(CONFIG["output_root"], f"01_Raw_Extracted_Data_{CONFIG['grid_size_km']}km.csv"), index=False)
    log.info(f"    Final samples after dropping NA: {len(df_final)}")
    return df_final


# ============================================================================
# 4. VIF filter
# ============================================================================
def fast_vif_filter(df, features_list, threshold=5.0):
    log.info(f">>> Phase 2: VIF screening (<{threshold})...")
    corr_abs = df[features_list].corr().abs()
    upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
    drop_init = [c for c in upper.columns if any(upper[c] > 0.98)]
    keep = [f for f in features_list if f not in drop_init]
    
    if drop_init: log.info(f"    Pre-dropped (|r|>0.98): {drop_init}")

    drop_log = []
    while len(keep) > 1:
        c_mat = df[keep].corr().values
        try:
            vifs = np.diag(np.linalg.inv(c_mat))
        except np.linalg.LinAlgError:
            vifs = np.diag(np.linalg.pinv(c_mat))
        vifs = np.nan_to_num(vifs, nan=np.inf, posinf=np.inf, neginf=np.inf)

        max_vif = float(np.max(vifs))
        if max_vif > threshold:
            idx = int(np.argmax(vifs))
            dropped = keep.pop(idx)
            drop_log.append({"Dropped_Feature": dropped, "VIF_at_drop": max_vif})
            log.info(f"    Dropping {dropped} (VIF={max_vif:.2f})")
        else:
            for f, v in zip(keep, vifs):
                drop_log.append({"Dropped_Feature": f"{f} (RETAINED)", "VIF_at_drop": float(v)})
            break

    pd.DataFrame(drop_log).to_csv(os.path.join(CONFIG["output_root"], "02_VIF_Filtering_Log.csv"), index=False)
    return keep


# ============================================================================
# 5. Nested CV + Bootstrap UQ
# ============================================================================
def perform_nested_cv_and_bootstrap(df, features_list, target_col):
    X, y = df[features_list], df[target_col]
    coords_proj = df[["X_proj", "Y_proj"]].values

    log.info(">>> Phase 3a: Spatial blocking (KMeans on raw projected coords)...")
    groups = KMeans(
        n_clusters=CONFIG["spatial_cv_clusters"],
        random_state=CONFIG["random_seed"],
        n_init=10,
    ).fit_predict(coords_proj)
    df = df.copy()
    df["Spatial_Block"] = groups

    log.info(">>> Phase 3b: Optimizing buffer distance via Moran's I (OOF residuals)...")
    tmp_model = make_xgb_regressor(CONFIG["use_gpu"], CONFIG["gpu_id"], CONFIG["random_seed"], n_estimators=200, max_depth=6)
    
    oof_y = cross_val_predict(tmp_model, X, y, cv=GroupKFold(n_splits=min(5, CONFIG["spatial_cv_clusters"])), groups=groups, n_jobs=1)
    res_sub = (y.values - oof_y).astype(float)

    moran_log = []
    optimal_buffer_m = CONFIG["buffer_search_start_km"] * 1000
    found = False
    cur = CONFIG["buffer_search_start_km"] * 1000
    
    while cur <= CONFIG["buffer_search_max_km"] * 1000:
        try:
            w = libpysal.weights.DistanceBand(coords_proj, threshold=cur, binary=True, silence_warnings=True)
            mi = Moran(res_sub, w, permutations=0)
            log.info(f"      Distance: {cur/1000:.0f} km | Residual Moran's I: {mi.I:.4f}")
            moran_log.append({"Distance_km": cur / 1000, "Moran_I": float(mi.I)})
            if (mi.I < CONFIG["target_moran_i"]) and (not found):
                optimal_buffer_m = cur
                found = True
                log.info(f"      Independence reached at {optimal_buffer_m/1000:.0f} km.")
        except Exception as e:
            log.warning(f"      Moran's I failed at {cur/1000:.0f} km: {e}")
            moran_log.append({"Distance_km": cur / 1000, "Moran_I": np.nan})
        cur += CONFIG["buffer_search_step_km"] * 1000

    if not found:
        optimal_buffer_m = CONFIG["buffer_search_max_km"] * 1000
        log.warning(f"    Moran's I never dropped below target; using max buffer {optimal_buffer_m/1000:.0f} km.")

    pd.DataFrame(moran_log).to_csv(os.path.join(CONFIG["output_root"], "03_Moran_Distance_Optimization_Curve.csv"), index=False)

    log.info(">>> Phase 4: Nested spatial CV (buffered inner + outer)...")
    outer_cv = BufferedGroupKFold(groups=groups, coords=coords_proj, buffer_distance_meters=optimal_buffer_m, n_splits=CONFIG["n_outer_folds"])
    xgb_base = make_xgb_regressor(CONFIG["use_gpu"], CONFIG["gpu_id"], CONFIG["random_seed"])

    oof_preds = np.full(len(df), np.nan)
    cv_fold_ids = np.full(len(df), -1, dtype=int)

    for fold_i, (tr_idx, te_idx) in enumerate(outer_cv.split(X, y)):
        if len(tr_idx) == 0: continue
        X_tr, y_tr, g_tr = X.iloc[tr_idx], y.iloc[tr_idx], groups[tr_idx]
        coords_tr = coords_proj[tr_idx]

        inner_cv = BufferedGroupKFold(groups=g_tr, coords=coords_tr, buffer_distance_meters=optimal_buffer_m, n_splits=CONFIG["n_inner_folds"])
        grid = GridSearchCV(estimator=xgb_base, param_grid=CONFIG["param_grid"], cv=inner_cv, scoring="r2", n_jobs=1, verbose=0)
        grid.fit(X_tr, y_tr)
        oof_preds[te_idx] = grid.best_estimator_.predict(X.iloc[te_idx])
        cv_fold_ids[te_idx] = fold_i + 1

    df["Nested_CV_Predicted"] = oof_preds
    df["CV_Fold_ID"] = cv_fold_ids

    valid = ~np.isnan(oof_preds)
    nested_r2 = float(r2_score(y[valid], oof_preds[valid]))
    nested_rmse = float(np.sqrt(mean_squared_error(y[valid], oof_preds[valid])))
    log.info(f"    Unbiased Nested CV  R2: {nested_r2:.4f} | RMSE: {nested_rmse:.4f}")

    pd.DataFrame({
        "Metric": ["Nested_CV_R2", "Nested_CV_RMSE", "Optimal_Buffer_km", "N_Samples_Used"],
        "Value":  [nested_r2, nested_rmse, optimal_buffer_m / 1000, int(valid.sum())],
    }).to_csv(os.path.join(CONFIG["output_root"], "04b_Nested_CV_Evaluation_Metrics.csv"), index=False)

    log.info(">>> Phase 5a: Final model fit (buffered group-CV grid search)...")
    final_cv = BufferedGroupKFold(groups=groups, coords=coords_proj, buffer_distance_meters=optimal_buffer_m, n_splits=CONFIG["n_outer_folds"])
    final_grid = GridSearchCV(estimator=xgb_base, param_grid=CONFIG["param_grid"], cv=final_cv, scoring="r2", n_jobs=1, verbose=0)
    final_grid.fit(X, y)
    final_model = final_grid.best_estimator_

    log.info(">>> Phase 5b: Spatial-block bootstrap UQ...")
    unique_blocks = np.unique(groups)
    boot = np.zeros((len(X), CONFIG["bootstrap_reps"]), dtype=np.float32)
    rng = np.random.RandomState(CONFIG["random_seed"])
    for b in tqdm(range(CONFIG["bootstrap_reps"]), desc="Bootstrapping"):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        boot_idx = np.concatenate([np.where(groups == blk)[0] for blk in sampled_blocks])
        m = clone(final_model)
        m.set_params(random_state=CONFIG["random_seed"] + b)
        m.fit(X.iloc[boot_idx], y.iloc[boot_idx])
        boot[:, b] = m.predict(X)

    df["Pred_Mean"]      = boot.mean(axis=1)
    df["Pred_StdDev"]    = boot.std(axis=1)
    df["Pred_Lower_95"]  = np.percentile(boot, 2.5, axis=1)
    df["Pred_Upper_95"]  = np.percentile(boot, 97.5, axis=1)
    df["Residuals"]      = df[target_col] - df["Pred_Mean"]

    out_cols = ["Lon", "Lat", target_col, "Pred_Mean", "Pred_StdDev", "Pred_Lower_95", "Pred_Upper_95", "Nested_CV_Predicted", "Residuals", "Spatial_Block", "CV_Fold_ID"]
    df[out_cols].rename(columns={target_col: "Target"}).to_csv(os.path.join(CONFIG["output_root"], "04_Global_Model_Predictions_with_UQ.csv"), index=False)

    return final_model, df, optimal_buffer_m


# ============================================================================
# 6. SHAP analytics
# ============================================================================
def run_univariate_cv(X_subset, y, coords_proj, groups, optimal_buffer_m, n_outer_folds):
    cv = BufferedGroupKFold(groups=groups, coords=coords_proj, buffer_distance_meters=optimal_buffer_m, n_splits=n_outer_folds)
    oof = np.full(len(y), np.nan)
    for tr, te in cv.split(X_subset, y):
        if len(tr) == 0: continue
        m = make_xgb_regressor(CONFIG["use_gpu"], CONFIG["gpu_id"], CONFIG["random_seed"], **CONFIG["univariate_params"])
        m.fit(X_subset.iloc[tr], y.iloc[tr])
        oof[te] = m.predict(X_subset.iloc[te])
    valid = ~np.isnan(oof)
    if valid.sum() < 2: return float("nan")
    return float(r2_score(y[valid], oof[valid]))


def analyze_shap_and_variance(model, df, features_list, optimal_buffer_m, target_col):
    log.info(">>> Phase 6: SHAP analytics...")
    X, y = df[features_list], df[target_col]
    coords_proj = df[["X_proj", "Y_proj"]].values
    groups = df["Spatial_Block"].values

    log.info("    Computing TreeExplainer SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X, check_additivity=False)
    shap_df = pd.DataFrame(shap_values, columns=features_list)

    mean_abs_shap = shap_df.abs().mean().sort_values(ascending=False)
    mean_abs_shap.rename("Mean_Abs_SHAP").reset_index().rename(columns={"index": "Feature"}).to_csv(
        os.path.join(CONFIG["output_root"], "05_SHAP_Global_Summary_Importance.csv"), index=False
    )

    raw = shap_df.copy()
    raw.columns = [f"{c}_SHAP" for c in raw.columns]
    raw.insert(0, "Lon", df["Lon"].values)
    raw.insert(1, "Lat", df["Lat"].values)
    raw.to_csv(os.path.join(CONFIG["output_root"], "12_SHAP_Values_Full_Raw.csv"), index=False)

    log.info("    Independent spatial variance...")
    var_log = []
    for f in tqdm(features_list, desc="Univariate variance"):
        r2v = run_univariate_cv(df[[f]], y, coords_proj, groups, optimal_buffer_m, CONFIG["n_outer_folds"])
        var_log.append({
            "Feature": f,
            "Independent_OOF_R2": r2v,
            "Variance_Explained_Pct": r2v * 100 if not np.isnan(r2v) else np.nan, 
        })

    top_1 = mean_abs_shap.index[0]
    top_2 = mean_abs_shap.index[1] if len(mean_abs_shap) > 1 else top_1
    r2_combo = run_univariate_cv(df[[top_1, top_2]], y, coords_proj, groups, optimal_buffer_m, CONFIG["n_outer_folds"])
    var_log.append({
        "Feature": f"Combined_Top2_{top_1}_{top_2}",
        "Independent_OOF_R2": r2_combo,
        "Variance_Explained_Pct": r2_combo * 100 if not np.isnan(r2_combo) else np.nan,
    })

    pd.DataFrame(var_log).sort_values("Independent_OOF_R2", ascending=False).to_csv(os.path.join(CONFIG["output_root"], "06_All_Features_Variance_Explained.csv"), index=False)

    top5 = mean_abs_shap.index[:5].tolist()
    dep = df[["Lon", "Lat", "Spatial_Block"] + top5].copy()
    for f in top5: dep[f"{f}_SHAP"] = shap_df[f].values
    dep.to_csv(os.path.join(CONFIG["output_root"], "07_SHAP_Dependence_Nonlinear_Top5.csv"), index=False)

    log.info("    Computing SHAP interaction matrix...")
    n_full = len(X)
    cap = CONFIG["shap_interaction_max_n"]
    if n_full > cap:
        rng = np.random.RandomState(CONFIG["random_seed"])
        sub_idx = rng.choice(n_full, size=cap, replace=False)
        log.info(f"    Subsampling to {cap}/{n_full} rows for interaction values.")
    else:
        sub_idx = np.arange(n_full)

    X_sub = X.iloc[sub_idx]
    df_sub = df.iloc[sub_idx].reset_index(drop=True)
    shap_inter = explainer.shap_interaction_values(X_sub)

    inter_abs = np.abs(shap_inter).sum(axis=0)
    np.fill_diagonal(inter_abs, 0)
    inter_norm = inter_abs / inter_abs.sum() if inter_abs.sum() > 0 else inter_abs

    pd.DataFrame(inter_norm, index=features_list, columns=features_list).to_csv(os.path.join(CONFIG["output_root"], "08_SHAP_Interaction_Matrix.csv"))

    i, j = np.unravel_index(np.argmax(inter_abs, axis=None), inter_abs.shape)
    f1, f2 = features_list[i], features_list[j]
    log.info(f"    Strongest interaction pair: {f1} <--> {f2}")

    pair = df_sub[["Lon", "Lat", f1, f2]].copy()
    pair[f"Interaction_SHAP_{f1}_{f2}"] = shap_inter[:, i, j] * 2 
    pair.to_csv(os.path.join(CONFIG["output_root"], "09_SHAP_Top_Interaction_Pair_Spatial.csv"), index=False)

    log.info("    Building dominant driver map...")
    map_df = df[["Lon", "Lat"]].copy()
    map_df["Biome_Name"] = df["Biome_ID"].map(CONFIG["BIOME_ID_MAP"]).fillna("Unknown")

    abs_shap = np.abs(shap_values)
    sorted_idx = np.argsort(abs_shap, axis=1)
    dom_idx = sorted_idx[:, -1]
    sub_idx2 = sorted_idx[:, -2]

    rows_arr = np.arange(len(dom_idx))
    map_df["Dominant_Driver"]      = [features_list[k] for k in dom_idx]
    map_df["Dominant_SHAP_Value"]  = shap_values[rows_arr, dom_idx]
    map_df["Dominant_AbsSHAP"]     = abs_shap[rows_arr, dom_idx]
    map_df["Dominant_Direction"]   = np.where(map_df["Dominant_SHAP_Value"] > 0, "Positive", "Negative")
    map_df["Sub_Dominant_Driver"]  = [features_list[k] for k in sub_idx2]
    map_df["Sub_Dominant_AbsSHAP"] = abs_shap[rows_arr, sub_idx2]
    map_df["Dominance_Margin"]     = map_df["Dominant_AbsSHAP"] - map_df["Sub_Dominant_AbsSHAP"]

    map_df.to_csv(os.path.join(CONFIG["output_root"], "10_SHAP_Directional_Dominant_Map.csv"), index=False)

    log.info("    Aggregating SHAP by biome...")
    shap_with_biome = shap_df.copy()
    shap_with_biome["Biome_Name"] = map_df["Biome_Name"].values
    valid_b = shap_with_biome[shap_with_biome["Biome_Name"] != "Unknown"]
    biome_imp = (valid_b.groupby("Biome_Name")[features_list].apply(lambda x: x.abs().mean()))
    biome_imp_pct = biome_imp.div(biome_imp.sum(axis=1), axis=0).mul(100)
    biome_imp_pct.to_csv(os.path.join(CONFIG["output_root"], "11_SHAP_Biome_Importance_Pct.csv"))


# ============================================================================
# 7. Main
# ============================================================================
def main():
    seed_everything(CONFIG["random_seed"])
    export_environment_snapshot(CONFIG["output_root"], CONFIG)

    df_full = load_data_optimized(
        CONFIG["target_tif_path"],
        CONFIG["biome_shp_path"],
        CONFIG["driver_mapping"],
    )

    non_features = {"Target", "Lon", "Lat", "X_raster", "Y_raster", "X_proj", "Y_proj", "Biome_ID", "Grid_ID"}
    env_features = [c for c in df_full.columns if c not in non_features]

    final_features = fast_vif_filter(df_full, env_features, CONFIG["vif_threshold"])
    log.info(f"Final feature set ({len(final_features)}): {final_features}")

    model, df_res, opt_buf = perform_nested_cv_and_bootstrap(df_full, final_features, "Target")
    analyze_shap_and_variance(model, df_res, final_features, opt_buf, "Target")

    log.info(f">>> Pipeline complete. Outputs in: {CONFIG['output_root']}")


if __name__ == "__main__":
    main()