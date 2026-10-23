#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 Part 4 / Pipeline-B (XGB-SHAP):  SLIM VERSION (Top-6 SHAP) + INTERACTIONS
   XGBoost (GPU) + SPATIAL BLOCK Nested CV + SHAP analysis

 What this script produces (per buffer):
   Main directory
     1_Block_CV_Evaluation_Metrics.csv          -- detailed, transparent CV metrics per fold
     1b_Block_CV_Per_Point_Predictions.csv      -- observed vs buffered-CV pred
     final_hyperparameters.json                 -- reproducibility sidecar
     final_xgb_model.json                       -- trained XGBoost booster
   final_shap/
     3_SHAP_Global_Summary_Importance.csv       -- global variable importance
     4_SHAP_Dependence_Nonlinear_Top6.csv       -- global Top-6 response data
     6_SHAP_Interaction_Top2_Pointwise.csv      -- Point-wise interaction data for the Top 2 features ONLY
     7_SHAP_Directional_Dominant_Map.csv        -- dominant driver map
     8_SHAP_Biome_Importance_Pct.csv            -- per-biome variable importance (ALL vars)
     9_SHAP_Biome_Dependence_Nonlinear_Top6.csv -- biome-specific Top-6 response
     10_SHAP_Beeswarm_Global_Top6.csv           -- beeswarm-ready data, Top-6 only
     11_SHAP_Beeswarm_Biome_Top6.csv            -- beeswarm-ready data per biome,
                                                   biome-specific Top-6 only

 Run-level (written ONCE at output_root, alongside per-buffer dirs):
   run_environment.json                         -- captured Python/lib versions,
                                                   host/OS info, GPU info,
                                                   CONFIG snapshot, data hashes

 Design notes:
   * NO SAMPLING: SHAP is computed on the FULL dataset.
   * Per-biome Top-N is biome-specific (each biome picks its own N drivers).
   * Computes shap_interaction_values ONLY for Top 2 global features in batches 
     to prevent Out-Of-Memory (OOM) crashes.
   * Nested CV Hyperparameter tuning evaluates 50 iterations via randomized search.

 Beeswarm CSV schema:
   10_SHAP_Beeswarm_Global_Top6.csv columns:
       feature, rank, mean_abs_SHAP,
       row_idx, lon, lat, BIOME_ID, BIOME_NAME,
       feature_value, shap_value
================================================================================
"""

from __future__ import annotations

import os
import sys
import json
import math
import hashlib
import logging
import warnings
import time
import platform
import tempfile
import subprocess
import random
from pathlib import Path
from datetime import datetime
from itertools import product
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold
from sklearn.neighbors import BallTree

import sklearn
import xgboost as xgb
import shap

try:
    from pyproj import Transformer
    import pyproj
    _PYPROJ_AVAILABLE = True
except ImportError:
    _PYPROJ_AVAILABLE = False
    pyproj = None
    warnings.warn("pyproj not installed. Falling back to approximate "
                  "equirectangular projection (latitude-distorted).")

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    # ---- I/O ----------------------------------------------------------------
    "b_csv_path":             "/data/global_mapping/models_V100/drive_fina/PC3/XGB_shap/xgb_data/XGB_Dataset_27Vars.csv",
    "output_root":            "/data/global_mapping/models_V100/drive_fina/PC3/XGB_shap/XGB_shap",

    "target_name":            "PC_target",
    # ---- Spatial block CV parameters (kilometers) ---------------------------
    "block_size_km":          1500,
    "block_buffer_list":      [800, 1000, 1200],

    # ---- Cross-validation ---------------------------------------------------
    "outer_folds":            10,
    "inner_folds":            5,
    "min_train_frac_warn":    0.30,

    # ---- Hyper-parameter grid (Expanded for 50 iterations) ------------------
    "param_grid": {
        'n_estimators': [500, 1000],  
        'max_depth': [4, 6, 8],          
        'learning_rate': [0.01, 0.05],   
        'subsample': [0.7, 0.9],              
        'colsample_bytree': [0.7, 0.9],  
        "reg_alpha": [0],        
        "reg_lambda": [1.00],        
    },
    # ---- How many features to keep in each export --------------------------
    "dependence_topN":        10,
    "beeswarm_topN":          10,

    # ---- Minimum samples for a biome to be included in biome SHAP exports ---
    "biome_min_samples":      1000,

    # ---- BIOME map ----------------------------------------------------------
    "BIOME_ID_MAP": {
        1: "Tropical Moist", 2: "Tropical Dry", 3: "Tropical Conifer",
        4: "Temperate Broadleaf", 5: "Temperate Conifer", 6: "Boreal Forests",
        7: "Tropical Grasslands", 8: "Temperate Grasslands", 9: "Flooded Grasslands",
        10: "Montane Grasslands", 11: "Tundra", 12: "Mediterranean Forests",
        13: "Deserts", 14: "Mangroves",
    },

    # ---- Allowed pipeline tags in the retained_json (defense-in-depth) ------
    "allowed_pipeline_tags":  [None, "XGB_SHAP", "XGB_SHAP_NO_SAMPLING"],

    # ---- RNG ----------------------------------------------------------------
    "seed":                   42,

    # ---- Version tag for THIS script (bumps when behavior changes) ----------
    "script_version":         "Part4_XGB_SHAP_v1.7_no_bootstrap_50iter",
}

# =============================================================================
# Logging
# =============================================================================
def setup_logger(out_dir: Path, name: str) -> logging.Logger:
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"XGB_Part4_log_{datetime.now():%Y%m%d_%H%M%S}.log"
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s", "%H:%M:%S")
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt); logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt); logger.addHandler(sh)
    logger.info(f"Log file: {log_file}")
    return logger

# =============================================================================
# Environment capture (for reproducibility)
# =============================================================================
def _safe_get_version(module_name: str) -> Optional[str]:
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None

def _get_gpu_info() -> dict:
    info = {"nvidia_smi_available": False, "devices": []}
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,driver_version,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT,
            timeout=5,
        ).decode("utf-8", errors="ignore")
        info["nvidia_smi_available"] = True
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                info["devices"].append({
                    "name":             parts[0],
                    "driver_version":   parts[1],
                    "memory_total_MiB": parts[2],
                    "compute_cap":      parts[3],
                })
    except Exception as e:
        info["error"] = str(e)
    return info

def _file_sha256(path: str, chunk: int = 2**20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return None

def export_environment(out_path: Path, use_gpu: bool, n_samples: int,
                       retained: list, logger: logging.Logger) -> None:
    env = {
        "script_version":   CONFIG["script_version"],
        "timestamp_iso":    datetime.now().isoformat(timespec="seconds"),
        "host": {
            "hostname":     platform.node(),
            "platform":     platform.platform(),
            "machine":      platform.machine(),
            "processor":    platform.processor(),
            "python":       sys.version.split()[0],
            "executable":   sys.executable,
            "cwd":          os.getcwd(),
        },
        "libraries": {
            "numpy":     np.__version__,
            "pandas":    pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost":   xgb.__version__,
            "shap":      shap.__version__,
            "pyproj":    getattr(pyproj, "__version__", None) if _PYPROJ_AVAILABLE else None,
            "scipy":     _safe_get_version("scipy"),
            "matplotlib": _safe_get_version("matplotlib"),
        },
        "gpu": {
            "xgb_uses_gpu": bool(use_gpu),
            "nvidia_smi":   _get_gpu_info(),
        },
        "data": {
            "b_csv_path":       CONFIG["b_csv_path"],
            "b_csv_sha256":     _file_sha256(CONFIG["b_csv_path"]),
            "retained_json":    CONFIG.get("retained_json", "HARDCODED_LIST"),
            "retained_sha256":  _file_sha256(CONFIG.get("retained_json", "")) if "retained_json" in CONFIG else None,
            "target_name":      CONFIG["target_name"],
            "n_samples_after_filtering": int(n_samples),
            "n_predictors":     int(len(retained)),
            "predictors":       list(retained),
        },
        "projection": {
            "use_pyproj":     bool(_PYPROJ_AVAILABLE),
            "epsg":           "EPSG:8857" if _PYPROJ_AVAILABLE else "equirectangular_fallback",
        },
        "config_snapshot": {
            k: v for k, v in CONFIG.items() if k != "BIOME_ID_MAP"
        },
        "biome_id_map": {str(k): v for k, v in CONFIG["BIOME_ID_MAP"].items()},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, default=str)
    logger.info(f"Environment captured: {out_path}")

# =============================================================================
# GPU detection
# =============================================================================
def gpu_available() -> bool:
    try:
        d = xgb.DMatrix(np.zeros((4, 2)), label=np.zeros(4))
        xgb.train({"tree_method": "hist", "device": "cuda"},
                  d, num_boost_round=1)
        return True
    except Exception:
        return False

def make_xgb(params: dict, use_gpu: bool, seed: int) -> xgb.XGBRegressor:
    base = dict(
        objective    = "reg:squarederror",
        tree_method  = "hist",
        device       = "cuda" if use_gpu else "cpu",
        n_jobs       = -1,
        random_state = seed,
        verbosity    = 0,
    )
    base.update(params)
    return xgb.XGBRegressor(**base)

# =============================================================================
# Projection (equal-area; EPSG:8857 Equal Earth)
# =============================================================================
EARTH_RADIUS_KM = 6371.0
EQUAL_AREA_EPSG = "EPSG:8857"

def _lon_lat_to_xy_approx(lon: np.ndarray, lat: np.ndarray):
    lat_mean = np.mean(lat)
    x = np.deg2rad(lon) * EARTH_RADIUS_KM * 1000.0 * np.cos(np.deg2rad(lat_mean))
    y = np.deg2rad(lat) * EARTH_RADIUS_KM * 1000.0
    return x, y

def _lon_lat_to_xy_proj(lon: np.ndarray, lat: np.ndarray):
    transformer = Transformer.from_crs("EPSG:4326", EQUAL_AREA_EPSG, always_xy=True)
    x, y = transformer.transform(lon, lat)
    return np.asarray(x), np.asarray(y)

def project_to_meters(lon: np.ndarray, lat: np.ndarray):
    if _PYPROJ_AVAILABLE:
        try:
            return _lon_lat_to_xy_proj(lon, lat)
        except Exception as e:
            logging.getLogger("XGB_Part4_setup").warning(
                f"pyproj projection failed ({e}); falling back to equirectangular.")
    return _lon_lat_to_xy_approx(lon, lat)

def assign_block_ids_km(x_m: np.ndarray, y_m: np.ndarray,
                        block_size_km: float) -> np.ndarray:
    block_size_m = block_size_km * 1000.0
    bx = np.floor(x_m / block_size_m).astype(np.int64)
    by = np.floor(y_m / block_size_m).astype(np.int64)
    pairs = pd.MultiIndex.from_arrays([bx, by])
    block_id, _ = pd.factorize(pairs)
    return block_id.astype(np.int64)

# =============================================================================
# Spatial block CV with optional Haversine buffer
# =============================================================================
def spatial_block_kfold(x_m: np.ndarray, y_m: np.ndarray,
                        lat: np.ndarray, lon: np.ndarray,
                        block_size_km: float, block_buffer_km: float,
                        n_splits: int, seed: int,
                        logger: logging.Logger | None = None):
    block_id = assign_block_ids_km(x_m, y_m, block_size_km)
    unique_blocks = np.unique(block_id)
    n_blocks = len(unique_blocks)
    if n_blocks < n_splits:
        raise ValueError(
            f"Not enough unique blocks ({n_blocks}) for n_splits={n_splits}. "
            f"Lower block_size_km.")

    rng = np.random.default_rng(seed)
    shuffled_blocks = unique_blocks.copy()
    rng.shuffle(shuffled_blocks)

    order = np.argsort(block_id, kind="stable")
    block_id_sorted = block_id[order]
    boundaries = np.concatenate(([0],
                                 np.where(np.diff(block_id_sorted) != 0)[0] + 1,
                                 [len(block_id_sorted)]))
    block_to_rows = {}
    for i in range(len(boundaries) - 1):
        rows = order[boundaries[i]:boundaries[i + 1]]
        block_to_rows[block_id_sorted[boundaries[i]]] = rows

    if block_buffer_km > 0:
        radians = np.deg2rad(np.column_stack([lat, lon]))
        tree = BallTree(radians, metric="haversine")
        buffer_rad = block_buffer_km / EARTH_RADIUS_KM
    else:
        tree = None
        buffer_rad = 0.0
        radians = None

    n = len(lon)
    kf = KFold(n_splits=n_splits, shuffle=False)
    for fold_idx, (_, test_block_pos) in enumerate(
            kf.split(np.arange(n_blocks)), start=1):
        test_blocks = shuffled_blocks[test_block_pos]
        test_idx = np.concatenate([block_to_rows[b] for b in test_blocks])

        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx] = True
        train_mask = ~test_mask
        
        buffer_idx = np.array([], dtype=np.int64)

        if block_buffer_km > 0 and tree is not None:
            test_pts = radians[test_idx]
            neighbours = tree.query_radius(test_pts, r=buffer_rad)
            if len(neighbours) > 0:
                too_close = np.unique(np.concatenate(neighbours))
                too_close_mask = np.zeros(n, dtype=bool)
                too_close_mask[too_close] = True
                
                buffer_mask = too_close_mask & (~test_mask)
                buffer_idx = np.where(buffer_mask)[0]
                
                train_mask = train_mask & (~too_close_mask)

        train_idx = np.where(train_mask)[0]
        yield fold_idx, train_idx, test_idx, buffer_idx

def spatial_block_kfold_on_subset(x_m_sub: np.ndarray, y_m_sub: np.ndarray,
                                  lat_sub: np.ndarray, lon_sub: np.ndarray,
                                  block_size_km: float,
                                  block_buffer_km: float,
                                  n_splits: int, seed: int):
    block_id = assign_block_ids_km(x_m_sub, y_m_sub, block_size_km)
    unique_blocks = np.unique(block_id)
    n_blocks = len(unique_blocks)
    if n_blocks < n_splits:
        return

    rng = np.random.default_rng(seed)
    shuffled_blocks = unique_blocks.copy()
    rng.shuffle(shuffled_blocks)

    order = np.argsort(block_id, kind="stable")
    block_id_sorted = block_id[order]
    boundaries = np.concatenate(([0],
                                 np.where(np.diff(block_id_sorted) != 0)[0] + 1,
                                 [len(block_id_sorted)]))
    block_to_rows = {}
    for i in range(len(boundaries) - 1):
        rows = order[boundaries[i]:boundaries[i + 1]]
        block_to_rows[block_id_sorted[boundaries[i]]] = rows

    if block_buffer_km > 0:
        radians = np.deg2rad(np.column_stack([lat_sub, lon_sub]))
        tree = BallTree(radians, metric="haversine")
        buffer_rad = block_buffer_km / EARTH_RADIUS_KM
    else:
        tree = None
        buffer_rad = 0.0
        radians = None

    n = len(lon_sub)
    kf = KFold(n_splits=n_splits, shuffle=False)
    for fold_idx, (_, test_block_pos) in enumerate(
            kf.split(np.arange(n_blocks)), start=1):
        test_blocks = shuffled_blocks[test_block_pos]
        test_idx = np.concatenate([block_to_rows[b] for b in test_blocks])

        test_mask = np.zeros(n, dtype=bool)
        test_mask[test_idx] = True
        train_mask = ~test_mask

        if block_buffer_km > 0 and tree is not None:
            test_pts = radians[test_idx]
            neighbours = tree.query_radius(test_pts, r=buffer_rad)
            if len(neighbours) > 0:
                too_close = np.unique(np.concatenate(neighbours))
                too_close_mask = np.zeros(n, dtype=bool)
                too_close_mask[too_close] = True
                train_mask = train_mask & (~too_close_mask)

        train_idx = np.where(train_mask)[0]
        yield fold_idx, train_idx, test_idx

# =============================================================================
# Inner spatial-block grid search
# =============================================================================
def is_single_point_grid(grid: dict) -> bool:
    return all(len(v) == 1 for v in grid.values())

def inner_block_grid_search(X: pd.DataFrame, y: np.ndarray,
                            x_m_tr: np.ndarray, y_m_tr: np.ndarray,
                            lat_tr: np.ndarray, lon_tr: np.ndarray,
                            grid: dict, n_inner: int,
                            block_size_km: float, block_buffer_km: float,
                            use_gpu: bool, seed: int,
                            logger: logging.Logger) -> tuple[Optional[dict], float]:
    keys = list(grid.keys())
    combos = list(product(*[grid[k] for k in keys]))
    
    # Enforce exactly 50 iterations by randomizing the parameter combinations
    if len(combos) > 50:
        rng = random.Random(seed)
        rng.shuffle(combos)
        combos = combos[:50]

    best_combo, best_score = None, -np.inf

    folds = list(spatial_block_kfold_on_subset(
        x_m_tr, y_m_tr, lat_tr, lon_tr,
        block_size_km, block_buffer_km, n_inner, seed))
    if len(folds) == 0:
        return None, -np.inf

    for combo in combos:
        params = dict(zip(keys, combo))
        scores = []
        for _, tr_idx, va_idx in folds:
            if len(tr_idx) < 50 or len(va_idx) < 20:
                continue
            mdl = make_xgb(params, use_gpu, seed)
            mdl.fit(X.iloc[tr_idx], y[tr_idx],
                    eval_set=[(X.iloc[va_idx], y[va_idx])],
                    verbose=False)
            pred = mdl.predict(X.iloc[va_idx])
            scores.append(r2_score(y[va_idx], pred))
        if len(scores) == 0:
            continue
        m = float(np.mean(scores))
        if m > best_score:
            best_score = m
            best_combo = params
    return best_combo, best_score

# =============================================================================
# Metric helpers
# =============================================================================
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    if len(y_true) < 2:
        return dict(R2=np.nan, RMSE=np.nan)
    return dict(
        R2  = float(r2_score(y_true, y_pred)),
        RMSE= float(math.sqrt(mean_squared_error(y_true, y_pred))),
    )

# =============================================================================
# SHAP helpers
# =============================================================================
def shap_for_model(model: xgb.XGBRegressor, X_subset: pd.DataFrame,
                   retained: list) -> tuple[np.ndarray, float]:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
        model.get_booster().save_model(tmp.name)
        booster_cpu = xgb.Booster()
        booster_cpu.load_model(tmp.name)
        if booster_cpu.feature_names != retained:
            raise ValueError(
                f"Feature-name mismatch in SHAP. "
                f"Model={booster_cpu.feature_names}, expected={retained}")
        explainer = shap.TreeExplainer(booster_cpu)
        try:
            sv_obj = explainer(X_subset)
            shap_values = np.asarray(sv_obj.values)
            if hasattr(sv_obj.base_values, "__len__"):
                base_value = float(np.asarray(sv_obj.base_values).ravel()[0])
            else:
                base_value = float(sv_obj.base_values)
        except Exception:
            shap_values = np.asarray(explainer.shap_values(X_subset))
            base_value = float(np.asarray(explainer.expected_value).ravel()[0])
    return shap_values, base_value

def compute_top2_interactions_chunked(model: xgb.XGBRegressor, X_subset: pd.DataFrame, 
                                      retained: list, top2_features: list, 
                                      logger: logging.Logger, chunk_size: int = 5000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Computes pure interaction values (N x F x F) strictly for the Top-2 features in chunks to prevent memory crash."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as tmp:
        model.get_booster().save_model(tmp.name)
        booster_cpu = xgb.Booster()
        booster_cpu.load_model(tmp.name)
        explainer = shap.TreeExplainer(booster_cpu)
        
        f1, f2 = top2_features[0], top2_features[1]
        idx1, idx2 = retained.index(f1), retained.index(f2)
        
        n_samples = len(X_subset)
        inter_vals = []
        main1_vals = []
        main2_vals = []
        
        logger.info(f"Computing explicit SHAP interaction values for Top-2 features ({f1}, {f2}) in batches...")
        
        for start in range(0, n_samples, chunk_size):
            end = min(start + chunk_size, n_samples)
            chunk = X_subset.iloc[start:end]
            chunk_int = np.asarray(explainer.shap_interaction_values(chunk))
            
            inter_vals.append(chunk_int[:, idx1, idx2] * 2)
            main1_vals.append(chunk_int[:, idx1, idx1])
            main2_vals.append(chunk_int[:, idx2, idx2])
            
        return np.concatenate(inter_vals), np.concatenate(main1_vals), np.concatenate(main2_vals)

# =============================================================================
# Beeswarm helpers (Top-N only)
# =============================================================================
def build_beeswarm_global(shap_values: np.ndarray, X_shap: pd.DataFrame,
                          retained: list, mean_abs_all: np.ndarray,
                          lon_shap: np.ndarray, lat_shap: np.ndarray,
                          biome_shap: np.ndarray, biome_name_shap: np.ndarray,
                          top_n: int) -> pd.DataFrame:
    order = np.argsort(mean_abs_all)[::-1]
    top_order = order[:int(top_n)]
    n = len(X_shap)
    frames = []
    for rank0, col_idx in enumerate(top_order):
        f = retained[col_idx]
        frames.append(pd.DataFrame({
            "feature":       f,
            "rank":          int(rank0 + 1),
            "mean_abs_SHAP": float(mean_abs_all[col_idx]),
            "row_idx":       np.arange(n, dtype=np.int64),
            "lon":           lon_shap,
            "lat":           lat_shap,
            "BIOME_ID":      biome_shap,
            "BIOME_NAME":    biome_name_shap,
            "feature_value": X_shap[f].to_numpy(dtype=np.float64),
            "shap_value":    shap_values[:, col_idx].astype(np.float64),
        }))
    return pd.concat(frames, ignore_index=True)

def build_beeswarm_biome(shap_values: np.ndarray, X_shap: pd.DataFrame,
                         retained: list, lon_shap: np.ndarray, lat_shap: np.ndarray,
                         biome_shap: np.ndarray, biome_name_shap: np.ndarray,
                         biome_min: int, top_n: int, logger: logging.Logger) -> Optional[pd.DataFrame]:
    unique_biomes = np.unique(biome_name_shap)
    frames = []
    for b_name in unique_biomes:
        mask = (biome_name_shap == b_name)
        n_b = int(mask.sum())
        if n_b < biome_min:
            continue
        b_indices = np.where(mask)[0]
        sub_shap = shap_values[b_indices]
        b_mean_abs = np.abs(sub_shap).mean(axis=0)
        order_b = np.argsort(b_mean_abs)[::-1]
        top_order_b = order_b[:int(top_n)]

        b_ids_here = np.unique(biome_shap[mask])
        b_ids_here = b_ids_here[~pd.isna(b_ids_here)]
        b_id = int(b_ids_here[0]) if len(b_ids_here) > 0 else -1

        for rank0, col_idx in enumerate(top_order_b):
            f = retained[col_idx]
            frames.append(pd.DataFrame({
                "BIOME_ID":               b_id,
                "BIOME_NAME":             str(b_name),
                "n_samples_in_biome":     n_b,
                "feature":                f,
                "rank_in_biome":          int(rank0 + 1),
                "mean_abs_SHAP_in_biome": float(b_mean_abs[col_idx]),
                "row_idx":                b_indices.astype(np.int64),
                "lon":                    lon_shap[b_indices],
                "lat":                    lat_shap[b_indices],
                "feature_value":          X_shap[f].iloc[b_indices].to_numpy(dtype=np.float64),
                "shap_value":             shap_values[b_indices, col_idx].astype(np.float64),
            }))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

# =============================================================================
# Final-model SHAP (Full Dataset - No Sampling)
# =============================================================================
def run_final_model_shap(final_model: xgb.XGBRegressor,
                         df: pd.DataFrame,
                         X: pd.DataFrame, y: np.ndarray,
                         lon: np.ndarray, lat: np.ndarray,
                         retained: list, final_params: dict,
                         use_gpu: bool, seed: int,
                         out_root: Path, logger: logging.Logger) -> list:
    out_root.mkdir(parents=True, exist_ok=True)
    biome_id   = df["BIOME_ID"].values
    biome_name = df["BIOME_NAME"].values

    X_shap = X.reset_index(drop=True)
    lon_shap = np.asarray(lon)
    lat_shap = np.asarray(lat)
    biome_shap = np.asarray(biome_id)
    biome_name_shap = np.asarray(biome_name)

    dep_topN = int(CONFIG["dependence_topN"])
    bee_topN = int(CONFIG["beeswarm_topN"])

    t0 = time.time()
    shap_values, base_value = shap_for_model(final_model, X_shap, retained)
    logger.info(f"  Base SHAP computation done in {time.time() - t0:.1f}s")

    # ---- (1) Global summary importance (ALL variables, Bootstrap deleted) ----
    mean_abs    = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    total = mean_abs.sum() if mean_abs.sum() > 0 else 1.0

    summary_df = pd.DataFrame({
        "feature":               retained,
        "mean_abs_SHAP":         mean_abs,
        "mean_signed_SHAP":      mean_signed,
        "contribution_pct":      100.0 * mean_abs / total,
    }).sort_values("mean_abs_SHAP", ascending=False).reset_index(drop=True)
    
    summary_df["rank"] = np.arange(1, len(summary_df) + 1)
    summary_cols = ["rank", "feature", "mean_abs_SHAP", "contribution_pct", "mean_signed_SHAP"]
    summary_df = summary_df[summary_cols]
    
    summary_csv = out_root / "3_SHAP_Global_Summary_Importance.csv"
    summary_df.to_csv(summary_csv, index=False)
    logger.info(f"Saved {summary_csv}")

    for _, row in summary_df.head(5).iterrows():
        logger.info(f"    {row['feature']:<10s}  rank={int(row['rank']):>2d}  "
                    f"mean_abs={row['mean_abs_SHAP']:.4f}")

    # ---- (1.5) Extract Pointwise Interactions for Top-2 Global Features ONLY ----
    try:
        t_int = time.time()
        top2_features = summary_df["feature"].head(2).tolist()
        if len(top2_features) == 2:
            f1, f2 = top2_features[0], top2_features[1]
            
            interaction_effect, main_effect_f1, main_effect_f2 = compute_top2_interactions_chunked(
                final_model, X_shap, retained, top2_features, logger, chunk_size=5000
            )
            
            inter_df = pd.DataFrame({
                "lon": lon_shap,
                "lat": lat_shap,
                "BIOME_ID": biome_shap,
                "BIOME_NAME": biome_name_shap,
                f"{f1}_value": X_shap[f1].values,
                f"{f2}_value": X_shap[f2].values,
                f"{f1}_main_SHAP": main_effect_f1,
                f"{f2}_main_SHAP": main_effect_f2,
                f"{f1}_{f2}_interaction_SHAP": interaction_effect
            })
            inter_csv = out_root / "6_SHAP_Interaction_Top2_Pointwise.csv"
            inter_df.to_csv(inter_csv, index=False)
            logger.info(f"Saved {inter_csv} ({time.time() - t_int:.1f}s)")
            
    except Exception as e:
        logger.warning(f"Failed to compute/export SHAP interactions: {e}")

    # ---- (2) Global Dependence data for Top-N features ----
    top_global = summary_df["feature"].head(dep_topN).tolist()
    dep_frames = []
    for f in top_global:
        col_idx = retained.index(f)
        dep_frames.append(pd.DataFrame({
            "feature":       f,
            "lon":           lon_shap,
            "lat":           lat_shap,
            "feature_value": X_shap[f].to_numpy(dtype=np.float64),
            "shap_value":    shap_values[:, col_idx].astype(np.float64),
        }))
    dep_csv = out_root / f"4_SHAP_Dependence_Nonlinear_Top{dep_topN}.csv"
    pd.concat(dep_frames, ignore_index=True).to_csv(dep_csv, index=False)

    # ---- (3) Dominant driver per location ----
    abs_shap = np.abs(shap_values)
    dom_idx = abs_shap.argmax(axis=1)
    row_arange = np.arange(len(shap_values))
    dom_signed = shap_values[row_arange, dom_idx]
    second_mask = abs_shap.copy()
    second_mask[row_arange, dom_idx] = -np.inf
    second_idx = second_mask.argmax(axis=1)
    retained_arr = np.array(retained, dtype=object)
    dom_df = pd.DataFrame({
        "lon":                lon_shap,
        "lat":                lat_shap,
        "BIOME_ID":           biome_shap,
        "BIOME_NAME":         biome_name_shap,
        "Dominant_Driver":    retained_arr[dom_idx],
        "Dominant_SHAP":      dom_signed,
        "Dominant_Direction": np.where(dom_signed > 0, "Positive", "Negative"),
        "Second_Driver":      retained_arr[second_idx],
        "Second_SHAP":        shap_values[row_arange, second_idx],
        "y_true":             y,
        "y_pred":             final_model.predict(X_shap),
    })
    dom_df.to_csv(out_root / "7_SHAP_Directional_Dominant_Map.csv", index=False)

    # =========================================================================
    # BIOME-LEVEL EXPORTS
    # =========================================================================
    biome_min = int(CONFIG["biome_min_samples"])
    unique_biomes = np.unique(biome_name_shap)

    biome_rows = []
    for b_name in unique_biomes:
        mask = (biome_name_shap == b_name)
        if mask.sum() < biome_min:
            continue
        sub_shap = shap_values[mask]
        b_mean_abs = np.abs(sub_shap).mean(axis=0)
        b_tot = b_mean_abs.sum() if b_mean_abs.sum() > 0 else 1.0
        b_pct = 100.0 * b_mean_abs / b_tot
        b_ids_here = np.unique(biome_shap[mask])
        b_ids_here = b_ids_here[~pd.isna(b_ids_here)]
        b_id = int(b_ids_here[0]) if len(b_ids_here) > 0 else -1

        for f, p_val, raw_val in zip(retained, b_pct, b_mean_abs):
            biome_rows.append({
                "BIOME_ID":       b_id,
                "BIOME_NAME":     str(b_name),
                "feature":        f,
                "mean_abs_SHAP":  float(raw_val),
                "importance_pct": float(p_val),
                "n_samples":      int(mask.sum()),
            })

    if biome_rows:
        biome_imp_df = (pd.DataFrame(biome_rows)
                          .sort_values(by=["BIOME_NAME", "mean_abs_SHAP"],
                                       ascending=[True, False]))
        biome_imp_df.to_csv(out_root / "8_SHAP_Biome_Importance_Pct.csv", index=False)

    biome_dep_frames = []
    for b_name in unique_biomes:
        mask = (biome_name_shap == b_name)
        if mask.sum() < biome_min:
            continue
        b_indices = np.where(mask)[0]
        sub_shap = shap_values[b_indices]
        b_mean_abs = np.abs(sub_shap).mean(axis=0)
        topN_idx_b = np.argsort(b_mean_abs)[::-1][:dep_topN]
        for col_idx in topN_idx_b:
            f = retained[col_idx]
            biome_dep_frames.append(pd.DataFrame({
                "BIOME_NAME":    str(b_name),
                "feature":       f,
                "lon":           lon_shap[b_indices],
                "lat":           lat_shap[b_indices],
                "feature_value": X_shap[f].iloc[b_indices].to_numpy(dtype=np.float64),
                "shap_value":    shap_values[b_indices, col_idx].astype(np.float64),
            }))

    if biome_dep_frames:
        pd.concat(biome_dep_frames, ignore_index=True).to_csv(
            out_root / f"9_SHAP_Biome_Dependence_Nonlinear_Top{dep_topN}.csv", index=False)

    # =========================================================================
    # BEESWARM EXPORTS
    # =========================================================================
    beeswarm_global = build_beeswarm_global(
        shap_values, X_shap, retained, mean_abs, lon_shap, lat_shap,
        biome_shap, biome_name_shap, bee_topN
    )
    beeswarm_global.to_csv(out_root / f"10_SHAP_Beeswarm_Global_Top{bee_topN}.csv", index=False)

    beeswarm_biome = build_beeswarm_biome(
        shap_values, X_shap, retained, lon_shap, lat_shap,
        biome_shap, biome_name_shap, biome_min, bee_topN, logger
    )
    if beeswarm_biome is not None:
        beeswarm_biome.to_csv(out_root / f"11_SHAP_Beeswarm_Biome_Top{bee_topN}.csv", index=False)

    return summary_df["feature"].tolist()

# =============================================================================
# Per-experiment driver: nested spatial-block CV
# =============================================================================
def run_experiment(block_buffer_km: float, block_size_km: float,
                   df: pd.DataFrame,
                   X: pd.DataFrame, y: np.ndarray,
                   lon: np.ndarray, lat: np.ndarray,
                   x_m: np.ndarray, y_m: np.ndarray,
                   retained: list, use_gpu: bool,
                   base_root: Path) -> tuple[Path, Optional[dict]]:
    out_root = base_root / f"block_{int(block_size_km)}km_buf_{int(block_buffer_km)}km"
    out_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_root, name=f"XGB_Part4_buf{int(block_buffer_km)}")

    biome_id   = df["BIOME_ID"].values
    biome_name = df["BIOME_NAME"].values
    single_grid = is_single_point_grid(CONFIG["param_grid"])

    fold_records = []
    pred_records = []
    temp_preds = {}
    chosen_param_history = []

    outer_folds_count = CONFIG["outer_folds"]
    roles_matrix = np.full((len(y), outer_folds_count), "Train", dtype=object)

    for fold_idx, tr_idx, te_idx, buf_idx in spatial_block_kfold(
            x_m, y_m, lat, lon, block_size_km, block_buffer_km,
            CONFIG["outer_folds"], CONFIG["seed"], logger):
        
        roles_matrix[te_idx, fold_idx - 1] = "Test"
        if len(buf_idx) > 0:
            roles_matrix[buf_idx, fold_idx - 1] = "Buffer"

        t0 = time.time()
        train_frac = len(tr_idx) / len(y)
        if len(tr_idx) < 100 or len(te_idx) < 30:
            continue

        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx],      y[te_idx]
        lon_tr, lat_tr = lon[tr_idx], lat[tr_idx]
        x_m_tr, y_m_tr = x_m[tr_idx], y_m[tr_idx]

        if single_grid:
            best_params = {k: v[0] for k, v in CONFIG["param_grid"].items()}
            best_inner_r2 = np.nan
        else:
            inner_seed = CONFIG["seed"] + fold_idx * 1000
            best_params, best_inner_r2 = inner_block_grid_search(
                X_tr, y_tr, x_m_tr, y_m_tr, lat_tr, lon_tr,
                CONFIG["param_grid"], CONFIG["inner_folds"],
                block_size_km, block_buffer_km,
                use_gpu, inner_seed, logger)
            if best_params is None:
                continue
        chosen_param_history.append((frozenset(best_params.items()), best_inner_r2))

        mdl = make_xgb(best_params, use_gpu, CONFIG["seed"])
        mdl.fit(X_tr, y_tr, verbose=False)
        y_pred = mdl.predict(X_te)

        m = compute_metrics(y_te, y_pred)
        
        # Transparent logging matching the requested format perfectly
        logger.info(f"|  fold {fold_idx}/{CONFIG['outer_folds']}  |  n_train={len(tr_idx):,}  n_test={len(te_idx):,}  (train_frac = {train_frac:.2f})")
        logger.info(f"     -> R2={m['R2']:+.4f}  RMSE={m['RMSE']:.4f}  ({time.time() - t0:.1f}s)")

        fold_records.append({
            "fold": fold_idx,
            "n_train": int(len(tr_idx)),
            "n_test":  int(len(te_idx)),
            "train_frac": float(train_frac),
            **m,
            **{f"hp_{k}": v for k, v in best_params.items()},
        })
        
        for i, gi in enumerate(te_idx):
            temp_preds[gi] = {
                "fold":       fold_idx,
                "global_idx": int(gi),
                "lon":        float(lon[gi]),
                "lat":        float(lat[gi]),
                "BIOME_ID":   float(biome_id[gi]) if pd.notna(biome_id[gi]) else np.nan,
                "BIOME_NAME": str(biome_name[gi]),
                "y_true":     float(y_te[i]),
                "y_pred_cv":  float(y_pred[i]),
                "residual":   float(y_pred[i] - y_te[i]),
            }

    for gi in sorted(temp_preds.keys()):
        record = temp_preds[gi]
        for f_idx in range(outer_folds_count):
            record[f"Fold_{f_idx + 1}_Role"] = roles_matrix[gi, f_idx]
        pred_records.append(record)

    if len(fold_records) == 0:
        for h in list(logger.handlers):
            h.close()
            logger.removeHandler(h)
        return out_root, None

    folds_df = pd.DataFrame(fold_records)
    metric_cols = ["R2", "RMSE"]
    summary = pd.DataFrame([
        {"fold": "MEAN",
         "n_train":    folds_df["n_train"].mean(),
         "n_test":     folds_df["n_test"].mean(),
         "train_frac": folds_df["train_frac"].mean(),
         **{c: folds_df[c].mean() for c in metric_cols}},
        {"fold": "STD",
         "n_train":    folds_df["n_train"].std(),
         "n_test":     folds_df["n_test"].std(),
         "train_frac": folds_df["train_frac"].std(),
         **{c: folds_df[c].std() for c in metric_cols}},
    ])
    
    # Detailed comprehensive CSV ensuring complete transparency
    pd.concat([folds_df, summary], ignore_index=True).to_csv(
        out_root / "1_Block_CV_Evaluation_Metrics.csv", index=False)

    pd.DataFrame(pred_records).to_csv(
        out_root / "1b_Block_CV_Per_Point_Predictions.csv", index=False)

    if len(chosen_param_history) > 0:
        counts = Counter([fp for fp, _ in chosen_param_history])
        max_count = max(counts.values())
        top_choices = [fp for fp, c in counts.items() if c == max_count]
        if len(top_choices) == 1:
            final_params = dict(top_choices[0])
        else:
            mean_r2 = {fp: np.mean([r for fp2, r in chosen_param_history if fp2 == fp])
                       for fp in top_choices}
            best_fp = max(mean_r2.items(), key=lambda kv: kv[1])[0]
            final_params = dict(best_fp)
    else:
        final_params = {k: v[0] for k, v in CONFIG["param_grid"].items()}

    with open(out_root / "final_hyperparameters.json", "w", encoding="utf-8") as f:
        json.dump({
            "pipeline":               "XGB_SHAP_NO_SAMPLING",
            "script_version":         CONFIG["script_version"],
            "cv_scheme":              "spatial_block_kfold",
            "projection":             EQUAL_AREA_EPSG if _PYPROJ_AVAILABLE else "equirectangular_fallback",
            "final_params":           final_params,
            "block_size_km":          block_size_km,
            "block_buffer_km":        block_buffer_km,
            "use_gpu":                use_gpu,
            "n_samples":              int(len(y)),
            "n_predictors":           int(len(retained)),
            "predictors":             retained,
            "outer_folds":            CONFIG["outer_folds"],
            "inner_folds":            CONFIG["inner_folds"],
            "dependence_topN":        CONFIG["dependence_topN"],
            "beeswarm_topN":          CONFIG["beeswarm_topN"],
            "biome_min_samples":      CONFIG["biome_min_samples"],
            "seed":                   CONFIG["seed"],
        }, f, indent=2)

    for h in list(logger.handlers):
        h.close()
        logger.removeHandler(h)

    return out_root, final_params

# =============================================================================
# MAIN
# =============================================================================
def main():
    base_root = Path(CONFIG["output_root"])
    base_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(base_root, name="XGB_Part4_setup")

    df = pd.read_csv(CONFIG["b_csv_path"])

    retained = [
        "HMI", "NTL", "POPD", "TTC",
        "MDR", "MTWeQ",
        "PWM", "PDM", "PS", "PWaQ", "PCQ",
        "ELEV", "SLP", "TRI", "TCURV", "TWI",
        "VPD", "SSR", "SWVL3", "WS10",
        "CEC", "CF", "CLAY", "PH","TN",  "SOC"
    ]

    target = CONFIG["target_name"]
    required_cols = [target] + list(retained) + ["lon", "lat"]
    df = df.dropna(subset=required_cols).reset_index(drop=True)
    
    if "BIOME_ID" not in df.columns:
        df["BIOME_ID"] = np.nan
    df["BIOME_NAME"] = df["BIOME_ID"].map(CONFIG["BIOME_ID_MAP"]).fillna("Unknown")
    df = df[df["BIOME_NAME"] != "Unknown"].reset_index(drop=True)

    X = df[retained].copy()
    y = df[target].values.astype(np.float64)
    lon = df["lon"].values
    lat = df["lat"].values

    x_m, y_m = project_to_meters(lon, lat)
    
    # Guarantee GPU status checks before pipeline execution
    use_gpu = gpu_available()
    if not use_gpu:
        logger.warning("GPU acceleration failed. Falling back to CPU processing.")
    else:
        logger.info("GPU acceleration strictly confirmed.")

    export_environment(
        out_path=base_root / "run_environment.json",
        use_gpu=use_gpu, n_samples=len(df), retained=retained, logger=logger,
    )

    np.random.seed(CONFIG["seed"])

    for buf_km in CONFIG["block_buffer_list"]:
        out_root, final_params = run_experiment(
            block_buffer_km=float(buf_km),
            block_size_km=float(CONFIG["block_size_km"]),
            df=df, X=X, y=y, lon=lon, lat=lat, x_m=x_m, y_m=y_m,
            retained=retained, use_gpu=use_gpu,
            base_root=base_root,
        )

        if final_params is None:
            continue

        exp_logger = setup_logger(out_root, name=f"XGB_Part4_buf{int(buf_km)}_post")
        final_model = make_xgb(final_params, use_gpu, CONFIG["seed"])
        final_model.fit(X, y, verbose=False)
        final_model.get_booster().save_model((out_root / "final_xgb_model.json").as_posix())
        
        final_shap_dir = out_root / "final_shap"
        run_final_model_shap(
            final_model, df, X, y, lon, lat, retained, 
            final_params=final_params, use_gpu=use_gpu, seed=CONFIG["seed"],
            out_root=final_shap_dir, logger=exp_logger
        )

        for h in list(exp_logger.handlers):
            h.close()
            exp_logger.removeHandler(h)

if __name__ == "__main__":
    main()