# -*- coding: utf-8 -*-
"""
TRA Inversion / Mapping Pipeline (FIXED, parity with AEF inversion)
Fixes vs. previous version:
  - JSON model fallback: when joblib fails to unpickle a model, the loader
    falls back to xgboost-native .json saved alongside (forward-compatible
    after xgboost / sklearn upgrades).
  - Three-level resume protection:
      (a) per biome-target -> skip if both Final_Mosaic files exist
      (b) per raster tile  -> skip if both *_MEAN.tif and *_CV.tif exist
      (c) per mosaic file  -> skip if mosaic already on disk
"""
import os
import glob
import gc
import joblib
import numpy as np
import pandas as pd
import rasterio
import warnings
import logging
from rasterio.merge import merge
from tqdm import tqdm
from sklearn.base import BaseEstimator, RegressorMixin
import xgboost as xgb

warnings.filterwarnings("ignore")
logging.getLogger('xgboost').setLevel(logging.ERROR)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("TRA_Mapping")

# ==============================================================================
# Configuration
# ==============================================================================
MASTER_RASTER_BANDS = []  # only used if TIF metadata is missing
MIN_MODEL_SUCCESS_RATIO = 0.8


class XGBWrapper(BaseEstimator, RegressorMixin):
    """Stub for joblib unpickling. The real fitted model lives in self.model."""
    def __init__(self, **kwargs):
        self.model = None

    def predict(self, X):
        return self.model.predict(X)


def normalize_name(name):
    if not name:
        return ""
    base = str(name).replace(".tif", "").replace(".TIF", "")
    return base.lower().strip()


# ==============================================================================
# Model loading (joblib first, JSON fallback) ¡ª parity with AEF
# ==============================================================================
def load_one_model(joblib_path):
    """*** FIXED: try joblib first, fall back to xgboost-native JSON. ***
    Returns an XGBWrapper-like object exposing .predict(X), or None on failure.
    """
    # 1) joblib
    try:
        wrapper = joblib.load(joblib_path)
        if hasattr(wrapper, 'model') and wrapper.model is not None:
            wrapper.model.set_params(device="cpu", n_jobs=-1)
        return wrapper
    except (OSError, EOFError, KeyError, AttributeError, ValueError, TypeError) as e:
        logger.warning(
            f"  joblib load failed for {os.path.basename(joblib_path)} "
            f"({type(e).__name__}: {e}); trying JSON fallback.")

    # 2) JSON fallback
    json_path = joblib_path.replace(".joblib", ".json")
    if not os.path.exists(json_path):
        logger.warning(f"  No JSON fallback for {os.path.basename(joblib_path)}.")
        return None
    try:
        booster = xgb.XGBRegressor()
        booster.load_model(json_path)
        booster.set_params(device="cpu", n_jobs=-1)
        wrapper = XGBWrapper()
        wrapper.model = booster
        return wrapper
    except Exception as e:
        logger.warning(
            f"  JSON fallback failed for {os.path.basename(json_path)}: "
            f"{type(e).__name__}: {e}")
        return None


# ==============================================================================
# Per-raster prediction
# ==============================================================================
def predict_raster(raster_path, loaded_models, output_dir, feature_indices):
    base = os.path.splitext(os.path.basename(raster_path))[0]
    out_m = os.path.join(output_dir, f"{base}_MEAN.tif")
    out_c = os.path.join(output_dir, f"{base}_CV.tif")

    # *** FIXED: per-raster resume ¡ª skip if both outputs exist ***
    if os.path.exists(out_m) and os.path.exists(out_c):
        return

    if feature_indices is None or not feature_indices:
        logger.error(f"{base}: empty feature_indices, skipping.")
        return

    try:
        with rasterio.open(raster_path) as src:
            meta = src.meta.copy()
            data = src.read()
    except (rasterio.errors.RasterioIOError, OSError) as e:
        logger.error(f"Error reading {raster_path}: {type(e).__name__}: {e}")
        return

    # Slice only the bands the model needs
    try:
        selected_data = data[feature_indices, :, :]
    except IndexError as e:
        logger.error(f"{base}: Index error slicing bands. Check feature mapping. {e}")
        return

    C, H, W = selected_data.shape
    flat = np.moveaxis(selected_data, 0, -1).reshape(-1, C)

    # Robust NoData masking
    nodata = meta.get('nodata')
    finite_mask = np.all(np.isfinite(flat), axis=1)
    if nodata is None or (isinstance(nodata, float) and np.isnan(nodata)):
        mask = finite_mask
    else:
        mask = finite_mask & np.all(flat != nodata, axis=1)
    idx = np.where(mask)[0]

    if len(idx) == 0:
        logger.warning(f"{base}: no valid pixels.")
        return

    X_v = flat[idx].astype(np.float32)

    all_preds = []
    fail_count = 0
    fail_reasons = {}
    for m in loaded_models:
        try:
            p = m.predict(X_v)
            all_preds.append(p)
        except (ValueError, TypeError, AttributeError, xgb.core.XGBoostError) as e:
            fail_count += 1
            reason = type(e).__name__
            fail_reasons[reason] = fail_reasons.get(reason, 0) + 1

    n_total = len(loaded_models)
    n_ok = len(all_preds)
    if fail_count > 0:
        logger.warning(
            f"{base}: {fail_count}/{n_total} models failed during predict | "
            f"reasons: {fail_reasons}")

    if n_total == 0 or n_ok / n_total < MIN_MODEL_SUCCESS_RATIO:
        logger.error(
            f"{base}: only {n_ok}/{n_total} models succeeded "
            f"(< {MIN_MODEL_SUCCESS_RATIO:.0%}). Skipping output.")
        return

    all_preds = np.array(all_preds)
    mean_val = np.mean(all_preds, axis=0)
    std_val = np.std(all_preds, axis=0)

    # Safe CV
    abs_mean = np.abs(mean_val)
    cv_threshold = np.percentile(abs_mean, 1)
    valid_for_cv = abs_mean > cv_threshold
    cv_val = np.full_like(mean_val, -9999, dtype=np.float32)
    cv_val[valid_for_cv] = (std_val[valid_for_cv] / abs_mean[valid_for_cv]) * 100.0

    meta.update({
        'count': 1,
        'dtype': 'float32',
        'nodata': -9999,
        'compress': 'lzw',
        'bigtiff': 'YES',
        'driver': 'GTiff',
        'tiled': True,
        'blockxsize': 256,
        'blockysize': 256,
        'predictor': 3,
    })

    for v, p in [(mean_val, out_m), (cv_val, out_c)]:
        out_data = np.full(H * W, -9999, dtype=np.float32)
        out_data[idx] = v
        tmp_p = p + ".tmp"
        with rasterio.open(tmp_p, 'w', **meta) as dst:
            dst.write(out_data.reshape(H, W), 1)
        os.replace(tmp_p, p)

    del data, selected_data, flat, X_v, all_preds, mean_val, std_val, cv_val
    gc.collect()


# ==============================================================================
# Mosaic
# ==============================================================================
def run_mosaic(pred_dir, b_id, var):
    out_v = os.path.join(pred_dir, "Final_Mosaic")
    os.makedirs(out_v, exist_ok=True)

    for s in ["MEAN", "CV"]:
        out_path = os.path.join(out_v, f"Biome{b_id}_{var}_{s}_Mosaic.tif")
        # *** FIXED: per-mosaic resume ***
        if os.path.exists(out_path):
            logger.info(f"  Mosaic exists, skip: {os.path.basename(out_path)}")
            continue

        fs = sorted(glob.glob(os.path.join(pred_dir, f"*_{s}.tif")))
        if not fs:
            continue

        srcs_to_merge = []
        try:
            for f in fs:
                srcs_to_merge.append(rasterio.open(f))

            mos, out_trans = merge(srcs_to_merge, nodata=-9999)

            out_meta = srcs_to_merge[0].meta.copy()
            out_meta.update({
                "height": mos.shape[1],
                "width": mos.shape[2],
                "transform": out_trans,
                "bigtiff": "YES",
                "compress": "lzw",
                "nodata": -9999,
                "tiled": True,
                "blockxsize": 256,
                "blockysize": 256,
                "predictor": 3,
            })

            tmp_path = out_path + ".tmp"
            with rasterio.open(tmp_path, "w", **out_meta) as dst:
                dst.write(mos)
            os.replace(tmp_path, out_path)
        finally:
            for sc in srcs_to_merge:
                try:
                    sc.close()
                except Exception:
                    pass


# ==============================================================================
# Resume helpers
# ==============================================================================
def biome_target_already_done(out_dir, b_id, var):
    """*** FIXED: a biome-target is considered done if BOTH mosaics exist. ***"""
    mos_dir = os.path.join(out_dir, "Final_Mosaic")
    mean_p = os.path.join(mos_dir, f"Biome{b_id}_{var}_MEAN_Mosaic.tif")
    cv_p = os.path.join(mos_dir, f"Biome{b_id}_{var}_CV_Mosaic.tif")
    return os.path.exists(mean_p) and os.path.exists(cv_p)


# ==============================================================================
# Main
# ==============================================================================
if __name__ == "__main__":
    CFG = {
        "base":  "/data/global_mapping/models_V100/",
        "vars": ["DBH", "H", "AGE", "BA", "Trees_N", "AGB", "V", "shannon", "CV_DBH"],
        "biomes": list(range(1, 13))
    }

    try:
        import cupy as cp
        _GPU_AVAILABLE = cp.cuda.is_available()
    except Exception:
        _GPU_AVAILABLE = False

    for current_var in CFG['vars']:
        logger.info(f"========== STARTING PREDICTION FOR TARGET: {current_var} ==========")

        for b_id in CFG['biomes']:
            r_d = os.path.join(CFG['base'], "1km_TRA_Variables")
            feat_path = os.path.join(CFG['base'], "TraditionC", f"biome{b_id}",
                                     current_var, "Global_Selected_Features.csv")
            m_d = os.path.join(CFG['base'], "TraditionC", f"biome{b_id}",
                               current_var, "Final_Models")
            o_d = os.path.join(CFG['base'], "Tradition_mappingA", f"biome{b_id}", current_var)
            os.makedirs(o_d, exist_ok=True)

            # *** FIXED: per biome-target resume ***
            if biome_target_already_done(o_d, b_id, current_var):
                logger.info(f"SKIP Biome {b_id} [{current_var}] - mosaics already produced.")
                continue

            logger.info(f">>>>>> MAPPING BIOME {b_id} [{current_var}] <<<<<<")

            # 1. Load Feature List Used by Model
            if not os.path.exists(feat_path):
                logger.error(f"Feature selection file missing: {feat_path}")
                continue

            selected_df = pd.read_csv(feat_path)
            model_features_raw = selected_df['Feature'].tolist()
            model_features_norm = [normalize_name(f) for f in model_features_raw]
            logger.info(f"Model requires {len(model_features_raw)} features.")

            # 2. Find Raster Files & Models
            tif_pattern = os.path.join(r_d, f"BIOME_{b_id}_ID_*.tif")
            tifs = sorted(glob.glob(tif_pattern))
            m_paths = sorted(glob.glob(os.path.join(m_d, "*.joblib")))

            if not tifs:
                logger.warning(f"No raster files for Biome {b_id} in {r_d}")
                continue
            if not m_paths:
                logger.warning(f"No model files for Biome {b_id} [{current_var}]")
                continue

            # 3. Calibrate Band Indices
            logger.info(f"Calibrating bands using: {os.path.basename(tifs[0])}")
            tif_band_names = []
            try:
                with rasterio.open(tifs[0]) as src:
                    raw_descs = list(src.descriptions)
                    n_bands = src.count

                    if not any(raw_descs) or raw_descs[0] is None:
                        logger.error("No embedded band descriptions found in TIF. "
                                     "Populate MASTER_RASTER_BANDS or write descriptions upstream.")
                        continue

                    tif_band_names = [normalize_name(d) for d in raw_descs]
                    logger.info(f"Reference TIF has {n_bands} bands with descriptions.")
            except (rasterio.errors.RasterioIOError, OSError) as e:
                logger.error(f"Failed to read raster metadata: {type(e).__name__}: {e}")
                continue

            # 4. Match Model Features to TIF Bands
            feature_indices = []
            missing_feats = []
            for m_feat_raw, m_feat_norm in zip(model_features_raw, model_features_norm):
                if m_feat_norm in tif_band_names:
                    idx = tif_band_names.index(m_feat_norm)
                    feature_indices.append(idx)
                else:
                    missing_feats.append(m_feat_raw)

            if missing_feats:
                logger.error(
                    f"[CRITICAL] {len(missing_feats)}/{len(model_features_raw)} "
                    f"features missing in raster: "
                    f"{missing_feats[:10]}{'...' if len(missing_feats) > 10 else ''}")
                logger.error(f"Available raster bands (sample): {tif_band_names[:10]}")
                continue

            if len(feature_indices) != len(model_features_raw):
                logger.error(
                    f"Feature index count {len(feature_indices)} != "
                    f"required {len(model_features_raw)}. Aborting biome.")
                continue
            logger.info(f"Successfully matched {len(feature_indices)} features.")

            # 5. *** FIXED: Load models with JSON fallback ***
            logger.info(f"Loading {len(m_paths)} models for {current_var} Biome {b_id}...")
            loaded_models = []
            load_fail = 0
            for mp in m_paths:
                wrapper = load_one_model(mp)
                if wrapper is None:
                    load_fail += 1
                else:
                    loaded_models.append(wrapper)

            if load_fail > 0:
                logger.warning(f"  Total {load_fail}/{len(m_paths)} models failed to load.")

            if not loaded_models:
                logger.error(f"  No usable models for Biome {b_id} [{current_var}]. Skipping.")
                continue

            # 6. Prediction Loop
            for tif in tqdm(tifs, desc=f"Mapping Biome {b_id} [{current_var}]"):
                predict_raster(tif, loaded_models, o_d, feature_indices)

            logger.info(f"Creating Mosaic for {current_var} Biome {b_id}...")
            run_mosaic(o_d, b_id, current_var)

            del loaded_models
            gc.collect()
            if _GPU_AVAILABLE:
                cp.get_default_memory_pool().free_all_blocks()

        logger.info(f"========== FINISHED TARGET: {current_var} ==========\n")