# -*- coding: utf-8 -*-
"""
AEF Modeling Pipeline (FIXED v2)
Fixes vs. previous version:
  - *** NEW FIX: JSON model save now uses booster.save_model() directly,
    bypassing the sklearn wrapper path that triggered
    "`_estimator_type` undefined" warnings on newer xgboost versions.
  - *** NEW FIX: XGBWrapper declares `_estimator_type = "regressor"` so any
    code path that does inspect the wrapper's sklearn metadata succeeds.
  - Target list restored to the full set; comment out targets you don't need.
  - Bootstrap models also saved as JSON (defends against joblib pickle drift
    after xgboost / sklearn upgrades).
  - point_bootstrap is GPU-native (no host-device round-trip per call).
  - SHAP subsampling seed is hashed per (biome, target).
  - Resume protection: biome-target combinations whose Metrics CSV already
    exists are skipped.
  - RNG isolation: bootstrap uses np.random.default_rng instead of global seed.
"""
import os
import sys
import time
import gc
import hashlib
import logging
import joblib
import warnings
import contextlib
import platform
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import pearsonr
import xgboost as xgb
import shap

warnings.filterwarnings("ignore")

# ==============================================================================
# Hardware
# ==============================================================================
try:
    import cupy as cp
    _CUDA_AVAILABLE = cp.cuda.is_available()
except ImportError:
    import numpy as cp
    _CUDA_AVAILABLE = False

# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class PipelineConfig:
    base_dir: str = "/data/global_mapping/models_V100/"
    output_dir: str = "/data/global_mapping/models_V100/AEF/"
    target_vars: List[str] = field(default_factory=lambda: [
        "AGE", "BA", "Trees_N", "AGB", "V", "shannon", "CV_DBH""DBH", "H"
    ])  # 
    biomes: List[int] = field(default_factory=lambda: list(range(1, 13)))
    n_bootstrap: int = 100
    seed: int = 42
    max_shap_samples: int = 10000

CONFIG = PipelineConfig()
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("AEF_Modeling")


def log_environment():
    try:
        import cupy as _cp
        cupy_ver = _cp.__version__
    except Exception:
        cupy_ver = "N/A"
    env_info = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgb.__version__,
        "shap": shap.__version__,
        "joblib": joblib.__version__,
        "cupy": cupy_ver,
        "cuda_available": _CUDA_AVAILABLE,
    }
    logger.info("================ ENVIRONMENT LOCK ================")
    for k, v in env_info.items():
        logger.info(f"  {k:<16}: {v}")
    logger.info("==================================================")
    return env_info


# ==============================================================================
# Utilities
# ==============================================================================
@contextlib.contextmanager
def gpu_mem_manager():
    try:
        yield
    finally:
        gc.collect()
        if _CUDA_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()


def to_cpu(data) -> np.ndarray:
    if _CUDA_AVAILABLE and hasattr(data, 'get'):
        return data.get()
    return np.asarray(data)


def stable_seed_from_id(base_seed: int, identifier) -> int:
    hash_bytes = hashlib.sha256(str(identifier).encode()).digest()
    offset = int.from_bytes(hash_bytes[:4], 'big') % 1000
    return base_seed + offset


def point_bootstrap(X, y, seed):
    """GPU-native bootstrap (no host-device copy per call)."""
    n = X.shape[0]
    if _CUDA_AVAILABLE:
        cp.random.seed(seed)
        idx = cp.random.choice(n, size=n, replace=True)
        return X[idx], y[idx]
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=n, replace=True)
        return X[idx], y[idx]


def safe_save_xgb_json(xgb_regressor, path: str) -> bool:
    """*** FIXED: Save xgboost model as JSON via the underlying Booster.

    Going through xgb.XGBRegressor.save_model() can raise
    "`_estimator_type` undefined" on newer xgboost versions when the model
    is wrapped inside a custom sklearn estimator. Calling
    booster.save_model() directly avoids that code path entirely and
    produces an identical JSON file that can be loaded later via:
        booster = xgb.Booster(); booster.load_model(path)
    """
    try:
        booster = xgb_regressor.get_booster()
        booster.save_model(path)
        return True
    except Exception as e:
        logger.warning(f"  Booster JSON save failed: {e}")
        return False


class XGBWrapper(BaseEstimator, RegressorMixin):
    # *** FIXED: explicit sklearn estimator type tag.
    # Some newer xgboost versions inspect this when save_model() is invoked
    # via the sklearn interface; declaring it here prevents
    # "`_estimator_type` undefined" errors on any code path.
    _estimator_type = "regressor"

    def __init__(self, n_estimators=500, max_depth=6, learning_rate=0.1, subsample=0.8,
                 colsample_bytree=0.8, gamma=0, reg_alpha=0, reg_lambda=1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.model = None

    def fit(self, X, y, verbose=False):
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            tree_method="hist",
            device="cuda" if _CUDA_AVAILABLE else "cpu",
            n_jobs=1 if _CUDA_AVAILABLE else -1,
            verbosity=0
        )
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


# ==============================================================================
# Pipeline
# ==============================================================================
class XGBPipeline:
    def __init__(self, biome_id, target_var):
        self.biome_id = biome_id
        self.target_var = target_var
        self.io_dir = os.path.join(CONFIG.output_dir, f"biome{biome_id}", self.target_var)
        self.feature_names = [f"A{i:02d}" for i in range(64)]

    def _bootstrap_seed(self, fold_id, b):
        identifier = f"biome{self.biome_id}_target{self.target_var}_fold{fold_id}_b{b}"
        return stable_seed_from_id(CONFIG.seed, identifier)

    def _shap_seed(self):
        """SHAP subsampling seed isolated per (biome, target)."""
        return stable_seed_from_id(
            CONFIG.seed, f"shap_biome{self.biome_id}_target{self.target_var}")

    def load_and_prep(self):
        csv_path = os.path.join(self.io_dir, "data_with_all_folds.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"File missing: {csv_path}")
            return False
        df = pd.read_csv(csv_path)

        missing_feats = [c for c in self.feature_names if c not in df.columns]
        if missing_feats:
            logger.error(
                f"Missing feature columns: {missing_feats[:5]}"
                f"{'...' if len(missing_feats) > 5 else ''} (total {len(missing_feats)}).")
            return False
        if self.target_var not in df.columns:
            logger.error(f"Target variable '{self.target_var}' missing from dataset.")
            return False
        for fold_col in ["fold_id_kmeans", "fold_id_random"]:
            if fold_col not in df.columns:
                logger.error(f"Required fold column '{fold_col}' not found.")
                return False

        raw_x = df[self.feature_names].values.astype(np.float32)
        raw_y = df[self.target_var].values.astype(np.float32).flatten()

        if (np.isnan(raw_x).any() or np.isnan(raw_y).any()
                or np.isinf(raw_x).any() or np.isinf(raw_y).any()):
            logger.error("NaN/Inf detected. Please clean upstream dataset.")
            return False

        self.X = cp.asarray(raw_x) if _CUDA_AVAILABLE else raw_x
        self.y = cp.asarray(raw_y) if _CUDA_AVAILABLE else raw_y
        self.f_kmeans = df["fold_id_kmeans"].values
        self.f_random = df["fold_id_random"].values
        return True

    def _grid_search(self, param_grid):
        params = list(ParameterGrid(param_grid))
        best_score, best_p = np.inf, None
        u_folds = np.unique(self.f_random)
        logger.info(f"Grid Search: {len(params)} combinations on {len(u_folds)} folds.")

        for i, p in enumerate(params, 1):
            t0 = time.time()
            rmses = []
            for f_id in u_folds:
                with gpu_mem_manager():
                    va, tr = (self.f_random == f_id), (self.f_random != f_id)
                    model = XGBWrapper(**p).fit(self.X[tr], self.y[tr])
                    pred = model.predict(self.X[va])
                    rmse = np.sqrt(mean_squared_error(to_cpu(self.y[va]), to_cpu(pred)))
                    rmses.append(rmse)
            avg_rmse = float(np.mean(rmses))
            if avg_rmse < best_score:
                best_score, best_p = avg_rmse, p
            logger.info(f"Trial {i}/{len(params)} | RMSE: {avg_rmse:.4f} | "
                        f"Time: {time.time()-t0:.2f}s")
        return best_p

    def run(self, param_grid):
        best_p = self._grid_search(param_grid)
        os.makedirs(self.io_dir, exist_ok=True)
        pd.DataFrame([best_p]).to_csv(
            os.path.join(self.io_dir, "best_params.csv"), index=False)

        logger.info("Generating Final Model and SHAP values...")
        t_shap_start = time.time()
        final_model = XGBWrapper(**best_p).fit(self.X, self.y)

        if self.X.shape[0] > CONFIG.max_shap_samples:
            shap_rng = np.random.default_rng(self._shap_seed())
            idx = shap_rng.choice(self.X.shape[0], CONFIG.max_shap_samples, replace=False)
            X_shap = to_cpu(self.X[idx])
        else:
            X_shap = to_cpu(self.X)

        # SHAP on CPU booster
        try:
            booster_cpu = final_model.model.get_booster().copy()
            booster_cpu.set_param({'device': 'cpu'})
            explainer = shap.TreeExplainer(booster_cpu)
        except Exception as e:
            logger.warning(f"Booster CPU copy failed ({e}); falling back to in-place switch.")
            try:
                final_model.model.get_booster().set_param({'device': 'cpu'})
            except Exception:
                pass
            explainer = shap.TreeExplainer(final_model.model)

        shap_v = explainer.shap_values(X_shap)
        shap_importance = np.abs(shap_v).mean(axis=0)
        pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': shap_importance
        }).to_csv(os.path.join(self.io_dir, "SHAP.csv"), index=False)
        logger.info(f"SHAP done | Time: {time.time()-t_shap_start:.2f}s")

        # CV evaluation
        strategies = [("Random", self.f_random), ("KMeans", self.f_kmeans)]
        results, all_preds = [], {}
        y_cpu = to_cpu(self.y)

        for name, folds in strategies:
            t_strat_start = time.time()
            logger.info(f">>> Strategy: {name} | Evaluating...")

            ps, pe = np.full(len(y_cpu), np.nan), np.full(len(y_cpu), np.nan)
            u_f = np.unique(folds)

            for fold_idx, fold_id in enumerate(u_f, 1):
                t_fold_start = time.time()
                va, tr = (folds == fold_id), (folds != fold_id)

                with gpu_mem_manager():
                    m_s = XGBWrapper(**best_p).fit(self.X[tr], self.y[tr])
                    ps[va] = to_cpu(m_s.predict(self.X[va]))
                    t_single = time.time() - t_fold_start

                    t_boot_start = time.time()
                    ens = np.zeros((va.sum(), CONFIG.n_bootstrap))
                    for b in range(CONFIG.n_bootstrap):
                        boot_seed = self._bootstrap_seed(f"{name}_{fold_id}", b)
                        xb, yb = point_bootstrap(self.X[tr], self.y[tr], boot_seed)
                        m_b = XGBWrapper(**best_p, random_state=boot_seed).fit(xb, yb)
                        ens[:, b] = to_cpu(m_b.predict(self.X[va]))
                        if (b + 1) % 10 == 0:
                            logger.info(
                                f"   [{name}] Fold {fold_idx}/{len(u_f)} | "
                                f"Bootstrap {b+1}/{CONFIG.n_bootstrap}...")

                    pe[va] = ens.mean(axis=1)
                    t_boot = time.time() - t_boot_start

                logger.info(
                    f"   [{name}] Fold {fold_idx} Finished | "
                    f"Single: {t_single:.1f}s | Bootstrap: {t_boot:.1f}s")

            for label, pv in [("Single", ps), ("Ensemble", pe)]:
                valid_idx = ~np.isnan(pv) & ~np.isnan(y_cpu)
                if valid_idx.sum() > 0:
                    results.append({
                        "Strategy": name, "Type": label,
                        "R": pearsonr(y_cpu[valid_idx], pv[valid_idx])[0],
                        "RMSE": float(np.sqrt(mean_squared_error(y_cpu[valid_idx], pv[valid_idx]))),
                        "MAE": float(mean_absolute_error(y_cpu[valid_idx], pv[valid_idx]))
                    })
                else:
                    logger.warning(f"[{name}/{label}] No valid predictions to score.")
            all_preds[f"{name}_S"], all_preds[f"{name}_E"] = ps, pe
            logger.info(f"DONE Strategy {name} | Total Time: {time.time()-t_strat_start:.2f}s")

        # Export production bootstrap models
        logger.info(f"Exporting {CONFIG.n_bootstrap} Final Bootstrap Models...")
        t_export_start = time.time()
        m_dir = os.path.join(self.io_dir, "Final_Models")
        os.makedirs(m_dir, exist_ok=True)
        json_fail_count = 0
        for b in range(CONFIG.n_bootstrap):
            with gpu_mem_manager():
                boot_seed = self._bootstrap_seed("FINAL", b)
                xb, yb = point_bootstrap(self.X, self.y, boot_seed)
                m = XGBWrapper(**best_p, random_state=boot_seed).fit(xb, yb)
                joblib.dump(m, os.path.join(m_dir, f"model_{b}.joblib"))
                # *** FIXED: save via booster.save_model() instead of sklearn wrapper ***
                ok = safe_save_xgb_json(m.model, os.path.join(m_dir, f"model_{b}.json"))
                if not ok:
                    json_fail_count += 1
            if (b + 1) % 10 == 0:
                logger.info(f"   Export Progress: {b+1}/{CONFIG.n_bootstrap}...")

        if json_fail_count > 0:
            logger.warning(f"   {json_fail_count}/{CONFIG.n_bootstrap} JSON saves failed "
                           f"(joblib copies are still complete).")
        logger.info(f"Export done | Time: {time.time()-t_export_start:.2f}s")
        return pd.DataFrame(results), all_preds


# ==============================================================================
# Main
# ==============================================================================
def is_already_completed(biome_id, target_var):
    """Resume protection: combo done iff Metrics CSV exists AND
    Final_Models has >= n_bootstrap joblib models."""
    io_dir = os.path.join(CONFIG.output_dir, f"biome{biome_id}", target_var)
    metrics_path = os.path.join(io_dir, f"B{biome_id}_Metrics.csv")
    m_dir = os.path.join(io_dir, "Final_Models")
    if not os.path.exists(metrics_path):
        return False
    if not os.path.isdir(m_dir):
        return False
    try:
        n_existing = len([f for f in os.listdir(m_dir) if f.endswith(".joblib")])
    except OSError:
        return False
    return n_existing >= CONFIG.n_bootstrap


if __name__ == "__main__":
    log_environment()

    grid = {
        "n_estimators": [800],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.01, 0.05],
        "subsample": [0.6, 0.8],
        "colsample_bytree": [0.5, 0.7],
        "reg_alpha": [0.5, 1.0, 5.0],
        "reg_lambda": [1.0, 5.0]
    }

    for current_target in CONFIG.target_vars:
        logger.info(f"********** STARTING PIPELINE FOR TARGET: {current_target} **********")

        for b_id in CONFIG.biomes:
            if is_already_completed(b_id, current_target):
                logger.info(f"SKIP Biome {b_id} [{current_target}] (already complete)")
                continue

            try:
                t_start = time.time()
                logger.info(f"================ START BIOME {b_id} [{current_target}] ================")
                pipeline = XGBPipeline(b_id, current_target)
                if pipeline.load_and_prep():
                    metrics, preds = pipeline.run(grid)

                    pd.DataFrame({"obs": to_cpu(pipeline.y), **preds}).to_csv(
                        os.path.join(pipeline.io_dir, f"B{b_id}_Preds.csv"), index=False)
                    metrics.to_csv(
                        os.path.join(pipeline.io_dir, f"B{b_id}_Metrics.csv"), index=False)

                    logger.info(
                        f"COMPLETE BIOME {b_id} [{current_target}] | "
                        f"Total Time: {(time.time()-t_start)/60:.2f}m")
                logger.info("====================================================\n")

                del pipeline
                gc.collect()
                if _CUDA_AVAILABLE:
                    cp.get_default_memory_pool().free_all_blocks()

            except Exception as e:
                logger.error(f"Failed Biome {b_id} [{current_target}]: {str(e)}", exc_info=True)
                continue

        logger.info(f"********** FINISHED TARGET {current_target} **********\n")
        gc.collect()
        if _CUDA_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()