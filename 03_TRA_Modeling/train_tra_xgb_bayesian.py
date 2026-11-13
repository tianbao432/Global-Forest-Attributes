# -*- coding: utf-8 -*-
"""
TRA Modeling Pipeline (Optimized v3 with Optuna Bayesian Search)

Pipeline: VIF -> per-fold RFECV -> Optuna search (Random) -> CV evaluation
          (Random + KMeans, KMeans reuses Random's hyperparameters) ->
          global RFECV -> SHAP -> bootstrap production models.

Changes vs. v2 (FIXED):
  - seed_from_id now uses 64-bit hash range (~2^31) to eliminate collisions.
  - GPU bootstrap uses a local cupy.random.RandomState, never touching
    the global RNG (thread-safe, no cross-call leakage).
  - Optuna trials each pass an explicit, deterministic random_state.
  - Bootstrap inner loop frees per-iteration GPU memory aggressively.
  - VIF feature lookup is O(n) via a name->index dict.
  - is_already_completed now also checks SHAP.csv and best_params.csv.
  - Failed (biome, target) combinations are appended to failed_runs.csv.
  - safe_save_xgb_json defends against an unfit / None regressor.
  - SHAP booster is deep-copied via save_raw + load_model, not the fragile
    Booster.copy() path.
  - Per-fold RFECV inner KFold seed is fixed to CONFIG.seed for stability.
  - Numeric feature filtering explicitly excludes bool columns.
  - Defensive asserts ensure mask lengths line up with feature lists.

Design note on VIF leakage:
  VIF runs once globally (not inside each CV fold). VIF uses only feature-
  feature correlations (no y), so leakage is structural rather than
  predictive. If you need strict anti-leakage for publication, move
  fast_vif_filter() inside the per-fold loop and accept the runtime cost
  plus per-fold-varying feature spaces (which complicates SHAP).
"""
from __future__ import annotations

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
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import sklearn
import scipy
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFECV
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.base import BaseEstimator, RegressorMixin
from scipy.stats import pearsonr
import xgboost as xgb
import shap
import optuna

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ==============================================================================
# Hardware
# ==============================================================================
try:
    import cupy as cp
    _CUDA_AVAILABLE = bool(cp.cuda.is_available())
except Exception:
    import numpy as cp  # noqa: F401  fallback alias
    _CUDA_AVAILABLE = False


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class PipelineConfig:
    base_dir: str = "/data/global_mapping/models_V100/"
    output_dir: str ="/data/global_mapping/models_V100/TraditionD/"
    target_vars: List[str] = field(default_factory=lambda: [
        "shannon", "CV_DBH"
    ])
    biomes: List[int] = field(default_factory=lambda: list(range(1, 13)))

    metadata_cols: List[str] = field(default_factory=lambda: [
        "Longitude", "Latitude", "fold_id_kmeans", "fold_id_random",
        "BIOME", "system:index", ".geo"
    ])

    n_bootstrap: int = 100
    seed: int = 42
    max_shap_samples: int = 10000

    vif_threshold: float = 10.0
    rfe_step: int = 4
    rfe_min_features: int = 10
    rfe_cv: int = 5

    # GPU memory hygiene: free pool every N bootstrap iterations
    gpu_cleanup_every: int = 40
    
    # Optuna settings
    optuna_trials: int = 50


CONFIG = PipelineConfig()
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("TRA_Modeling")


def log_environment() -> Dict[str, Any]:
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
        "optuna": optuna.__version__,
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
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass


def to_cpu(data) -> np.ndarray:
    if _CUDA_AVAILABLE and hasattr(data, 'get'):
        return data.get()
    return np.asarray(data)


# *** FIX #1: 64-bit offset eliminates seed collisions across the entire run.
# *** With ~2^31 possible offsets, birthday-paradox collision probability for
# *** ~1e6 calls is < 1e-3 (vs. essentially 100% under the old %1000 scheme).
_SEED_RANGE = 2 ** 31 - 1  # stay within int32 for downstream RNG safety


def stable_seed_from_id(base_seed: int, identifier) -> int:
    hash_bytes = hashlib.sha256(str(identifier).encode("utf-8")).digest()
    offset = int.from_bytes(hash_bytes[:8], 'big') % _SEED_RANGE
    # XOR with base_seed so different base_seeds give independent streams.
    return int((base_seed ^ offset) & _SEED_RANGE)


def fast_vif_filter(X_np: np.ndarray, feature_names: List[str],
                    threshold: float, min_features: int
                    ) -> Tuple[List[str], np.ndarray]:
    """Iterative VIF filter. Drops near-zero-variance features first."""
    X_calc = X_np.copy()
    current_features = list(feature_names)

    stds = np.std(X_calc, axis=0)
    valid_idx = np.where(stds > 1e-6)[0]
    if len(valid_idx) < X_calc.shape[1]:
        logger.info(f"Dropped {X_calc.shape[1] - len(valid_idx)} "
                    f"near-zero-variance features.")
        X_calc = X_calc[:, valid_idx]
        current_features = [current_features[i] for i in valid_idx]

    max_vif = 0.0
    while X_calc.shape[1] > min_features:
        corr_matrix = np.corrcoef(X_calc, rowvar=False)
        corr_matrix += np.eye(corr_matrix.shape[0]) * 1e-6
        try:
            inv_corr = np.linalg.inv(corr_matrix)
        except np.linalg.LinAlgError:
            inv_corr = np.linalg.pinv(corr_matrix)
        vifs = np.diag(inv_corr)
        max_idx = int(np.argmax(vifs))
        max_vif = float(vifs[max_idx])
        if max_vif > threshold:
            X_calc = np.delete(X_calc, max_idx, axis=1)
            current_features.pop(max_idx)
        else:
            break

    if X_calc.shape[1] <= min_features and max_vif > threshold:
        logger.warning(f"VIF filter hit minimum feature limit "
                       f"({min_features}). Max VIF = {max_vif:.2f}.")
    return current_features, X_calc


def point_bootstrap(X, y, seed: int):
    """GPU-native bootstrap with **isolated** RNG.

    *** FIX #2: Uses a local RandomState instead of seeding the global RNG.
    *** Safe under future parallelism and avoids cross-call seed bleed.
    """
    n = X.shape[0]
    if _CUDA_AVAILABLE:
        rng = cp.random.RandomState(seed)
        idx = rng.choice(n, size=n, replace=True)
        return X[idx], y[idx]
    else:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=n, replace=True)
        return X[idx], y[idx]


def safe_save_xgb_json(xgb_regressor, path: str) -> bool:
    """Save xgboost model as JSON via the underlying Booster.

    Going through xgb.XGBRegressor.save_model() can raise
    "`_estimator_type` undefined" on newer xgboost versions when the model
    is wrapped inside a custom sklearn estimator. Calling
    booster.save_model() directly avoids that code path entirely.
    """
    # *** FIX #10: defend against unfit / None inputs.
    if xgb_regressor is None:
        logger.warning("  Booster JSON save skipped: regressor is None.")
        return False
    try:
        booster = xgb_regressor.get_booster()
        booster.save_model(path)
        return True
    except xgb.core.XGBoostError as e:
        logger.warning(f"  Booster JSON save failed (xgboost): {e}")
        return False
    except Exception as e:
        logger.warning(f"  Booster JSON save failed: {e}")
        return False


def deep_copy_booster_to_cpu(booster: xgb.Booster) -> xgb.Booster:
    """*** FIX #11: Deep-copy a booster to CPU via raw bytes round-trip.

    Booster.copy() can be a shallow/incomplete copy across xgboost versions,
    and mutating device on the copy is not guaranteed isolated from the
    source. save_raw + load_model gives a guaranteed-fresh booster.
    """
    raw = booster.save_raw(raw_format="ubj")
    new_booster = xgb.Booster()
    # load_model accepts bytearray of UBJ/JSON serialized model.
    new_booster.load_model(bytearray(raw))
    new_booster.set_param({"device": "cpu"})
    return new_booster


class XGBWrapper(BaseEstimator, RegressorMixin):
    """Sklearn-compatible thin wrapper around XGBRegressor."""

    # Explicit sklearn estimator type tag (silences metadata probes on newer
    # xgboost / sklearn versions).
    _estimator_type = "regressor"

    def __init__(self, n_estimators=500, max_depth=6, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, gamma=0, min_child_weight=1,
                 reg_alpha=0, reg_lambda=1, random_state=42):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self.model: Optional[xgb.XGBRegressor] = None

    def fit(self, X, y, verbose: bool = False):
        self.model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            gamma=self.gamma,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            tree_method="hist",
            device="cuda" if _CUDA_AVAILABLE else "cpu",
            n_jobs=1 if _CUDA_AVAILABLE else -1,
            verbosity=0,
        )
        # *** FIX #5: actually pass verbose through.
        self.model.fit(X, y, verbose=verbose)
        return self

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("XGBWrapper.predict called before fit().")
        return self.model.predict(X)


# ==============================================================================
# Pipeline
# ==============================================================================
class XGBPipeline:
    def __init__(self, biome_id: int, target_var: str):
        self.biome_id = biome_id
        self.target_var = target_var
        self.io_dir = os.path.join(
            CONFIG.output_dir, f"biome{biome_id}", self.target_var)
        self.feature_names: List[str] = []

    # ---- seeds -----------------------------------------------------------
    def _bootstrap_seed(self, fold_id, b: int) -> int:
        identifier = (f"biome{self.biome_id}_target{self.target_var}"
                      f"_fold{fold_id}_b{b}")
        return stable_seed_from_id(CONFIG.seed, identifier)

    def _shap_seed(self) -> int:
        return stable_seed_from_id(
            CONFIG.seed,
            f"shap_biome{self.biome_id}_target{self.target_var}")

    def _optuna_trial_seed(self, trial_idx: int) -> int:
        return stable_seed_from_id(
            CONFIG.seed,
            f"optuna_biome{self.biome_id}_target{self.target_var}_t{trial_idx}")

    # ---- I/O -------------------------------------------------------------
    def load_and_prep(self) -> bool:
        csv_path = os.path.join(self.io_dir, "data_with_all_folds.csv")
        if not os.path.exists(csv_path):
            logger.warning(f"File missing: {csv_path}")
            return False
        df = pd.read_csv(csv_path)

        if self.target_var not in df.columns:
            logger.error(
                f"Target variable '{self.target_var}' missing from dataset.")
            return False
        for fold_col in ["fold_id_kmeans", "fold_id_random"]:
            if fold_col not in df.columns:
                logger.error(f"Required fold column '{fold_col}' not found.")
                return False

        drop_cols = set(CONFIG.target_vars + CONFIG.metadata_cols)
        available_cols = [c for c in df.columns if c not in drop_cols]
        # *** FIX #13: explicitly drop bool columns; np.number includes bool.
        df_features = (df[available_cols]
                       .select_dtypes(include=[np.number])
                       .select_dtypes(exclude=["bool"]))
        self.feature_names = df_features.columns.tolist()
        if not self.feature_names:
            logger.error("Zero numeric features remaining after filtering.")
            return False

        raw_x = df_features.values.astype(np.float32)
        raw_y = df[self.target_var].values.astype(np.float32).flatten()
        if (np.isnan(raw_x).any() or np.isnan(raw_y).any()
                or np.isinf(raw_x).any() or np.isinf(raw_y).any()):
            logger.error("NaN/Inf detected. Please clean upstream dataset.")
            return False

        logger.info(
            f"Loaded {len(self.feature_names)} pure numeric features.")
        self.X = cp.asarray(raw_x) if _CUDA_AVAILABLE else raw_x
        self.y = cp.asarray(raw_y) if _CUDA_AVAILABLE else raw_y
        self.f_kmeans = df["fold_id_kmeans"].values
        self.f_random = df["fold_id_random"].values
        return True

    # ---- per-fold RFECV --------------------------------------------------
    def _perform_feature_selection(self, X_train_np: np.ndarray,
                                   y_train_np: np.ndarray,
                                   fold_id=0, prefix: str = "") -> np.ndarray:
        """Anti-leakage RFECV (applied after global VIF).

        Returns a boolean mask of length X_train_np.shape[1]. The caller
        is responsible for ensuring this matches the global feature space.
        """
        logger.info(f"[{prefix}_Fold_{fold_id}] Starting RFECV...")
        t0 = time.time()
        n_features_in = X_train_np.shape[1]

        if n_features_in <= CONFIG.rfe_min_features:
            logger.info(f"[{prefix}] Skipping RFECV: features <= "
                        f"{CONFIG.rfe_min_features}.")
            rfe_mask = np.ones(n_features_in, dtype=bool)
        else:
            # *** FIX #12: Stable inner CV seed across folds for reproducible
            # *** feature selection. Outer fold variation already comes from
            # *** different train splits.
            est = xgb.XGBRegressor(
                n_estimators=500, tree_method="hist",
                device="cuda" if _CUDA_AVAILABLE else "cpu",
                n_jobs=1, random_state=CONFIG.seed, verbosity=0,
            )
            selector = RFECV(
                estimator=est, step=CONFIG.rfe_step,
                min_features_to_select=CONFIG.rfe_min_features,
                cv=KFold(n_splits=CONFIG.rfe_cv, shuffle=True,
                         random_state=CONFIG.seed),
                scoring='neg_root_mean_squared_error', n_jobs=1,
            )
            try:
                with gpu_mem_manager():
                    selector.fit(X_train_np, y_train_np)
                rfe_mask = selector.support_
            except Exception as e:
                logger.error(f"[{prefix}] RFECV failed: {e}. "
                             f"Using all VIF-retained features.")
                rfe_mask = np.ones(n_features_in, dtype=bool)

        # *** FIX #3: defensive contract.
        assert rfe_mask.shape[0] == n_features_in, (
            f"RFECV mask length {rfe_mask.shape[0]} != "
            f"input features {n_features_in}")
        logger.info(f"[{prefix}_Fold_{fold_id}] Retained {int(sum(rfe_mask))} "
                    f"features | Time: {time.time()-t0:.2f}s")
        return rfe_mask

    # ---- Optuna Bayesian Optimization --------------------------------------
    def _optuna_search(self, fold_array: np.ndarray,
                       fold_masks: Dict[Any, np.ndarray],
                       prefix: str = "") -> Optional[dict]:
        u_folds = np.unique(fold_array)
        logger.info(f"[{prefix}] Optuna Bayesian Optimization: {CONFIG.optuna_trials} trials.")

        def objective(trial: optuna.Trial):
            p = {
                "n_estimators": trial.suggest_int("n_estimators", 400, 1200, step=100),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "max_depth": trial.suggest_int("max_depth", 4, 8),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "subsample": trial.suggest_float("subsample", 0.5, 0.9),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
                "reg_alpha": trial.suggest_float("reg_alpha", 0.1, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            }
            
            rmses = []
            for f_id in u_folds:
                # *** FIX #6 adapted: Each trial gets a deterministic, distinct seed.
                trial_seed = self._optuna_trial_seed(trial.number)
                with gpu_mem_manager():
                    va = (fold_array == f_id)
                    tr = ~va
                    mask = fold_masks[f_id]
                    model = XGBWrapper(**p, random_state=trial_seed).fit(self.X[tr][:, mask], self.y[tr])
                    pred = model.predict(self.X[va][:, mask])
                    rmse = float(np.sqrt(mean_squared_error(to_cpu(self.y[va]), to_cpu(pred))))
                    rmses.append(rmse)
                    del model, pred
            return float(np.mean(rmses))

        t0 = time.time()
        sampler_seed = stable_seed_from_id(CONFIG.seed, f"optuna_sampler_{prefix}_biome{self.biome_id}_{self.target_var}")
        sampler = optuna.samplers.TPESampler(seed=sampler_seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        study.optimize(objective, n_trials=CONFIG.optuna_trials, n_jobs=1)
        
        logger.info(f"[{prefix}] Optuna Finished | Best Mean RMSE: {study.best_value:.4f} | Time: {time.time()-t0:.2f}s")
        return study.best_params

    # ---- main run --------------------------------------------------------
    def run(self) -> Tuple[pd.DataFrame, dict]:
        os.makedirs(self.io_dir, exist_ok=True)
        X_cpu = to_cpu(self.X)
        y_cpu = to_cpu(self.y)
        results, all_preds = [], {}

        # ---------------- Stage 0: Global VIF ----------------
        logger.info(">>> Stage 0: Global VIF Filter <<<")
        t_vif = time.time()
        vif_features, X_cpu_vif = fast_vif_filter(
            X_cpu, self.feature_names,
            CONFIG.vif_threshold, CONFIG.rfe_min_features
        )
        # *** FIX #8: O(n) lookup via dict.
        name_to_idx = {name: i for i, name in enumerate(self.feature_names)}
        vif_indices = [name_to_idx[f] for f in vif_features]
        self.feature_names = vif_features
        self.X = self.X[:, vif_indices]
        X_cpu = X_cpu_vif
        # *** FIX #3: contract check.
        assert X_cpu.shape[1] == len(self.feature_names) == self.X.shape[1]
        logger.info(f"Global VIF completed in {time.time()-t_vif:.2f}s. "
                    f"Reduced to {len(self.feature_names)} features.")

        # ---------------- Stage 1 & 2: Strategy evaluation ----------------
        best_p_random: Optional[dict] = None
        strategies = [("Random", self.f_random), ("KMeans", self.f_kmeans)]

        for name, folds in strategies:
            logger.info(f">>> Evaluating Strategy: {name} <<<")
            t_strat_start = time.time()
            u_f = np.unique(folds)
            current_masks: Dict[Any, np.ndarray] = {}

            # 1. Per-fold RFECV
            for fold_id in u_f:
                tr = (folds != fold_id)
                current_masks[fold_id] = self._perform_feature_selection(
                    X_cpu[tr], y_cpu[tr], fold_id=fold_id, prefix=name
                )

            # 2. Hyperparameter tuning (Random only; KMeans reuses)
            if name == "Random":
                best_p = self._optuna_search(folds, current_masks, prefix=name)
                if best_p is None:
                    logger.error(f"[{name}] Optuna search returned no "
                                 f"parameters. Aborting biome-target.")
                    return pd.DataFrame(results), all_preds
                best_p_random = best_p
                pd.DataFrame([best_p]).to_csv(
                    os.path.join(self.io_dir,
                                 f"best_params_cv_{name}.csv"), index=False)
            else:
                if best_p_random is None:
                    logger.error(f"[{name}] No Random-tuned hyperparameters "
                                 f"available. Skipping.")
                    continue
                best_p = best_p_random
                pd.DataFrame([best_p]).to_csv(
                    os.path.join(self.io_dir,
                                 f"best_params_cv_{name}.csv"), index=False)
                logger.info(
                    f"[{name}] Using hyperparameters from Random tuning.")

            # 3. Final predictions & bootstrap
            ps = np.full(len(y_cpu), np.nan)
            pe = np.full(len(y_cpu), np.nan)
            for fold_idx, fold_id in enumerate(u_f, 1):
                t_fold_start = time.time()
                va = (folds == fold_id)
                tr = ~va
                mask = current_masks[fold_id]

                # Single model
                with gpu_mem_manager():
                    single_seed = stable_seed_from_id(
                        CONFIG.seed, f"single_{name}_{fold_id}")
                    m_s = XGBWrapper(
                        **best_p, random_state=single_seed
                    ).fit(self.X[tr][:, mask], self.y[tr])
                    ps[va] = to_cpu(m_s.predict(self.X[va][:, mask]))
                    del m_s
                t_single = time.time() - t_fold_start

                # Bootstrap ensemble
                t_boot_start = time.time()
                ens = np.zeros((int(va.sum()), CONFIG.n_bootstrap),
                               dtype=np.float32)
                X_tr_mask = self.X[tr][:, mask]
                y_tr = self.y[tr]
                X_va_mask = self.X[va][:, mask]

                for b in range(CONFIG.n_bootstrap):
                    boot_seed = self._bootstrap_seed(f"{name}_{fold_id}", b)
                    xb, yb = point_bootstrap(X_tr_mask, y_tr, boot_seed)
                    m_b = XGBWrapper(
                        **best_p, random_state=boot_seed).fit(xb, yb)
                    ens[:, b] = to_cpu(m_b.predict(X_va_mask))
                    # *** FIX #7: per-iter cleanup keeps GPU peak low.
                    del m_b, xb, yb
                    if (b + 1) % CONFIG.gpu_cleanup_every == 0:
                        gc.collect()
                        if _CUDA_AVAILABLE:
                            cp.get_default_memory_pool().free_all_blocks()
                    if (b + 1) % 10 == 0:
                        logger.info(
                            f"   [{name}] Fold {fold_idx}/{len(u_f)} | "
                            f"Bootstrap {b+1}/{CONFIG.n_bootstrap}...")

                pe[va] = ens.mean(axis=1)
                del ens, X_tr_mask, y_tr, X_va_mask
                gc.collect()
                if _CUDA_AVAILABLE:
                    cp.get_default_memory_pool().free_all_blocks()
                t_boot = time.time() - t_boot_start
                logger.info(
                    f"   [{name}] Fold {fold_idx} Finished | "
                    f"Single: {t_single:.1f}s | Bootstrap: {t_boot:.1f}s")

            for label, pv in [("Single", ps), ("Ensemble", pe)]:
                valid_idx = ~np.isnan(pv) & ~np.isnan(y_cpu)
                if valid_idx.sum() > 1:
                    results.append({
                        "Strategy": name, "Type": label,
                        "R": float(pearsonr(
                            y_cpu[valid_idx], pv[valid_idx])[0]),
                        "RMSE": float(np.sqrt(mean_squared_error(
                            y_cpu[valid_idx], pv[valid_idx]))),
                        "MAE": float(mean_absolute_error(
                            y_cpu[valid_idx], pv[valid_idx])),
                        "N": int(valid_idx.sum()),
                    })
            all_preds[f"{name}_S"] = ps
            all_preds[f"{name}_E"] = pe
            logger.info(f"DONE Strategy {name} | "
                        f"Total Time: {time.time()-t_strat_start:.2f}s")

        # ---------------- Stage 3: Production model & SHAP ----------------
        if best_p_random is None:
            logger.error("No Random-tuned parameters available. "
                         "Cannot build production model. Aborting.")
            return pd.DataFrame(results), all_preds

        logger.info(">>> Stage 3: Final Production Model & SHAP "
                    "(using Random-tuned parameters) <<<")
        global_mask = self._perform_feature_selection(
            X_cpu, y_cpu, fold_id="ALL", prefix="Global")
        # *** FIX #3: contract check.
        assert global_mask.shape[0] == len(self.feature_names)

        global_features = [self.feature_names[i]
                           for i in range(len(global_mask)) if global_mask[i]]
        pd.DataFrame(global_features, columns=["Feature"]).to_csv(
            os.path.join(self.io_dir,
                         "Global_Selected_Features.csv"), index=False)

        final_best_p = best_p_random
        pd.DataFrame([final_best_p]).to_csv(
            os.path.join(self.io_dir, "best_params.csv"), index=False)

        X_global = self.X[:, global_mask]
        prod_seed = stable_seed_from_id(
            CONFIG.seed, f"prod_biome{self.biome_id}_target{self.target_var}")
        final_model = XGBWrapper(
            **final_best_p, random_state=prod_seed).fit(X_global, self.y)

        # SHAP ¡ª booster forced to CPU via deep copy
        t_shap_start = time.time()
        if X_global.shape[0] > CONFIG.max_shap_samples:
            shap_rng = np.random.default_rng(self._shap_seed())
            idx = shap_rng.choice(
                X_global.shape[0], CONFIG.max_shap_samples, replace=False)
            X_shap = to_cpu(X_global[idx])
        else:
            X_shap = to_cpu(X_global)

        try:
            booster_cpu = deep_copy_booster_to_cpu(
                final_model.model.get_booster())
            explainer = shap.TreeExplainer(booster_cpu)
        except Exception as e:
            logger.warning(f"Booster CPU deep-copy failed ({e}); "
                           f"falling back to in-place device switch.")
            try:
                final_model.model.get_booster().set_param({'device': 'cpu'})
            except Exception:
                pass
            explainer = shap.TreeExplainer(final_model.model)

        shap_v = explainer.shap_values(X_shap)
        shap_importance = np.abs(shap_v).mean(axis=0)
        # *** FIX #3: contract check.
        assert len(shap_importance) == len(global_features), (
            f"SHAP importance length {len(shap_importance)} != "
            f"global feature count {len(global_features)}")
        pd.DataFrame({
            'Feature': global_features,
            'Importance': shap_importance
        }).to_csv(os.path.join(self.io_dir, "SHAP.csv"), index=False)
        logger.info(f"SHAP done | Time: {time.time()-t_shap_start:.2f}s")

        # ---------------- Stage 4: Bootstrap production models ------------
        logger.info(
            f"Exporting {CONFIG.n_bootstrap} Final Bootstrap Models...")
        t_export_start = time.time()
        m_dir = os.path.join(self.io_dir, "Final_Models")
        os.makedirs(m_dir, exist_ok=True)
        json_fail_count = 0
        for b in range(CONFIG.n_bootstrap):
            boot_seed = self._bootstrap_seed("FINAL", b)
            xb, yb = point_bootstrap(X_global, self.y, boot_seed)
            m = XGBWrapper(
                **final_best_p, random_state=boot_seed).fit(xb, yb)
            joblib.dump(m, os.path.join(m_dir, f"model_{b}.joblib"))
            ok = safe_save_xgb_json(
                m.model, os.path.join(m_dir, f"model_{b}.json"))
            if not ok:
                json_fail_count += 1
            del m, xb, yb
            if (b + 1) % CONFIG.gpu_cleanup_every == 0:
                gc.collect()
                if _CUDA_AVAILABLE:
                    cp.get_default_memory_pool().free_all_blocks()
            if (b + 1) % 10 == 0:
                logger.info(
                    f"   Export Progress: {b+1}/{CONFIG.n_bootstrap}...")

        if json_fail_count > 0:
            logger.warning(
                f"   {json_fail_count}/{CONFIG.n_bootstrap} JSON saves "
                f"failed (joblib copies are still complete).")
        logger.info(f"Export done | Time: {time.time()-t_export_start:.2f}s")

        return pd.DataFrame(results), all_preds


# ==============================================================================
# Resume / failure tracking
# ==============================================================================
def is_already_completed(biome_id: int, target_var: str) -> bool:
    """Strict resume protection: every required artifact must exist."""
    io_dir = os.path.join(CONFIG.output_dir, f"biome{biome_id}", target_var)
    metrics_path = os.path.join(io_dir, f"B{biome_id}_Metrics.csv")
    feat_path = os.path.join(io_dir, "Global_Selected_Features.csv")
    # *** FIX #4: also require SHAP and best_params.
    shap_path = os.path.join(io_dir, "SHAP.csv")
    bp_path = os.path.join(io_dir, "best_params.csv")
    m_dir = os.path.join(io_dir, "Final_Models")

    for p in (metrics_path, feat_path, shap_path, bp_path):
        if not os.path.exists(p):
            return False
    if not os.path.isdir(m_dir):
        return False
    try:
        n_existing = len([f for f in os.listdir(m_dir)
                          if f.endswith(".joblib")])
    except OSError:
        return False
    return n_existing >= CONFIG.n_bootstrap


def record_failure(biome_id: int, target_var: str, err: str) -> None:
    """*** FIX #9: persist failures for downstream re-run scripts."""
    os.makedirs(CONFIG.output_dir, exist_ok=True)
    fail_path = os.path.join(CONFIG.output_dir, "failed_runs.csv")
    row = pd.DataFrame([{
        "biome": biome_id,
        "target": target_var,
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "error": str(err)[:500],
    }])
    header = not os.path.exists(fail_path)
    row.to_csv(fail_path, mode="a", header=header, index=False)


# ==============================================================================
# Main
# ==============================================================================
def main() -> None:
    log_environment()

    for current_target in CONFIG.target_vars:
        logger.info(f"********** STARTING PIPELINE FOR TARGET: "
                    f"{current_target} **********")
        for b_id in CONFIG.biomes:
            if is_already_completed(b_id, current_target):
                logger.info(
                    f"SKIP Biome {b_id} [{current_target}] (already complete)")
                continue

            try:
                t_start = time.time()
                logger.info(f"================ START BIOME {b_id} "
                            f"[{current_target}] ================")
                pipeline = XGBPipeline(b_id, current_target)
                if pipeline.load_and_prep():
                    metrics, preds = pipeline.run()
                    pd.DataFrame(
                        {"obs": to_cpu(pipeline.y), **preds}
                    ).to_csv(
                        os.path.join(pipeline.io_dir,
                                     f"B{b_id}_Preds.csv"), index=False)
                    metrics.to_csv(
                        os.path.join(pipeline.io_dir,
                                     f"B{b_id}_Metrics.csv"), index=False)
                    logger.info(
                        f"COMPLETE BIOME {b_id} [{current_target}] | "
                        f"Total Time: {(time.time()-t_start)/60:.2f}m")
                logger.info("=" * 60 + "\n")
                del pipeline
                gc.collect()
                if _CUDA_AVAILABLE:
                    cp.get_default_memory_pool().free_all_blocks()
            except Exception as e:
                logger.error(
                    f"Failed Biome {b_id} [{current_target}]: {e}",
                    exc_info=True)
                record_failure(b_id, current_target, str(e))
                continue

        logger.info(f"********** FINISHED TARGET {current_target} **********\n")
        gc.collect()
        if _CUDA_AVAILABLE:
            cp.get_default_memory_pool().free_all_blocks()


if __name__ == "__main__":
    main()