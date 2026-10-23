#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
================================================================================
 Cross-buffer summary (Anchored at 800km mean +/- cross-buffer SD)
--------------------------------------------------------------------------------
 Reads per buffer:
   1_Block_CV_Evaluation_Metrics.csv      -> Fold-level Mean R2 & RMSE
   1b_Block_CV_Per_Point_Predictions.csv  -> Global OOF CV R2 & RMSE

 Outputs two CSVs:
   Summary_Across_Buffers_numeric.csv   -- Raw numeric global metrics for all buffers
   Summary_Across_Buffers_paper.csv     -- Final paper table format (800km value +/- SD across 3 buffers)
================================================================================
"""

import os
import math
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error

# ============================================================================
# Configuration
# ============================================================================
BASE_ROOT  = Path("/data/global_mapping/models_V100/drive_fina/PC2/XGB_shap/XGB_shap/")
BLOCK_SIZE = 1500
BUFFERS    = [800, 1000, 1200]

OUT_DIR    = Path("/data/global_mapping/models_V100/drive_fina/PC2/XGB_shap/XGB_shap/Block_CV")
OUT_NUMERIC = OUT_DIR / "Summary_Across_Buffers_numeric.csv"
OUT_PAPER   = OUT_DIR / "Summary_Across_Buffers_paper.csv"

DECIMALS    = 3
SEP         = " \u00B1 "


def summarize_one_buffer(folder: Path, buf_km: int) -> dict:
    row = {"Buffer_km": buf_km}

    # 1. Fold-level Metrics
    cv_path = folder / "1_Block_CV_Evaluation_Metrics.csv"
    if cv_path.exists():
        df_cv = pd.read_csv(cv_path)
        folds = df_cv[~df_cv["fold"].astype(str).isin(["MEAN", "STD"])]
        if len(folds) > 0:
            row["Fold_Mean_R2"]   = float(folds["R2"].mean())
            row["Fold_Mean_RMSE"] = float(folds["RMSE"].mean())
    else:
        print(f"  [WARN] missing: {cv_path.name}")

    # 2. Global (Out-of-Fold) Metrics
    pred_path = folder / "1b_Block_CV_Per_Point_Predictions.csv"
    if pred_path.exists():
        df_pred = pd.read_csv(pred_path)
        y_true = df_pred["y_true"].values
        y_pred = df_pred["y_pred_cv"].values
        
        row["n_samples"]   = len(y_true)
        row["Global_R2"]   = float(r2_score(y_true, y_pred))
        row["Global_RMSE"] = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    else:
        print(f"  [WARN] missing: {pred_path.name}")

    return row


def reorder_columns(df: pd.DataFrame) -> pd.DataFrame:
    base_cols = [
        "Buffer_km", "n_samples", 
        "Global_R2", "Global_RMSE", 
        "Fold_Mean_R2", "Fold_Mean_RMSE"
    ]
    base_cols = [c for c in base_cols if c in df.columns]
    feat_cols = [c for c in df.columns if c not in base_cols]
    return df[base_cols + feat_cols]


def make_paper_table(df_num: pd.DataFrame) -> pd.DataFrame:
    if 800 not in df_num["Buffer_km"].values:
        print("\n[ERROR] Missing 800km buffer baseline data, cannot generate table!")
        return pd.DataFrame()

    row_800 = df_num[df_num["Buffer_km"] == 800].iloc[0]
    metrics = [c for c in df_num.columns if c not in ["Buffer_km", "n_samples"]]
    summary_dict = {"Baseline": f"800km Metric {SEP} Cross-Buffer SD"}

    for metric in metrics:
        val_800 = row_800[metric]
        
        if pd.isna(val_800):
            continue
            
        cross_sd = df_num[metric].std(ddof=1)
        
        if pd.isna(cross_sd):
            cross_sd = 0.0

        summary_dict[metric] = f"{val_800:.{DECIMALS}f}{SEP}{cross_sd:.{DECIMALS}f}"

    return pd.DataFrame([summary_dict])


def main():
    print(f"Base path: {BASE_ROOT}")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for buf in BUFFERS:
        folder = BASE_ROOT / f"block_{BLOCK_SIZE}km_buf_{buf}km"
        if not folder.exists():
            print(f"[SKIP] missing folder: {folder.name}")
            continue
        print(f"[READ] {folder.name}")
        rows.append(summarize_one_buffer(folder, buf))

    if not rows:
        print("ERROR: no data collected.")
        return

    df_num   = reorder_columns(pd.DataFrame(rows))
    df_paper = make_paper_table(df_num)

    df_num.to_csv(OUT_NUMERIC, index=False)
    if not df_paper.empty:
        df_paper.to_csv(OUT_PAPER, index=False, encoding="utf-8-sig")

    print("=" * 78)
    print(f"Saved Numeric: {OUT_NUMERIC}")
    print(f"Saved Paper:   {OUT_PAPER}")
    print("=" * 78)
    
    print("\nNumeric version (Global Metrics per Buffer):")
    df_show = df_num.copy()
    for c in df_show.columns:
        if df_show[c].dtype.kind == "f":
            df_show[c] = df_show[c].round(DECIMALS + 1)
    print(df_show.to_string(index=False))

    if not df_paper.empty:
        print("\nPaper version (800km Value +/- Cross-Buffer SD):")
        print(df_paper.to_string(index=False))


if __name__ == "__main__":
    main()