"""
Created on Tue Jan 13 05:34:41 2026

Baseline runner for the pump-and-dump modeling pipeline.

Author: Luca Persia (USI/ZHAW)
"""

from __future__ import annotations
import os, json, warnings
from typing import Dict, Any, List, Set
import numpy as np, polars as pl, pandas as pd
from joblib import Parallel, delayed
warnings.filterwarnings("ignore")
# Trying with sklearnex to speed up the RF
_HAS_SKLEARNEX = False
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    _HAS_SKLEARNEX = True
except Exception:
    _HAS_SKLEARNEX = False
# Utils
import pipeline_utils as utils

# config
DATE_COL = "date"
LABEL_COL = "flag"
GROUP_COL = "symbol"
DROP_ENGINEERED_NULLS = True
ENGINEERED_FEATURES = [
    "std_rush_order", "avg_rush_order",
    "std_trades",
    "std_volume", "avg_volume",
    "std_price", "avg_price",
    "avg_price_max",
    "hour_of_the_day",
]
# Time split proportions
SPLIT_TRAIN = 0.60
SPLIT_VAL = 0.20
SPLIT_TEST = 0.20
# held out CV
N_SPLITS_WF = 5
EMBARGO_N = 5
# seeds
SEEDS: List[int] = [11, 22, 33, 44, 55, 66, 77, 88, 99]
SEED_PARALLEL_JOBS = 1
MODEL_N_JOBS = max(1, (os.cpu_count() or 4) // max(1, SEED_PARALLEL_JOBS))
# Models to run
MODELS_TO_RUN: Set[str] = {"RF"}

## Datasets
DATASETS: Dict[str, Dict[str, Any]] = {
    "long": {
        "path": r"C:\Users\pess\Desktop\PhD\Fraud Detection\project\outputs\panel_engineered_full.opt.parquet",
        "engineered": True
    },
}
## Explicit features
FEATURE_COLS = [
    # Market based
    "open", "high", "low", "close",
    "volume", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote",
    # Costructed
    "std_rush_order", "avg_rush_order", "std_trades",
    "std_volume", "std_price", "avg_volume", "avg_price",
    "avg_price_max", 
    # Time
    "hour_of_the_day"
]

## Grids 
RF_GRID = {
    # capacity
    "n_estimators": [500, 1000],
    "max_depth": [12, None],
    # regularization / smoothing
    "min_samples_leaf": [1, 10],
    # stochasticity
    "max_features": ["sqrt", 0.8],
}

XGB_GRID = {
    # capacity
    "n_estimators": [500, 1000],
    "max_depth": [4, 10],
    # optimization / shrinkage
    "learning_rate": [0.05, 0.1],
    # stochasticity
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    # regularization
    "reg_lambda": [1.0, 10.0],
}

# Preprocessing control
SCALE_TREES = False   # if only use trees makes sense to avoid std
# Saving dirs
SAVE_DIR = r"C:\Users\pess\Desktop\PhD\Fraud Detection\project\outputs\model_artifacts"
SAVE_MODELS = True
SAVE_PREDICTIONS = True

## RUN ------------------------------------------------------------------------
print(
    f"[CPU] os.cpu_count()={os.cpu_count()} | "
    f"SEED_PARALLEL_JOBS={SEED_PARALLEL_JOBS} | MODEL_N_JOBS={MODEL_N_JOBS}"
)
print(f"[MODELS] RUNNING {sorted(MODELS_TO_RUN)}")
print(f"[SAVE] dir={SAVE_DIR} | models={SAVE_MODELS} | preds={SAVE_PREDICTIONS}")

all_rows: List[Dict[str, Any]] = []
for ds_name, spec in DATASETS.items():
    path = spec["path"]
    is_eng = bool(spec.get("engineered", False))
    print("\n" + "=" * 100)
    print(f"[DATASET] {ds_name} | engineered={is_eng}")
    print(f"[LOAD] {path}")

    # Loading polars dataset
    df = pl.read_parquet(path)
    # Date parse + global time sort
    if df[DATE_COL].dtype == pl.Utf8:
        df = df.with_columns(pl.col(DATE_COL).str.strptime(pl.Datetime, strict=False))
    df = df.drop_nulls(subset=[DATE_COL]).sort(DATE_COL)
    # Label sanity
    if LABEL_COL not in df.columns:
        raise ValueError(f"[{ds_name}] Missing '{LABEL_COL}' column.")
    df = df.with_columns(pl.col(LABEL_COL).cast(pl.Int8).clip(0, 1))
    # Identifiers aligned with X_all/y_all 
    if GROUP_COL not in df.columns:
        raise ValueError(f"[{ds_name}] Missing '{GROUP_COL}' column (needed for id_df).")
    id_df = df.select([DATE_COL, GROUP_COL]).to_pandas()
    # positives info for imbalance ration
    nrows = df.height
    pos = int(df.select(pl.col(LABEL_COL).sum()).item())
    print(f"[CLEAN] rows={nrows:,} | positives={pos:,} ({(pos / max(nrows, 1)) * 100:.5f}%)")
    # log any missing features just to know what we are running
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{ds_name}] Missing feature columns: {missing}")
    print(f"[DATA] feature cols ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    # memory: convert once to contiguous float32
    X_all = df.select(FEATURE_COLS).to_numpy()
    if X_all.dtype != np.float32:
        X_all = X_all.astype(np.float32, copy=False)
    X_all = np.ascontiguousarray(X_all)
    y_all = df.select(LABEL_COL).to_numpy().reshape(-1).astype(np.uint8, copy=False)

    if not np.isfinite(X_all).all():
        bad = int(np.sum(~np.isfinite(X_all)))
        print(f"[WARN] X_all has {bad:,} non-finite values. Preprocess will impute them.")
    
    # ensure datetime + hourly floor for true hourly prediction
    id_df[DATE_COL] = pd.to_datetime(id_df[DATE_COL], errors="coerce")
    id_df[DATE_COL] = id_df[DATE_COL].dt.floor("h")
    # drop NaT rows consistently
    good = id_df[DATE_COL].notna().to_numpy()
    if not good.all():
        id_df = id_df.loc[good].reset_index(drop=True)
        X_all = X_all[good]
        y_all = y_all[good]
        
    ## Splitting --------------------------------------------------------------
    # changed to follow GNN preprocessing, should be consistent with held-out
    unique_dates = id_df[DATE_COL].unique()
    n_dates = len(unique_dates)
    train_end_idx = int(SPLIT_TRAIN * n_dates)
    val_end_idx   = int((SPLIT_TRAIN + SPLIT_VAL) * n_dates)
    dates_train = unique_dates[:train_end_idx]
    dates_val   = unique_dates[train_end_idx + EMBARGO_N : val_end_idx]
    dates_test  = unique_dates[val_end_idx + EMBARGO_N :]
    print(f"Train: {len(dates_train)} h | Val: {len(dates_val)} h | Test: {len(dates_test)} h")
    # Map each row to a split
    time_code, _ = pd.factorize(id_df[DATE_COL].to_numpy(), sort=False)
    val_start  = train_end_idx + EMBARGO_N
    test_start = val_end_idx + EMBARGO_N
    mask_train = time_code < train_end_idx
    mask_val   = (time_code >= val_start) & (time_code < val_end_idx)
    mask_test  = time_code >= test_start
    
    idx_train = np.flatnonzero(mask_train)
    idx_val   = np.flatnonzero(mask_val)
    idx_test  = np.flatnonzero(mask_test)
    if idx_val.size == 0 or idx_test.size == 0:
        raise ValueError(
            f"Empty VAL/TEST after embargo. n_dates={n_dates}, "
            f"train_end_idx={train_end_idx}, val_end_idx={val_end_idx}, embargo={EMBARGO_N}"
        )
    idx_trainval = np.concatenate([idx_train, idx_val])
    print(f"[ROWS] train={idx_train.size:,} val={idx_val.size:,} test={idx_test.size:,}")
    print(
        f"[FLAGS] train={int(y_all[idx_train].sum())} | "
        f"val={int(y_all[idx_val].sum())} | test={int(y_all[idx_test].sum())}"
    )

    
    # Running -----------------------------------------------------------------
    seed_results = Parallel(n_jobs=SEED_PARALLEL_JOBS, prefer="threads")(
        delayed(utils.run_seed)(
            seed=seed,
            X_all=X_all,
            y_all=y_all,

            idx_train=idx_train,
            idx_val=idx_val,
            idx_test=idx_test,

            models_to_run=MODELS_TO_RUN,

            rf_grid=RF_GRID,
            xgb_grid=XGB_GRID,

            # TODO - need to refactor this to deleter WF
            n_splits_wf=N_SPLITS_WF,
            embargo_n=EMBARGO_N,
            model_n_jobs=MODEL_N_JOBS,
            scale_trees=SCALE_TREES,

            # persistence
            ds_name=ds_name,
            save_root=SAVE_DIR,
            save_models=SAVE_MODELS,
            save_predictions=SAVE_PREDICTIONS,
            id_df=id_df,
            feature_cols=FEATURE_COLS,
        )
        for seed in SEEDS
    )

    print("\n[SUMMARY] " + ds_name)
    def _fmt(md: Any) -> str:
        if not isinstance(md, dict) or md.get("ok") is not True:
            return "NA"
        return (
            f"P={md['precision']:.3f} R={md['recall']:.3f} "
            f"F1={md['f1']:.3f} AP={md['ap']:.3f} "
            f"thr={md['threshold']:.5f}"
        )

    for seed, metrics in zip(SEEDS, seed_results):
        print(
            f"Seed {seed} | "
            f"RF: {_fmt(metrics.get('RF'))} | "
            f"XGB: {_fmt(metrics.get('XGB'))} | "
        )
        # Metrics CSV rows
        for model_name, md in metrics.items():
            if not isinstance(md, dict) or md.get("ok") is not True:
                continue
            all_rows.append({
                "dataset": ds_name,
                "engineered": int(is_eng),
                "seed": seed,
                "model": model_name,
                "precision": md["precision"],
                "recall": md["recall"],
                "f1": md["f1"],
                "ap": md["ap"],
                "threshold": md["threshold"],
                "pred_pos_rate": md.get("pred_pos_rate", np.nan),
                "sklearnex": int(_HAS_SKLEARNEX),
                "params_json": json.dumps(md.get("params", {}), ensure_ascii=False),
                "oof_f1": md.get("oof_f1", np.nan),
                "oof_ap": md.get("oof_ap", np.nan),
            })

out_csv = "test_RF_XGBOOST_long_panel.csv"
pd.DataFrame(all_rows).to_csv(out_csv, index=False)
print(f"\n[SAVE] {out_csv}")
