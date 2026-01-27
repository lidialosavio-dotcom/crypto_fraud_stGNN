"""
Created on Tue Jan 13 05:34:41 2026

Baseline runner for the pump-and-dump modeling pipeline.

Author: Luca Persia (USI/ZHAW)
"""

# libraries
from __future__ import annotations
import os, logging, warnings
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import numpy as np, polars as pl, pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import precision_score, recall_score, f1_score
warnings.filterwarnings("ignore")

# Trying with sklearnex to speed up the RF
_HAS_SKLEARNEX = False
try:
    from sklearnex import patch_sklearn
    patch_sklearn()
    _HAS_SKLEARNEX = True
except Exception:
    _HAS_SKLEARNEX = False

# Utils from pipeline_utils
import pipeline_utils as utils

# add logging
# TODO: add to utils
def setup_logger(name: str = "pipeline_inline") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logger.propagate = False
    return logger

def mean_std(values: List[float]) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0

def compute_prf_from_scores(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> Tuple[float, float, float]:
    y_true = np.asarray(y_true).astype(np.uint8, copy=False)
    y_score = np.asarray(y_score).astype(np.float32, copy=False)
    y_pred = (y_score >= float(thr)).astype(np.uint8)
    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    return p, r, f1

def compute_per_token_prf(
    ids_test: pd.DataFrame,
    y_true: np.ndarray,
    y_score: np.ndarray,
    thr: float,
    tokens_keep: List[str],
    group_col: str,
) -> pd.DataFrame:
    """
    Returns per-token P/R/F1 computed on the SAME test predictions,
    for tokens in tokens_keep only.
    """
    ids = ids_test[[group_col]].copy()
    ids[group_col] = ids[group_col].astype(str)

    y_true = np.asarray(y_true).astype(np.uint8, copy=False)
    y_score = np.asarray(y_score).astype(np.float32, copy=False)
    y_pred = (y_score >= float(thr)).astype(np.uint8)

    row = ids.copy()
    row["y_true"] = y_true
    row["y_pred"] = y_pred
    row = row[row[group_col].isin(tokens_keep)]
    if row.empty:
        return pd.DataFrame(columns=[group_col, "precision", "recall", "f1", "n_obs", "n_pumps"])

    g = row.groupby(group_col, sort=False)
    n_obs = g["y_true"].size().astype(int)
    n_pumps = g["y_true"].sum().astype(int)

    tp = (row["y_true"].eq(1) & row["y_pred"].eq(1)).groupby(row[group_col]).sum().astype(int)
    fp = (row["y_true"].eq(0) & row["y_pred"].eq(1)).groupby(row[group_col]).sum().astype(int)
    fn = (row["y_true"].eq(1) & row["y_pred"].eq(0)).groupby(row[group_col]).sum().astype(int)

    out = pd.DataFrame({group_col: n_obs.index.astype(str)})
    out["n_obs"] = out[group_col].map(n_obs).astype(int).to_numpy()
    out["n_pumps"] = out[group_col].map(n_pumps).astype(int).to_numpy()

    out["tp"] = out[group_col].map(tp).fillna(0).astype(int).to_numpy()
    out["fp"] = out[group_col].map(fp).fillna(0).astype(int).to_numpy()
    out["fn"] = out[group_col].map(fn).fillna(0).astype(int).to_numpy()

    denom_p = out["tp"] + out["fp"]
    denom_r = out["tp"] + out["fn"]

    out["precision"] = np.where(denom_p > 0, out["tp"] / denom_p, 0.0)
    out["recall"] = np.where(denom_r > 0, out["tp"] / denom_r, 0.0)
    out["f1"] = np.where(
        (out["precision"] + out["recall"]) > 0,
        2 * out["precision"] * out["recall"] / (out["precision"] + out["recall"]),
        0.0,
    )

    return out[[group_col, "n_obs", "n_pumps", "precision", "recall", "f1"]]

### config -------------------------------------------------------------------------
DATE_COL = "date"
LABEL_COL = "flag"
GROUP_COL = "symbol"

SPLIT_TRAIN = 0.60
SPLIT_VAL = 0.20
SPLIT_TEST = 0.20

# kept for compatibility with utils.run_seed signature
N_SPLITS_WF = 5
EMBARGO_N = 5

SEEDS: List[int] = [11] # set seeds
SEED_PARALLEL_JOBS = 1  # TODO: RF doesn't handle GPU from scikt, this doens't change speed
MODEL_N_JOBS = max(1, (os.cpu_count() or 4) // max(1, SEED_PARALLEL_JOBS))

MODELS_TO_RUN: Set[str] = {"XGB"}  # {"RF","XGB"}

DATASETS: Dict[str, Dict[str, Any]] = {
    "long": {
        "path": r"C:\Users\pess\Desktop\PhD\Fraud Detection\project\outputs\panel_engineered_full.opt.parquet",
        "engineered": True,
    },
}

FEATURE_COLS = [
    "open", "high", "low", "close",
    "volume", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote",
    "std_rush_order", "avg_rush_order", "std_trades",
    "std_volume", "std_price", "avg_volume", "avg_price",
    "avg_price_max",
    "hour_of_the_day",
]

RF_GRID = {
    "n_estimators": [500, 1000],
    "max_depth": [12, None],
    "min_samples_leaf": [1, 10],
    "max_features": ["sqrt", 0.8],
}

XGB_GRID = {
    "n_estimators": [500, 1000],
    "max_depth": [4, 10],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "reg_lambda": [1.0, 10.0],
}

SCALE_TREES = False
# SAVE ONLY MODELS
SAVE_DIR = r"C:\Users\pess\Desktop\PhD\Fraud Detection\project\outputs\model_artifacts"
SAVE_MODELS = True
SAVE_PREDICTIONS = False  # no pred saving

logger = setup_logger("pipeline_inline")

# run
logger.info(
    f"[CPU] os.cpu_count()={os.cpu_count()} | "
    f"SEED_PARALLEL_JOBS={SEED_PARALLEL_JOBS} | MODEL_N_JOBS={MODEL_N_JOBS}"
)
logger.info(f"[MODELS] RUNNING {sorted(MODELS_TO_RUN)}")
logger.info(f"[SAVE] dir={SAVE_DIR} | models={SAVE_MODELS} | preds={SAVE_PREDICTIONS} | sklearnex={int(_HAS_SKLEARNEX)}")

for ds_name, spec in DATASETS.items():
    path = spec["path"]
    is_eng = bool(spec.get("engineered", False))

    logger.info("\n" + "=" * 100)
    logger.info(f"[DATASET] {ds_name} | engineered={is_eng}")
    logger.info(f"[LOAD] {path}")

    df = pl.read_parquet(path)

    if df[DATE_COL].dtype == pl.Utf8:
        df = df.with_columns(pl.col(DATE_COL).str.strptime(pl.Datetime, strict=False))
    df = df.drop_nulls(subset=[DATE_COL]).sort(DATE_COL)

    if LABEL_COL not in df.columns:
        raise ValueError(f"[{ds_name}] Missing '{LABEL_COL}' column.")
    df = df.with_columns(pl.col(LABEL_COL).cast(pl.Int8).clip(0, 1))

    if GROUP_COL not in df.columns:
        raise ValueError(f"[{ds_name}] Missing '{GROUP_COL}' column (needed for id_df).")
    id_df = df.select([DATE_COL, GROUP_COL]).to_pandas()

    nrows = df.height
    pos = int(df.select(pl.col(LABEL_COL).sum()).item())
    logger.info(f"[CLEAN] rows={nrows:,} | positives={pos:,} ({(pos / max(nrows, 1)) * 100:.5f}%)")

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"[{ds_name}] Missing feature columns: {missing}")
    logger.info(f"[DATA] feature cols ({len(FEATURE_COLS)}): {FEATURE_COLS}")

    X_all = df.select(FEATURE_COLS).to_numpy()
    if X_all.dtype != np.float32:
        X_all = X_all.astype(np.float32, copy=False)
    X_all = np.ascontiguousarray(X_all)
    y_all = df.select(LABEL_COL).to_numpy().reshape(-1).astype(np.uint8, copy=False)

    if not np.isfinite(X_all).all():
        bad = int(np.sum(~np.isfinite(X_all)))
        logger.warning(f"[WARN] X_all has {bad:,} non-finite values. Preprocess will impute them.")

    id_df[DATE_COL] = pd.to_datetime(id_df[DATE_COL], errors="coerce")
    id_df[DATE_COL] = id_df[DATE_COL].dt.floor("h")

    good = id_df[DATE_COL].notna().to_numpy()
    if not good.all():
        id_df = id_df.loc[good].reset_index(drop=True)
        X_all = X_all[good]
        y_all = y_all[good]

    ### split to be consistent with Lidia's logic -----------------------------
    unique_dates = id_df[DATE_COL].unique()
    n_dates = len(unique_dates)
    train_end_idx = int(SPLIT_TRAIN * n_dates)
    val_end_idx = int((SPLIT_TRAIN + SPLIT_VAL) * n_dates)

    dates_train = unique_dates[:train_end_idx]
    dates_val = unique_dates[train_end_idx + EMBARGO_N: val_end_idx]
    dates_test = unique_dates[val_end_idx + EMBARGO_N:]

    logger.info(f"[SPLIT] Train: {len(dates_train)} h | Val: {len(dates_val)} h | Test: {len(dates_test)} h")

    time_code, _ = pd.factorize(id_df[DATE_COL].to_numpy(), sort=False)
    val_start = train_end_idx + EMBARGO_N
    test_start = val_end_idx + EMBARGO_N

    mask_train = time_code < train_end_idx
    mask_val = (time_code >= val_start) & (time_code < val_end_idx)
    mask_test = time_code >= test_start

    idx_train = np.flatnonzero(mask_train)
    idx_val = np.flatnonzero(mask_val)
    idx_test = np.flatnonzero(mask_test)

    if idx_val.size == 0 or idx_test.size == 0:
        raise ValueError(
            f"Empty VAL/TEST after embargo. n_dates={n_dates}, "
            f"train_end_idx={train_end_idx}, val_end_idx={val_end_idx}, embargo={EMBARGO_N}"
        )

    logger.info(f"[ROWS] train={idx_train.size:,} val={idx_val.size:,} test={idx_test.size:,}")
    logger.info(
        f"[FLAGS] train={int(y_all[idx_train].sum())} | "
        f"val={int(y_all[idx_val].sum())} | test={int(y_all[idx_test].sum())}"
    )

    # Tokens in TEST with >=3 pumps in TEST
    ids_test = id_df.iloc[idx_test].copy().reset_index(drop=True)
    ids_test[GROUP_COL] = ids_test[GROUP_COL].astype(str)
    y_test_true_panel = y_all[idx_test]

    pumps_by_token = (
        pd.DataFrame({GROUP_COL: ids_test[GROUP_COL].values, "y_true": y_test_true_panel})
        .groupby(GROUP_COL, sort=False)["y_true"]
        .sum()
        .astype(int)
    )
    tokens_ge3 = sorted(pumps_by_token[pumps_by_token >= 3].index.tolist())
    logger.info(f"[TOKENS] test_tokens_pumps>=3={len(tokens_ge3):,}")

    # train/eval
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
            n_splits_wf=N_SPLITS_WF,
            embargo_n=EMBARGO_N,
            model_n_jobs=MODEL_N_JOBS,
            scale_trees=SCALE_TREES,
            ds_name=ds_name,
            save_root=SAVE_DIR,
            save_models=SAVE_MODELS,
            save_predictions=SAVE_PREDICTIONS,
            id_df=id_df,
            feature_cols=FEATURE_COLS,
        )
        for seed in SEEDS
    )

    # overall per-seed + overall mean±std
    logger.info("\n" + "-" * 100)
    logger.info(f"[OVERALL TEST METRICS] dataset={ds_name}")

    overall_by_model: Dict[str, Dict[str, List[float]]] = {m: {"precision": [], "recall": [], "f1": []} for m in MODELS_TO_RUN}

    for seed, metrics in zip(SEEDS, seed_results):
        parts = [f"Seed {seed:>3}"]
        for model_name in sorted(MODELS_TO_RUN):
            md = metrics.get(model_name)
            if not isinstance(md, dict) or md.get("ok") is not True:
                parts.append(f"{model_name}: NA")
                continue
            p = float(md["precision"]); r = float(md["recall"]); f1v = float(md["f1"])
            overall_by_model[model_name]["precision"].append(p)
            overall_by_model[model_name]["recall"].append(r)
            overall_by_model[model_name]["f1"].append(f1v)
            parts.append(f"{model_name}: P={p:.3f} R={r:.3f} F1={f1v:.3f}")
        logger.info(" | ".join(parts))

    logger.info("\n[OVERALL TEST] mean ± std across seeds")
    for model_name in sorted(MODELS_TO_RUN):
        mP, sP = mean_std(overall_by_model[model_name]["precision"])
        mR, sR = mean_std(overall_by_model[model_name]["recall"])
        mF, sF = mean_std(overall_by_model[model_name]["f1"])
        logger.info(f"{model_name}: P={mP:.3f}±{sP:.3f}  R={mR:.3f}±{sR:.3f}  F1={mF:.3f}±{sF:.3f}")

    ### per-token per seed, then per-token mean±std across seeds
    logger.info("\n" + "-" * 100)
    logger.info(f"[PER-TOKEN TEST METRICS] dataset={ds_name} | filter: tokens in TEST with n_pumps>=3")

    if not tokens_ge3:
        logger.info("[PER-TOKEN] No tokens satisfy n_pumps>=3 in TEST. Done.")
        continue

    # token --> seed --> (P,R,F1)
    per_token_seed_rows: Dict[str, Dict[int, Tuple[float, float, float]]] = {tok: {} for tok in tokens_ge3}

    for seed, metrics in zip(SEEDS, seed_results):
        logger.info(f"\nSeed {seed:>3} | per-token metrics (P/R/F1) for tokens_ge3")
        for model_name in sorted(MODELS_TO_RUN):
            md = metrics.get(model_name)
            if not isinstance(md, dict) or md.get("ok") is not True:
                logger.info(f"  {model_name}: NA")
                continue
            if not all(k in md for k in ("test_y", "test_scores", "threshold")):
                logger.info(f"  {model_name}: missing test arrays -> NA")
                continue

            y_true = np.asarray(md["test_y"])
            y_score = np.asarray(md["test_scores"])
            thr = float(md["threshold"])

            per_tok = compute_per_token_prf(
                ids_test=ids_test,
                y_true=y_true,
                y_score=y_score,
                thr=thr,
                tokens_keep=tokens_ge3,
                group_col=GROUP_COL,
            )

            # Print in stable token order
            logger.info(f"  {model_name} (thr={thr:.5f}):")
            for tok in tokens_ge3:
                row = per_tok[per_tok[GROUP_COL] == tok]
                if row.empty:
                    logger.info(f"    {tok}: NA")
                    continue
                p = float(row["precision"].iloc[0])
                r = float(row["recall"].iloc[0])
                f1v = float(row["f1"].iloc[0])
                per_token_seed_rows[tok][seed] = (p, r, f1v)
                logger.info(f"    {tok}: P={p:.3f} R={r:.3f} F1={f1v:.3f} (n_pumps={int(row['n_pumps'].iloc[0])})")

    # Aggregate per token across seeds (mean±std)
    logger.info("\n" + "-" * 100)
    logger.info("[PER-TOKEN] mean ± std across seeds (P/R/F1)")

    for tok in tokens_ge3:
        vals_p = [per_token_seed_rows[tok][s][0] for s in per_token_seed_rows[tok].keys()]
        vals_r = [per_token_seed_rows[tok][s][1] for s in per_token_seed_rows[tok].keys()]
        vals_f = [per_token_seed_rows[tok][s][2] for s in per_token_seed_rows[tok].keys()]

        mP, sP = mean_std(vals_p)
        mR, sR = mean_std(vals_r)
        mF, sF = mean_std(vals_f)

        n_seeds_eff = len(vals_f)
        n_pumps = int(pumps_by_token.get(tok, 0))
        logger.info(
            f"{tok}: "
            f"P={mP:.3f}±{sP:.3f}  R={mR:.3f}±{sR:.3f}  F1={mF:.3f}±{sF:.3f}  "
            f"(n_seeds={n_seeds_eff}, n_pumps_test={n_pumps})"
        )

out_csv = "test_RF_XGBOOST_long_panel.csv"
pd.DataFrame(all_rows).to_csv(out_csv, index=False)
print(f"\n[SAVE] {out_csv}")
