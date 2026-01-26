"""
Created on Fri Jan 02 05:35:55 2026

Utilities:
- TimeSeriesSplit on TRAIN+VAL (X_tv).
- embargo to each fold's validation indices.
- Fit preprocessing only on fold-train, apply to fold-val.
- Produce OOF probabilities.
- Choose threshold on OOF to maximize F1.
- Select best hyperparams by (OOF F1).
- Refit on full TRAIN+VAL, evaluate on TEST.

Preprocessing:
- Always imputes non-finites using TRAIN medians.
- Scaling, if needed.

Saves per (dataset, model, seed):
    * meta.json
    * preprocess.joblib
    * model.joblib (RF/XGB)
    * test_predictions.parquet
    
Author: Luca Persia (USI/ZHAW)
"""

from __future__ import annotations
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np, pandas as pd
import joblib, json
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    precision_recall_curve,
)
from sklearn.preprocessing import StandardScaler
# models
import models as model_lib


### Preprocessing function
def fit_preprocess(X_train: np.ndarray, *, scale: bool) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Fit preprocessing on TRAIN only.

    Do
    -----
    - Convert to float32
    - Replace non-finite with NaN
    - Impute NaNs with column-wise TRAIN medians
       - If a full column is NaN, its median is NaN; we replace it with 0.0
    - Optionally apply StandardScaler on TRAIN (mind is not done for tree algorithms)

    Returns
    -------
    X_train_processed : np.ndarray
        Processed TRAIN data (float32).
    artifacts : dict
        Preprocessing artifacts for apply_preprocess()
    """
    # memory
    X = X_train.astype(np.float32, copy=True)
    # nan imputation
    X[~np.isfinite(X)] = np.nan
    medians = np.nanmedian(X, axis=0)
    medians = np.where(np.isfinite(medians), medians, 0.0).astype(np.float32, copy=False)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        X[nan_mask] = np.take(medians, np.where(nan_mask)[1])
    # scaling
    scaler = None
    if scale:
        scaler = StandardScaler(copy=False)
        scaler.fit(X)
        X = scaler.transform(X)
    X = np.ascontiguousarray(X.astype(np.float32, copy=False))
    
    return X, {"medians": medians, "scaler": scaler, "scale": bool(scale)}


def apply_preprocess(X: np.ndarray, artifacts: Dict[str, Any]) -> np.ndarray:
    """
    Apply TRAIN-fitted preprocessing to any split (VAL/TEST).

    Steps
    -----
    - float32 + non-finite -> NaN
    - median imputation using TRAIN medians
    - optional scaler transform 

    Returns
    -------
    np.ndarray
        Processed data (float32, contiguous).
    """
    Xp = X.astype(np.float32, copy=True)
    Xp[~np.isfinite(Xp)] = np.nan
    medians = artifacts["medians"]
    nan_mask = np.isnan(Xp)
    if nan_mask.any():
        Xp[nan_mask] = np.take(medians, np.where(nan_mask)[1])
    if artifacts["scale"]:
        Xp = artifacts["scaler"].transform(Xp)

    return np.ascontiguousarray(Xp.astype(np.float32, copy=False))


# Embargo function (DePardo)
def embargo_validation(val_idx: np.ndarray, *, max_train_idx: int, embargo_n: int) -> np.ndarray:
    """
    Apply an embargo to walk-forward validation indices.

    We drop the first `embargo_n` rows after the end of the training fold to
    reduce adjacency leakage in time series settings.

    Parameters
    ----------
    val_idx : np.ndarray
        Validation indices for the fold.
    max_train_idx : int
        Maximum index in the training fold.
    embargo_n : int
        Number of samples to drop immediately after train boundary.

    Returns
    -------
    np.ndarray
        Filtered validation indices.
    """
    if val_idx.size == 0:
        return val_idx
    cutoff = int(max_train_idx) + int(embargo_n)
    return val_idx[val_idx > cutoff]


### Threshold selection
def select_threshold_f1(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, Tuple[float, float, float]]:
    """
    Choose probability threshold that maximizes F1.

    Always the same across RF/XGB:
    - we select the threshold on OOF scores (TRAIN+VAL only)
    - then we apply that threshold unchanged on TEST

    Returns
    -------
    thr : float
        Chosen threshold.
    (precision, recall, f1) : tuple
        Metrics at the chosen threshold.
    """
    y_true = y_true.astype(np.uint8, copy=False)
    scores = scores.astype(np.float32, copy=False)

    # # fall back to high percentile threshold
    # if (y_true.sum() == 0) or (y_true.sum() == len(y_true)):
    #     thr = float(np.percentile(scores, 99))
    
    # compute prec rec curve
    prec, rec, thr = precision_recall_curve(y_true, scores)
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * (prec * rec) / (prec + rec + 1e-12)

    if len(thr) == 0:
        thr_star = float(np.percentile(scores, 99))
        preds = (scores >= thr_star).astype(np.uint8)
        p = precision_score(y_true, preds, zero_division=0)
        r = recall_score(y_true, preds, zero_division=0)
        f = f1_score(y_true, preds, zero_division=0)
        return thr_star, (float(p), float(r), float(f))

    best_idx = int(np.nanargmax(f1[1:])) + 1
    thr_star = float(thr[best_idx - 1])
    
    # predictions on tuned thr
    preds = (scores >= thr_star).astype(np.uint8)
    p = precision_score(y_true, preds, zero_division=0)
    r = recall_score(y_true, preds, zero_division=0)
    f = f1_score(y_true, preds, zero_division=0)

    return thr_star, (float(p), float(r), float(f))


def compute_metrics(y_true: np.ndarray, scores: np.ndarray, thr: float) -> Dict[str, float]:
    """
    Compute thresholded metrics (P/R/F1) + threshold-free AP.

    Returns
    -------
    dict
        precision, recall, f1, ap, pred_pos_rate
    """
    y_true = y_true.astype(np.uint8, copy=False)
    scores = scores.astype(np.float32, copy=False)
    preds = (scores >= float(thr)).astype(np.uint8)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1v = f1_score(y_true, preds, zero_division=0)
    ap = 0.0
    if len(np.unique(y_true)) > 1:
        ap = float(average_precision_score(y_true, scores))

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1v),
        "ap": float(ap),
        "pred_pos_rate": float(preds.mean()),
    }
      
  
def save_artifacts(
    *,
    save_root: str,
    ds_name: str,
    seed: int,
    model_name: str,
    result: Dict[str, Any],
    id_df: pd.DataFrame,
    idx_test: np.ndarray,
    feature_cols: List[str],
    save_models: bool,
    save_predictions: bool,
) -> None:
    """ 
    model + preprocess + test predictions for post-run analysis.

    Folder structure
    ----------------
    <save_root>/<ds_name>/<model_name>/seed_<seed>/
      - meta.json
      - preprocess.joblib
      - model.joblib
      - test_predictions.parquet
    """
    if not isinstance(result, dict) or result.get("ok") is not True:
        return

    base = Path(save_root) / ds_name / model_name / f"seed_{seed}"
    base.mkdir(parents=True, exist_ok=True)

    feature_cols_used = result.get("persist_feature_cols", None)
    if not feature_cols_used:
        feature_cols_used = list(feature_cols)

    # meta metrics
    meta = {
        "dataset": ds_name,
        "seed": int(seed),
        "model": model_name,
        "feature_cols": list(feature_cols_used),
        "metrics": {
            "precision": result.get("precision"),
            "recall": result.get("recall"),
            "f1": result.get("f1"),
            "ap": result.get("ap"),
            "threshold": result.get("threshold"),
            "pred_pos_rate": result.get("pred_pos_rate"),
            "oof_f1": result.get("oof_f1"),
            "oof_ap": result.get("oof_ap"),
        },
        "params": result.get("params", {}),
    }
    (base / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    # preprocess
    preprocess = result.get("persist_preprocess", None)
    if save_models and preprocess is not None:
        joblib.dump(preprocess, base / "preprocess.joblib", compress=3)
    # models
    if save_models:
        model = result.get("persist_model", None)
        if model is not None:
            try:
                joblib.dump(model, base / "model.joblib", compress=3)
            except Exception as e:
                print(f"[WARN] Failed saving {model_name} model for {ds_name}/{seed}: {e}")
    # test predictions
    if save_predictions:
        scores = result.get("test_scores", None)
        pred = result.get("test_pred", None)
        y = result.get("test_y", None)

        if scores is not None and pred is not None and y is not None:
            ids_test = id_df.iloc[idx_test].copy().reset_index(drop=True)
            out = ids_test
            out["y_true"] = np.asarray(y).astype(np.uint8)
            out["y_score"] = np.asarray(scores).astype(np.float32)
            out["y_pred"] = np.asarray(pred).astype(np.uint8)
            out["threshold"] = float(result.get("threshold", np.nan))

            out.to_parquet(base / "test_predictions.parquet", index=False)


### Tuning --------------------------------------------------------------------
def iter_grid(grid: Dict[str, List[Any]]) -> Iterable[Dict[str, Any]]:
    """
    Deterministic grid iterator.
    """
    items = sorted(grid.items())
    keys = [k for k, _ in items]
    values = [v for _, v in items]
    for combo in product(*values):
        yield dict(zip(keys, combo))

def tune_with_oof(
    *,
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    t_hour_tv: Optional[np.ndarray],  # (datetime64[h] per row)
    seed: int,
    model_kind: str,
    grid: Dict[str, List[Any]],
    n_splits_wf: int,
    embargo_n: int,  # interpreted as HOURS when t_hour_tv is provided
    model_n_jobs: int,
    scale_features: bool,
) -> Tuple[Dict[str, Any], float, float, float]:
    """
    Tuning on TRAIN+VAL with OOF probabilities.

    If t_hour_tv is provided, folds + embargo are applied at HOURLY granularity
    (all symbols within an hour stick together; embargo_n drops first N HOURS of each val fold).
    """
    y_tv = y_tv.astype(np.uint8, copy=False)

    # Hour-aware split 
    if t_hour_tv is not None:
        t_hour_tv = np.asarray(t_hour_tv).astype("datetime64[h]")
        if t_hour_tv.shape[0] != y_tv.shape[0]:
            raise ValueError("t_hour_tv must have same length as y_tv")
        # Factorize to hour ids 0..H-1
        unique_hours, hour_code = np.unique(t_hour_tv, return_inverse=True)
        H = unique_hours.size

        if H <= int(n_splits_wf):
            raise ValueError(
                f"Not enough unique hours ({H}) for n_splits_wf={n_splits_wf}. "
                f"Need > n_splits_wf."
            )

        # Precompute row indices per hour for fast fold assembly
        rows_by_hour = [np.where(hour_code == h)[0] for h in range(H)]

        tscv = TimeSeriesSplit(n_splits=int(n_splits_wf))
        best_params: Optional[Dict[str, Any]] = None
        best_thr: float = 0.5
        best_f1: float = -1.0
        best_ap: float = -1.0

        for params in iter_grid(grid):
            oof = np.full(y_tv.shape[0], np.nan, dtype=np.float32)

            # Split on HOURS, not rows
            for tr_h, va_h in tscv.split(np.arange(H)):
                # Embargo in HOURS: drop first embargo_n hours of the validation fold
                if int(embargo_n) > 0 and va_h.size > 0:
                    va_h = va_h[int(embargo_n):]
                if va_h.size == 0:
                    continue
                tr_idx = np.concatenate([rows_by_hour[h] for h in tr_h]) if tr_h.size else np.empty(0, dtype=int)
                va_idx = np.concatenate([rows_by_hour[h] for h in va_h]) if va_h.size else np.empty(0, dtype=int)
                if va_idx.size == 0 or tr_idx.size == 0:
                    continue

                X_tr, y_tr = X_tv[tr_idx], y_tv[tr_idx]
                X_va, y_va = X_tv[va_idx], y_tv[va_idx]
                # Preprocess on fold-train only
                X_tr_p, artifacts = fit_preprocess(X_tr, scale=bool(scale_features))
                X_va_p = apply_preprocess(X_va, artifacts)
                if model_kind == "RF":
                    model = model_lib.build_rf(seed=seed, n_jobs=model_n_jobs, params=params)
                    model.fit(X_tr_p, y_tr)
                    s = model.predict_proba(X_va_p)[:, 1].astype(np.float32, copy=False)
                elif model_kind == "XGB":
                    model = model_lib.build_xgb(seed=seed, n_jobs=model_n_jobs, params=params)
                    if model is None:
                        return {}, 0.5, -1.0, -1.0
                    model.fit(X_tr_p, y_tr)
                    s = model.predict_proba(X_va_p)[:, 1].astype(np.float32, copy=False)
                else:
                    raise ValueError("There should be an unknown model kind!")

                oof[va_idx] = s

            mask = ~np.isnan(oof)
            if mask.sum() == 0:
                continue

            y_oof = y_tv[mask]
            s_oof = oof[mask]
            thr, (_, _, f1v) = select_threshold_f1(y_oof, s_oof)
            apv = 0.0
            if len(np.unique(y_oof)) > 1:
                apv = float(average_precision_score(y_oof, s_oof))
            if (f1v > best_f1) or (np.isclose(f1v, best_f1) and apv > best_ap):
                best_f1 = float(f1v)
                best_ap = float(apv)
                best_params = dict(params)
                best_thr = float(thr)
        if best_params is None:
            raise RuntimeError(f"No valid config found for {model_kind}. Missing grid or unknown model kind.")

        return best_params, float(best_thr), float(best_f1), float(best_ap)


def fit_and_eval_holdout(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    model_kind: str,
    grid: Dict[str, List[Any]],
    model_n_jobs: int,
    scale_features: bool,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """
    Changed to held out as in GNN: 
      - fit preprocess on TRAIN only
      - for each hyperparam combo:
          fit model on TRAIN
          score on VAL
          choose threshold on VAL to maximize F1
      - select best combo by VAL F1
      - evaluate on TEST using the chosen threshold
      
    TODO: address the old WF, kept it for now as it doesnt affect.
    """
    y_train = y_train.astype(np.uint8, copy=False)
    y_val   = y_val.astype(np.uint8, copy=False)
    y_test  = y_test.astype(np.uint8, copy=False)

    # Preprocess fitted ONLY on TRAIN (matches your GNN scaling philosophy)
    X_train_p, artifacts = fit_preprocess(X_train, scale=bool(scale_features))
    X_val_p  = apply_preprocess(X_val, artifacts)
    X_test_p = apply_preprocess(X_test, artifacts)

    best_params: Optional[Dict[str, Any]] = None
    best_thr: float = 0.5
    best_f1: float = -1.0
    best_ap: float = -1.0
    best_model = None

    for params in iter_grid(grid):
        if model_kind == "RF":
            model = model_lib.build_rf(seed=seed, n_jobs=model_n_jobs, params=params)
        elif model_kind == "XGB":
            model = model_lib.build_xgb(seed=seed, n_jobs=model_n_jobs, params=params)
            if model is None:
                return {"ok": False, "reason": "xgboost_not_installed"}
        else:
            raise ValueError("There should be an unknown model kind!")

        model.fit(X_train_p, y_train)
        val_scores = model.predict_proba(X_val_p)[:, 1].astype(np.float32, copy=False)

        thr, (_, _, f1v) = select_threshold_f1(y_val, val_scores)
        apv = 0.0
        if len(np.unique(y_val)) > 1:
            apv = float(average_precision_score(y_val, val_scores))

        if (f1v > best_f1) or (np.isclose(f1v, best_f1) and apv > best_ap):
            best_f1 = float(f1v)
            best_ap = float(apv)
            best_thr = float(thr)
            best_params = dict(params)
            best_model = model  # already trained on TRAIN

    if best_params is None or best_model is None:
        raise RuntimeError(f"No valid config found for {model_kind}.")

    # Test evaluation (model trained on TRAIN, threshold chosen on VAL)
    test_scores = best_model.predict_proba(X_test_p)[:, 1].astype(np.float32, copy=False)
    metrics = compute_metrics(y_test, test_scores, best_thr)
    pred_test = (test_scores >= float(best_thr)).astype(np.uint8)

    metrics.update({
        "ok": True,
        "threshold": float(best_thr),
        "params": dict(best_params),
        "val_f1": float(best_f1),
        "val_ap": float(best_ap),

        # Persistables
        "persist_model": best_model,
        "persist_preprocess": artifacts,
        "persist_feature_cols": list(feature_cols),
        "test_scores": test_scores,
        "test_pred": pred_test,
        "test_y": y_test.astype(np.uint8, copy=False),
    })
    return metrics



def fit_and_eval(
    *,
    X_tv: np.ndarray,
    y_tv: np.ndarray,
    t_hour_tv: Optional[np.ndarray],  # <-- NEW
    X_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
    model_kind: str,
    grid: Dict[str, List[Any]],
    n_splits_wf: int,
    embargo_n: int,
    model_n_jobs: int,
    scale_features: bool,
    feature_cols: List[str]
) -> Dict[str, Any]:
    """
    End-to-end routine: tune (OOF) -> refit -> evaluate on TEST.
    """

    best_params, best_thr, best_oof_f1, best_oof_ap = tune_with_oof(
        X_tv=X_tv,
        y_tv=y_tv,
        t_hour_tv=t_hour_tv,  # <-- NEW
        seed=seed,
        model_kind=model_kind,
        grid=grid,
        n_splits_wf=n_splits_wf,
        embargo_n=embargo_n,
        model_n_jobs=model_n_jobs,
        scale_features=scale_features
    )

    # Optional deps missing
    if model_kind == "XGB" and best_oof_f1 < 0:
        return {"ok": False, "reason": "xgboost_not_installed"}

    # Models
    X_tv_p, artifacts = fit_preprocess(X_tv, scale=bool(scale_features))
    X_test_p = apply_preprocess(X_test, artifacts)

    model = None
    scores_test = None

    if model_kind == "RF":
        model = model_lib.build_rf(seed=seed, n_jobs=model_n_jobs, params=best_params)
        model.fit(X_tv_p, y_tv)
        scores_test = model.predict_proba(X_test_p)[:, 1].astype(np.float32, copy=False)
    elif model_kind == "XGB":
        model = model_lib.build_xgb(seed=seed, n_jobs=model_n_jobs, params=best_params)
        if model is None:
            return {"ok": False, "reason": "xgboost_not_installed"}
        model.fit(X_tv_p, y_tv)
        scores_test = model.predict_proba(X_test_p)[:, 1].astype(np.float32, copy=False)
    else:
        raise ValueError("There should be an unkwon model_kind!")
        
    # compute metrics for test set
    metrics = compute_metrics(y_test, scores_test, best_thr)
    pred_test = (scores_test >= float(best_thr)).astype(np.uint8)
    metrics.update({
        "ok": True,
        "threshold": float(best_thr),
        "params": dict(best_params),
        "oof_f1": float(best_oof_f1),
        "oof_ap": float(best_oof_ap),
        # Persistables
        "persist_model": model,
        "persist_preprocess": artifacts,
        "persist_feature_cols": list(feature_cols),
        "test_scores": scores_test,
        "test_pred": pred_test,
        "test_y": y_test.astype(np.uint8, copy=False),
    })
    return metrics


def run_seed(
    *,
    seed: int,
    X_all: np.ndarray,
    y_all: np.ndarray,

    # explicit splits 
    idx_train: np.ndarray,
    idx_val: np.ndarray,
    idx_test: np.ndarray,

    models_to_run: Set[str],
    rf_grid: Dict[str, List[Any]],
    xgb_grid: Dict[str, List[Any]],
    n_splits_wf: int,   # TODO - address WF
    embargo_n: int,     # TODO - address WF
    model_n_jobs: int,
    scale_trees: bool,

    # Persistence inputs
    ds_name: str,
    save_root: str,
    save_models: bool,
    save_predictions: bool,
    id_df: pd.DataFrame,
    feature_cols: List[str],
) -> Dict[str, Any]:
    """
    Run selected models for one seed (TRAIN->VAL selection, TEST evaluation).
    """
    np.random.seed(int(seed))

    X_train = X_all[idx_train]
    y_train = y_all[idx_train]
    X_val   = X_all[idx_val]
    y_val   = y_all[idx_val]
    X_test  = X_all[idx_test]
    y_test  = y_all[idx_test]

    out: Dict[str, Any] = {}

    if "RF" in models_to_run:
        out["RF"] = fit_and_eval_holdout(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            seed=seed, model_kind="RF",
            grid=rf_grid,
            model_n_jobs=model_n_jobs,
            scale_features=bool(scale_trees),
            feature_cols=feature_cols,
        )
        save_artifacts(
            save_root=save_root,
            ds_name=ds_name,
            seed=seed,
            model_name="RF",
            result=out["RF"],
            id_df=id_df,
            idx_test=idx_test,
            feature_cols=feature_cols,
            save_models=save_models,
            save_predictions=save_predictions,
        )

    if "XGB" in models_to_run:
        out["XGB"] = fit_and_eval_holdout(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            seed=seed, model_kind="XGB",
            grid=xgb_grid,
            model_n_jobs=model_n_jobs,
            scale_features=bool(scale_trees),
            feature_cols=feature_cols,
        )
        save_artifacts(
            save_root=save_root,
            ds_name=ds_name,
            seed=seed,
            model_name="XGB",
            result=out["XGB"],
            id_df=id_df,
            idx_test=idx_test,
            feature_cols=feature_cols,
            save_models=save_models,
            save_predictions=save_predictions,
        )

    return out
