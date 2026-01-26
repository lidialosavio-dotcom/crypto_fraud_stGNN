"""
Created on Fri Jan 02 05:35:55 2026

This script holds the baseline models used for the classification of
pump and dumps.

Author: Luca Persia (USI/ZHAW)
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

### RANDOM FOREST
def build_rf(
    *,
    seed: int,
    n_jobs: int,
    params: Dict[str, Any]
) -> RandomForestClassifier:
    """
    RandomForestClassifier.
    """
    return RandomForestClassifier(
        random_state=int(seed),
        n_jobs=int(n_jobs),
        **params,
    )

# XGBOOST
def build_xgb(
    *,
    seed: int,
    n_jobs: int,
    params: Dict[str, Any]
) -> Optional["XGBClassifier"]:
    """
    XGBClassifier.
    """
    return XGBClassifier(
        random_state=int(seed),
        n_jobs=int(n_jobs),
        eval_metric="logloss",
        tree_method="hist",
        verbosity=0,
        **params,
    )

