"""
Created on Fri Jan 02 05:35:55 2026

This script holds the baseline models used for the classification of
pump and dumps.

Author: Lidia Losavio (USI), Luca Persia (USI/ZHAW)
"""

from __future__ import annotations
from typing import Any, Dict, Optional
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

### Random Forest function
def build_rf(
    *,
    seed: int,
    n_jobs: int,
    params: Dict[str, Any],
    scale_pos_weight: float,
) -> RandomForestClassifier:
  
    """
    Build a standard RandomForestClassifier for binary classification.

    Notes
    -----
    - Currently the model in not weighting classes. To let the RF 
    to actually account for imbalance through class weights set:
    class_weight={0: 1.0, 1: scale_pos_weight} inside this builder.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_jobs : int
        Number of CPU threads to train trees in parallel.
    params : Dict[str, Any]
        Hyperparameters for RandomForestClassifier.
    scale_pos_weight : float
        Imbalance ratio (#neg / #pos).

    Returns
    -------
    RandomForestClassifier
        Unfitted sklearn RandomForestClassifier instance.
    """
    
    return RandomForestClassifier(
        random_state=int(seed),
        n_jobs=int(n_jobs),
        **params,
    )


### XGBoost Function
def build_xgb(
    *,
    seed: int,
    n_jobs: int,
    params: Dict[str, Any],
    scale_pos_weight: float,
) -> Optional["XGBClassifier"]:
  
    """
    Build a standard XGBoost classifier for binary classification.

    Notes
    -----
    - Currently the models in not weighting classes. To let the XGB to handle 
    imbalance  pass: scale_pos_weight=scale_pos_weight in the constructor.

    Parameters
    ----------
    seed : int
        Random seed for reproducibility.
    n_jobs : int
        Number of CPU threads XGBoost can use.
    params : Dict[str, Any]
        Hyperparameters for XGBClassifier.
    scale_pos_weight : float
        Imbalance ratio (#neg / #pos).

    Returns
    -------
    Optional["XGBClassifier"]
        Unfitted XGBClassifier instance.
    """

    return XGBClassifier(
        random_state=int(seed),
        n_jobs=int(n_jobs),
        eval_metric="logloss",   # training metric
        tree_method="hist",      # CPU histogram algorithm
        verbosity=0,
        **params,
    )


### Simple feedforward NN function
def build_nn(
    *,
    input_dim: int,
    seed: int,
    params: Dict[str, Any],
) -> Optional["tf.keras.Model"]:
    
    """
    Build a simple feedforward neural network for binary classification.
    This is a classic MLP classifier:
        x(features)->Dense(ReLU)->Dropout->Dense(ReLU)->Dropout->Sigmoid.

    Parameters
    ----------
    input_dim : int
        Number of input features (columns). The model input has shape (input_dim,).
    seed : int
        Random seed for reproducibility.
    params : Dict[str, Any]
        Hyperparameters for the NN architecture and optimizer.
        Required keys:
          - "hidden_units": int, size of the first hidden layer
          - "dropout"     : float in [0,1], dropout probability
          - "lr"          : float, learning rate for optimizer

    Returns
    -------
    Optional["tf.keras.Model"]
        A compiled tf.keras.Model with:
          - loss: binary_crossentropy
          - optimizer: Adam(learning_rate=lr)
          - metric: PR-AUC (AUC under precision-recall curve)
    """
  
    # Set seeds for reproducibility
    tf.keras.utils.set_random_seed(int(seed))
    # Hyperparameters
    hidden_units = int(params["hidden_units"])  # width of first hidden layer
    dropout = float(params["dropout"])       
    lr = float(params["lr"])                    

    # Two layers model
    m = models.Sequential([
        layers.Input(shape=(int(input_dim),)),
        layers.Dense(hidden_units, activation="relu"),
        layers.Dropout(dropout),
        # enforce a minimum of 8 units so the second layer doesn’t collapse
        layers.Dense(max(8, hidden_units // 2), activation="relu"),
        layers.Dropout(dropout),
        layers.Dense(1, activation="sigmoid"),
    ])

    # Compilation
    m.compile(
        optimizer=optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(curve="PR", name="pr_auc")],
    )
    return m


def nn_callbacks(train_only: bool = False) -> list:
  
    """
    Construct callbacks for NN training.

    Parameters
    ----------
    train_only : bool
        If False (default):
          We assume we have validation data available and we early-stop on
          validation PR-AUC, trying to maximize it.
        If True:
          We assume we are fitting on TRAIN+VAL without a validation set and we
          early-stop on training loss, trying to minimize it.

    Returns
    -------
    list
        Keras callbacks.
    """
  
    if train_only:
        # No validation set
        return [
            callbacks.EarlyStopping(
                monitor="loss",           # training loss
                patience=3,               # stop after 3 epochs without improvement
                restore_best_weights=True # go back to best epoch weights
            )
        ]

    # With validation monitor val PR-AUC
    return [
        callbacks.EarlyStopping(
            monitor="val_pr_auc",
            mode="max",
            patience=5,
            restore_best_weights=True,
        )
    ]


def nn_clear_session() -> None:
  
    """
    Clear the TensorFlow/Keras session to save some memory in local.

    Returns
    -------
    None
    """
  
    tf.keras.backend.clear_session()
