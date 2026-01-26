"""
Created on Tue Jan 6 09:01:35 2026

This module builds engineered features used to detect pump-and-dump behavior
from hourly Binance kline panels.

It follows the “change in rolling statistics” idea used in the Sapienza
pump-and-dump dataset work: for each symbol, compute rolling mean/std over a
window and then take the percentage change of that rolling statistic.

Author: Luca Persia (USI/ZHAW)
"""

from __future__ import annotations
from typing import Literal, List
import numpy as np
import pandas as pd

### Helpers
def _pct_change_of_rolling_stat(
    x: pd.Series,
    *,
    window: int,
    stat: Literal["std", "mean"],
) -> pd.Series:
    
    """
    Compute percentage change of a rolling statistic for a single time series.

    Intuition
    ---------
    1) Compute rolling statistic over `window` observations.
    2) Take pct_change of the rolling statistic.

    This produce information about how fast is variability/average 
    changing type feature.

    Input
    -----
    x : pd.Series
        Time-ordered values for a single symbol.
    window : int
        Rolling window length (in rows). With hourly data, using 12.
    stat : {"std", "mean"}
        Which rolling statistic to compute.

    Output
    ------
    pd.Series
        Series aligned with x index, containing pct_change(rolling_stat(x)).
        Leading values are NaN due to rolling window + pct_change.
    """
    
    if stat == "std":
        r = x.rolling(window=window).std()
    elif stat == "mean":
        r = x.rolling(window=window).mean()
    else:
        raise ValueError(f"Unknown stat={stat!r}. Use 'std' or 'mean'.")

    return r.pct_change()


def _require_columns(df: pd.DataFrame, required: List[str]) -> None:
    
    """
    Validate that required columns exist in df.

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


### Enriching function
def add_pump_features(
    df: pd.DataFrame,
    *,
    window: int = 12,
) -> pd.DataFrame:
    
    """
    Add pump-and-dump engineered features to an hourly kline panel.

    For each symbol, compute rolling mean/std over the specified window and then
    take percentage changes to get fast regime shifts in microstructure.
    
    Note:
    - In our implementation we only have aggregated data, 
    the idea is to reconstruct rush orders by considering the 
    "buy pressure variability changes"
    ----> the ratio: taker_buy_quote / volume

    Required columns
    ----------------
    date, symbol, close, high, volume, num_trades, taker_buy_quote

    Input
    -----
    df : pd.DataFrame
        Hourly panel with one row per (symbol, date).
    window : int, default=12
        Rolling window length (rows).

    Output
    ------
    pd.DataFrame
        Copy of df with additional columns:

        buy_pressure
            taker_buy_quote / volume (with volume=0 treated as NaN)
        std_rush_order
            pct_change(rolling_std(buy_pressure))
        avg_rush_order
            pct_change(rolling_mean(buy_pressure))
        std_trades
            pct_change(rolling_std(num_trades))
        std_volume
            pct_change(rolling_std(volume))
        avg_volume
            pct_change(rolling_mean(volume))
        std_price
            pct_change(rolling_std(close))
        avg_price
            pct_change(rolling_mean(close))
        avg_price_max
            pct_change(rolling_mean(high))

    IMPORTANT
    -----
    - NaNs at the beginning of each symbol’s history due to rolling windows.
    - Percentage change can also produce inf if the rolling statistic hits zero. 
      Downstream preprocessing should handle inf/nan.
    """
    
    required = ["date", "symbol", "close", "high", "volume", 
                "num_trades", "taker_buy_quote"]
    _require_columns(df, required)

    # Ensure deterministic time ordering per symbol before rolling
    df_out = df.sort_values(["symbol", "date"]).copy()
    g = df_out.groupby("symbol", group_keys=False)

    ### "Microstructure" proxy for rush-order pressure:
    # taker_buy_quote represents the quote-volume bought by taker
    # It should give an idea of aggressive buys
    # Dividing by total volume normalizes it. If volume==0, treat as NaN.
    vol_safe = df_out["volume"].replace(0, np.nan)
    df_out["buy_pressure"] = df_out["taker_buy_quote"] / vol_safe

    ## Rush-order features
    df_out["std_rush_order"] = g["buy_pressure"].transform(
        lambda x: _pct_change_of_rolling_stat(x, window=window, stat="std")
    )
    df_out["avg_rush_order"] = g["buy_pressure"].transform(
        lambda x: _pct_change_of_rolling_stat(x, window=window, stat="mean")
    )

    ## Trades
    df_out["std_trades"] = g["num_trades"].transform(
        lambda x: _pct_change_of_rolling_stat(x, window=window, stat="std")
    )

    ## Volume (liquidity / attention shifts)
    df_out["std_volume"] = g["volume"].transform(
        lambda x: _pct_change_of_rolling_stat(x, window=window, stat="std")
    )
    df_out["avg_volume"] = g["volume"].transform(
        lambda x: _pct_change_of_rolling_stat(x, window=window, stat="mean")
    )

    ## Price dynamics
    df_out["std_price"] = g["close"].transform(
        lambda x: _pct_change_of_rolling_stat(x, window=window, stat="std")
    )
    df_out["avg_price"] = g["close"].transform(
        lambda x: _pct_change_of_rolling_stat(x, window=window, stat="mean")
    )
    df_out["avg_price_max"] = g["high"].transform(
        lambda x: _pct_change_of_rolling_stat(x, window=window, stat="mean")
    )

    return df_out
