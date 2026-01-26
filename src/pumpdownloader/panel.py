"""
Created on Tue Jan 6 07:31:39 2026

Panel builder for pump-and-dump datasets. This module builds two panel families:

(A) GLOBAL panels
    One large time window per token:
      [global_first_pump - days_before, global_last_pump + days_after]
    Pump hours are flagged inside that global window.

(B) Per - Token panels
    Each token has its own timeframe. The final dataset stacks all the tokens together.
    The ideas is that some token could also be not traded at a specific point in time.

Author: Lidia Losavio (USI), Luca Persia (USI/ZHAW)
"""

from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Literal
from zoneinfo import ZoneInfo
from pumpdownloader.binance import fetch_klines
from pumpdownloader.features import add_pump_features

WindowMode = Literal["per_token", "dataset_global", "both"]

### Config
@dataclass(frozen=True)
class PanelConfig:
    """
    Configuration for panel building.

    Parameters
    ----------
    interval : str
        Binance kline interval (e.g., "1m", "5m", "1h", "1d").
    days_before : int
        Global window extension before the earliest pump time.
    days_after : int
        Global window extension after the latest pump time.
    window_days : int
        Half-window size for per-pump chunking in windowed panels.
    quote : str
        Quote currency suffix appended to symbols (e.g., BTC -> ETHBTC).
        We only used BTC, one could explore USD or ETH.
    exchange_filter : str
        Keep only pump records matching this exchange string.
    timezone : str
        Timezone used to interpret pump times.
    round_pump_time : bool
        If True, round pump timestamps to the nearest candle interval.
    """
    
    interval: str = "1h"
    days_before: int = 7
    days_after: int = 7
    window_days: int = 3
    quote: str = "BTC"
    exchange_filter: str = "binance"
    timezone: str = "UTC"
    round_pump_time: bool = True


# Global cols
BASE_COLS = [
    "date", "symbol",
    "open", "high", "low", "close",
    "volume", "quote_asset_volume", "num_trades",
    "taker_buy_base", "taker_buy_quote",
    "flag",
]


# Time helpers
def _interval_to_pandas_freq(interval: str) -> str:
    
    """
    Convert Binance interval notation to a pandas rounding frequency string.

    Input
    -----
    interval : str
        Binance-style interval such as "15m", "1h", "1d", "1w", "1M".

    Output
    ------
    str
        Pandas frequency string compatible with Timestamp.round().

    Raises
    ------
    ValueError
        If the interval is not supported for rounding.
    """
    
    if interval.endswith("m"):
        return f"{int(interval[:-1])}min"
    if interval.endswith("h"):
        return f"{int(interval[:-1])}h"
    if interval.endswith("d"):
        return f"{int(interval[:-1])}d"
    if interval.endswith("w"):
        return f"{int(interval[:-1])}w"
    if interval.endswith("M"):
        # Month start frequency
        return f"{int(interval[:-1])}MS"
    raise ValueError(f"This interval is not supported for rounding: {interval!r}")


def _pump_time(row: pd.Series, 
               tz: str, *, 
               interval: str, 
               do_round: bool) -> pd.Timestamp:
    """
    Build pump timestamp from the raw CSV row.

    The CSV stores:
      - row["date"] like "YYYY-mm-dd"
      - row["hour"] like "HH:MM"

    Input
    -----
    row : pd.Series
        Pump record row from pump_telegram.csv.
    tz : str
        IANA timezone string (e.g., "UTC", "Europe/Zurich").
        We used UTC. This should be consistend within the number of panels.
    interval : str
        Binance interval, used only if rounding is enabled.
    do_round : bool
        If True, round pump time to the nearest candle boundary.

    Output
    ------
    pd.Timestamp
        Timezone-aware pump timestamp.
    """
    
    t = pd.to_datetime(f"{row['date']} {row['hour']}", format="%Y-%m-%d %H:%M")
    t = t.tz_localize(ZoneInfo(tz))

    if do_round:
        freq = _interval_to_pandas_freq(interval)
        t = t.round(freq)

    return t


def _to_utc_ms(ts: pd.Timestamp) -> int:
    
    """
    Convert a timestamp to UTC milliseconds (Binance).

    Input
    -----
    ts : pd.Timestamp
        Timestamp (tz-aware).

    Output
    ------
    int
        Milliseconds since epoch in UTC.
    """
    
    if ts.tzinfo is None:
        ts = ts.tz_localize(ZoneInfo("UTC"))
    ts_utc = ts.tz_convert(ZoneInfo("UTC"))
    return int(ts_utc.timestamp() * 1000)


def _convert_open_time_to_cfg_tz(open_time_ms: pd.Series,
                                 cfg: PanelConfig) -> pd.Series:
    """
    Convert Binance open_time to timezone-aware timestamps in cfg.timezone.

    Works with either:
      - numeric milliseconds since epoch, or
      - datetime-like inputs (fetch_klines already returns tz-aware timestamps).
    """
    # Robust to numeric millisecond inputs *and* datetime-like inputs.
    if pd.api.types.is_numeric_dtype(open_time_ms):
        dt = pd.to_datetime(open_time_ms, unit="ms", utc=True)
    else:
        dt = pd.to_datetime(open_time_ms, utc=True)

    if cfg.timezone.upper() != "UTC":
        dt = dt.dt.tz_convert(ZoneInfo(cfg.timezone))
    return dt


def _pump_times_in_cfg_tz(pump_times: np.ndarray,
                          cfg: PanelConfig) -> pd.DatetimeIndex:
    """
    Ensure pump_times is a tz-aware DatetimeIndex in cfg.timezone.

    Input
    -----
    pump_times : np.ndarray
        Array-like of timestamps.
    cfg : PanelConfig
        Panel configuration.

    Output
    ------
    pd.DatetimeIndex
        Pump times aligned to cfg.timezone.
    """
    
    pts = pd.to_datetime(pump_times)
    if cfg.timezone.upper() != "UTC":
        pts = pts.tz_convert(ZoneInfo(cfg.timezone))
    return pts


### Set of download helper functions
def _fetch_token_panel(
    symbol_rest: str,
    pump_times: np.ndarray,
    cfg: PanelConfig,
    start_ms: int,
    end_ms: int,
) -> pd.DataFrame:
    
    """
    Download a single GLOBAL window for one token, flagging pump hours.

    Input
    -----
    symbol_rest : str
        Binance symbol without slash, e.g. "ETHBTC".
    pump_times : np.ndarray
        Pump timestamps for this token.
    cfg : PanelConfig
        Panel configuration.
    start_ms, end_ms : int
        UTC milliseconds window boundaries for Binance.

    Output
    ------
    pd.DataFrame
        Base panel for this token with BASE_COLS.
        Empty DataFrame if no klines are available.
    """
    print(
        f"[FETCH] {symbol_rest} [{cfg.interval}] "
        f"{pd.to_datetime(start_ms, unit='ms', utc=True)} -> {pd.to_datetime(end_ms, unit='ms', utc=True)}"
    )

    kdf = fetch_klines(symbol_rest, cfg.interval, start_ms, end_ms, limit=1000)
    if kdf.empty:
        return pd.DataFrame(columns=BASE_COLS)

    kdf["symbol"] = symbol_rest
    kdf["date"] = _convert_open_time_to_cfg_tz(kdf["open_time"], cfg)
    pts = _pump_times_in_cfg_tz(pump_times, cfg)
    kdf["flag"] = kdf["date"].isin(pts).astype(int)

    return kdf[BASE_COLS].copy()


def build_panels(csv_path: Path, cfg: PanelConfig, *, window_mode: WindowMode = "dataset_global") -> Dict[str, pd.DataFrame]:
    src = pd.read_csv(csv_path)

    pumps = src[src["exchange"].astype(str).str.lower() == cfg.exchange_filter.lower()].copy()
    if pumps.empty:
        empty_base = pd.DataFrame(columns=BASE_COLS)
        if window_mode == "both":
            return {
                "panel_base_token": empty_base.copy(),
                "panel_engineered_token": empty_base.copy(),
                "panel_base_global": empty_base.copy(),
                "panel_engineered_global": empty_base.copy(),
            }
        return {"panel_base": empty_base, "panel_engineered": empty_base.copy()}

    pumps["symbol_rest"] = pumps["symbol"].astype(str) + cfg.quote

    pumps["pump_time"] = pumps.apply(
        lambda r: _pump_time(r, cfg.timezone, interval=cfg.interval, do_round=cfg.round_pump_time),
        axis=1,
    )

    pumps = pumps.drop_duplicates(subset=["symbol_rest", "pump_time"])

    grouped = pumps.groupby("symbol_rest", sort=True)
    symbols = list(grouped.groups.keys())
    print(f"[TOKENS]: {len(symbols)}")

    td_before = pd.Timedelta(days=cfg.days_before)
    td_after = pd.Timedelta(days=cfg.days_after)

    global_start = pumps["pump_time"].min() - td_before
    global_end = pumps["pump_time"].max() + td_after

    def _build_base(mode: Literal["per_token", "dataset_global"]) -> pd.DataFrame:
        failures = 0
        frames: List[pd.DataFrame] = []

        for sym, gsym in grouped:
            try:
                pts = gsym["pump_time"].to_numpy()

                if mode == "per_token":
                    start = gsym["pump_time"].min() - td_before
                    end = gsym["pump_time"].max() + td_after
                else:
                    start = global_start
                    end = global_end

                df_sym = _fetch_token_panel(sym, pts, cfg, _to_utc_ms(start), _to_utc_ms(end))
                if not df_sym.empty:
                    frames.append(df_sym)

            except Exception as e:
                failures += 1
                print(f"[WARN] Failed ({mode}) {sym}: {e}")

        if failures:
            print(f"[INFO] Completed {mode} with {failures} failures.")

        if not frames:
            return pd.DataFrame(columns=BASE_COLS)

        return (
            pd.concat(frames, ignore_index=True)
            .sort_values(["date", "symbol"])
            .reset_index(drop=True)
        )

    if window_mode in ("per_token", "dataset_global"):
        base = _build_base(window_mode)
        engineered = add_pump_features(base, window=12) if not base.empty else base.copy()
        return {"panel_base": base, "panel_engineered": engineered}

    # both
    base_token = _build_base("per_token")
    eng_token = add_pump_features(base_token, window=12) if not base_token.empty else base_token.copy()

    base_global = _build_base("dataset_global")
    eng_global = add_pump_features(base_global, window=12) if not base_global.empty else base_global.copy()

    return {
        "panel_base_token": base_token,
        "panel_engineered_token": eng_token,
        "panel_base_global": base_global,
        "panel_engineered_global": eng_global,
    }


def build_panel(csv_path: Path, cfg: PanelConfig, *, window_mode: Literal["per_token", "dataset_global"] = "dataset_global") -> pd.DataFrame:
    outs = build_panels(csv_path, cfg, window_mode=window_mode)
    return outs["panel_engineered"]["panel_engineered"]
