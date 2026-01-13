"""
Created on Tue Jan 6 07:31:39 2026

Panel builder for pump-and-dump datasets. This module builds two panel families:

(A) GLOBAL panels
    One large time window per token:
      [global_first_pump - days_before, global_last_pump + days_after]
    Pump hours are flagged inside that global window.

(B) WINDOWED panels
    Stacked per-pump chunks per token:
      for each pump time t, download [t-window_days, t+window_days]
    Chunks are stacked and pump hours are flagged.
    Rolling engineered features are computed per chunk in this case.

We splits tokens into:
  - multi-pump tokens (>= 2 pump events)
  - single-pump tokens (= 1 pump event)

Author: Lidia Losavio (USI), Luca Persia (USI/ZHAW)
"""

from __future__ import annotations
import numpy as np, pandas as pd
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Dict, List
from pumpdownloader.binance import fetch_klines
from pumpdownloader.features import add_pump_features


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
    Convert Binance open_time (ms UTC) to timezone-aware timestamps in cfg.timezone.

    Input
    -----
    open_time_ms : pd.Series
        Binance open_time values in ms since epoch.
    cfg : PanelConfig
        Panel configuration.

    Output
    ------
    pd.Series
        tz-aware datetime series in cfg.timezone.
    """
    
    dt = pd.to_datetime(open_time_ms, unit="ms", utc=True)
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


def _fetch_token_windows(
    symbol_rest: str,
    pump_times: np.ndarray,
    cfg: PanelConfig,
) -> pd.DataFrame:
    
    """
    Download per-pump chunks for one token, stack them, flagging pump hours.

    For each pump time t, download:
      [t - cfg.window_days, t + cfg.window_days]

    A chunk id is attached so engineered rolling features can be computed per chunk
    without mixing different pump windows.

    Input
    -----
    symbol_rest : str
        Binance symbol without slash, e.g. "ETHBTC".
    pump_times : np.ndarray
        Pump timestamps for this token.
    cfg : PanelConfig
        Panel configuration.

    Output
    ------
    pd.DataFrame
        Base windowed panel with BASE_COLS + ["__chunk"].
        Empty DataFrame if nothing was fetched.
    """
    
    pts = _pump_times_in_cfg_tz(pump_times, cfg)

    frames: List[pd.DataFrame] = []
    for t in pts:
        start = t - pd.Timedelta(days=cfg.window_days)
        end = t + pd.Timedelta(days=cfg.window_days)

        kdf = fetch_klines(symbol_rest, cfg.interval, _to_utc_ms(start), _to_utc_ms(end), limit=1000)
        if kdf.empty:
            continue

        kdf["symbol"] = symbol_rest
        kdf["date"] = _convert_open_time_to_cfg_tz(kdf["open_time"], cfg)
        kdf["flag"] = kdf["date"].isin(pts).astype(int)

        # Unique chunk id per pump event
        kdf["__chunk"] = f"{symbol_rest}||{t.isoformat()}"

        frames.append(kdf[BASE_COLS + ["__chunk"]].copy())

    if not frames:
        return pd.DataFrame(columns=BASE_COLS + ["__chunk"])

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["__chunk", "date"]).reset_index(drop=True)
    return out


# Feature engineering for windowed panels
def _engineer_per_chunk(window_df_with_chunk: pd.DataFrame, *, 
                        window: int = 12) -> pd.DataFrame:
    
    """
    Compute engineered rolling features per chunk (no mixing across pump windows).
    add_pump_features() computes rolling stats grouped by "symbol".
    For windowed data we want grouping by chunk, not by token.
    We temporarily set:
      symbol := chunk_id
    so the feature function groups per chunk, then restore symbol afterwards.

    Input
    -----
    window_df_with_chunk : pd.DataFrame
        DataFrame with BASE_COLS + ["__chunk"].
    window : int, default=12
        Rolling window size in rows (hours if interval="1h").

    Output
    ------
    pd.DataFrame
        Engineered panel with BASE_COLS + engineered features (no __chunk).
    """
    
    if window_df_with_chunk.empty:
        return window_df_with_chunk.drop(columns=["__chunk"], errors="ignore").copy()

    dfw = window_df_with_chunk.copy()
    dfw = dfw.sort_values(["__chunk", "date"]).reset_index(drop=True)

    # Stable row id to restore symbol even
    dfw["__rowid"] = np.arange(len(dfw), dtype=np.int64)
    dfw["__token_symbol"] = dfw["symbol"].astype(str)

    out_parts: List[pd.DataFrame] = []

    for chunk_id, g in dfw.groupby("__chunk", sort=False):
        g = g.sort_values("date").copy()

        # Force grouping in add_pump_features() to be per chunk
        g["symbol"] = chunk_id

        eng = add_pump_features(g, window=window)

        # Restore token symbol
        if "__token_symbol" in eng.columns:
            eng["symbol"] = eng["__token_symbol"].astype(str)
        else:
            eng = eng.merge(
                g[["__rowid", "__token_symbol"]],
                on="__rowid",
                how="left",
                validate="one_to_one",
            )
            eng["symbol"] = eng["__token_symbol"].astype(str)

        # Drop internal helper columns
        eng = eng.drop(columns=["__chunk", "__token_symbol"], errors="ignore")
        out_parts.append(eng)

    engineered = pd.concat(out_parts, ignore_index=True)
    engineered = engineered.sort_values(["date", "symbol"]).reset_index(drop=True)
    return engineered


### Buld panels here
def build_panels(csv_path: Path, cfg: PanelConfig) -> Dict[str, pd.DataFrame]:
    
    """
    Build and return all panels.

    Panels returned
    ---------------
    GLOBAL (single window per token):
      - panel_base
      - panel_engineered
      - single_pump_base
      - single_pump_engineered

    WINDOWED (stacked per-pump chunks):
      - panel_base_w
      - panel_engineered_w
      - single_pump_base_w
      - single_pump_engineered_w

    Input
    -----
    csv_path : Path
        Path to pump_telegram.csv.
    cfg : PanelConfig
        Panel configuration.

    Output
    ------
    Dict[str, pd.DataFrame]
        Dictionary mapping keys above to DataFrames.
    """
    src = pd.read_csv(csv_path)

    pumps = src[src["exchange"].astype(str).str.lower() == cfg.exchange_filter.lower()].copy()
    if pumps.empty:
        empty_base = pd.DataFrame(columns=BASE_COLS)
        return {
            "panel_base": empty_base,
            "panel_engineered": empty_base.copy(),
            "single_pump_base": empty_base.copy(),
            "single_pump_engineered": empty_base.copy(),
            "panel_base_w": empty_base.copy(),
            "panel_engineered_w": empty_base.copy(),
            "single_pump_base_w": empty_base.copy(),
            "single_pump_engineered_w": empty_base.copy(),
        }

    # Binance REST symbol (ETH + BTC => ETHBTC)
    pumps["symbol_rest"] = pumps["symbol"].astype(str) + cfg.quote

    # Pump timestamp
    pumps["pump_time"] = pumps.apply(
        lambda r: _pump_time(r, cfg.timezone, interval=cfg.interval, do_round=cfg.round_pump_time),
        axis=1,
    )

    # Split tokens by pump count
    counts = pumps.groupby("symbol_rest").size()
    multi_syms = counts[counts >= 2].index.tolist()
    single_syms = counts[counts == 1].index.tolist()

    print(f"[TOKENS] multi-pump (>=2): {len(multi_syms)} | single-pump (=1): {len(single_syms)}")

    # Global window boundaries
    global_min = pumps["pump_time"].min()
    global_max = pumps["pump_time"].max()

    start = global_min - pd.Timedelta(days=cfg.days_before)
    end = global_max + pd.Timedelta(days=cfg.days_after)

    start_ms = _to_utc_ms(start)
    end_ms = _to_utc_ms(end)

    print(f"[WINDOW] GLOBAL: {start} -> {end}  (before={cfg.days_before}, after={cfg.days_after})")
    print(f"[WINDOW] PER-PUMP: +/- {cfg.window_days} days")

    failures = 0

    # GLOBAL panels
    multi_frames: List[pd.DataFrame] = []
    single_frames: List[pd.DataFrame] = []

    for sym in multi_syms:
        try:
            pts = pumps.loc[pumps["symbol_rest"] == sym, "pump_time"].to_numpy()
            df_sym = _fetch_token_panel(sym, pts, cfg, start_ms, end_ms)
            if not df_sym.empty:
                multi_frames.append(df_sym)
        except Exception as e:
            failures += 1
            print(f"[WARN] Failed (GLOBAL) {sym}: {e}")

    for sym in single_syms:
        try:
            pts = pumps.loc[pumps["symbol_rest"] == sym, "pump_time"].to_numpy()
            df_sym = _fetch_token_panel(sym, pts, cfg, start_ms, end_ms)
            if not df_sym.empty:
                single_frames.append(df_sym)
        except Exception as e:
            failures += 1
            print(f"[WARN] Failed (GLOBAL) {sym}: {e}")

    panel_base = (
        pd.concat(multi_frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
        if multi_frames else pd.DataFrame(columns=BASE_COLS)
    )
    single_pump_base = (
        pd.concat(single_frames, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
        if single_frames else pd.DataFrame(columns=BASE_COLS)
    )

    panel_engineered = add_pump_features(panel_base, window=12) if not panel_base.empty else panel_base.copy()
    single_pump_engineered = (
        add_pump_features(single_pump_base, window=12) if not single_pump_base.empty else single_pump_base.copy()
    )

    # WINDOWED panels
    multi_w_frames: List[pd.DataFrame] = []
    single_w_frames: List[pd.DataFrame] = []

    for sym in multi_syms:
        try:
            pts = pumps.loc[pumps["symbol_rest"] == sym, "pump_time"].to_numpy()
            df_w = _fetch_token_windows(sym, pts, cfg)
            if not df_w.empty:
                multi_w_frames.append(df_w)
        except Exception as e:
            failures += 1
            print(f"[WARN] Failed (WINDOWED) {sym}: {e}")

    for sym in single_syms:
        try:
            pts = pumps.loc[pumps["symbol_rest"] == sym, "pump_time"].to_numpy()
            df_w = _fetch_token_windows(sym, pts, cfg)
            if not df_w.empty:
                single_w_frames.append(df_w)
        except Exception as e:
            failures += 1
            print(f"[WARN] Failed (WINDOWED) {sym}: {e}")

    panel_base_w_raw = (
        pd.concat(multi_w_frames, ignore_index=True).reset_index(drop=True)
        if multi_w_frames else pd.DataFrame(columns=BASE_COLS + ["__chunk"])
    )
    single_pump_base_w_raw = (
        pd.concat(single_w_frames, ignore_index=True).reset_index(drop=True)
        if single_w_frames else pd.DataFrame(columns=BASE_COLS + ["__chunk"])
    )

    # Base window outputs should not include __chunk
    panel_base_w = (
        panel_base_w_raw.drop(columns=["__chunk"], errors="ignore")
        .sort_values(["date", "symbol"]).reset_index(drop=True)
        if not panel_base_w_raw.empty else pd.DataFrame(columns=BASE_COLS)
    )
    single_pump_base_w = (
        single_pump_base_w_raw.drop(columns=["__chunk"], errors="ignore")
        .sort_values(["date", "symbol"]).reset_index(drop=True)
        if not single_pump_base_w_raw.empty else pd.DataFrame(columns=BASE_COLS)
    )

    # Engineered features computed per chunk
    panel_engineered_w = (
        _engineer_per_chunk(panel_base_w_raw, window=12) if not panel_base_w_raw.empty else panel_base_w.copy()
    )
    single_pump_engineered_w = (
        _engineer_per_chunk(single_pump_base_w_raw, window=12) if not single_pump_base_w_raw.empty else single_pump_base_w.copy()
    )

    if failures:
        print(f"[INFO] Completed with {failures} failures.")

    print(f"[DONE] GLOBAL  multi rows={len(panel_base):,} | engineered rows={len(panel_engineered):,}")
    print(f"[DONE] GLOBAL  single rows={len(single_pump_base):,} | engineered rows={len(single_pump_engineered):,}")
    print(f"[DONE] WINDOW multi rows={len(panel_base_w):,} | engineered rows={len(panel_engineered_w):,}")
    print(f"[DONE] WINDOW single rows={len(single_pump_base_w):,} | engineered rows={len(single_pump_engineered_w):,}")

    return {
        "panel_base": panel_base,
        "panel_engineered": panel_engineered,
        "single_pump_base": single_pump_base,
        "single_pump_engineered": single_pump_engineered,
        "panel_base_w": panel_base_w,
        "panel_engineered_w": panel_engineered_w,
        "single_pump_base_w": single_pump_base_w,
        "single_pump_engineered_w": single_pump_engineered_w,
    }


def build_panel(csv_path: Path, cfg: PanelConfig) -> pd.DataFrame:
    """
    Backward-compatible helper: return GLOBAL engineered multi-pump dataset.

    Input
    -----
    csv_path : Path
        Path to pump_telegram.csv.
    cfg : PanelConfig
        Panel configuration.

    Output
    ------
    pd.DataFrame
        Engineered global multi-pump panel.
    """
    
    outs = build_panels(csv_path, cfg)
    return outs["panel_engineered"]
