from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from pumpdownloader.binance import fetch_klines


@dataclass(frozen=True)
class PanelConfig:
    """
    Configuration for building a pump-event panel from Binance klines.

    Input
    -----
    interval : str
        Candle frequency (e.g. '1h').
    days_before : int
        Days to include BEFORE each pump timestamp.
    days_after : int
        Days to include AFTER each pump timestamp.
    quote : str
        Quote currency used to build Binance REST symbol, e.g. 'ETH' + 'BTC' -> 'ETHBTC'.
    exchange_filter : str
        Keep only CSV rows where exchange matches this value (case-insensitive).
    timezone : str
        Timezone used to interpret pump timestamps from the CSV (default 'UTC').
    round_pump_time : bool
        If True, round pump timestamps to the candle grid implied by `interval`.
        This reproduces your original "round-to-hour" behavior when interval='1h'.
    """
    interval: str = "1h"
    days_before: int = 3
    days_after: int = 3
    # note: assuming we have BTC pairs token, this is reasonable for our analysis
    # but could be less robust to different methodologies. One could consider USD or ETH pairs.
    quote: str = "BTC"
    exchange_filter: str = "binance"
    timezone: str = "UTC"
    round_pump_time: bool = True


def _interval_to_pandas_freq(interval: str) -> str:
    """
    Convert Binance interval string to a pandas frequency string.

    Input
    -----
    interval : str
        Binance kline interval, e.g. '1m', '5m', '1h', '1d'.

    Output
    ------
    freq : str
        Pandas frequency string, e.g. '1min', '5min', '1H', '1D'.

    Notes
    -----
    Covers the common intervals used for intraday/daily observations.
    """
    
    # could be that minute are encoded differently in binance
    if interval.endswith("m"):
        return f"{int(interval[:-1])}min"
    
    # intraday is at lower case in the new version
    if interval.endswith("h"):
        return f"{int(interval[:-1])}h"
    
    # daily to monthly observations could be encoded differently
    if interval.endswith("d"):
        return f"{int(interval[:-1])}d"
    if interval.endswith("w"):
        return f"{int(interval[:-1])}w"
      
    # monthly is less common; we include a reasonable default
    if interval.endswith("M"):
        return f"{int(interval[:-1])}MS"
    raise ValueError(f"Unsupported interval for rounding: {interval}")


def _pump_time(row: pd.Series, tz: str, *, interval: str, do_round: bool) -> pd.Timestamp:
    """
    Parse pump timestamp from a CSV row and optionally round it to the candle grid.

    Input
    -----
    row : pd.Series
        One row of the pump CSV. Must contain: 'date' and 'hour'.
    tz : str
        Timezone name, e.g. 'UTC'.
    interval : str
        Binance interval string used to define the candle grid for rounding.
    do_round : bool
        If True, round the pump timestamp to the interval grid. Rounding is
        necessary for our analysis.

    Output
    ------
    pump_time : pd.Timestamp
        Timezone-aware pump timestamp.
    """
    t = pd.to_datetime(f"{row['date']} {row['hour']}", format="%Y-%m-%d %H:%M")
    t = t.tz_localize(ZoneInfo(tz))

    if do_round:
        freq = _interval_to_pandas_freq(interval)
        t = t.round(freq)

    return t


def build_panel(csv_path: Path, cfg: PanelConfig) -> pd.DataFrame:
    """
    Build ONE panel dataset by downloading Binance klines around each pump event.

    Input
    -----
    csv_path : Path
        Path to pump_telegram.csv.
        Required columns:
        - exchange
        - symbol   (base asset, e.g. 'ETH', 'XRP')
        - date     (YYYY-MM-DD)
        - hour     (HH:MM)
        Optional columns:
        - group (note that we keep the group for possible future uses)
    cfg : PanelConfig
        Download window and symbol construction parameters.

    Output
    ------
    panel : pd.DataFrame
        Stacked kline data across all pump events, sorted by (date, symbol).
        Columns:
        date, symbol, open, high, low, close, volume,
        quote_asset_volume, num_trades, taker_buy_base, taker_buy_quote,
        flag, [group if present]
    """
    src = pd.read_csv(csv_path)
    pumps = (src[src["exchange"]
                 .astype(str)
                 .str
                 .lower() == cfg.exchange_filter.lower()].copy())

    # handles general REST symbols like ETHBTC
    pumps["symbol_rest"] = pumps["symbol"].astype(str) + cfg.quote

    frames: list[pd.DataFrame] = []
    failures = 0

    # fetching loop
    for _, row in pumps.iterrows():
        symbol = row["symbol_rest"]

        # our rounding trick here
        pump_time = _pump_time(
            row,
            cfg.timezone,
            interval=cfg.interval,
            do_round=cfg.round_pump_time,
        )

        start_ms = int((pump_time - pd.Timedelta(days=cfg.days_before)).timestamp() * 1000)
        end_ms = int((pump_time + pd.Timedelta(days=cfg.days_after)).timestamp() * 1000)

        print(f"Fetching {symbol} around {pump_time} ...")

        try:
            kdf = fetch_klines(symbol, cfg.interval, start_ms, end_ms, limit=1000)
            if kdf.empty:
                continue

            kdf["symbol"] = symbol
            kdf["date"] = pd.to_datetime(kdf["open_time"])

            # flag on rounded pump time
            kdf["flag"] = np.where(kdf["date"] == pump_time, 1, 0)

            keep = [
                "date", "symbol",
                # pricing related
                "open", "high", "low", "close",
                # volume related
                "volume", "quote_asset_volume", "num_trades",
                "taker_buy_base", "taker_buy_quote",
                # pump
                "flag",
            ]
            kdf = kdf[keep].copy()

            # keep optional metadata columns from pump CSV
            if "group" in row.index:
                kdf["group"] = row["group"]

            frames.append(kdf)

        except Exception as e:
            failures += 1
            print(f"[WARN] Failed {symbol}: {e}")

    if not frames:
        base_cols = [
            "date", "symbol",
            "open", "high", "low", "close", 
            "volume", "quote_asset_volume", "num_trades",
            "taker_buy_base", "taker_buy_quote",
            "flag",
        ]
        return pd.DataFrame(columns=base_cols)

    panel = (
        pd.concat(frames, ignore_index=True)
        .sort_values(["date", "symbol"])
        .reset_index(drop=True)
    )

    if failures:
        print(f"[INFO] Completed with {failures} failures.")
    return panel
