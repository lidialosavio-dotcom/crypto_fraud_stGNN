"""
Created on Fri Dec 12 17:15:01 2025

This is a downloader for the report "Detecting Pump-and-Dump Schemes in Crypto
Markets". The goal is to fetch market information for tickers 
involved in a Pump&Dump scheme using the Binance API. The tickers are reported 
in the file pump_telegram.csv collected by the System Lab Sapienza.

Reference: 'https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset/tree/master'

Author: Luca Persia (USI/ZHAW)

Note: the downaloader is currently handling ONLY /BTC currency pair.
"""

import time
import requests
import pandas as pd

BASE = "https://api.binance.com"  # spot


def fetch_klines(
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int = 1000,
    *,
    sleep_s: float = 0.1,
    backoff_s: float = 15.0,
    timeout_s: float = 30.0,
) -> pd.DataFrame:
    
    """
    Fetch historical OHLCV candles from BINANCE Spot REST API.

    Input
    -----
    symbol : str
        Binance REST symbol (NO slash), e.g. 'ETHBTC', 'BTCUSDT'.
    interval : str
        Candle frequency supported by Binance, e.g. '1m', '5m', '1h', '1d'.
    start_ms : int
        Start time in milliseconds since epoch (UTC).
    end_ms : int
        End time in milliseconds since epoch (UTC).
    limit : int, optional
        Max candles per API call (Binance default max is 1000).
    sleep_s : float, optional
        Sleep between requests (politeness / avoid throttling).
    backoff_s : float, optional
        Sleep time when rate-limited (HTTP 429).
    timeout_s : float, optional
        HTTP request timeout in seconds.

    Output
    ------
    df : pd.DataFrame
        Candle-level market data with columns:
        - open_time (UTC timestamp)
        - open, high, low, close (float)
        - volume (float)
        - close_time (UTC timestamp)
        - quote_asset_volume (float)
        - num_trades (float)
        - taker_buy_base (float)
        - taker_buy_quote (float)
        - ignore
        If no data is available in the requested window, returns an empty DataFrame.
    """
    
    url = f"{BASE}/api/v3/klines"
    out = []
    since = start_ms

    while since < end_ms:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": since,
            "endTime": end_ms,
            "limit": limit,
        }
        r = requests.get(url, params=params, timeout=timeout_s)
        if r.status_code == 429:
            time.sleep(backoff_s)
            continue
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        since = batch[-1][6] + 1  # last close_time + 1 ms
        time.sleep(sleep_s)

    # Note: aggregated information
    cols = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "num_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ]
    
    df = pd.DataFrame(out, columns=cols)
    
    if df.empty:
        return df

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    num_cols = [
        "open", "high", "low", "close", "volume", "num_trades",
        "quote_asset_volume", "taker_buy_base", "taker_buy_quote",
    ]
    df[num_cols] = df[num_cols].astype(float)
    
    return df
