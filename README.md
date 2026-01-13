# Pump & Dump Detection Paper

## Dataset Pipeline
This section documents how we build the **panel datasets** used for pump-and-dump detection: starting from `pump_telegram.csv`, we use Binance to download aggregated market data at hourly frequency, flagging pump hours, and (optionally) adding rolling engineered features.

### Input
- `pump_telegram.csv` (minimum columns used): `symbol`, `exchange`, `date` (`YYYY-mm-dd`), `hour` (`HH:MM`)
- Only rows where `exchange` matches `--exchange-filter` (default: `binance`) are used.
- Binance REST symbol is built as `symbol + quote` (we specifically only use quote: `BTC`, e.g. `ETHBTC`).

### Time handling
- Pump timestamps are parsed from `date` + `hour`, localized to `--timezone`. We keep `UTC`.
- By default pump times are rounded to the nearest candle boundary (`--interval`, e.g. `1h`). Disable with `--no-round`.
- Binance candle times are UTC ms; we convert to `--timezone` before matching/flagging.

### What we build
We output **8 panels** = 2 (GLOBAL vs WINDOWED) × 2 (multi-pump vs single-pump) × 2 (base vs engineered).

**Token split**
- multi-pump: tokens with `>= 2` pump events  
- single-pump: tokens with `== 1` pump event

**GLOBAL panels**
- One window per token using dataset-wide bounds:  
  `[min(pump_time)-days_before, max(pump_time)+days_after]`
- Pump label: `flag = 1` if candle open time equals one of the token’s pump times.

**WINDOWED panels**
- Per pump event time `t`, download `[t-window_days, t+window_days]`, then stack chunks.
- Rolling features (engineered) are computed **per chunk**, to avoid mixing across pump windows.

**Base schema**
`date, symbol, open, high, low, close, volume, quote_asset_volume, num_trades, taker_buy_base, taker_buy_quote, flag`

### Engineered features
We add pct-change of rolling mean/std features, including:
- `buy_pressure = taker_buy_quote / volume` (volume=0 → NaN)
- rolling/pct-change features on buy pressure, trades, volume, price (e.g., `std_rush_order`, `avg_volume`, `std_price`, …)

### How to run
Build all panels by providing the outputs you want:

```bash
python -m pumpdownloader.cli \
  --input data/pump_telegram.csv \
  --out-base outputs/panel_base.csv \
  --out-engineered outputs/panel_engineered.csv \
  --out-single-base outputs/single_pump_base.csv \
  --out-single-engineered outputs/single_pump_engineered.csv \
  --out-base-w3 outputs/panel_base_w3.csv \
  --out-engineered-w3 outputs/panel_engineered_w3.csv \
  --out-single-base-w3 outputs/single_pump_base_w3.csv \
  --out-single-engineered-w3 outputs/single_pump_engineered_w3.csv \
  --interval 1h --quote BTC \
  --days-before 7 --days-after 7 \
  --window-days 3 \
  --exchange-filter binance --timezone UTC
```

### Reference
Pump event source: SystemsLab Sapienza pump-and-dump dataset
https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset
