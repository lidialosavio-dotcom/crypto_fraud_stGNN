# Fraud Detection in Cryptocurrency Markets with Spatio-Temporal Graph Neural Networks --- SDS2026 Conference Paper

This repository contains the code used to build the datasets and reproduce the experiments for the paper:
"Fraud Detection in Cryptocurrency Markets with Spatio-Temporal Graph Neural Networks".

## Repository structure

```text
PUMP-AND-DUMP-PROJECT/
├── data/
├── outputs/
├── src/
│   ├── baselines/
│   │   ├── models.py
│   │   ├── pipeline_baselines.py
│   │   └── pipeline_utils.py
│   ├── pumpdownloader/
│   └── temp-only-baselines/
│       ├── GRUbaseline.py
│       └── TransformerOnly_baseline.py
├── tests/
├── dynamic_numTrades_corr_solution.py
├── dynamic_volume_corr_solution.py
├── node_embeddings_solution.py
├── static_numTrades_corr_solution.py
├── static_volume_corr_solution.py
├── environment.yml
└── README.md
```

## Dataset Pipeline

This section documents how we build the **panel datasets** used for pump-and-dump detection: starting from `pump_telegram.csv`, we use Binance to download aggregated market data at hourly frequency, flagging pump hours, and optionally adding rolling engineered features.

### Input

* `pump_telegram.csv` (minimum columns used): `symbol`, `exchange`, `date` (`YYYY-mm-dd`), `hour` (`HH:MM`)

  * Each row of this file contains:

    * `symbol`: the symbol (`SYM`) of the pumped coin
    * `group`: the code of the group that arranged the pump and dump. More information about groups can be found in `group.csv`
    * `date`: the pump-and-dump date
    * `hour`: the pump-and-dump hour in UTC
    * `exchange`: the exchange targeted by the group
  * All pump-and-dump events in the dataset are on the trading pair `SYM/BTC`
* Only rows where `exchange` matches `--exchange-filter` (default: `binance`) are used
* Binance REST symbol is built as `symbol + quote` (we specifically use quote `BTC`, e.g. `ETHBTC`)

### Time handling

* Pump timestamps are parsed from `date` + `hour`, localized to `--timezone`. We keep `UTC`
* By default, pump times are rounded to the nearest candle boundary (`--interval`, e.g. `1h`). Disable with `--no-round`
* Binance candle times are UTC ms; we convert to `--timezone` before matching and flagging

### What we build

We output **8 panels** = 2 (GLOBAL vs WINDOWED) × 2 (multi-pump vs single-pump) × 2 (base vs engineered)

**Token split**

* multi-pump: tokens with `>= 2` pump events
* single-pump: tokens with `== 1` pump event

**GLOBAL panels**

* One window per token using dataset-wide bounds:
  `[min(pump_time)-days_before, max(pump_time)+days_after]`
* Pump label: `flag = 1` if candle open time equals one of the token’s pump times

**WINDOWED panels**

* Per pump event time `t`, download `[t-window_days, t+window_days]`, then stack chunks
* Rolling features (engineered) are computed **per chunk**, to avoid mixing across pump windows

**Base schema**

```text
date, symbol, open, high, low, close, volume, quote_asset_volume, num_trades, taker_buy_base, taker_buy_quote, flag
```

### Engineered features

We add pct-change of rolling mean/std features, including:

* `buy_pressure = taker_buy_quote / volume` (`volume=0 -> NaN`)
* rolling/pct-change features on buy pressure, trades, volume, price (e.g. `std_rush_order`, `avg_volume`, `std_price`, ...)

### How to run the dataset builder

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

## Environment setup

We provide an `environment.yml` file for reproducibility.

### Create the environment

```bash
conda env create -f environment.yml
conda activate pumpdump-stgnn
```

### Update the environment

```bash
conda env update -n pumpdump-stgnn -f environment.yml --prune
conda activate pumpdump-stgnn
```

### Basic environment checks

```bash
python -c "import sys; print(sys.version)"
python -c "import torch; print('torch:', torch.__version__)"
python -c "import torch_geometric; print('pyg:', torch_geometric.__version__)"
python -c "import xgboost; print('xgboost:', xgboost.__version__)"
python -c "from sklearn.ensemble import RandomForestClassifier; print('RandomForest OK')"
```

### Smoke test

```bash
python - <<'PY'
import torch
import torch_geometric
import xgboost
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.nn import TransformerConv

print("torch:", torch.__version__)
print("pyg:", torch_geometric.__version__)
print("cuda available:", torch.cuda.is_available())
print("xgboost:", xgboost.__version__)
print("RandomForest OK:", RandomForestClassifier())

x = torch.randn(4, 8)
edge_index = torch.tensor([[0,1,2,3],[1,2,3,0]], dtype=torch.long)
edge_attr = torch.randn(4, 1)
conv = TransformerConv(8, 16, heads=2, edge_dim=1)
out = conv(x, edge_index, edge_attr)
print("TransformerConv OK:", out.shape)

print("Environment OK")
PY
```

## Data placement before running experiments

After downloading or generating the panel dataset, place it in the appropriate local directory.

Most experiment scripts expect a local parquet or CSV file and use a hardcoded `file_path`. Before running, you must update the local path to match your machine.

### Important path changes

* For all model scripts **outside the tree baselines**, update the `file_path` variable inside each `.py` file to your local dataset path
* For the tree baselines, update the data path in:

  * `src/baselines/pipeline_baselines.py`
  * specifically around **line 142**

## Reproducing the paper experiments

The paper experiments are run with **9 seeds**:

```text
11, 22, 33, 44, 55, 66, 77, 88, 99
```

The updated experiment scripts already run the full multi-seed evaluation and save:

* one folder per seed
* raw prediction arrays
* ROC/PR curves
* summary metrics
* aggregated CSV files across seeds
* per-token metrics

### Tree baselines

Tree baselines are run inside:

```text
src/baselines/pipeline_baselines.py
```

This script runs:

* **XGBoost**
* **Random Forest**

in sequence.

Before running it, update the dataset path inside `pipeline_baselines.py` (around line 142).

Run:

```bash
python src/baselines/pipeline_baselines.py
```

### Temp-only baselines

The temporal-only baselines are in:

```text
src/temp-only-baselines/GRUbaseline.py
src/temp-only-baselines/TransformerOnly_baseline.py
```

These are the time-only ablations:

* **GRU time-only**
* **TransformerEncoder time-only**

Before running them, update the local `file_path` inside each script.

Run:

```bash
python src/temp-only-baselines/GRUbaseline.py
python src/temp-only-baselines/TransformerOnly_baseline.py
```

### Graph-based models

The graph-based models used in the paper are:

```text
dynamic_numTrades_corr_solution.py
dynamic_volume_corr_solution.py
node_embeddings_solution.py
static_numTrades_corr_solution.py
static_volume_corr_solution.py
```

Before running them, update the local `file_path` inside each script.

Run:

```bash
python dynamic_numTrades_corr_solution.py
python dynamic_volume_corr_solution.py
python node_embeddings_solution.py
python static_numTrades_corr_solution.py
python static_volume_corr_solution.py
```

## Expected outputs

Each multi-seed experiment script saves:

### Per-seed outputs

Inside a folder like:

```text
metrics_outputs_.../seed_011/
metrics_outputs_.../seed_022/
...
metrics_outputs_.../seed_099/
```

you will find:

* `test_probs.npy`
* `test_true.npy`
* `roc_fpr.npy`
* `roc_tpr.npy`
* `roc_thresholds.npy`
* `pr_precision.npy`
* `pr_recall.npy`
* `pr_thresholds.npy`
* `summary_metrics.txt`
* `roc_curve.png`
* `pr_curve.png`

### Aggregated outputs

At the script root level, each experiment also saves CSV files such as:

* full results for all seeds
* aggregated summary over seeds
* per-token metrics

## Recommended execution order

A typical full reproduction workflow is:

1. Create and activate the conda environment
2. Download the source pump dataset
3. Build the panel datasets with `pumpdownloader`
4. Update local dataset paths inside the experiment scripts
5. Run tree baselines:

   * `src/baselines/pipeline_baselines.py`
6. Run temporal-only baselines:

   * `src/temp-only-baselines/GRUbaseline.py`
   * `src/temp-only-baselines/TransformerOnly_baseline.py`
7. Run graph-based models:

   * `dynamic_numTrades_corr_solution.py`
   * `dynamic_volume_corr_solution.py`
   * `node_embeddings_solution.py`
   * `static_numTrades_corr_solution.py`
   * `static_volume_corr_solution.py`

## Notes on reproducibility

* All main experiments are evaluated over the same 9 seeds:
  `11, 22, 33, 44, 55, 66, 77, 88, 99`
* Temporal splits use:

  * `60%` train
  * `20%` validation
  * `20%` test
  * embargo of `5`
* Threshold tuning is performed on the **validation set**
* The best checkpoint is selected using validation F1
* Final metrics are computed on the test set using the best validation threshold

## Reference

Pump event source: SystemsLab Sapienza pump-and-dump dataset
[https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset](https://github.com/SystemsLab-Sapienza/pump-and-dump-dataset)
