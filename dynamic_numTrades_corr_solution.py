import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn.functional as F
import random
from torch.nn import Linear
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import copy
import os

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve,
    roc_auc_score, roc_curve, auc,
    average_precision_score
)

# Headless plotting + saving
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
    """
    Return the number of model parameters.
    - trainable_only=True counts only parameters with requires_grad=True
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def print_parameter_summary(model: torch.nn.Module, top_k: int = 50) -> None:
    """
    Print a summary of total, trainable, and frozen parameters.
    Also print the first top_k parameter tensors by size.
    """
    total = count_parameters(model, trainable_only=False)
    trainable = count_parameters(model, trainable_only=True)
    frozen = total - trainable

    print("\n--- Parameter Count ---")
    print(f"Total params:      {total:,}")
    print(f"Trainable params:  {trainable:,}")
    print(f"Frozen params:     {frozen:,}")

    named = [(name, p.numel(), p.requires_grad) for name, p in model.named_parameters()]
    named.sort(key=lambda x: x[1], reverse=True)

    print(f"\nTop-{min(top_k, len(named))} parameter tensors by size:")
    for name, n, req in named[:top_k]:
        flag = "trainable" if req else "frozen"
        print(f"  {name:50s} {n:12,d}  ({flag})")


# ----------------------------
# 1) SEED
# ----------------------------
def set_seed(seed=44):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed fixed to {seed}.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99]


# ----------------------------
# 2) CONFIG
# ----------------------------
SPLIT_TRAIN = 0.60
SPLIT_VAL   = 0.20
SPLIT_TEST  = 0.20

EMBARGO_STEPS = 5
WINDOW_SIZE = 5

MIN_CORR_THRESHOLD = 0.15
TARGET_DENSITY = 0.95

# ----------------------------
# DYNAMIC GRAPH CONFIG
# ----------------------------
PUMP_WINDOW_HOURS = 12
CORR_FEATURE = "num_trades"
CORR_METHOD = "pearson"

UPDATE_EXISTING_WITH_LOW_CORR = True
USE_ABS_FOR_NEW_EDGES = False

HIDDEN_CHANNELS = 64
HEADS = 2
LEARNING_RATE = 0.001
EPOCHS = 50
DROPOUT = 0.3
BATCH_SIZE = 64

# Not used anymore as the main threshold; kept as fallback
FIXED_THRESHOLD = 0.5

DATE_COL   = "date"
LABEL_COL  = "flag"
GROUP_COL  = "symbol"

DROP_COLS_TRAIN = [
    DATE_COL, LABEL_COL, GROUP_COL,
    "group",
    "log_ret", "ret_BTC", "vola_BTC", "buy_pressure"
]

# ----------------------------
# OUTPUT DIR
# ----------------------------
OUTPUT_ROOT = "metrics_outputs_timing"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

GRID_RESULTS_FILE = "grid_results_dynamic_num_trades_timing.csv"
GRID_SUMMARY_FILE = "grid_summary_dynamic_num_trades_timing.csv"
GRID_PER_TOKEN_FILE = "grid_per_token_results_dynamic_num_trades_timing.csv"


# ----------------------------
# 3) LOAD DATA 
# ----------------------------
file_path = r"/home/lidialosav/pump-and-dump-dataset/project/panel_engineered_full.opt.parquet"

print("\n--- Loading Data ---")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Path not found: {file_path}")

df = pd.read_parquet(file_path)
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.sort_values(DATE_COL).reset_index(drop=True)
df[LABEL_COL] = df[LABEL_COL].astype(int).clip(0, 1)

unique_symbols = df[GROUP_COL].unique()
symbol_to_idx = {sym: i for i, sym in enumerate(unique_symbols)}
idx_to_symbol = {i: sym for sym, i in symbol_to_idx.items()}
num_nodes = len(unique_symbols)
print(f"Dataset: {len(df)} rows, {num_nodes} unique tokens.")

# ----------------------------
# 4) FEATURES (RAW)
# ----------------------------
print("\n--- Feature Analysis ---")
raw_feature_cols = [
    c for c in df.columns
    if c not in DROP_COLS_TRAIN and np.issubdtype(df[c].dtype, np.number)
]
print(f"Selected Features ({len(raw_feature_cols)}): {raw_feature_cols}")

df_proc = df.copy()

# ----------------------------
# 5) TEMPORAL SPLIT
# ----------------------------
unique_dates = df[DATE_COL].unique()
n_dates = len(unique_dates)
train_end_idx = int(SPLIT_TRAIN * n_dates)
val_end_idx   = int((SPLIT_TRAIN + SPLIT_VAL) * n_dates)

dates_train = unique_dates[:train_end_idx]
dates_val   = unique_dates[train_end_idx + EMBARGO_STEPS : val_end_idx]
dates_test  = unique_dates[val_end_idx + EMBARGO_STEPS :]

print(f"Train: {len(dates_train)} h | Val: {len(dates_val)} h | Test: {len(dates_test)} h")

# ----------------------------
# 6) DYNAMIC GRAPH (correlation updates on windows around pump events in TRAIN)
# ----------------------------
print()
print("--- Building DYNAMIC Graph (Correlation on pump windows) ---")

base_edge_index = torch.tensor([list(range(num_nodes)), list(range(num_nodes))], dtype=torch.long)
base_edge_attr  = torch.ones(num_nodes, dtype=torch.float)

df_train = df[df[DATE_COL].isin(dates_train)].copy()

if CORR_FEATURE not in df.columns:
    raise ValueError(f"Column '{CORR_FEATURE}' not found in dataframe.")

train_dates_idx = pd.DatetimeIndex(dates_train).sort_values()

pump_dates = (
    df_train.loc[df_train[LABEL_COL] == 1, DATE_COL]
    .dropna()
    .unique()
)
pump_dates = pd.DatetimeIndex(pump_dates).sort_values()

print(f"Pump events in TRAIN: {len(pump_dates)} (unique timestamps)")


def _window_dates_around_pump(pump_date, train_dates_index, window_h=PUMP_WINDOW_HOURS):
    pos = train_dates_index.searchsorted(pump_date)
    if pos == len(train_dates_index) or train_dates_index[pos] != pump_date:
        pos = max(0, pos - 1)
    left = max(0, pos - window_h)
    right = pos
    return train_dates_index[left:right+1]


def _corr_matrix_on_dates(df_full, dates_window, feature_col, method=CORR_METHOD, symbols_order=None):
    sub = df_full[df_full[DATE_COL].isin(dates_window)].copy()
    pivot = sub.pivot_table(index=DATE_COL, columns=GROUP_COL, values=feature_col).fillna(0)
    if symbols_order is not None:
        pivot = pivot.reindex(columns=symbols_order, fill_value=0)
    pivot_log = np.log1p(pivot)
    corr = pivot_log.corr(method=method).fillna(0)
    np.fill_diagonal(corr.values, 0)
    return corr.values


# def _effective_threshold(corr_vals, target_density=TARGET_DENSITY, min_thr=MIN_CORR_THRESHOLD):
#     flat = corr_vals.reshape(-1)
#     perc_thr = np.percentile(flat, target_density * 100)
#     return float(max(min_thr, perc_thr))


def _effective_threshold(corr_vals, target_density=TARGET_DENSITY, min_thr=MIN_CORR_THRESHOLD):
    # Paper: Q_rho over off-diagonal correlations {C_ij}_{i<j}
    iu = np.triu_indices_from(corr_vals, k=1)  # strictly upper triangular (i<j)
    offdiag = corr_vals[iu]

    # Optional but safe: if offdiag is empty for any reason
    if offdiag.size == 0:
        perc_thr = 0.0
    else:
        perc_thr = np.percentile(offdiag, target_density * 100)

    return float(max(min_thr, perc_thr))


def _edge_stats_to_tensors(edge_stats_dict, num_nodes):
    if len(edge_stats_dict) == 0:
        return base_edge_index, base_edge_attr

    sources, targets, weights = [], [], []
    for (i, j), (s, c) in edge_stats_dict.items():
        mean_w = float(s) / float(c)
        sources.extend([i, j])
        targets.extend([j, i])
        weights.extend([mean_w, mean_w])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr  = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_attr


if len(pump_dates) == 0:
    print("WARNING: no pump in TRAIN. Falling back to a static graph built on the full train period.")

    df_structure = df[df[DATE_COL].isin(dates_train)].copy()
    vol_pivot = df_structure.pivot_table(index=DATE_COL, columns=GROUP_COL, values=CORR_FEATURE).fillna(0)
    vol_pivot = vol_pivot.reindex(columns=unique_symbols, fill_value=0)
    vol_log = np.log1p(vol_pivot)

    corr_matrix = vol_log.corr(method=CORR_METHOD).fillna(0)
    np.fill_diagonal(corr_matrix.values, 0)

    percentile_thresh = np.percentile(corr_matrix.values.flatten(), TARGET_DENSITY * 100)
    effective_thresh = max(MIN_CORR_THRESHOLD, percentile_thresh)

    sources, targets, weights = [], [], []
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            val = corr_matrix.iloc[i, j]
            if val > effective_thresh:
                sources.extend([i, j])
                targets.extend([j, i])
                weights.extend([val, val])

    if not sources:
        final_edge_index, final_edge_attr = base_edge_index, base_edge_attr
    else:
        final_edge_index = torch.tensor([sources, targets], dtype=torch.long)
        final_edge_attr  = torch.tensor(weights, dtype=torch.float)

    dynamic_graph_updates = {}
else:
    edge_stats = {}
    dynamic_graph_updates = {}
    symbols_order = unique_symbols

    for k, pdate in enumerate(pump_dates):
        win_dates = _window_dates_around_pump(pdate, train_dates_idx, window_h=PUMP_WINDOW_HOURS)

        corr_vals = _corr_matrix_on_dates(
            df_full=df,
            dates_window=win_dates,
            feature_col=CORR_FEATURE,
            method=CORR_METHOD,
            symbols_order=symbols_order
        )

        thr = _effective_threshold(corr_vals, target_density=TARGET_DENSITY, min_thr=MIN_CORR_THRESHOLD)

        if UPDATE_EXISTING_WITH_LOW_CORR and len(edge_stats) > 0:
            for (i, j), sc in edge_stats.items():
                s, c = sc
                v = float(corr_vals[i, j])
                edge_stats[(i, j)][0] = s + v
                edge_stats[(i, j)][1] = c + 1

        mat = corr_vals
        if USE_ABS_FOR_NEW_EDGES:
            sel = np.abs(mat) > thr
        else:
            sel = mat > thr

        iu = np.triu_indices(num_nodes, k=1)
        cand_mask = sel[iu]
        cand_i = iu[0][cand_mask]
        cand_j = iu[1][cand_mask]

        new_added = 0
        for i, j in zip(cand_i, cand_j):
            key = (int(i), int(j))
            if key not in edge_stats:
                edge_stats[key] = [float(mat[i, j]), 1]
                new_added += 1
            else:
                if not UPDATE_EXISTING_WITH_LOW_CORR:
                    edge_stats[key][0] += float(mat[i, j])
                    edge_stats[key][1] += 1

        curr_edge_index, curr_edge_attr = _edge_stats_to_tensors(edge_stats, num_nodes)
        dynamic_graph_updates[pdate] = (curr_edge_index, curr_edge_attr)

        print(f"[{k+1}/{len(pump_dates)}] pump @ {pdate} | "
              f"win={len(win_dates)}h | thr={thr:.4f} | "
              f"new_edges={new_added} | total_undirected_edges={len(edge_stats)}")

    last_pump = pump_dates[-1]
    final_edge_index, final_edge_attr = dynamic_graph_updates[last_pump]

print(f"Final graph edges (directed): {final_edge_index.size(1)}")

# ----------------------------
# 7) SCALING + WINDOWING (FIX float64 issue: CLEAN + CLIP as in the previous code)
# ----------------------------
print("\n--- Scaling & Temporal Window Construction ---")

df_train_subset = df_proc[df_proc[DATE_COL].isin(dates_train)].copy()

# 1) Force numeric conversion
for c in raw_feature_cols:
    df_train_subset[c] = pd.to_numeric(df_train_subset[c], errors="coerce")

# 2) Replace inf/-inf with NaN
df_train_subset[raw_feature_cols] = df_train_subset[raw_feature_cols].replace([np.inf, -np.inf], np.nan)

# 3) Clip outliers (train only)
lower = df_train_subset[raw_feature_cols].quantile(0.0005)
upper = df_train_subset[raw_feature_cols].quantile(0.9995)
df_train_subset[raw_feature_cols] = df_train_subset[raw_feature_cols].clip(lower=lower, upper=upper, axis=1)

# 4) Fit scaler on clean float64 data
scaler = StandardScaler()
scaler.fit(df_train_subset[raw_feature_cols].fillna(0).astype(np.float64))
print("Scaler fit OK (after inf replacement + outlier clipping).")


def get_temporal_snapshots(target_dates, window_size=WINDOW_SIZE, graph_mode="dynamic"):
    snapshots = []
    all_dates_list = df_proc[DATE_COL].unique()

    min_date_idx = np.searchsorted(all_dates_list, target_dates[0])
    relevant_start_idx = max(0, min_date_idx - window_size + 1)
    relevant_dates = all_dates_list[
        relevant_start_idx : np.searchsorted(all_dates_list, target_dates[-1]) + 1
    ]

    subset = df_proc[df_proc[DATE_COL].isin(relevant_dates)].copy()

    # Same preprocessing as train: force numeric + replace inf + clip (train thresholds) + scaler
    for c in raw_feature_cols:
        subset[c] = pd.to_numeric(subset[c], errors="coerce")
    subset[raw_feature_cols] = subset[raw_feature_cols].replace([np.inf, -np.inf], np.nan)
    subset[raw_feature_cols] = subset[raw_feature_cols].clip(lower=lower, upper=upper, axis=1)
    subset[raw_feature_cols] = scaler.transform(subset[raw_feature_cols].fillna(0).astype(np.float64))

    grouped = subset.groupby(DATE_COL)

    date_to_matrix = {}
    for date, group in grouped:
        mat = np.zeros((num_nodes, len(raw_feature_cols)), dtype=np.float32)
        indices = [symbol_to_idx[s] for s in group[GROUP_COL]]
        mat[indices] = group[raw_feature_cols].values.astype(np.float32)
        date_to_matrix[date] = mat

    pump_dates_sorted = pd.DatetimeIndex(list(dynamic_graph_updates.keys())).sort_values()

    def graph_for_date(date):
        if len(pump_dates_sorted) == 0:
            return final_edge_index, final_edge_attr
        pos = pump_dates_sorted.searchsorted(date, side="right") - 1
        if pos < 0:
            return base_edge_index, base_edge_attr
        p = pump_dates_sorted[pos]
        return dynamic_graph_updates[p]

    for date in tqdm(target_dates, desc="Generating Windows"):
        curr_idx = np.where(all_dates_list == date)[0][0]

        window_matrices = []
        for w in range(window_size):
            lookback_idx = curr_idx - (window_size - 1) + w
            if lookback_idx >= 0:
                d = all_dates_list[lookback_idx]
                window_matrices.append(date_to_matrix.get(
                    d, np.zeros((num_nodes, len(raw_feature_cols)), dtype=np.float32)
                ))
            else:
                window_matrices.append(np.zeros((num_nodes, len(raw_feature_cols)), dtype=np.float32))

        x_seq = np.stack(window_matrices, axis=1)  # [N, W, F]

        current_group = df[df[DATE_COL] == date]
        y_t = np.zeros(num_nodes, dtype=np.float32)
        mask_t = np.zeros(num_nodes, dtype=np.bool_)

        indices = [symbol_to_idx[s] for s in current_group[GROUP_COL]]
        y_t[indices] = current_group[LABEL_COL].values.astype(np.float32)
        mask_t[indices] = True

        if graph_mode == "dynamic":
            eidx, eattr = graph_for_date(date)
        else:
            eidx, eattr = final_edge_index, final_edge_attr

        data = Data(
            x=torch.tensor(x_seq, dtype=torch.float),
            y=torch.tensor(y_t, dtype=torch.float),
            mask=torch.tensor(mask_t, dtype=torch.bool),
            edge_index=eidx,
            edge_attr=eattr
        )
        snapshots.append(data)

    return snapshots


train_snapshots = get_temporal_snapshots(dates_train, graph_mode="dynamic")
val_snapshots   = get_temporal_snapshots(dates_val,   graph_mode="final")
test_snapshots  = get_temporal_snapshots(dates_test,  graph_mode="final")


def make_loaders(seed: int):
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = DataLoader(train_snapshots, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    val_loader   = DataLoader(val_snapshots,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_snapshots,  batch_size=BATCH_SIZE, shuffle=False)
    return train_loader, val_loader, test_loader


# ----------------------------
# 8) MODEL (GNN + Temporal Transformer)
# ----------------------------
class TemporalGraphModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, heads=HEADS, window_size=WINDOW_SIZE):
        super().__init__()
        self.lin_in = Linear(num_features, hidden_channels)
        self.conv1 = TransformerConv(hidden_channels, hidden_channels, heads=heads, edge_dim=1)
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=1)

        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=8,
            dim_feedforward=128,
            dropout=DROPOUT,
            batch_first=True
        )
        self.temporal_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.out = Linear(hidden_channels, 1)
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, window_size, hidden_channels))

    def forward(self, x, edge_index, edge_weight):
        batch_nodes, window, feats = x.size()
        edge_attr = edge_weight.view(-1, 1)

        temporal_embeddings = []
        for t in range(window):
            x_t = self.lin_in(x[:, t, :])
            h = self.conv1(x_t, edge_index, edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=DROPOUT, training=self.training)
            h = self.conv2(h, edge_index, edge_attr)
            h = F.relu(h)
            temporal_embeddings.append(h)

        sequence = torch.stack(temporal_embeddings, dim=1)
        sequence = sequence + self.pos_embedding
        time_out = self.temporal_transformer(sequence)
        last_step = time_out[:, -1, :]
        out = self.out(last_step)
        return torch.sigmoid(out).view(-1)


PER_TOKEN_ROWS = []


# ----------------------------
# 9) SINGLE RUN
# ----------------------------
def run_one(seed: int):
    set_seed(seed)

    OUTPUT_DIR = os.path.join(OUTPUT_ROOT, f"seed_{seed:03d}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_loader, val_loader, test_loader = make_loaders(seed)

    model = TemporalGraphModel(len(raw_feature_cols), HIDDEN_CHANNELS).to(device)
    print_parameter_summary(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = torch.nn.BCELoss()

    @torch.no_grad()
    def collect_probs_and_true(loader):
        model.eval()
        probs_list, true_list = [], []
        for batch in loader:
            batch = batch.to(device)
            probs = model(batch.x, batch.edge_index, batch.edge_attr)
            probs_list.append(probs[batch.mask].detach().cpu().numpy())
            true_list.append(batch.y[batch.mask].detach().cpu().numpy())
        probs = np.concatenate(probs_list, axis=0) if probs_list else np.array([])
        true  = np.concatenate(true_list, axis=0)  if true_list else np.array([])
        return probs, true

    @torch.no_grad()
    def evaluate_loss_f1(loader, thresh=0.5):
        model.eval()
        total_loss = 0.0
        all_preds, all_true = [], []

        for batch in loader:
            batch = batch.to(device)
            probs = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(probs[batch.mask], batch.y[batch.mask])
            total_loss += float(loss.item())

            preds = (probs >= thresh).float()
            all_preds.extend(preds[batch.mask].cpu().numpy())
            all_true.extend(batch.y[batch.mask].cpu().numpy())

        avg_loss = total_loss / max(1, len(loader))
        f1 = f1_score(all_true, all_preds, zero_division=0) if len(all_true) else 0.0
        return avg_loss, f1

    @torch.no_grad()
    def tune_threshold_on_validation(val_loader):
        """
        Threshold tuning on VALIDATION (precision_recall_curve) -> maximize F1.
        Returns (best_threshold, best_f1).
        """
        y_prob, y_true = collect_probs_and_true(val_loader)
        if len(y_true) == 0:
            return FIXED_THRESHOLD, 0.0

        prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-12)

        best_idx = int(np.argmax(f1s))
        if best_idx >= len(thresholds):
            best_thresh = FIXED_THRESHOLD
        else:
            best_thresh = float(thresholds[best_idx])

        best_f1 = float(np.max(f1s))
        return best_thresh, best_f1

    print(f"\n--- Start Training (Dynamic Graph) ({EPOCHS} epochs) | Seed {seed} ---")

    best_val_f1 = -1.0
    best_model_state = None
    best_val_threshold = FIXED_THRESHOLD

    pbar = tqdm(range(EPOCHS), desc=f"Training seed {seed}", unit="epoch", leave=False)
    for epoch in pbar:
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            probs = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(probs[batch.mask], batch.y[batch.mask])

            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        avg_train_loss = train_loss / max(1, len(train_loader))

        # Tune threshold on VALIDATION (no leakage)
        tuned_t, tuned_val_f1 = tune_threshold_on_validation(val_loader)

        # Logging: val loss and test f1 with tuned_t (TEST does NOT influence selection)
        val_loss, _ = evaluate_loss_f1(val_loader, thresh=tuned_t)
        _, test_f1 = evaluate_loss_f1(test_loader, thresh=tuned_t)

        if tuned_val_f1 > best_val_f1:
            best_val_f1 = tuned_val_f1
            best_model_state = copy.deepcopy(model.state_dict())
            best_val_threshold = tuned_t

        pbar.set_postfix({
            "Ltr": f"{avg_train_loss:.3f}",
            "ValLoss": f"{val_loss:.3f}",
            "ValF1@t*": f"{tuned_val_f1:.3f}",
            "t*": f"{tuned_t:.3f}",
            "TstF1@t*": f"{test_f1:.3f}",
            "BestVal": f"{best_val_f1:.3f}"
        })

    # ----------------------------
    # FINAL TEST (best checkpoint + best threshold from VALIDATION)
    # ----------------------------
    print(f"\n\n--- Final Results (Dynamic Graph) | Seed {seed} ---")
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_threshold = best_val_threshold

    test_probs, test_true = collect_probs_and_true(test_loader)
    final_preds = (test_probs >= final_threshold).astype(int)

    test_precision = precision_score(test_true, final_preds, zero_division=0)
    test_recall = recall_score(test_true, final_preds, zero_division=0)
    test_f1 = f1_score(test_true, final_preds, zero_division=0)

    print(f"Best threshold from VALIDATION (best checkpoint): {final_threshold:.3f}")
    print(f"Precision: {test_precision:.4f}")
    print(f"Recall:    {test_recall:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(test_true, final_preds))

    # ----------------------------
    # ROC-AUC + PR-AUC + SAVE CURVES
    # ----------------------------
    fpr, tpr, roc_thresholds = roc_curve(test_true, test_probs)
    roc_auc_val = auc(fpr, tpr)  # equivalent to roc_auc_score for binary
    pr_prec, pr_rec, pr_thresholds = precision_recall_curve(test_true, test_probs)
    pr_auc_val = average_precision_score(test_true, test_probs)

    print(f"ROC-AUC:   {roc_auc_val:.4f}")
    print(f"PR-AUC(AP): {pr_auc_val:.4f}")

    # Save raw arrays to allow recomputing/plotting later
    np.save(os.path.join(OUTPUT_DIR, "test_probs.npy"), test_probs)
    np.save(os.path.join(OUTPUT_DIR, "test_true.npy"), test_true)

    np.save(os.path.join(OUTPUT_DIR, "roc_fpr.npy"), fpr)
    np.save(os.path.join(OUTPUT_DIR, "roc_tpr.npy"), tpr)
    np.save(os.path.join(OUTPUT_DIR, "roc_thresholds.npy"), roc_thresholds)

    np.save(os.path.join(OUTPUT_DIR, "pr_precision.npy"), pr_prec)
    np.save(os.path.join(OUTPUT_DIR, "pr_recall.npy"), pr_rec)
    np.save(os.path.join(OUTPUT_DIR, "pr_thresholds.npy"), pr_thresholds)

    # Save a small summary as text (seed-safe because in seed folder)
    with open(os.path.join(OUTPUT_DIR, "summary_metrics.txt"), "w") as f:
        f.write(f"seed={seed}\n")
        f.write(f"final_threshold={final_threshold:.6f}\n")
        f.write(f"precision={test_precision:.6f}\n")
        f.write(f"recall={test_recall:.6f}\n")
        f.write(f"f1={test_f1:.6f}\n")
        f.write(f"roc_auc={roc_auc_val:.6f}\n")
        f.write(f"pr_auc_ap={pr_auc_val:.6f}\n")

    # Save ROC figure (no show)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc_val:.3f})")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "roc_curve.png"), dpi=200)
    plt.close()

    # Save PR figure (no show)
    plt.figure()
    plt.plot(pr_rec, pr_prec, label=f"PR (AP = {pr_auc_val:.3f})")
    baseline = (np.sum(test_true) / max(1, len(test_true))) if len(test_true) else 0.0
    plt.hlines(baseline, 0, 1, linestyles="--", label=f"Baseline={baseline:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "pr_curve.png"), dpi=200)
    plt.close()

    print(f"\nSaved ROC/PR curves + raw arrays into: {OUTPUT_DIR}")

    # ----------------------------
    # PER-TOKEN METRICS (only tokens with >=3 pumps in TEST)
    # ----------------------------
    @torch.no_grad()
    def collect_per_token_predictions(loader, thresh=0.5):
        model.eval()
        token_to_true = {}
        token_to_pred = {}

        for batch in loader:
            batch = batch.to(device)
            probs = model(batch.x, batch.edge_index, batch.edge_attr)
            preds = (probs >= thresh).long()

            node_global_idx = torch.arange(batch.x.size(0), device=device)
            node_local_idx = (node_global_idx % num_nodes).long()

            masked_idx = batch.mask.nonzero(as_tuple=False).view(-1)
            masked_local = node_local_idx[masked_idx].detach().cpu().numpy()
            masked_true  = batch.y[masked_idx].detach().cpu().numpy().astype(int)
            masked_pred  = preds[masked_idx].detach().cpu().numpy().astype(int)

            for li, yt, yp in zip(masked_local, masked_true, masked_pred):
                sym = idx_to_symbol[int(li)]
                token_to_true.setdefault(sym, []).append(int(yt))
                token_to_pred.setdefault(sym, []).append(int(yp))

        return token_to_true, token_to_pred

    token_to_true, token_to_pred = collect_per_token_predictions(test_loader, thresh=final_threshold)

    eligible_tokens = []
    for sym, ys in token_to_true.items():
        if int(np.sum(np.array(ys) == 1)) >= 3:
            eligible_tokens.append(sym)

    print("\n--- Per-Token Metrics (only tokens with >3 pumps in TEST) ---")
    if not eligible_tokens:
        print("No token with more than 3 pumps in the test set.")
    else:
        for sym in sorted(eligible_tokens):
            ys = np.array(token_to_true[sym], dtype=int)
            ps = np.array(token_to_pred[sym], dtype=int)

            p = precision_score(ys, ps, zero_division=0)
            r = recall_score(ys, ps, zero_division=0)
            f = f1_score(ys, ps, zero_division=0)
            pumps = int(np.sum(ys == 1))
            n_obs = int(len(ys))

            print(f"{sym} | n={n_obs} | pumps={pumps} | Precision={p:.4f} | Recall={r:.4f} | F1={f:.4f}")
            PER_TOKEN_ROWS.append({
                "seed": seed,
                "target_density": TARGET_DENSITY,
                "dropout": DROPOUT,
                "pump_window_hours": PUMP_WINDOW_HOURS,
                "best_val_threshold": final_threshold,
                "token": sym,
                "n_obs": n_obs,
                "pumps": pumps,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
            })

    return {
        "seed": seed,
        "target_density": TARGET_DENSITY,
        "dropout": DROPOUT,
        "pump_window_hours": PUMP_WINDOW_HOURS,
        "best_val_f1": float(best_val_f1),
        "best_val_threshold": float(final_threshold),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
    }


# ----------------------------
# 10) MULTI-SEED EXPERIMENT
# ----------------------------
def run_multi_seed():
    results = []
    total = len(SEEDS)
    pbar = tqdm(total=total, desc="MultiSeed", unit="run")

    for seed in SEEDS:
        r = run_one(seed)
        results.append(r)

        pbar.set_postfix({
            "seed": seed,
            "valF1": f"{r['best_val_f1']:.3f}",
            "testF1": f"{r['test_f1']:.3f}",
        })
        pbar.update(1)

    pbar.close()

    df_res = pd.DataFrame(results)
    df_res.to_csv(GRID_RESULTS_FILE, index=False)

    if len(PER_TOKEN_ROWS) > 0:
        pd.DataFrame(PER_TOKEN_ROWS).to_csv(GRID_PER_TOKEN_FILE, index=False)
    else:
        pd.DataFrame(columns=[
            "seed", "target_density", "dropout", "pump_window_hours", "best_val_threshold",
            "token", "n_obs", "pumps", "precision", "recall", "f1"
        ]).to_csv(GRID_PER_TOKEN_FILE, index=False)

    summary = (
        df_res
        .groupby(["target_density", "dropout", "pump_window_hours"], as_index=False)
        .agg(
            val_f1_mean=("best_val_f1", "mean"),
            val_f1_std=("best_val_f1", "std"),
            test_f1_mean=("test_f1", "mean"),
            test_f1_std=("test_f1", "std"),
            test_precision_mean=("test_precision", "mean"),
            test_recall_mean=("test_recall", "mean"),
        )
        .sort_values(["val_f1_mean", "test_f1_mean"], ascending=False)
        .reset_index(drop=True)
    )
    summary.to_csv(GRID_SUMMARY_FILE, index=False)

    best = summary.iloc[0].to_dict()
    print("\n==============================")
    print("BEST CONFIG (mean over seeds)")
    print("==============================")
    print(f"TARGET_DENSITY    = {best['target_density']}")
    print(f"DROPOUT           = {best['dropout']}")
    print(f"PUMP_WINDOW_HOURS = {int(best['pump_window_hours'])}")
    print(f"Val F1 (mean±std)  = {best['val_f1_mean']:.4f} ± {best['val_f1_std']:.4f}")
    print(f"Test F1 (mean±std) = {best['test_f1_mean']:.4f} ± {best['test_f1_std']:.4f}")
    print(f"Test Precision mean = {best['test_precision_mean']:.4f}")
    print(f"Test Recall mean    = {best['test_recall_mean']:.4f}")

    print("\nSaved:")
    print(f"- {GRID_RESULTS_FILE}")
    print(f"- {GRID_SUMMARY_FILE}")
    print(f"- {GRID_PER_TOKEN_FILE}")
    print(f"- {OUTPUT_ROOT}/seed_***/")


if __name__ == "__main__":
    print("\n--- Running Multi-Seed Dynamic Graph Experiment ---")
    run_multi_seed()