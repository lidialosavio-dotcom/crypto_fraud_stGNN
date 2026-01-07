
# =====================================================================================
# Script overview (general description):
#
# This script trains a temporal node-level binary classifier over a universe of tokens
# ("symbols") using:
#   (1) Temporal windowed node features: for each timestamp, each node has a sequence
#       of length WINDOW_SIZE with numeric features.
#   (2) A graph neural network (TransformerConv) applied at each time step using an
#       edge-weighted graph.
#   (3) A temporal TransformerEncoder applied across the WINDOW_SIZE sequence of
#       spatial embeddings to produce the final per-node probability.
#
# Graph construction approach (dynamic kNN + Gaussian similarity graph):
# - The graph is built from kNN similarity computed on TRAIN data only.
# - We identify "pump events" (timestamps in train where at least one node has label=1).
# - For each pump timestamp p, we compute kNN and Gaussian weights on a time window
#   [p - PUMP_WINDOW_HOURS, p] using all node features.
# - We maintain an evolving edge set (edge_stats) that accumulates evidence across pump windows.
# - For each pump, we store a snapshot of the current graph (mean weights so far).
# - During training snapshots, we use the graph "state" corresponding to the most recent
#   pump at or before the current timestamp. Before any pump, a safe base graph is used.
# - Validation and test snapshots use the final graph (state after the last train pump).
#
# Temporal split:
# - Split is performed by unique timestamps (not by rows), with an embargo gap between
#   splits to reduce temporal leakage.
#
# Metrics:
# - The model is selected by best validation F1 (computed at a fixed threshold).
# - Final test metrics are reported using the same fixed threshold.
# =====================================================================================

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
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
)


warnings.filterwarnings("ignore")

# ----------------------------
# 1) SEED
# ----------------------------
def set_seed(seed=11):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Seed fixed to {seed}.")

set_seed(99)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ----------------------------
# 2) CONFIG
# ----------------------------
# Temporal split proportions over unique timestamps.
SPLIT_TRAIN = 0.60
SPLIT_VAL   = 0.20
SPLIT_TEST  = 0.20

# Embargo gap (in number of timestamps) between train/val and val/test to reduce leakage.
EMBARGO_STEPS = 5

# Number of past steps included in each sample window (sequence length for temporal model).
WINDOW_SIZE = 5

# kNN Graph Config
KNN_K = 10                  # Number of neighbors

# ----------------------------
# DYNAMIC GRAPH CONFIG
# ----------------------------
# Window size for correlation computation around each pump timestamp in TRAIN.
# The window is [pump - 48h, pump] (clipped within train timestamps).
PUMP_WINDOW_HOURS = 48           # ONLY 48 hours LOOKBACK from each pump in TRAIN (no right side)

# Model/training hyperparameters.
HIDDEN_CHANNELS = 64
HEADS = 2
LEARNING_RATE = 0.001
EPOCHS = 50
DROPOUT = 0.4
BATCH_SIZE = 64

# Fixed probability threshold used for classification and F1 evaluation.
# Only this fixed threshold is used in training selection and final test evaluation.
FIXED_THRESHOLD = 0.5

# Column names used in the CSV.
DATE_COL   = "date"
LABEL_COL  = "flag"
GROUP_COL  = "symbol"

# Columns excluded from feature selection (targets, identifiers, non-feature columns).
DROP_COLS_TRAIN = [
    DATE_COL, LABEL_COL, GROUP_COL,
    "high", "low", "close", "group",
    "log_ret", "ret_BTC", "vola_BTC"
]

# ----------------------------
# 3) LOAD DATA
# ----------------------------
# Path to the dataset CSV (hourly pump & dump).
file_path = r"/home/lidialosav/pump-and-dump-dataset/project/hourly_pump&dump_15112025.csv"

print("\n--- Caricamento Dati ---")
# Load dataset and perform basic cleaning and sorting.
df = pd.read_csv(file_path, delimiter=",")
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.sort_values(DATE_COL).reset_index(drop=True)

# Ensure labels are binary integers in {0,1}.
df[LABEL_COL] = df[LABEL_COL].astype(int).clip(0, 1)

# Define the node universe as the set of unique symbols; map each symbol to an index.
unique_symbols = df[GROUP_COL].unique()
symbol_to_idx = {sym: i for i, sym in enumerate(unique_symbols)}
idx_to_symbol = {i: sym for sym, i in symbol_to_idx.items()}  # <<< ADDED (per metriche token)
num_nodes = len(unique_symbols)
print(f"Dataset: {len(df)} righe, {num_nodes} token unici.")

# ----------------------------
# 4) FEATURES (RAW)
# ----------------------------
print("\n--- Analisi Features ---")
# Select numeric feature columns excluding known drop columns.
raw_feature_cols = [
    c for c in df.columns
    if c not in DROP_COLS_TRAIN and np.issubdtype(df[c].dtype, np.number)
]
print(f"Features Selezionate ({len(raw_feature_cols)}): {raw_feature_cols}")

print(raw_feature_cols)
# Work on a copy for downstream processing.
df_proc = df.copy()

# ----------------------------
# 5) TEMPORAL SPLIT
# ----------------------------
# Split by unique timestamps to preserve temporal ordering (no random shuffling of time).
unique_dates = df[DATE_COL].unique()
n_dates = len(unique_dates)
train_end_idx = int(SPLIT_TRAIN * n_dates)
val_end_idx   = int((SPLIT_TRAIN + SPLIT_VAL) * n_dates)

# Train timestamps.
dates_train = unique_dates[:train_end_idx]
# Validation timestamps (with embargo after train).
dates_val   = unique_dates[train_end_idx + EMBARGO_STEPS : val_end_idx]
# Test timestamps (with embargo after validation).
dates_test  = unique_dates[val_end_idx + EMBARGO_STEPS :]

print(f"Train: {len(dates_train)} h | Val: {len(dates_val)} h | Test: {len(dates_test)} h")

# ----------------------------
# 6) DYNAMIC GRAPH (kNN updates on windows around pump events in TRAIN)
# ----------------------------
print("\n--- Dynamic graph construction (kNN + Gaussian on pump windows) ---\n")

# Base graph used as a safe fallback (self-loops only).
# This is used before the first pump event (or in edge-empty situations).
base_edge_index = torch.tensor([list(range(num_nodes)), list(range(num_nodes))], dtype=torch.long)
base_edge_attr  = torch.ones(num_nodes, dtype=torch.float)

# Subset train rows for pump detection and window slicing.
df_train = df[df[DATE_COL].isin(dates_train)].copy()

# Sorted DatetimeIndex of TRAIN timestamps for robust index-based window slicing.
train_dates_idx = pd.DatetimeIndex(dates_train).sort_values()

# Pump timestamps in TRAIN: any timestamp where at least one token has label=1.
pump_dates = (
    df_train.loc[df_train[LABEL_COL] == 1, DATE_COL]
    .dropna()
    .unique()
)
pump_dates = pd.DatetimeIndex(pump_dates).sort_values()

print(f"Pump events nel TRAIN: {len(pump_dates)} (timestamps unici)")

def _window_dates_around_pump(pump_date, train_dates_index, window_h=PUMP_WINDOW_HOURS):
    """Return the TRAIN timestamps within the window [pump-48, pump] (clipped)."""
    # Locate the pump timestamp position in the sorted train index.
    pos = train_dates_index.searchsorted(pump_date)
    # If pump_date is not exactly present, move to the closest prior index.
    if pos == len(train_dates_index) or train_dates_index[pos] != pump_date:
        pos = max(0, pos - 1)

    # Clip left boundary within valid train index bounds.
    left = max(0, pos - window_h)

    # IMPORTANT: to avoid leakage, we updated the code so that the right boundary is the pump itself (no future).
    right = pos

    return train_dates_index[left:right+1]

# Pre-compute global stats for standardization (to be consistent across windows)
# shape: {feat: (mean, std)}
feat_stats = {}
for feat in raw_feature_cols:
    feat_pivot = df_train.pivot_table(index=DATE_COL, columns=GROUP_COL, values=feat).fillna(0)
    vals = feat_pivot.values
    feat_stats[feat] = (np.mean(vals), np.std(vals) + 1e-12)

def compute_knn_gaussian_edges_for_window(df_full, dates_window):
    """Compute kNN + Gaussian edges for a given time window."""
    # Construct feature matrix [Nodes, Time * Features]
    feature_matrices = []

    # take subset
    sub = df_full[df_full[DATE_COL].isin(dates_window)].copy()

    for feat in raw_feature_cols:
        # Pivot: [Time, Nodes]
        pivot = sub.pivot_table(index=DATE_COL, columns=GROUP_COL, values=feat).fillna(0)
        # Ensure correct column order
        pivot = pivot.reindex(columns=unique_symbols, fill_value=0)

        # Standardize using GLOBAL train stats
        mean_val, std_val = feat_stats[feat]
        vals_scaled = (pivot.values - mean_val) / std_val

        # Transpose to [Nodes, Time] and append
        feature_matrices.append(vals_scaled.T)

    # Concatenate features: [Nodes, Time * Num_Features]
    Pts = np.concatenate(feature_matrices, axis=1)

    # kNN search
    # We need k+1 neighbors (including self)
    # Using 'euclidean' metric
    nbrs = NearestNeighbors(n_neighbors=KNN_K + 1, algorithm='auto', metric='euclidean').fit(Pts)
    distances, indices = nbrs.kneighbors(Pts)

    # Gaussian similarity
    # Local scaling sigma: distance to the k-th neighbor
    sigmas = distances[:, KNN_K]

    new_edges = {} # (u, v) -> weight

    for i in range(num_nodes):
        sigma_i = sigmas[i] if sigmas[i] > 0 else 1e-12

        for k_idx in range(1, KNN_K + 1):
            j = indices[i, k_idx]
            d = distances[i, k_idx]

            # Skip self-loops
            if i == j: continue

            sigma_j = sigmas[j] if sigmas[j] > 0 else 1e-12

            # Symmetric Sigma
            sigma_edge = max(sigma_i, sigma_j)

            # Gaussian similarity (weights)
            # s_i(j) = exp(-4 * (d_i(j)^2) / (sigma_i(j)^2))
            w = np.exp(-4 * (d**2) / (sigma_edge**2))

            # Enforce symmetry by storing ordered pair
            u, v = (i, j) if i < j else (j, i)

            # If multiple directions/occurrences in same window (unlikely for kNN unless symmetric), max or avg?
            # take max weight if both see each other (or overwrite).
            if (u, v) in new_edges:
                new_edges[(u, v)] = max(new_edges[(u, v)], w)
            else:
                new_edges[(u, v)] = w

    return new_edges


def _edge_stats_to_tensors(edge_stats_dict, num_nodes):
    """
    Convert accumulated undirected edge stats into a directed PyG edge list.

    edge_stats_dict: {(i,j): [sum, count]} with i<j
    Output:
      - edge_index [2, E*2] containing both directions (i->j and j->i)
      - edge_attr  [E*2] with mean weight replicated per direction
    """
    if len(edge_stats_dict) == 0:
        # If no edges exist, fall back to the base self-loop graph.
        return base_edge_index, base_edge_attr

    sources, targets, weights = [], [], []
    for (i, j), (s, c) in edge_stats_dict.items():
        mean_w = float(s) / float(c)
        # Add both directions to make message passing symmetric (bidirectional).
        sources.extend([i, j])
        targets.extend([j, i])
        weights.extend([mean_w, mean_w])

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr  = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_attr

# If there are no pump events in TRAIN, use base graph.
if len(pump_dates) == 0:
    print("ATTENZIONE: nessun pump nel TRAIN. Fallback a grafo statico (base).")
    final_edge_index, final_edge_attr = base_edge_index, base_edge_attr
    dynamic_graph_updates = {}
else:
    # edge_stats accumulates per-undirected-edge statistics across pump windows:
    # key=(i,j) with i<j, value=[sum_weight, count_updates]
    edge_stats = {}
    dynamic_graph_updates = {}  # pump_date -> (edge_index, edge_attr) AFTER applying this pump update

    for k, pdate in enumerate(pump_dates):
        # Determine the train timestamps within the pump-centered window.
        win_dates = _window_dates_around_pump(pdate, train_dates_idx, window_h=PUMP_WINDOW_HOURS)

        # Compute kNN edges for this window
        window_edges = compute_knn_gaussian_edges_for_window(df, win_dates)

        new_added = 0
        for (u, v), w in window_edges.items():
            if (u, v) not in edge_stats:
                # First time this edge is observed: initialize sum and count.
                #  (2) Add new edges that are in the top-k
                edge_stats[(u, v)] = [float(w), 1]
                new_added += 1
            else:
                # Accumulate
                # (1) Update existing edges (if they are still in Top-K this window)
                edge_stats[(u, v)][0] += float(w)
                edge_stats[(u, v)][1] += 1

        # (3) Store the current graph state after this pump:
        # Convert accumulated stats to edge_index/edge_attr (mean weight per undirected edge).
        curr_edge_index, curr_edge_attr = _edge_stats_to_tensors(edge_stats, num_nodes)
        dynamic_graph_updates[pdate] = (curr_edge_index, curr_edge_attr)

        print(f"[{k+1}/{len(pump_dates)}] pump @ {pdate} | "
              f"win={len(win_dates)}h | "
              f"new_edges={new_added} | total_undirected_edges={len(edge_stats)}")

    # Final graph is the state after the last pump in TRAIN.
    last_pump = pump_dates[-1]
    final_edge_index, final_edge_attr = dynamic_graph_updates[last_pump]

# Report final graph size in directed edges (because each undirected edge becomes 2 directed edges).
print(f"Final graph edges (directed): {final_edge_index.size(1)}")

# ----------------------------
# 7) SCALING + WINDOWING
# ----------------------------
print("\n--- Scaling & Creazione Finestre Temporali ---")

# Fit scaler on TRAIN only to avoid leakage.
df_train_subset = df_proc[df_proc[DATE_COL].isin(dates_train)]
scaler = StandardScaler()
scaler.fit(df_train_subset[raw_feature_cols].fillna(0))

def get_temporal_snapshots(target_dates, window_size=WINDOW_SIZE, graph_mode="dynamic"):
    """
    Each Data object contains:
      - x: [N, W, F] windowed node features
      - y: [N] node labels for the current timestamp
      - mask: [N] indicates which nodes are present at the current timestamp
      - edge_index, edge_attr: either dynamic graph (per date) or fixed final graph
    """
    snapshots = []

    all_dates_list = df_proc[DATE_COL].unique()

    # Include enough history before the first target date to build the initial window.
    min_date_idx = np.searchsorted(all_dates_list, target_dates[0])
    relevant_start_idx = max(0, min_date_idx - window_size + 1)
    relevant_dates = all_dates_list[
        relevant_start_idx : np.searchsorted(all_dates_list, target_dates[-1]) + 1
    ]

    # Subset data to the relevant time range and scale numeric features.
    subset = df_proc[df_proc[DATE_COL].isin(relevant_dates)].copy()
    subset[raw_feature_cols] = scaler.transform(subset[raw_feature_cols].fillna(0))
    grouped = subset.groupby(DATE_COL)

    # Precompute a dense node-feature matrix per date: [N, F], zeros for missing nodes.
    date_to_matrix = {}
    for date, group in grouped:
        mat = np.zeros((num_nodes, len(raw_feature_cols)), dtype=np.float32)
        indices = [symbol_to_idx[s] for s in group[GROUP_COL]]
        mat[indices] = group[raw_feature_cols].values.astype(np.float32)
        date_to_matrix[date] = mat

    # Sorted list of pump timestamps for selecting the latest available graph update.
    pump_dates_sorted = pd.DatetimeIndex(list(dynamic_graph_updates.keys())).sort_values()

    def graph_for_date(date):
        # If no dynamic updates exist (no pump in train), always use the final graph.
        if len(pump_dates_sorted) == 0:
            return final_edge_index, final_edge_attr

        # Find the most recent pump <= date.
        pos = pump_dates_sorted.searchsorted(date, side="right") - 1
        if pos < 0:
            # Before the first pump: use base graph.
            return base_edge_index, base_edge_attr
        p = pump_dates_sorted[pos]
        return dynamic_graph_updates[p]

    for date in tqdm(target_dates, desc="Generating Windows"):
        # Locate current date index in the full date list.
        curr_idx = np.where(all_dates_list == date)[0][0]

        # Build a window of W matrices aligned in time ending at current date.
        window_matrices = []
        for w in range(window_size):
            lookback_idx = curr_idx - (window_size - 1) + w
            if lookback_idx >= 0:
                d = all_dates_list[lookback_idx]
                window_matrices.append(date_to_matrix.get(
                    d, np.zeros((num_nodes, len(raw_feature_cols)), dtype=np.float32)
                ))
            else:
                # Pad with zeros when lookback exceeds dataset start.
                window_matrices.append(np.zeros((num_nodes, len(raw_feature_cols)), dtype=np.float32))

        # Stack into [N, W, F].
        x_seq = np.stack(window_matrices, axis=1)  # [N, W, F]

        # Extract labels and a presence mask for the current timestamp.
        current_group = df[df[DATE_COL] == date]
        y_t = np.zeros(num_nodes, dtype=np.float32)
        mask_t = np.zeros(num_nodes, dtype=np.bool_)

        indices = [symbol_to_idx[s] for s in current_group[GROUP_COL]]
        y_t[indices] = current_group[LABEL_COL].values.astype(np.float32)
        mask_t[indices] = True

        # Choose graph: dynamic (date-dependent) or final (fixed).
        if graph_mode == "dynamic":
            eidx, eattr = graph_for_date(date)
        else:
            eidx, eattr = final_edge_index, final_edge_attr

        # Create PyG Data object.
        data = Data(
            x=torch.tensor(x_seq, dtype=torch.float),
            y=torch.tensor(y_t, dtype=torch.float),
            mask=torch.tensor(mask_t, dtype=torch.bool),
            edge_index=eidx,
            edge_attr=eattr
        )
        snapshots.append(data)

    return snapshots

# Use dynamic graphs for training snapshots; use final fixed graph for validation/test snapshots.
train_snapshots = get_temporal_snapshots(dates_train, graph_mode="dynamic")
val_snapshots   = get_temporal_snapshots(dates_val,   graph_mode="final")
test_snapshots  = get_temporal_snapshots(dates_test,  graph_mode="final")

# Create DataLoaders. Training uses shuffle=True to randomize snapshot ordering in optimization.
train_loader = DataLoader(train_snapshots, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_snapshots,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_snapshots,  batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# 8) MODEL (GNN + Temporal Transformer)
# ----------------------------
class TemporalGraphModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, heads=HEADS, window_size=WINDOW_SIZE):
        super().__init__()

        # Project raw numeric features into hidden dimension.
        self.lin_in = Linear(num_features, hidden_channels)

        # Spatial message passing using attention-based convolution on edge-weighted graphs.
        # edge_dim=1 expects edge_attr shaped as [E, 1].
        self.conv1 = TransformerConv(hidden_channels, hidden_channels, heads=heads, edge_dim=1)
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=1)

        # Temporal encoder over the sequence dimension (window_size).
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=8,
            dim_feedforward=128,
            dropout=DROPOUT,
            batch_first=True
        )
        self.temporal_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output head: one probability per node.
        self.out = Linear(hidden_channels, 1)

        # Learnable positional embedding for temporal order in the window.
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, window_size, hidden_channels))

    def forward(self, x, edge_index, edge_weight):
        # x: [TotalNodesInBatch, W, F] (PyG batches concatenate nodes across graphs).
        batch_nodes, window, feats = x.size()

        # Convert edge weights to shape [E, 1] for TransformerConv.
        edge_attr = edge_weight.view(-1, 1)

        # Compute spatial embeddings for each time step independently.
        temporal_embeddings = []
        for t in range(window):
            # Project features at time t.
            x_t = self.lin_in(x[:, t, :])

            # First spatial layer with attention and dropout.
            h = self.conv1(x_t, edge_index, edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=DROPOUT, training=self.training)

            # Second spatial layer (heads=1).
            h = self.conv2(h, edge_index, edge_attr)
            h = F.relu(h)

            temporal_embeddings.append(h)

        # Stack embeddings into a temporal sequence: [TotalNodesInBatch, W, H].
        sequence = torch.stack(temporal_embeddings, dim=1)  # [BNodes, W, H]

        # Add learnable positional encodings.
        sequence = sequence + self.pos_embedding

        # Temporal transformer over the window.
        time_out = self.temporal_transformer(sequence)      # [BNodes, W, H]

        # Use the last time step representation for classification.
        last_step = time_out[:, -1, :]
        out = self.out(last_step)                           # [BNodes, 1]

        # Sigmoid for binary probability output per node.
        return torch.sigmoid(out).view(-1)                  # [BNodes]

# Instantiate model, optimizer, and loss.
model = TemporalGraphModel(len(raw_feature_cols), HIDDEN_CHANNELS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()

# ----------------------------
# 9) EVAL UTILS (threshold on VAL)
# ----------------------------
@torch.no_grad()
def collect_probs_and_true(loader):
    # Collect predicted probabilities and ground-truth labels for masked nodes only.
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

def find_best_threshold_from_pr(y_true, y_prob):
    """
    Find the threshold that maximizes F1 using validation only.
    """
    if len(y_true) == 0:
        return 0.5, 0.0

    prec, rec, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * (prec * rec) / (prec + rec + 1e-12)

    best_idx = int(np.argmax(f1s))
    if best_idx >= len(thresholds):
        best_thresh = 0.5
    else:
        best_thresh = float(thresholds[best_idx])

    best_f1 = float(np.max(f1s))
    return best_thresh, best_f1

@torch.no_grad()
def evaluate_loss_f1(loader, thresh=0.5):
    # Compute average loss and F1 at a given probability threshold over masked nodes.
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

# ----------------------------
# 10) TRAINING (test logging enabled, FIXED THRESHOLD)
# ----------------------------
print(f"\n--- Start Training (Dynamic Graph) ({EPOCHS} epochs) ---")

best_val_f1 = -1.0
best_model_state = None

pbar = tqdm(range(EPOCHS), desc="Training", unit="epoch")
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

    # Use FIXED_THRESHOLD directly (no per-epoch threshold search on validation).
    val_loss, val_f1 = evaluate_loss_f1(val_loader, thresh=FIXED_THRESHOLD)
    _, test_f1 = evaluate_loss_f1(test_loader, thresh=FIXED_THRESHOLD)

    # Track best checkpoint by validation F1 (computed at the fixed threshold).
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = copy.deepcopy(model.state_dict())

    pbar.set_postfix({
        "Ltr": f"{avg_train_loss:.3f}",
        "ValF1@0.5": f"{val_f1:.3f}",
        "TstF1@0.5": f"{test_f1:.3f}",
        "BestVal": f"{best_val_f1:.3f}"
    })

# ----------------------------
# 11) FINAL TEST (best checkpoint + FIXED THRESHOLD)
# ----------------------------
print("\n\n--- Risultati Finali (Dynamic Graph) ---")
if best_model_state is not None:
    model.load_state_dict(best_model_state)

test_probs, test_true = collect_probs_and_true(test_loader)
final_preds = (test_probs >= FIXED_THRESHOLD).astype(int)

print(f"Soglia fissa: {FIXED_THRESHOLD:.2f}")
print(f"Precision: {precision_score(test_true, final_preds, zero_division=0):.4f}")
print(f"Recall:    {recall_score(test_true, final_preds, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(test_true, final_preds, zero_division=0):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(test_true, final_preds))

# ----------------------------
# 12) METRICHE PER TOKEN (solo token con pump nel TEST >= 3)  <<< ADDED
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

        # Ogni snapshot ha sempre N nodi nello stesso ordine (0..N-1).
        # Nel batch concatenato, l'indice "locale" del nodo è (global_idx % num_nodes).
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

token_to_true, token_to_pred = collect_per_token_predictions(test_loader, thresh=FIXED_THRESHOLD)

eligible_tokens = []
for sym, ys in token_to_true.items():
    if int(np.sum(np.array(ys) == 1)) >= 3:
        eligible_tokens.append(sym)

print("\n--- Metriche per Token (solo token con >=3 pump nel TEST) ---")
if not eligible_tokens:
    print("Nessun token con >=3 pump nel test set.")
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
