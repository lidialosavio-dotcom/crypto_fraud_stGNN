# =====================================================================================
# Graph WaveNet reference (adaptive adjacency via learnable node embeddings):
# - Wu, Zonghan, et al. "Graph WaveNet for Deep Spatial-Temporal Graph Modeling"
#   (Graph WaveNet introduces an adaptive adjacency matrix learned from node embeddings,
#   typically via a low-rank factorization and softmax-normalized connectivity scores.)
#
# What this script builds (high-level):
# - A temporal node-classification pipeline over a universe of assets/tokens ("symbols").
# - For each timestamp, we construct a node feature tensor using a fixed window of
#   lookback snapshots: x_seq has shape [N_nodes, WINDOW_SIZE, N_features].
# - We learn a self-adaptive (data-driven) graph structure using two learnable node
#   embedding tables (E1, E2). Their bilinear product yields a dense score matrix
#   [N, N] that is row-wise softmax-normalized into a probabilistic adjacency A. -- mean F1 = 78
#   This is conceptually aligned with Graph WaveNet's adaptive adjacency mechanism:
#   node embeddings parameterize pairwise affinities, producing a learned connectivity
#   pattern rather than relying on a fixed, externally provided graph.
# - We then sparsify A by thresholding (instead of TopK) to obtain edge_index/edge_attr.
# - Spatial modeling per time step: TransformerConv is applied on the adaptive graph.
# - Temporal modeling across the window: a TransformerEncoder processes the sequence of
#   spatial embeddings, and a sigmoid head outputs a probability per node.
#
# Notes on the node embedding mechanism and similarity to Graph WaveNet:
# - E1 and E2 are learnable node representations; scores = E1 @ E2^T defines directed
#   affinities. Row-wise softmax yields normalized outgoing connection strengths.
# - Graph WaveNet uses a similar idea: learnable node embeddings form an adaptive
#   adjacency (often via softmax(ReLU(E1 E2^T))). This script follows that pattern,
#   then uses a threshold to keep edges above a minimum weight for message passing.
# =====================================================================================

###  MLP, RF, XGBoost, LSTM -- Luca

### static similarity, static correlation (volume), static correlation (num_trades)

### dynamic with similarity -- Dim, dynamic with correlation (volume), dynamic with correlation (num_trades), dynamic with node embeddings (WaveNet)

### Kind of results:
    ### F1 score on the entire dataset (mean across 9 seeds) --- table, F1 score per tokens* --- candle stick charts and barplot,
    ### F1 score + additional features**, F1 score with minutes
    

### *Select only tokens with 5 or more pump events
    ### A = [t32 = 1, t100 = 0, ...] --- |A| = 10 --- F1scoreTokenA = ...
    ### B = [t32 = 1, t101 = 0, ...] --- |B| = 3 --- F1scoreTokenB = ...
    ### Possible limitation = these performances are computed using the dataset construction around pump events, to better understand the performances on a single
    ### token history we can consider only one token per time and having the entire history for that token 

### **As Sapienza + prices -- Luca

### Contributions:
    ### Comprehensive comparison of methods for P&D event with modern static/dynamic spatio-temporal gnns applied to pump-and-dump crime
    ### pipeline for pump-and-dump crime evaluation/detection
    ### Code and data given

import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
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
    precision_score, recall_score, f1_score, confusion_matrix
)


warnings.filterwarnings("ignore")

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


set_seed(11)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ----------------------------
# 2) CONFIG
# ----------------------------

SPLIT_TRAIN = 0.60
SPLIT_VAL   = 0.20
SPLIT_TEST  = 0.20


EMBARGO_STEPS = 5

# Number of past steps included in each sample window.
WINDOW_SIZE = 5

# Model/training hyperparameters.
HIDDEN_CHANNELS = 64
HEADS = 2
LEARNING_RATE = 0.001
EPOCHS = 50
DROPOUT = 0.4
BATCH_SIZE = 64

# Fixed probability threshold for final binary decisions.
FIXED_THRESHOLD = 0.5

# ----------------------------
# SELF-ADAPTIVE ADJACENCY CONFIG
# ----------------------------
# Dimension of the learnable node embeddings used to generate adaptive adjacency.
ADP_DIM = 32 #64, 128

# Optional ReLU on scores before softmax (removes negative affinities).
ADP_USE_RELU = True

# Optional addition of self-loops in the adaptive graph.
ADP_ADD_SELF_LOOPS = False

# Threshold on adjacency weights after softmax: keep edges only if weight > threshold.
ADP_GRAPH_THRESHOLD = 0.015   

# Column names and drop list for feature selection.
DATE_COL   = "date"
LABEL_COL  = "flag"
GROUP_COL  = "symbol"
DROP_COLS_TRAIN = [
    DATE_COL, LABEL_COL, GROUP_COL,
    "high", "low", "close", "group",
    "log_ret", "ret_BTC", "vola_BTC"
]

# ----------------------------
# 3) LOAD DATA
# ----------------------------
# Path to the CSV with hourly time series and labels.
file_path = r"/home/lidialosav/pump-and-dump-dataset/project/hourly_pump&dump_15112025.csv"

print("\n--- Caricamento Dati ---")

if not os.path.exists(file_path):
    raise FileNotFoundError(f"Path non trovato: {file_path}")

# Load dataset, parse datetime, sort temporally, enforce binary labels.
df = pd.read_csv(file_path, delimiter=",")
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.sort_values(DATE_COL).reset_index(drop=True)
df[LABEL_COL] = df[LABEL_COL].astype(int).clip(0, 1)

# Define node universe as unique symbols; map each symbol to an integer node index.
unique_symbols = df[GROUP_COL].unique()
symbol_to_idx = {sym: i for i, sym in enumerate(unique_symbols)}
num_nodes = len(unique_symbols)
print(f"Dataset: {len(df)} righe, {num_nodes} token unici.")

# ----------------------------
# 4) FEATURES (RAW)
# ----------------------------
print("\n--- Analisi Features ---")
# Select numeric feature columns excluding dropped columns and known metadata/targets.
raw_feature_cols = [
    c for c in df.columns
    if c not in DROP_COLS_TRAIN and np.issubdtype(df[c].dtype, np.number)
]
print(f"Features Selezionate ({len(raw_feature_cols)}): {raw_feature_cols}")

# Work on a copy to avoid accidental modifications to the original DataFrame.
df_proc = df.copy()

# ----------------------------
# 5) TEMPORAL SPLIT
# ----------------------------
# Split by unique timestamps to respect temporal ordering.
unique_dates = df[DATE_COL].unique()
n_dates = len(unique_dates)
train_end_idx = int(SPLIT_TRAIN * n_dates)
val_end_idx   = int((SPLIT_TRAIN + SPLIT_VAL) * n_dates)

# Train dates: from start to train_end_idx.
dates_train = unique_dates[:train_end_idx]
# Validation dates: after an embargo gap following train, until val_end_idx.
dates_val   = unique_dates[train_end_idx + EMBARGO_STEPS : val_end_idx]
# Test dates: after an embargo gap following validation.
dates_test  = unique_dates[val_end_idx + EMBARGO_STEPS :]

print(f"Train: {len(dates_train)} h | Val: {len(dates_val)} h | Test: {len(dates_test)} h")

# ----------------------------
# 6) SCALING + WINDOWING
# ----------------------------
print("\n--- Scaling & Creazione Finestre Temporali ---")

# Fit the scaler only on training timestamps to prevent leakage.
df_train_subset = df_proc[df_proc[DATE_COL].isin(dates_train)]
scaler = StandardScaler()
scaler.fit(df_train_subset[raw_feature_cols].fillna(0))

# Dummy edge_index for Data initialization.
# The model does NOT use this; it constructs its own adaptive graph in forward().
dummy_edge_index = torch.tensor([list(range(num_nodes)), list(range(num_nodes))], dtype=torch.long)
dummy_edge_attr  = torch.ones(num_nodes, dtype=torch.float)

def get_temporal_snapshots(target_dates, window_size=WINDOW_SIZE):
    # Build PyG Data objects, one per timestamp in target_dates.
    # Each snapshot contains:
    # - x: node feature sequence [N, W, F]
    # - y: node label vector [N]
    # - mask: boolean mask indicating which nodes are present/valid at that timestamp
    # - edge_index/edge_attr: placeholders (unused by the model)
    snapshots = []
    all_dates_list = df_proc[DATE_COL].unique()

    # Determine a relevant range that includes lookback needed for the first target date.
    min_date_idx = np.searchsorted(all_dates_list, target_dates[0])
    relevant_start_idx = max(0, min_date_idx - window_size + 1)
    relevant_dates = all_dates_list[
        relevant_start_idx : np.searchsorted(all_dates_list, target_dates[-1]) + 1
    ]

    # Subset data, apply scaling to numeric features.
    subset = df_proc[df_proc[DATE_COL].isin(relevant_dates)].copy()
    subset[raw_feature_cols] = scaler.transform(subset[raw_feature_cols].fillna(0))
    grouped = subset.groupby(DATE_COL)

    # Precompute a dense node-feature matrix per date: [N, F], with zeros for missing symbols.
    date_to_matrix = {}
    for date, group in grouped:
        mat = np.zeros((num_nodes, len(raw_feature_cols)), dtype=np.float32)
        indices = [symbol_to_idx[s] for s in group[GROUP_COL]]
        mat[indices] = group[raw_feature_cols].values.astype(np.float32)
        date_to_matrix[date] = mat

    # For each target date, build a window of W matrices stacked into [N, W, F].
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

        # Stack along time axis: x_seq shape is [N, W, F].
        x_seq = np.stack(window_matrices, axis=1)  # [N, W, F]

        # Extract labels and mask for the current timestamp.
        # Only symbols present in the raw df at this timestamp are marked valid.
        current_group = df[df[DATE_COL] == date]
        y_t = np.zeros(num_nodes, dtype=np.float32)
        mask_t = np.zeros(num_nodes, dtype=np.bool_)

        indices = [symbol_to_idx[s] for s in current_group[GROUP_COL]]
        y_t[indices] = current_group[LABEL_COL].values.astype(np.float32)
        mask_t[indices] = True

        # Create a PyG Data object holding the windowed node features and labels.
        data = Data(
            x=torch.tensor(x_seq, dtype=torch.float),
            y=torch.tensor(y_t, dtype=torch.float),
            mask=torch.tensor(mask_t, dtype=torch.bool),
            edge_index=dummy_edge_index,
            edge_attr=dummy_edge_attr
        )
        snapshots.append(data)

    return snapshots

# Build datasets for each temporal split.
train_snapshots = get_temporal_snapshots(dates_train)
val_snapshots   = get_temporal_snapshots(dates_val)
test_snapshots  = get_temporal_snapshots(dates_test)

# Build loaders. Training loader uses shuffle=True (windows are still created by date,
# but batches are shuffled during optimization).
train_loader = DataLoader(train_snapshots, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_snapshots,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_snapshots,  batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# 7) MODEL (Self-adaptive Dynamic Graph)
# ----------------------------
class TemporalGraphModelSelfAdaptive(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_nodes,
                 heads=HEADS, window_size=WINDOW_SIZE,
                 adp_dim=ADP_DIM, adp_threshold=ADP_GRAPH_THRESHOLD):
        super().__init__()
        # Number of nodes (symbols) in the global universe.
        self.num_nodes = num_nodes

        # Threshold used to sparsify the softmax adjacency into a sparse edge list.
        self.adp_threshold = adp_threshold

        # Learnable node embeddings used to construct adaptive adjacency.
        # E1 and E2 implement a low-rank factorization of an affinity matrix.
        self.E1 = nn.Parameter(torch.empty(num_nodes, adp_dim))
        self.E2 = nn.Parameter(torch.empty(num_nodes, adp_dim))
        nn.init.xavier_uniform_(self.E1)
        nn.init.xavier_uniform_(self.E2)

        # Feature projection from raw feature dimension -> hidden dimension.
        self.lin_in = Linear(num_features, hidden_channels)

        # Spatial layers: TransformerConv uses attention-based message passing.
        # edge_dim=1 means edge_attr is expected as a 1D scalar per edge (here: adaptive weight).
        self.conv1 = TransformerConv(hidden_channels, hidden_channels, heads=heads, edge_dim=1)
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=1, edge_dim=1)

        # Temporal encoder: Transformer over the sequence dimension (window size).
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=8,
            dim_feedforward=128,
            dropout=DROPOUT,
            batch_first=True
        )
        self.temporal_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)

        # Output head: predict one logit per node (then sigmoid).
        self.out = Linear(hidden_channels, 1)

        # Learnable positional embedding for the temporal window.
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, hidden_channels))

    def build_adaptive_graph_single(self):
        # Construct an adaptive graph for a single "node universe" of size N.
        #
        # Steps:
        # 1) Compute dense affinity scores via bilinear product (directed):
        #    scores[i, j] = <E1[i], E2[j]>
        # 2) Optionally apply ReLU to remove negative scores.
        # 3) Apply row-wise softmax to obtain a probabilistic adjacency A (outgoing distribution).
        # 4) Threshold A to select edges; convert to sparse edge_index/edge_attr.
        #
        # This mirrors the core idea used in Graph WaveNet: adaptive adjacency derived from
        # learned node embeddings, rather than a fixed prior graph.
        scores = self.E1 @ self.E2.t()
        
        if ADP_USE_RELU:
            scores = torch.relu(scores)

        A = torch.softmax(scores, dim=1)

        # Threshold-based sparsification (instead of TopK).
        mask = A > self.adp_threshold
        
        # Convert dense masked matrix into COO-like sparse representation.
        indices = mask.nonzero(as_tuple=False).t()
        weights = A[mask]

        edge_index = indices
        edge_attr  = weights

        # Optional self-loops (not enabled by default here).
        if ADP_ADD_SELF_LOOPS:
            sl = torch.arange(self.num_nodes, device=A.device)
            sl_edge_index = torch.stack([sl, sl], dim=0)
            sl_edge_attr  = torch.ones(self.num_nodes, device=A.device, dtype=torch.float)

            edge_index = torch.cat([edge_index, sl_edge_index], dim=1)
            edge_attr  = torch.cat([edge_attr, sl_edge_attr], dim=0)

        return edge_index, edge_attr

    def replicate_graph_for_batch(self, edge_index, edge_attr, batch_size):
        # The adaptive graph is built for a single set of N nodes.
        # In PyG batching, multiple graphs are concatenated; nodes are offset by +k*N.
        # This method replicates the same learned graph structure across the batch.
        n = self.num_nodes
        E = edge_index.size(1)
        
        # If no edges pass threshold, avoid runtime errors by creating per-node self-loops.
        if E == 0:
            dummy_src = torch.arange(n * batch_size, device=edge_index.device)
            dummy_dst = dummy_src
            edge_index_b = torch.stack([dummy_src, dummy_dst], dim=0)
            edge_attr_b = torch.ones(n * batch_size, device=edge_index.device)
            return edge_index_b, edge_attr_b

        # Offsets for each graph in the batch.
        offsets = (torch.arange(batch_size, device=edge_index.device) * n).view(-1, 1)

        # Replicate and offset source and destination indices.
        src = edge_index[0].view(1, E).repeat(batch_size, 1) + offsets
        dst = edge_index[1].view(1, E).repeat(batch_size, 1) + offsets
        edge_index_b = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)

        # Replicate edge weights accordingly.
        edge_attr_b = edge_attr.view(1, E).repeat(batch_size, 1).reshape(-1)
        return edge_index_b, edge_attr_b

    def forward(self, x):
        # x arrives from DataLoader as [B*N, W, F] because batch concatenates nodes.
        batch_nodes, window, feats = x.size()
        B = batch_nodes // self.num_nodes

        # Build adaptive graph once per forward pass, then replicate across batch.
        eidx, eattr = self.build_adaptive_graph_single()
        eidx, eattr = self.replicate_graph_for_batch(eidx, eattr, batch_size=B)

        # TransformerConv expects edge_attr with shape [E, edge_dim].
        edge_attr = eattr.view(-1, 1)

        # Spatial encoding per time step.
        temporal_embeddings = []
        for t in range(window):
            # Project raw features at time t into hidden space.
            x_t = self.lin_in(x[:, t, :])

            # First spatial attention-based convolution.
            h = self.conv1(x_t, eidx, edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=DROPOUT, training=self.training)

            # Second spatial convolution (heads=1; input is hidden_channels*HEADS from conv1).
            h = self.conv2(h, eidx, edge_attr)
            h = F.relu(h)

            # Collect embedding for this time step.
            temporal_embeddings.append(h)

        # Stack over time: [B*N, W, hidden_channels].
        sequence = torch.stack(temporal_embeddings, dim=1)

        # Add learnable positional encodings to preserve order information.
        sequence = sequence + self.pos_embedding

        # Temporal Transformer: models dependencies across the W time steps.
        time_out = self.temporal_transformer(sequence)

        # Use the representation at the last time step for prediction.
        last_step = time_out[:, -1, :]

        # Linear + sigmoid yields probability per node.
        out = self.out(last_step)
        return torch.sigmoid(out).view(-1)

# Instantiate model.
model = TemporalGraphModelSelfAdaptive(
    num_features=len(raw_feature_cols),
    hidden_channels=HIDDEN_CHANNELS,
    num_nodes=num_nodes
).to(device)

# Optimizer and loss for binary classification with probabilities.
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()

@torch.no_grad()
def evaluate_f1(loader, thresh=0.5):
    # Evaluate average loss and F1 score at a fixed probability threshold.
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)

        # Forward pass: probabilities per node in the batch.
        probs = model(batch.x)

        # Compute loss only on valid nodes (mask).
        loss = criterion(probs[batch.mask], batch.y[batch.mask])
        total_loss += float(loss.item())

        # Threshold probabilities to get binary predictions.
        preds = (probs >= thresh).float()
        all_preds.extend(preds[batch.mask].cpu().numpy())
        all_true.extend(batch.y[batch.mask].cpu().numpy())

    # F1 over all masked nodes across the loader.
    f1 = f1_score(all_true, all_preds, zero_division=0) if len(all_true) else 0.0
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, f1

@torch.no_grad()
def collect_probs_and_true(loader):
    # Collect concatenated probabilities and ground truth labels over masked nodes.
    model.eval()
    probs_list, true_list = [], []
    for batch in loader:
        batch = batch.to(device)
        probs = model(batch.x)
        probs_list.append(probs[batch.mask].detach().cpu().numpy())
        true_list.append(batch.y[batch.mask].detach().cpu().numpy())
    probs = np.concatenate(probs_list, axis=0) if probs_list else np.array([])
    true  = np.concatenate(true_list, axis=0)  if true_list else np.array([])
    return probs, true

# ----------------------------
# 8) TRAINING
# ----------------------------
print(f"\n--- Start Training (Pure Softmax Graph) ({EPOCHS} epochs) ---")

best_val_f1 = -1.0
best_model_state = None

# Train for a fixed number of epochs, keeping the checkpoint with best validation F1.
pbar = tqdm(range(EPOCHS), desc="Training", unit="epoch")
for epoch in pbar:
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward and compute loss on masked nodes only.
        probs = model(batch.x)
        loss = criterion(probs[batch.mask], batch.y[batch.mask])

        # Backprop and update parameters (includes E1/E2 and all network weights).
        loss.backward()
        optimizer.step()
        train_loss += float(loss.item())

    avg_train_loss = train_loss / max(1, len(train_loader))

    # Evaluate using a fixed threshold; report both val and test for monitoring.
    val_loss, val_f1 = evaluate_f1(val_loader, thresh=FIXED_THRESHOLD)
    _, test_f1 = evaluate_f1(test_loader, thresh=FIXED_THRESHOLD)

    # Model selection by best validation F1.
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = copy.deepcopy(model.state_dict())

    # Progress bar postfix with losses and F1 metrics.
    pbar.set_postfix({
        "Ltr": f"{avg_train_loss:.3f}",
        "ValF1@0.5": f"{val_f1:.3f}",
        "TstF1@0.5": f"{test_f1:.3f}",
        "BestVal": f"{best_val_f1:.3f}"
    })

# ----------------------------
# 9) FINAL TEST
# ----------------------------
print("\n\n--- Risultati Finali (Pure Softmax Graph) ---")

# Restore best validation checkpoint before computing final test metrics.
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Collect test probabilities/labels and compute metrics at the fixed threshold.
test_probs, test_true = collect_probs_and_true(test_loader)
final_preds = (test_probs >= FIXED_THRESHOLD).astype(int)

print(f"Soglia fissa: {FIXED_THRESHOLD:.2f}")
print(f"Precision: {precision_score(test_true, final_preds, zero_division=0):.4f}")
print(f"Recall:    {recall_score(test_true, final_preds, zero_division=0):.4f}")
print(f"F1 Score:  {f1_score(test_true, final_preds, zero_division=0):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(test_true, final_preds))
