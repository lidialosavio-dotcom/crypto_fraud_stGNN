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
    print(f"Seed fissato a {seed}.")

set_seed(99)  # Puoi provare anche 66 o 42 per vedere se ora è più stabile

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ----------------------------
# 2) CONFIG
# ----------------------------
SPLIT_TRAIN = 0.60
SPLIT_VAL   = 0.20
SPLIT_TEST  = 0.20
EMBARGO_STEPS = 5

WINDOW_SIZE = 5

HIDDEN_CHANNELS = 64
HEADS = 2
LEARNING_RATE = 0.001
EPOCHS = 50
DROPOUT = 0.4
BATCH_SIZE = 64

# >>> threshold fisso per la classificazione finale
FIXED_THRESHOLD = 0.5

# ----------------------------
# SELF-ADAPTIVE ADJACENCY CONFIG
# ----------------------------
ADP_DIM = 32
ADP_USE_RELU = True
ADP_ADD_SELF_LOOPS = False
ADP_GRAPH_THRESHOLD = 0.015  # <--- NUOVO: Tiengo archi solo se peso > 1%

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
file_path = r"/home/lidialosav/pump-and-dump-dataset/project/hourly_pump&dump_15112025.csv"

print("\n--- Caricamento Dati ---")
if not os.path.exists(file_path):
    raise FileNotFoundError(f"Path non trovato: {file_path}")

df = pd.read_csv(file_path, delimiter=",")
df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.sort_values(DATE_COL).reset_index(drop=True)
df[LABEL_COL] = df[LABEL_COL].astype(int).clip(0, 1)

unique_symbols = df[GROUP_COL].unique()
symbol_to_idx = {sym: i for i, sym in enumerate(unique_symbols)}
idx_to_symbol = {i: sym for sym, i in symbol_to_idx.items()}  # <<< ADDED (per metriche per token)
num_nodes = len(unique_symbols)
print(f"Dataset: {len(df)} righe, {num_nodes} token unici.")

# ----------------------------
# 4) FEATURES (RAW)
# ----------------------------
print("\n--- Analisi Features ---")
raw_feature_cols = [
    c for c in df.columns
    if c not in DROP_COLS_TRAIN and np.issubdtype(df[c].dtype, np.number)
]
print(f"Features Selezionate ({len(raw_feature_cols)}): {raw_feature_cols}")
df_proc = df.copy()

# ----------------------------
# 5) SPLIT TEMPORALE
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
# 6) SCALING + WINDOWING
# ----------------------------
print("\n--- Scaling & Creazione Finestre Temporali ---")

df_train_subset = df_proc[df_proc[DATE_COL].isin(dates_train)]
scaler = StandardScaler()
scaler.fit(df_train_subset[raw_feature_cols].fillna(0))

# Dummy edge index per inizializzare l'oggetto Data (il modello lo ignorerà e userà quello adattivo)
dummy_edge_index = torch.tensor([list(range(num_nodes)), list(range(num_nodes))], dtype=torch.long)
dummy_edge_attr  = torch.ones(num_nodes, dtype=torch.float)

def get_temporal_snapshots(target_dates, window_size=WINDOW_SIZE):
    snapshots = []
    all_dates_list = df_proc[DATE_COL].unique()

    # buffer per riempire la prima finestra
    min_date_idx = np.searchsorted(all_dates_list, target_dates[0])
    relevant_start_idx = max(0, min_date_idx - window_size + 1)
    relevant_dates = all_dates_list[
        relevant_start_idx : np.searchsorted(all_dates_list, target_dates[-1]) + 1
    ]

    subset = df_proc[df_proc[DATE_COL].isin(relevant_dates)].copy()
    subset[raw_feature_cols] = scaler.transform(subset[raw_feature_cols].fillna(0))
    grouped = subset.groupby(DATE_COL)

    date_to_matrix = {}
    for date, group in grouped:
        mat = np.zeros((num_nodes, len(raw_feature_cols)), dtype=np.float32)
        indices = [symbol_to_idx[s] for s in group[GROUP_COL]]
        mat[indices] = group[raw_feature_cols].values.astype(np.float32)
        date_to_matrix[date] = mat

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

        data = Data(
            x=torch.tensor(x_seq, dtype=torch.float),
            y=torch.tensor(y_t, dtype=torch.float),
            mask=torch.tensor(mask_t, dtype=torch.bool),
            edge_index=dummy_edge_index,
            edge_attr=dummy_edge_attr
        )
        snapshots.append(data)

    return snapshots

train_snapshots = get_temporal_snapshots(dates_train)
val_snapshots   = get_temporal_snapshots(dates_val)
test_snapshots  = get_temporal_snapshots(dates_test)

train_loader = DataLoader(train_snapshots, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_snapshots,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_snapshots,  batch_size=BATCH_SIZE, shuffle=False)

# ----------------------------
# 7) MODELLO (Self-adaptive Graph NO TOPK -> Threshold)
# ----------------------------
class TemporalGraphModelSelfAdaptive(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, num_nodes,
                 heads=HEADS, window_size=WINDOW_SIZE,
                 adp_dim=ADP_DIM, adp_threshold=ADP_GRAPH_THRESHOLD):
        super().__init__()
        self.num_nodes = num_nodes
        self.adp_threshold = adp_threshold

        # Inizializzazione Xavier per maggiore stabilità
        self.E1 = nn.Parameter(torch.empty(num_nodes, adp_dim))
        self.E2 = nn.Parameter(torch.empty(num_nodes, adp_dim))
        nn.init.xavier_uniform_(self.E1)
        nn.init.xavier_uniform_(self.E2)

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
        self.pos_embedding = nn.Parameter(torch.randn(1, window_size, hidden_channels))

    def build_adaptive_graph_single(self):
        # 1. Calcolo Scores: E1 @ E2.T  [N, N]
        scores = self.E1 @ self.E2.t()
        
        # 2. ReLU (opzionale, ma aiuta a pulire i negativi)
        if ADP_USE_RELU:
            scores = torch.relu(scores)

        # 3. Softmax per riga: Otteniamo pesi probabilistici
        A = torch.softmax(scores, dim=1)

        # 4. >>> RIMOZIONE TOPK <<<
        # Invece di prendere i K migliori, prendiamo TUTTI quelli che superano la soglia.
        # Questo permette al gradiente di fluire meglio e non "taglia" connessioni buone 
        # solo perché sono al posto K+1.
        if ADP_USE_RELU:
            mask = A > self.adp_threshold
        else:
            mask = (A > self.adp_threshold) | (A < self.adp_threshold) 
        
        # Convertiamo la matrice densa mascherata in formato sparso (edge_index)
        # indices sarà [2, Num_Edges_Sopra_Soglia]
        indices = mask.nonzero(as_tuple=False).t()
        weights = A[mask]

        edge_index = indices
        edge_attr  = weights

        # 5. Self loops (opzionale ma consigliato per GNN)
        if ADP_ADD_SELF_LOOPS:
            sl = torch.arange(self.num_nodes, device=A.device)
            sl_edge_index = torch.stack([sl, sl], dim=0)
            sl_edge_attr  = torch.ones(self.num_nodes, device=A.device, dtype=torch.float)

            edge_index = torch.cat([edge_index, sl_edge_index], dim=1)
            edge_attr  = torch.cat([edge_attr, sl_edge_attr], dim=0)

        return edge_index, edge_attr

    def replicate_graph_for_batch(self, edge_index, edge_attr, batch_size):
        n = self.num_nodes
        E = edge_index.size(1)
        
        # Se E=0 (nessun arco sopra soglia), gestiamo il crash creando dummy self-loops per tutto il batch
        if E == 0:
            dummy_src = torch.arange(n * batch_size, device=edge_index.device)
            dummy_dst = dummy_src
            edge_index_b = torch.stack([dummy_src, dummy_dst], dim=0)
            edge_attr_b = torch.ones(n * batch_size, device=edge_index.device)
            return edge_index_b, edge_attr_b

        offsets = (torch.arange(batch_size, device=edge_index.device) * n).view(-1, 1)

        src = edge_index[0].view(1, E).repeat(batch_size, 1) + offsets
        dst = edge_index[1].view(1, E).repeat(batch_size, 1) + offsets
        edge_index_b = torch.stack([src.reshape(-1), dst.reshape(-1)], dim=0)

        edge_attr_b = edge_attr.view(1, E).repeat(batch_size, 1).reshape(-1)
        return edge_index_b, edge_attr_b

    def forward(self, x):
        batch_nodes, window, feats = x.size()
        B = batch_nodes // self.num_nodes

        # Costruisci grafo dinamico (Threshold-based)
        eidx, eattr = self.build_adaptive_graph_single()
        eidx, eattr = self.replicate_graph_for_batch(eidx, eattr, batch_size=B)
        edge_attr = eattr.view(-1, 1)

        temporal_embeddings = []
        for t in range(window):
            x_t = self.lin_in(x[:, t, :])
            h = self.conv1(x_t, eidx, edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=DROPOUT, training=self.training)
            h = self.conv2(h, eidx, edge_attr)
            h = F.relu(h)
            temporal_embeddings.append(h)

        sequence = torch.stack(temporal_embeddings, dim=1)
        sequence = sequence + self.pos_embedding

        time_out = self.temporal_transformer(sequence)
        last_step = time_out[:, -1, :]
        out = self.out(last_step)
        return torch.sigmoid(out).view(-1)

model = TemporalGraphModelSelfAdaptive(
    num_features=len(raw_feature_cols),
    hidden_channels=HIDDEN_CHANNELS,
    num_nodes=num_nodes
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()

@torch.no_grad()
def evaluate_f1(loader, thresh=0.5):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0.0

    for batch in loader:
        batch = batch.to(device)
        probs = model(batch.x)
        loss = criterion(probs[batch.mask], batch.y[batch.mask])
        total_loss += float(loss.item())

        preds = (probs >= thresh).float()
        all_preds.extend(preds[batch.mask].cpu().numpy())
        all_true.extend(batch.y[batch.mask].cpu().numpy())

    f1 = f1_score(all_true, all_preds, zero_division=0) if len(all_true) else 0.0
    avg_loss = total_loss / max(1, len(loader))
    return avg_loss, f1

@torch.no_grad()
def collect_probs_and_true(loader):
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

pbar = tqdm(range(EPOCHS), desc="Training", unit="epoch")
for epoch in pbar:
    model.train()
    train_loss = 0.0

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        probs = model(batch.x)
        loss = criterion(probs[batch.mask], batch.y[batch.mask])

        loss.backward()
        optimizer.step()
        train_loss += float(loss.item())

    avg_train_loss = train_loss / max(1, len(train_loader))

    val_loss, val_f1 = evaluate_f1(val_loader, thresh=FIXED_THRESHOLD)
    _, test_f1 = evaluate_f1(test_loader, thresh=FIXED_THRESHOLD)

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
# 9) TEST FINALE
# ----------------------------
print("\n\n--- Risultati Finali (Pure Softmax Graph) ---")
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
# 10) METRICHE PER TOKEN (solo token con pump nel TEST >= 3)  <<< ADDED
# ----------------------------
@torch.no_grad()
def collect_per_token_predictions(loader, thresh=0.5):
    model.eval()
    token_to_true = {}
    token_to_pred = {}

    for batch in loader:
        batch = batch.to(device)
        probs = model(batch.x)
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