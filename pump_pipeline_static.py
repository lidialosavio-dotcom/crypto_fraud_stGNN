




# # #GNN + GRU 
# # -*- coding: utf-8 -*-
# # """
# # Pump-and-Dump Detection: Spatio-Temporal Graph Transformer (GNN + GRU)
# # """
import sys
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn.functional as F
import random
from torch.nn import Linear, GRU
from torch_geometric.nn import TransformerConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import copy
import os
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.csgraph import connected_components
from sklearn.neighbors import NearestNeighbors
from torch_geometric.utils import to_scipy_sparse_matrix

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve
)

warnings.filterwarnings("ignore")

# --- 1. SEED FISSO ---
def set_seed(seed=44):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed fissato a {seed}.")

set_seed(99)

# --- 2. CONFIGURAZIONE ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Split temporale
SPLIT_TRAIN = 0.60
SPLIT_VAL   = 0.20
SPLIT_TEST  = 0.20
EMBARGO_STEPS = 5 

# Temporal Config (NUOVO)
WINDOW_SIZE = 5  # Guardiamo le 5 ore precedenti (inclusa l'attuale)

# Graph Config
MIN_CORR_THRESHOLD = 0.15 
# Where is target density used?
TARGET_DENSITY = 0.95 

# Model Hyperparameters
# How can we select optimally these hyperparameters?
HIDDEN_CHANNELS = 64
HEADS = 2            
LEARNING_RATE = 0.001 
EPOCHS = 50          
DROPOUT = 0.4
BATCH_SIZE = 64 # Riduco leggermente il batch size perché i dati ora sono più pesanti (x5)

# Columns
DATE_COL   = "date"
LABEL_COL  = "flag"
GROUP_COL  = "symbol"
DROP_COLS_TRAIN  = [DATE_COL, LABEL_COL, GROUP_COL, 
                    'high', 'low', 'close', 'group', 
                    'log_ret', 'ret_BTC', 'vola_BTC']

## --- 3. CARICAMENTO DATI ---
file_path = r"hourly_pump&dump_15112025.csv"

print("\n--- Caricamento Dati ---")
try:
    df = pd.read_csv(file_path, delimiter=',')
except FileNotFoundError:
    raise FileNotFoundError(f"File non trovato: {file_path}")

df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
df = df.sort_values(DATE_COL).reset_index(drop=True)
df[LABEL_COL] = df[LABEL_COL].astype(int).clip(0, 1)

unique_symbols = df[GROUP_COL].unique()
symbol_to_idx = {sym: i for i, sym in enumerate(unique_symbols)}
num_nodes = len(unique_symbols)
print(f"Dataset: {len(df)} righe, {num_nodes} token unici.")

## --- 4. GESTIONE FEATURES (RAW - NO LOG) ---
print("\n--- Analisi Features ---")
raw_feature_cols = [c for c in df.columns if c not in DROP_COLS_TRAIN and np.issubdtype(df[c].dtype, np.number)]
print(f"Features Selezionate ({len(raw_feature_cols)}): {raw_feature_cols}")
df_proc = df.copy() # Usiamo feature RAW come richiesto

## --- 5. SPLITTING ---
unique_dates = df[DATE_COL].unique()
n_dates = len(unique_dates)
train_end_idx = int(SPLIT_TRAIN * n_dates)
val_end_idx   = int((SPLIT_TRAIN + SPLIT_VAL) * n_dates)

dates_train = unique_dates[:train_end_idx]
dates_val   = unique_dates[train_end_idx + EMBARGO_STEPS : val_end_idx]
dates_test  = unique_dates[val_end_idx + EMBARGO_STEPS :]

print(f"Train: {len(dates_train)} h | Val: {len(dates_val)} h | Test: {len(dates_test)} h")

## --- 6. COSTRUZIONE GRAFO (STRUTTURA) ---
def Graph_construction(method_graph=0, kNN=10):

    # Keep the training data    
    df_structure = df[df[DATE_COL].isin(dates_train)].copy()
    # Pivot table for volume
    vol_pivot = df_structure.pivot_table(index=DATE_COL, columns=GROUP_COL, values='volume').fillna(0)
    # Reindec
    vol_pivot = vol_pivot.reindex(columns=unique_symbols, fill_value=0) 
    # Get the log before computing correlation
    vol_log = np.log1p(vol_pivot)

    if method_graph == 0:
        print("--- Graph construction based on correlation of volume, method = {method_graph} ---")
        corr_matrix = vol_log.corr(method='pearson').fillna(0)
        sim_matrix = corr_matrix.values
        
    elif method_graph == 1 or method_graph == 2:
        print(f"--- Graph construction based on similarity of all features, method = {method_graph}, kNN = {kNN} ---")
        # Gaussian similarity, according to: Zelnik-Manor and Perona (2005)
        
        # Feature matrix
        # Pts: [N_nodes, Time * N_features]
        feature_matrices = []
        
        for feat in raw_feature_cols:
            # Pivot table: [Time, Nodes]
            feat_pivot = df_structure.pivot_table(index=DATE_COL, columns=GROUP_COL, values=feat).fillna(0)
            feat_pivot = feat_pivot.reindex(columns=unique_symbols, fill_value=0)
            
            # Scaling (z-score) across time and nodes
            # Subtract the mean and divide by the standard deviation (all global)
            # I think we can do this with the StandardScaler as well, gotta check
            # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
            vals = feat_pivot.values
            vals_scaled = (vals - np.mean(vals)) / (np.std(vals) + 1e-12)
            
            # Transpose to [Nodes, Time]
            feature_matrices.append(vals_scaled.T)
            
        # Concatenate features: [Nodes, Time * N_features]
        Pts = np.concatenate(feature_matrices, axis=1)
        
        # sklearn NearestNeighbors
        # We need kNN neighbors + 1 (self) to find the k-th neighbor
        # Let's think more about which distance to use 
        # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html
        nbrs = NearestNeighbors(n_neighbors=kNN + 1, algorithm='auto', metric='euclidean').fit(Pts)
        distances, indices = nbrs.kneighbors(Pts)
        
        # Sigma scaling acc. to the distance to the k-th neighbor
        # indices[:, 0] is self, indices[:, kNN] is the k-th neighbor
        sigmas = distances[:, kNN]
        
        if method_graph == 1:
            # Collect all unique edges found by kNN (Symmetric)
            edge_dict = {} # (u, v) -> distance
            
            for i in range(num_nodes):
                for k in range(1, kNN + 1):
                    j = indices[i, k]
                    d = distances[i, k]
                    
                    if i < j:
                        edge_dict[(i, j)] = d
                    else:
                        edge_dict[(j, i)] = d
            
            sources, targets, weights = [], [], []
            
            for (u, v), d in edge_dict.items():
                sigma_edge = max(sigmas[u], sigmas[v])
                if sigma_edge == 0: sigma_edge = 1e-12
                
                w = np.exp(-4 * (d**2) / (sigma_edge**2))
                
                sources.extend([u, v])
                targets.extend([v, u])
                weights.extend([w, w])

        elif method_graph == 2:
            # Asymmetric (Directed) without forcing symmetry
            sources, targets, weights = [], [], []
            
            for i in range(num_nodes):
                for k in range(1, kNN + 1):
                    j = indices[i, k]
                    d = distances[i, k]
                    
                    # For directed graph: source=i, target=j
                    # Weight depends on source's capacity (sigmas[i])
                    
                    sigma_edge = sigmas[i] 
                    if sigma_edge == 0: sigma_edge = 1e-12
                    
                    w = np.exp(-4 * (d**2) / (sigma_edge**2))
                    
                    sources.append(i)
                    targets.append(j)
                    weights.append(w)

    # Build the edge list
    # For method 0 (correlation), we have the sim_matrix, for method 1 (kNN), we have the edge list.
    
    if method_graph == 0:
        np.fill_diagonal(sim_matrix, 0)
        percentile_thresh = np.percentile(sim_matrix.flatten(), TARGET_DENSITY * 100)
        effective_thresh = max(MIN_CORR_THRESHOLD, percentile_thresh)

        sources, targets, weights = [], [], []
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                val = sim_matrix[i, j]
                if val > effective_thresh:
                    sources.extend([i, j])
                    targets.extend([j, i])
                    weights.extend([val, val])

    if not sources:
        sources, targets = list(range(num_nodes)), list(range(num_nodes))
        weights = [1.0] * num_nodes

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_attr  = torch.tensor(weights, dtype=torch.float)
    
    return edge_index, edge_attr



# method_graph = 0 (correlation) or 1 (kNN symmetric) or 2 (kNN asymmetric)
method_graph = 2
kNN = 1
edge_index, edge_attr = Graph_construction(method_graph, kNN)
print("========================================================================")
print(f"Adjacency matrix size [nodes x nodes]: [{num_nodes}, {num_nodes}]")
print(f"Total number of edges: {edge_index.shape[1]} for method: {method_graph}")

# Check connectivity
adj = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)
# For directed graphs (method 2), check weak connectivity (no isolated nodes)
is_directed = (method_graph == 2)
n_components, labels = connected_components(adj, connection='weak', directed=is_directed)
print(f"Graph directed: {is_directed}")
print(f"Graph connected (weak): {n_components == 1}")
print(f"Number of connected components: {n_components}")
print("========================================================================")


sys.exit("Done with graph construction")

## --- 7. SCALING & SEQUENCING (NUOVO: WINDOWING) ---
print("\n--- Scaling & Creazione Finestre Temporali ---")

# Fit Scaler solo su TRAIN
df_train_subset = df_proc[df_proc[DATE_COL].isin(dates_train)]
scaler = StandardScaler()
scaler.fit(df_train_subset[raw_feature_cols].fillna(0))

def get_temporal_snapshots(target_dates, window_size=WINDOW_SIZE):
    """
    Crea snapshot contenenti sequenze temporali.
    Ogni oggetto Data avrà x di forma [Num_Nodes, Window_Size, Features]
    """
    snapshots = []
    
    # Prendi tutti i dati necessari (inclusi un po' di dati PRIMA della prima data target per riempire la finestra)
    # Troviamo l'indice della prima data nel df globale
    first_date = target_dates[0]
    start_idx_df = df_proc[df_proc[DATE_COL] == first_date].index[0]
    
    # Dobbiamo tornare indietro di (window_size - 1) ore per avere la storia completa del primo elemento
    # Se non abbiamo abbastanza storia, padderemo con zeri, ma idealmente prendiamo dal df completo
    
    # Semplifichiamo: lavoriamo sulle date uniche
    all_dates_list = df_proc[DATE_COL].unique()
    target_dates_set = set(target_dates)
    
    # Prepariamo un dizionario {data: matrice_features_scalate}
    date_to_matrix = {}
    
    # Pre-scaliamo tutto il necessario per velocità
    # Consideriamo le date target + un buffer precedente
    min_date_idx = np.searchsorted(all_dates_list, target_dates[0])
    relevant_start_idx = max(0, min_date_idx - window_size + 1)
    relevant_dates = all_dates_list[relevant_start_idx : np.searchsorted(all_dates_list, target_dates[-1]) + 1]
    
    subset = df_proc[df_proc[DATE_COL].isin(relevant_dates)].copy()
    subset[raw_feature_cols] = scaler.transform(subset[raw_feature_cols].fillna(0))
    grouped = subset.groupby(DATE_COL)
    
    for date, group in grouped:
        mat = np.zeros((num_nodes, len(raw_feature_cols)))
        indices = [symbol_to_idx[s] for s in group[GROUP_COL]]
        mat[indices] = group[raw_feature_cols].values
        date_to_matrix[date] = mat

    # Creazione Snapshot Sequenziali
    for i, date in enumerate(tqdm(target_dates, desc="Generating Windows")):
        # Trova l'indice di questa data nella lista completa delle date
        curr_idx = np.where(all_dates_list == date)[0][0]
        
        # Costruisci la finestra temporale [t - window + 1 : t + 1]
        window_matrices = []
        for w in range(window_size):
            lookback_idx = curr_idx - (window_size - 1) + w
            if lookback_idx >= 0:
                d = all_dates_list[lookback_idx]
                if d in date_to_matrix:
                    window_matrices.append(date_to_matrix[d])
                else:
                     # Se manca la data (es. buco nel dataset), usa zeri
                    window_matrices.append(np.zeros((num_nodes, len(raw_feature_cols))))
            else:
                # Padding iniziale se siamo all'inizio assoluto del dataset
                window_matrices.append(np.zeros((num_nodes, len(raw_feature_cols))))
        
        # Stack lungo la dimensione temporale -> [Num_Nodes, Window, Features]
        x_seq = np.stack(window_matrices, axis=1) 
        
        # Recupera label e mask per l'ora CORRENTE (l'ultima della finestra)
        # Nota: dobbiamo recuperare i label originali (non scalati)
        current_group = df[df[DATE_COL] == date]
        y_t = np.zeros(num_nodes)
        mask_t = np.zeros(num_nodes)
        indices = [symbol_to_idx[s] for s in current_group[GROUP_COL]]
        y_t[indices] = current_group[LABEL_COL].values
        mask_t[indices] = 1.0
        
        data = Data(
            x=torch.tensor(x_seq, dtype=torch.float), # [N, W, F]
            y=torch.tensor(y_t, dtype=torch.float),
            mask=torch.tensor(mask_t, dtype=torch.bool),
            edge_index=edge_index, 
            edge_attr=edge_attr
        )
        snapshots.append(data)
        
    return snapshots

train_snapshots = get_temporal_snapshots(dates_train)
val_snapshots   = get_temporal_snapshots(dates_val)
test_snapshots  = get_temporal_snapshots(dates_test)

train_loader = DataLoader(train_snapshots, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_snapshots,   batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_snapshots,  batch_size=BATCH_SIZE, shuffle=False)


class TemporalGraphModel(torch.nn.Module):
    def __init__(self, num_features, hidden_channels, heads=HEADS, window_size=WINDOW_SIZE):
        super().__init__()
        
        # 1. GNN (Spazio) - Resta uguale
        self.lin_in = Linear(num_features, hidden_channels)
        self.conv1 = TransformerConv(hidden_channels, hidden_channels, heads=heads, edge_dim=1)
        self.conv2 = TransformerConv(hidden_channels*heads, hidden_channels, heads=1, edge_dim=1)
        
        # 2. TEMPORAL TRANSFORMER (Sostituisce la GRU)
        # Questo layer impara le relazioni tra i 5 step temporali usando l'attenzione
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=hidden_channels, 
            nhead=8,                # 4 teste di attenzione temporale
            dim_feedforward=128, 
            dropout=0.4,
            batch_first=True        # Input shape: [Batch, Time, Feats]
        )
        self.temporal_transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 3. Output (si usa spesso un Global Average Pooling o solo l'ultimo step)
        self.out = Linear(hidden_channels, 1)

        # Positional Encoding (Opzionale ma utile per i transformer)
        self.pos_embedding = torch.nn.Parameter(torch.randn(1, window_size, hidden_channels))

    def forward(self, x, edge_index, edge_weight):
        batch_nodes, window, feats = x.size()
        edge_attr = edge_weight.view(-1, 1)
        
        # --- SPACIAL ATTENTION (GNN) ---
        temporal_embeddings = []
        for t in range(window):
            x_t = x[:, t, :]
            x_t = self.lin_in(x_t)
            h = F.relu(self.conv1(x_t, edge_index, edge_attr))
            h = F.dropout(h, p=DROPOUT, training=self.training)
            h = F.relu(self.conv2(h, edge_index, edge_attr))
            temporal_embeddings.append(h)
            
        # Stack: [Batch, Window, Hidden]
        sequence = torch.stack(temporal_embeddings, dim=1)
        
        # Aggiungo Positional Encoding (per dire al Transformer l'ordine 1,2,3,4,5)
        sequence = sequence + self.pos_embedding
        
        # --- TEMPORAL ATTENTION (Transformer) ---
        # Capisce le relazioni temporali complesse
        # Output shape: [Batch, Window, Hidden]
        time_out = self.temporal_transformer(sequence)
        
        # Prendo l'ultimo step (o faccio la media)
        # out = self.out(last_step_out)
       #return torch.sigmoid(out).squeeze()
        last_step = time_out[:, -1, :]
        out = self.out(last_step)
        
        return torch.sigmoid(out).squeeze()

model = TemporalGraphModel(len(raw_feature_cols), HIDDEN_CHANNELS).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.BCELoss()

## --- 9. TRAINING LOOP ---
def evaluate_set(loader, thresh=0.5):
    model.eval()
    all_preds, all_true = [], []
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            # x ha dimensione [N, Window, F]
            out = model(batch.x, batch.edge_index, batch.edge_attr)
            loss = criterion(out[batch.mask], batch.y[batch.mask])
            total_loss += loss.item()
            
            preds = (out > thresh).float()
            valid_mask = batch.mask
            all_preds.extend(preds[valid_mask].cpu().numpy())
            all_true.extend(batch.y[valid_mask].cpu().numpy())
            
    return total_loss / len(loader), f1_score(all_true, all_preds, zero_division=0)

print(f"\n--- Start Training GNN+GRU ({EPOCHS} epochs) ---")
best_val_f1 = -1.0
best_model_state = None
pbar = tqdm(range(EPOCHS), desc="Training", unit="epoch")

for epoch in pbar:
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        
        loss = criterion(out[batch.mask], batch.y[batch.mask])
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)
    val_loss, val_f1 = evaluate_set(val_loader)
    _, test_f1 = evaluate_set(test_loader) 
    
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = copy.deepcopy(model.state_dict())
    
    pbar.set_postfix({
        'L': f"{avg_train_loss:.3f}", 
        'Val': f"{val_f1:.3f}", 
        'Tst': f"{test_f1:.3f}", 
        'Best': f"{best_val_f1:.3f}"
    })

## --- 10. TEST FINALE ---
print("\n\n--- Risultati Finali Spazio-Temporali ---")
if best_model_state:
    model.load_state_dict(best_model_state)

model.eval()
test_probs, test_true = [], []
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_attr)
        test_probs.extend(out[batch.mask].cpu().numpy())
        test_true.extend(batch.y[batch.mask].cpu().numpy())

test_probs = np.array(test_probs)
test_true = np.array(test_true)

prec, rec, thresholds = precision_recall_curve(test_true, test_probs)
f1_scores = 2 * (prec * rec) / (prec + rec + 1e-12)
best_idx = np.argmax(f1_scores)
best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
final_preds = (test_probs >= best_thresh).astype(int)

print(f"Soglia Ottimale: {best_thresh:.4f}")
print(f"Precision: {precision_score(test_true, final_preds):.4f}")
print(f"Recall:    {recall_score(test_true, final_preds):.4f}")
print(f"F1 Score:  {f1_score(test_true, final_preds):.4f}")
print("Confusion Matrix:")
print(confusion_matrix(test_true, final_preds))
