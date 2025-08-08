# =========================
# UFGAT7.py  —  高效版整合
# =========================

import os
import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    accuracy_score, classification_report, confusion_matrix
)
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import pandas as pd
import igraph as ig
from tqdm import tqdm

# ---- 底層平行度：可依 CPU 核心數微調 ----
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["OPENBLAS_NUM_THREADS"] = "8"
os.environ["MKL_NUM_THREADS"] = "8"
os.environ["NUMEXPR_NUM_THREADS"] = "8"

# =========================
# 參數區
# =========================
file_path = r"C:\Users\Leon\Desktop\程式語言資料\python\UF-FAE\SAML-D.parquet"  # 交易資料集
GRAPH_FEATURE_NODE_THRESHOLD = 3
SEQ_LEN = 20                    # LSTM 時序步數
LSTM_INPUT_DIM = 17             # 對應 get_tx_features 的特徵維度
LSTM_HIDDEN = 32
EMBED_DIM = 16
GAT_HIDDEN = 64
EPOCHS = 10
TRIPLET_SAMPLES = 64
TRIPLET_MARGIN = 1.0
ALPHA = 0.1
THR_MIN = 0.1
THR_MAX = 1.0
THR_STEP = 0.01

# ---- 圖特徵加速控制 ----
USE_CENTRALITY = False     # 先關掉中心性，要用再打開（True）
CENT_SIZE_MIN = 300        # 群組節點 >= 才算中心性
CENT_UPDATE_EVERY = 10000  # 每群組每處理多少條邊，才重算一次中心性

# =========================
# UF-FAE 動態分群核心
# =========================
class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        merged = (xr != yr)
        if merged:
            self.parent[yr] = xr
        return merged

# =========================
# LSTM 群組嵌入
# =========================
class GroupLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, emb_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, emb_dim)
    def forward(self, x_seq):
        output, (h_n, _) = self.lstm(x_seq)
        emb = self.fc(h_n[-1])
        return emb

# =========================
# GAT 多模態圖模型
# =========================
class UFGAT(nn.Module):
    def __init__(self, in_node, in_edge, hidden=64):
        super().__init__()
        self.gat1 = GATConv(in_node, hidden, heads=1)
        self.gat2 = GATConv(hidden, hidden, heads=1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden * 2 + in_edge, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x, edge_index, edge_attr):
        h = F.relu(self.gat1(x, edge_index))
        h = F.relu(self.gat2(h, edge_index))
        h_u, h_v = h[edge_index[0]], h[edge_index[1]]
        edge_input = torch.cat([h_u, h_v, edge_attr], dim=-1)
        return self.mlp(edge_input).squeeze(), h

# =========================
# Triplet Loss
# =========================
def triplet_contrastive_loss(node_emb, triplet_idx, margin=1.0):
    if len(triplet_idx) == 0:
        return torch.tensor(0.0, device=node_emb.device)
    anchor_idx, pos_idx, neg_idx = zip(*triplet_idx)
    anchor = node_emb[list(anchor_idx)]
    positive = node_emb[list(pos_idx)]
    negative = node_emb[list(neg_idx)]
    pos_dist = torch.norm(anchor - positive, p=2, dim=-1)
    neg_dist = torch.norm(anchor - negative, p=2, dim=-1)
    loss = torch.relu(pos_dist - neg_dist + margin).mean()
    return loss

# =========================
# 群組統計快取（增量 O(1) / 降頻中心性）
# =========================
group_stats = defaultdict(lambda: {
    "edges": 0,
    "bidirect": 0,
    "last_centrality_update": -1,
    "closeness": None,
    "betweenness": None
})

# =========================
# 交易＋圖論特徵（增量＋快取）
# =========================
def get_tx_features(row, G=None, s=None, r=None, node_map=None,
                    node_threshold=GRAPH_FEATURE_NODE_THRESHOLD, stats=None):
    # 基礎交易特徵
    features = [
        np.log1p(float(row["Amount"])),
        1 if row["Payment_type"] == "Cash Deposit" else 0,
        1 if row["Payment_type"] == "Credit card" else 0,
        1 if row["Payment_type"] == "Cross-border" else 0,
        1 if row["Payment_type"] == "Cheque" else 0,
        float(str(row["Sender_account"])[-4:]) % 10000 / 10000,
        float(str(row["Receiver_account"])[-4:]) % 10000 / 10000,
        float(row["Amount"]) / 100000
    ]

    # 預設補零
    s_deg = r_deg = s_close = r_close = s_between = r_between = 0.0
    group_size = group_edge_count = 0
    bidirect_ratio = 0.0

    if G is not None and G.vcount() >= 3 and node_map is not None:
        group_size = G.vcount()
        group_edge_count = G.ecount()

        if group_size > node_threshold:
            # 映射 idx
            try:
                s_idx = node_map[s]
                r_idx = node_map[r]
            except KeyError:
                # 新節點剛加還沒寫進 node_map
                return features + [0,0,0,0,0,0, group_size, group_edge_count, 0.0]

            # 便宜：度數每次算
            s_deg = G.degree(s_idx)
            r_deg = G.degree(r_idx)

            # 雙向邊比率：用快取（增量維護）
            if stats is not None and group_edge_count > 0:
                bidirect_ratio = stats["bidirect"] / group_edge_count

            # 中心性：僅在快取存在時讀取，否則補 0
            if stats is not None and stats.get("closeness") is not None and stats.get("betweenness") is not None:
                closeness_vals = stats["closeness"]
                betweenness_vals = stats["betweenness"]
                if 0 <= s_idx < len(closeness_vals): s_close = closeness_vals[s_idx]
                if 0 <= r_idx < len(closeness_vals): r_close = closeness_vals[r_idx]
                if 0 <= s_idx < len(betweenness_vals): s_between = betweenness_vals[s_idx]
                if 0 <= r_idx < len(betweenness_vals): r_between = betweenness_vals[r_idx]

    features += [s_deg, r_deg, s_close, r_close, s_between, r_between, group_size, group_edge_count, bidirect_ratio]
    return features

# =========================
# 1. 讀檔＋帳戶編號
# =========================
print("掃描帳戶全集...", flush=True)
df = pl.read_parquet(file_path).sort(["Date", "Time"])
account_set = set()
for row in df.iter_rows(named=True):
    account_set.add(row["Sender_account"])
    account_set.add(row["Receiver_account"])
node_idx_map = {acc: i for i, acc in enumerate(sorted(account_set))}
node_count = len(node_idx_map)
print(f"Total nodes (accounts): {node_count}", flush=True)

# =========================
# 2. UF-FAE 分群＋群組序列收集（含：增量雙向、中心性降頻）
# =========================
uf = UnionFind()
group_graphs = defaultdict(lambda: ig.Graph(directed=True))
group_node_maps = defaultdict(dict)
group_tx_seq = defaultdict(list)
edge_records = []
merge_edges, nonmerge_edges = [], []

for idx, row in enumerate(tqdm(df.iter_rows(named=True), total=df.height, desc="動態分群/特徵蒐集ING")):
    s, r = row["Sender_account"], row["Receiver_account"]
    is_laundering = int(row["Is_laundering"])
    merged = uf.union(s, r)
    gid = uf.find(s)
    G = group_graphs[gid]
    node_map = group_node_maps[gid]

    # 新帳號補點
    for acc in [s, r]:
        if acc not in node_map:
            v_idx = G.vcount()
            G.add_vertices(1)
            node_map[acc] = v_idx
    s_idx, r_idx = node_map[s], node_map[r]

    # 加邊
    G.add_edges([(s_idx, r_idx)])  # igraph 允許重複邊
    st = group_stats[gid]
    st["edges"] += 1

    # 雙向：O(1) 檢查反向是否存在（are_connected 已棄用 → are_adjacent）
    if G.are_adjacent(r_idx, s_idx):
        st["bidirect"] += 1

    # 中心性：依條件降頻重算
    if USE_CENTRALITY and G.vcount() >= CENT_SIZE_MIN:
        if st["last_centrality_update"] < 0 or (st["edges"] - st["last_centrality_update"] >= CENT_UPDATE_EVERY):
            st["closeness"] = G.closeness()
            st["betweenness"] = G.betweenness()
            st["last_centrality_update"] = st["edges"]

    # 取特徵（讀快取而非每次重算）
    feat = get_tx_features(
        row, G=G, s=s, r=r, node_map=node_map,
        node_threshold=GRAPH_FEATURE_NODE_THRESHOLD,
        stats=group_stats[gid]
    )
    group_tx_seq[gid].append(feat)
    edge_records.append([node_idx_map[s], node_idx_map[r], feat, is_laundering])

    if merged:
        merge_edges.append((node_idx_map[s], node_idx_map[r]))
    else:
        nonmerge_edges.append((node_idx_map[s], node_idx_map[r]))

print(f"總樣本數: {len(edge_records)}，正樣本: {np.sum([rec[3] for rec in edge_records])}", flush=True)

# =========================
# 3. LSTM 群組嵌入
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

group_lstm = GroupLSTM(input_dim=LSTM_INPUT_DIM, hidden_dim=LSTM_HIDDEN, emb_dim=EMBED_DIM).to(device)
group_embeddings = {}

for gid, seq in group_tx_seq.items():
    padded = np.zeros((SEQ_LEN, LSTM_INPUT_DIM))
    seq_np = np.array(seq[-SEQ_LEN:])
    padded[-len(seq_np):] = seq_np
    seq_tensor = torch.tensor(padded, dtype=torch.float).unsqueeze(0).to(device)
    with torch.no_grad():
        group_emb = group_lstm(seq_tensor)
        group_embeddings[gid] = group_emb.squeeze(0).cpu().numpy()

# =========================
# 4. 節點特徵組裝（群組嵌入）
# =========================
node_features = np.zeros((node_count, EMBED_DIM))
for acc, idx in node_idx_map.items():
    gid = uf.find(acc)
    node_features[idx] = group_embeddings.get(gid, np.zeros(EMBED_DIM))

# =========================
# 5. PyG Data
# =========================
edges = np.array([[rec[0], rec[1]] for rec in edge_records]).T
edge_features = np.array([rec[2] for rec in edge_records])
labels = np.array([rec[3] for rec in edge_records])

data = Data(
    x=torch.tensor(node_features, dtype=torch.float).to(device),
    edge_index=torch.tensor(edges, dtype=torch.long).to(device),
    edge_attr=torch.tensor(edge_features, dtype=torch.float).to(device),
    y=torch.tensor(labels, dtype=torch.float).to(device)
)

# =========================
# 6. 訓練（+Triplet）
# =========================
model = UFGAT(in_node=node_features.shape[1], in_edge=edge_features.shape[1], hidden=GAT_HIDDEN).to(device)
optimizer = torch.optim.Adam(list(model.parameters()) + list(group_lstm.parameters()), lr=0.001)

n_pos = np.sum(labels == 1)
n_neg = np.sum(labels == 0)
pos_weight = torch.tensor([n_neg / n_pos]).to(device) if n_pos > 0 else torch.tensor([1.0]).to(device)
loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

def get_triplet_samples(merge_edges, nonmerge_edges, k=TRIPLET_SAMPLES):
    triplets = []
    if len(merge_edges) == 0 or len(nonmerge_edges) == 0:
        return triplets
    sample_merges = random.choices(merge_edges, k=min(k, len(merge_edges)))
    sample_nonmerges = random.choices(nonmerge_edges, k=min(k, len(nonmerge_edges)))
    for (a, p), (n1, n2) in zip(sample_merges, sample_nonmerges):
        triplets.append((a, p, n1))
    return triplets

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    out, node_emb = model(data.x, data.edge_index, data.edge_attr)
    loss = loss_fn(out, data.y)
    triplet_idx = get_triplet_samples(merge_edges, nonmerge_edges, TRIPLET_SAMPLES)
    contrastive = triplet_contrastive_loss(node_emb, triplet_idx, margin=TRIPLET_MARGIN)
    total_loss = loss + ALPHA * contrastive
    total_loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS} Loss={total_loss.item():.4f}", flush=True)

# =========================
# 7. 推論與指標/繪圖（高效版）
# =========================
print("Inference start", flush=True)
with torch.no_grad():
    logits, _ = model(data.x, data.edge_index, data.edge_attr)
    pred = torch.sigmoid(logits).cpu().numpy()
    y_true_np = data.y.cpu().numpy()
print("Inference done", flush=True)

# AUC（排序一次）
auc_val = roc_auc_score(y_true_np, pred)

# 向量化門檻搜尋
search_grid = np.arange(THR_MIN, THR_MAX + THR_STEP, THR_STEP)
thrs = search_grid[:, None]                          # [T,1]
pred_bin_mat = (pred[None, :] > thrs).astype(np.uint8)  # [T,N]
y = y_true_np[None, :].astype(np.uint8)

TP = np.sum((pred_bin_mat == 1) & (y == 1), axis=1)
FP = np.sum((pred_bin_mat == 1) & (y == 0), axis=1)
TN = np.sum((pred_bin_mat == 0) & (y == 0), axis=1)
FN = np.sum((pred_bin_mat == 0) & (y == 1), axis=1)

precision_arr = np.where(TP+FP>0, TP/(TP+FP), 0.0)
recall_arr    = np.where(TP+FN>0, TP/(TP+FN), 0.0)
f1_arr        = np.where(precision_arr+recall_arr>0, 2*precision_arr*recall_arr/(precision_arr+recall_arr), 0.0)
acc_arr       = (TP+TN)/(TP+FP+TN+FN)

best_idx = int(np.argmax(f1_arr))
best_thr = search_grid[best_idx]
best_pred_label = (pred > best_thr).astype(np.uint8)

# 最終整體指標
f1_val = f1_score(y_true_np, best_pred_label)
precision_val = precision_score(y_true_np, best_pred_label, zero_division=0)
recall_val = recall_score(y_true_np, best_pred_label, zero_division=0)
acc_val = accuracy_score(y_true_np, best_pred_label)

print("\n--- GAT (UF-FAE + LSTM + 圖論特徵) 分群全融合指標 ---")
print(f"Best Threshold (max F1): {best_thr:.4f}")
print("AUC:      ", auc_val)
print("F1:       ", f1_val)
print("Precision:", precision_val)
print("Recall:   ", recall_val)
print("Accuracy: ", acc_val)

print("\n=== 各類別詳細指標（Best Threshold） ===")
target_names = ["Class 0 (非洗錢)", "Class 1 (洗錢)"]
print(classification_report(y_true_np, best_pred_label, target_names=target_names, digits=4))
print("\n=== 混淆矩陣 ===")
print(confusion_matrix(y_true_np, best_pred_label))

# per-class sweep（0/1）
prec_1 = np.where(TP+FP>0, TP/(TP+FP), 0.0)
rec_1  = np.where(TP+FN>0, TP/(TP+FN), 0.0)
f1_1   = np.where(prec_1+rec_1>0, 2*prec_1*rec_1/(prec_1+rec_1), 0.0)

prec_0 = np.where(TN+FN>0, TN/(TN+FN), 0.0)
rec_0  = np.where(TN+FP>0, TN/(TN+FP), 0.0)
f1_0   = np.where(prec_0+rec_0>0, 2*prec_0*rec_0/(prec_0+rec_0), 0.0)

result_df = pd.DataFrame({
    "threshold": search_grid,
    "Precision_0": prec_0, "Recall_0": rec_0, "F1_0": f1_0,
    "Precision_1": prec_1, "Recall_1": rec_1, "F1_1": f1_1,
    "Accuracy": acc_arr
})
result_df.to_csv("threshold_sweep_per_class.csv", index=False)
print("每個門檻下各類別指標已存檔：threshold_sweep_per_class.csv", flush=True)

# 近似 ROC（分箱法）
print("Approx ROC (binned) start", flush=True)
bins = 512
edges = np.linspace(0.0, 1.0, bins+1)
pos_mask = (y_true_np == 1)
neg_mask = ~pos_mask
pos_hist, _ = np.histogram(pred[pos_mask], bins=edges)
neg_hist, _ = np.histogram(pred[neg_mask], bins=edges)
tpr = np.cumsum(pos_hist[::-1]) / max(pos_hist.sum(), 1)
fpr = np.cumsum(neg_hist[::-1]) / max(neg_hist.sum(), 1)
tpr = tpr[::-1]; fpr = fpr[::-1]

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label='Approx ROC (binned)')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (binned approx)")
plt.legend(loc='lower right')
metrics_text = f"AUC={auc_val:.3f}\nF1={f1_val:.3f}\nPre={precision_val:.3f}\nRec={recall_val:.3f}\nThr={best_thr:.3f}"
plt.gca().text(0.02, 0.98, metrics_text, fontsize=13, va='top', ha='left',
               bbox=dict(facecolor='white', alpha=0.7))
plt.savefig("roc_full.png")
plt.close()
print("Approx ROC saved -> roc_full.png", flush=True)
