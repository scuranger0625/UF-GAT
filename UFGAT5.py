import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve
from collections import defaultdict
import matplotlib.pyplot as plt
import random
import pandas as pd
import igraph as ig
from tqdm import tqdm
import os

# =========================
# 參數區：你可以調整
# =========================
file_path = r"C:\Users\Leon\Desktop\程式語言資料\python\TD-UF\SAML-D.parquet"  # 交易資料集
GRAPH_FEATURE_NODE_THRESHOLD = 3  # 這裡你可以統一調整
SEQ_LEN = 20            # LSTM 時序步數（每個群組記錄最近N筆交易序列）
LSTM_INPUT_DIM = 17     # ⚠️ 根據你get_tx_features產生的特徵數量調整
LSTM_HIDDEN = 32        # LSTM隱藏維度
EMBED_DIM = 16          # group embedding維度
GAT_HIDDEN = 64         # GAT 隱藏維度
EPOCHS = 10             # 訓練回合
TRIPLET_SAMPLES = 64    # 對比學習樣本數
TRIPLET_MARGIN = 1.0    # triplet loss間距
ALPHA = 0.1             # triplet loss權重
THR_MIN = 0.1           # threshold搜尋區間
THR_MAX = 1.0
THR_STEP = 0.01

# =========================
# UF-FAE 動態分群核心
# =========================
class UnionFind:
    """
    維護動態群組分群關係（path compression + union by root）
    """
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
    """
    每個群組一條時序序列（SEQ_LEN步），
    用LSTM壓縮為一個固定長度的group embedding
    """
    def __init__(self, input_dim, hidden_dim, emb_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, emb_dim)
    def forward(self, x_seq):
        # x_seq: [batch, seq_len, input_dim]
        output, (h_n, _) = self.lstm(x_seq)
        emb = self.fc(h_n[-1])
        return emb  # [batch, emb_dim]

# =========================
# GAT 多模態圖模型
# =========================
class UFGAT(nn.Module):
    """
    用GAT處理分群之間的異常邊(edge)，
    節點特徵是群組LSTM embedding
    邊特徵包含交易/圖論資訊
    """
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
# Triplet Loss 對比學習
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
# 交易＋圖論特徵（自定義融合）
# =========================
def get_tx_features(row, G=None, s=None, r=None, node_map=None, node_threshold=GRAPH_FEATURE_NODE_THRESHOLD):
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
    if G is not None and G.vcount() >= 3 and node_map is not None:
        group_size = G.vcount()
        group_edge_count = G.ecount()
        if group_size > node_threshold:
            try:
                s_idx = node_map[s]
                r_idx = node_map[r]
            except KeyError:
                # 新節點剛加還沒來得及寫進去
                return features + [0]*9
            s_deg = G.degree(s_idx)
            r_deg = G.degree(r_idx)
            closeness_vals = G.closeness()
            betweenness_vals = G.betweenness()
            s_close = closeness_vals[s_idx] if group_size > s_idx else 0
            r_close = closeness_vals[r_idx] if group_size > r_idx else 0
            s_between = betweenness_vals[s_idx] if group_size > s_idx else 0
            r_between = betweenness_vals[r_idx] if group_size > r_idx else 0
            bidirect_count = sum(
                1 for e in G.es
                if G.are_connected(G.vs[e.source], G.vs[e.target]) and G.are_connected(G.vs[e.target], G.vs[e.source])
            )
            bidirect_ratio = bidirect_count / group_edge_count if group_edge_count > 0 else 0
        else:
            # 群組太小，全補0
            s_deg = r_deg = s_close = r_close = s_between = r_between = group_size = group_edge_count = bidirect_ratio = 0
    else:
        s_deg = r_deg = s_close = r_close = s_between = r_between = group_size = group_edge_count = bidirect_ratio = 0

    features += [
        s_deg, r_deg, s_close, r_close, s_between, r_between, group_size, group_edge_count, bidirect_ratio
    ]
    return features

# =========================
# 1. 讀檔＋帳戶編號
# =========================
print("掃描帳戶全集...")
df = pl.read_parquet(file_path).sort(["Date", "Time"])
account_set = set()
for row in df.iter_rows(named=True):
    account_set.add(row["Sender_account"])
    account_set.add(row["Receiver_account"])
node_idx_map = {acc: i for i, acc in enumerate(sorted(account_set))}
node_count = len(node_idx_map)
print(f"Total nodes (accounts): {node_count}")

# =========================
# 2. UF-FAE分群＋群組序列收集
# =========================
uf = UnionFind()
group_graphs = defaultdict(lambda: ig.Graph(directed=True))
group_node_maps = defaultdict(dict)  # 每個群組: 帳號str -> idx
group_tx_seq = defaultdict(list)          # 每個群組一個時序特徵序列
edge_records = []
merge_edges, nonmerge_edges = [], []

for idx, row in enumerate(tqdm(df.iter_rows(named=True), total=df.height, desc="動態分群/特徵蒐集ING")):
    s, r = row["Sender_account"], row["Receiver_account"]
    is_laundering = int(row["Is_laundering"])
    merged = uf.union(s, r)
    gid = uf.find(s)
    G = group_graphs[gid]
    node_map = group_node_maps[gid]

    # 新帳號補進去頂點
    for acc in [s, r]:
        if acc not in node_map:
            v_idx = G.vcount()
            G.add_vertices(1)
            node_map[acc] = v_idx
    s_idx, r_idx = node_map[s], node_map[r]
    G.add_edges([(s_idx, r_idx)])  # igraph允許重複邊

    feat = get_tx_features(row, G=G, s=s, r=r, node_map=node_map, node_threshold=GRAPH_FEATURE_NODE_THRESHOLD)
    group_tx_seq[gid].append(feat)
    edge_records.append([node_idx_map[s], node_idx_map[r], feat, is_laundering])
    if merged:
        merge_edges.append((node_idx_map[s], node_idx_map[r]))
    else:
        nonmerge_edges.append((node_idx_map[s], node_idx_map[r]))

print(f"總樣本數: {len(edge_records)}，正樣本: {np.sum([rec[3] for rec in edge_records])}")

# =========================
# 3. 用LSTM壓每個群組的時序成embedding
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
group_lstm = GroupLSTM(input_dim=LSTM_INPUT_DIM, hidden_dim=LSTM_HIDDEN, emb_dim=EMBED_DIM).to(device)
group_embeddings = {}

for gid, seq in group_tx_seq.items():
    # 只取最新SEQ_LEN步，不足則前面補零
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
    if gid in group_embeddings:
        node_features[idx] = group_embeddings[gid]
    else:
        node_features[idx] = np.zeros(EMBED_DIM)   # 沒特徵就補0

# =========================
# 5. PyG Data 結構
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
# 6. GAT訓練（加Triplet Loss）
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
    print(f"Epoch {epoch+1}/{EPOCHS} Loss={total_loss.item():.4f}")

# =========================
# 7. 推論與指標/繪圖
# =========================
with torch.no_grad():
    pred = torch.sigmoid(model(data.x, data.edge_index, data.edge_attr)[0]).cpu().numpy()
    y_true_np = data.y.cpu().numpy()

fpr, tpr, thresholds = roc_curve(y_true_np, pred)
thr_mask = (thresholds >= THR_MIN) & (thresholds <= THR_MAX)
if np.any(thr_mask):
    masked_thresholds = thresholds[thr_mask]
    f1s = [f1_score(y_true_np, (pred > thr).astype(int)) for thr in masked_thresholds]
    best_idx = int(np.argmax(f1s))
    best_thr = masked_thresholds[best_idx]
    best_pred_label = (pred > best_thr).astype(int)
else:
    best_thr = THR_MAX
    best_pred_label = (pred > best_thr).astype(int)

auc_val = roc_auc_score(y_true_np, pred)
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

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (全資料)")
plt.legend(loc='lower right')
metrics_text = f"AUC={auc_val:.3f}\nF1={f1_val:.3f}\nPre={precision_val:.3f}\nRec={recall_val:.3f}\nThr={best_thr:.3f}"
plt.gca().text(0.02, 0.98, metrics_text, fontsize=13, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7))
plt.savefig("roc_full.png")
plt.close()

# === 可選：自動 threshold sweep、報表 ===
search_grid = np.arange(THR_MIN, THR_MAX + THR_STEP, THR_STEP)
f1s, precisions, recalls, accs = [], [], [], []
for thr in search_grid:
    pred_bin = (np.array(pred) > thr).astype(int)
    f1s.append(f1_score(y_true_np, pred_bin))
    precisions.append(precision_score(y_true_np, pred_bin, zero_division=0))
    recalls.append(recall_score(y_true_np, pred_bin, zero_division=0))
    accs.append(accuracy_score(y_true_np, pred_bin))
result_df = pd.DataFrame({
    "threshold": search_grid,
    "F1": f1s,
    "Precision": precisions,
    "Recall": recalls,
    "Accuracy": accs
})
result_df.to_csv("threshold_sweep.csv", index=False)
print("所有 threshold 報表已存檔：threshold_sweep.csv")
