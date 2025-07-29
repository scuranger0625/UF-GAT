import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score, roc_curve
from collections import defaultdict
import matplotlib.pyplot as plt

# === 參數 ===
file_path = r"C:\Users\Leon\Desktop\程式語言資料\python\TD-UF\Anti Money Laundering Transaction Data (SAML-D)\SAML-D.parquet"
BATCH_SIZE = 100000     # 建議 10~50萬為一批
epochs_per_batch = 5    # 每 batch 訓練幾個 epoch
max_tx = None           # None 代表全資料，設數字則提前結束（如測試）

# === UF-FAE核心 ===
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
        if xr != yr:
            self.parent[yr] = xr
            return True
        return False

# === 建帳戶全集（一次建立即可，避免 index 問題） ===
print("掃描帳戶全集...")
df = pl.read_parquet(file_path).sort(["Date", "Time"])
account_set = set()
for row in df.iter_rows(named=True):
    account_set.add(row["Sender_account"])
    account_set.add(row["Receiver_account"])
node_idx_map = {acc: i for i, acc in enumerate(sorted(account_set))}
node_count = len(node_idx_map)
print(f"Total nodes (accounts): {node_count}")

# === GAT 多模態模型 ===
class UFGAT(nn.Module):
    def __init__(self, in_node, in_edge, hidden=64, dropout=0.3):
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
        return self.mlp(edge_input).squeeze()

# === 初始化全域模型 ===
model = None
optimizer = None
loss_fn = None

# === 分批處理/訓練 ===
total_samples = df.height
if max_tx:  # 若只想測前 N 筆
    total_samples = min(total_samples, max_tx)
start_idx = 0
global_epoch = 1

while start_idx < total_samples:
    end_idx = min(start_idx + BATCH_SIZE, total_samples)
    print(f"\n=== 處理第 {start_idx+1} ~ {end_idx} 筆資料（batch size: {end_idx-start_idx}） ===")
    batch_df = df[start_idx:end_idx]
    uf = UnionFind()
    group_graphs = defaultdict(nx.DiGraph)
    edge_records = []
    node_features_cache = np.zeros((node_count, 4))
    
    for idx, row in enumerate(batch_df.iter_rows(named=True)):
        s, r = row["Sender_account"], row["Receiver_account"]
        amount = float(row["Amount"])
        paytype = row["Payment_type"]
        is_laundering = int(row["Is_laundering"])
        merged = uf.union(s, r)
        gid = uf.find(s)
        group_graphs[gid].add_edge(s, r, weight=amount)
        G = group_graphs[gid]
        group_size = G.number_of_nodes()
        # 只對較大群組才算中心性，避免計算太慢
        if group_size >= 5:
            closeness = nx.closeness_centrality(G)
            betweenness = nx.betweenness_centrality(G)
            avg_closeness = np.mean(list(closeness.values()))
            avg_betweenness = np.mean(list(betweenness.values()))
        else:
            avg_closeness = 0
            avg_betweenness = 0
        node_features_cache[node_idx_map[s], 0] = G.degree(s)
        node_features_cache[node_idx_map[r], 0] = G.degree(r)
        node_features_cache[node_idx_map[s], 1] = 1 if merged else 0
        node_features_cache[node_idx_map[r], 1] = 1 if merged else 0
        edge_feat = [
            np.log1p(amount),
            1 if paytype == "Cash Deposit" else 0,
            1 if paytype == "Credit card" else 0,
            1 if paytype == "Cross-border" else 0,
            1 if paytype == "Cheque" else 0,
            avg_closeness, avg_betweenness, group_size
        ]
        edge_records.append([
            node_idx_map[s], node_idx_map[r], edge_feat, is_laundering
        ])
    print(f"本批樣本數: {len(edge_records)}，正樣本: {np.sum([rec[3] for rec in edge_records])}")
    # === 構建 PyG Data ===
    edges = np.array([[rec[0], rec[1]] for rec in edge_records]).T
    edge_features = np.array([rec[2] for rec in edge_records])
    labels = np.array([rec[3] for rec in edge_records])
    node_features = node_features_cache
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edges, dtype=torch.long),
        edge_attr=torch.tensor(edge_features, dtype=torch.float),
        y=torch.tensor(labels, dtype=torch.float)
    )
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)
    if n_pos == 0:
        pos_weight = torch.tensor([1.0])
        print("Warning: 沒有正樣本！")
    else:
        pos_weight = torch.tensor([n_neg / n_pos])
    print(f"正負樣本分布：正樣本={n_pos}，負樣本={n_neg}，pos_weight={pos_weight.item():.2f}")
    # === 建立/重用模型 ===
    if model is None:
        model = UFGAT(in_node=node_features.shape[1], in_edge=edge_features.shape[1])
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # === batch 訓練 ===
    for epoch in range(epochs_per_batch):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.edge_attr)
        loss = loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        print(f"Batch {start_idx//BATCH_SIZE+1} | Epoch {global_epoch} - Loss: {loss.item():.4f}")
        global_epoch += 1
    # === 評估（可選，或最後全做） ===
    with torch.no_grad():
        pred = torch.sigmoid(model(data.x, data.edge_index, data.edge_attr)).cpu().numpy()
        y_true_np = data.y.cpu().numpy()
    fpr, tpr, thresholds = roc_curve(y_true_np, pred)
    f1s = []
    for thr in thresholds:
        pred_label = (pred > thr).astype(int)
        f1s.append(f1_score(y_true_np, pred_label))
    best_idx = int(np.argmax(f1s))
    best_thr = thresholds[best_idx]
    best_pred_label = (pred > best_thr).astype(int)
    print(f"Batch 評估 -- Best Threshold (max F1): {best_thr:.4f}")
    print("AUC:      ", roc_auc_score(y_true_np, pred))
    print("F1:       ", f1_score(y_true_np, best_pred_label))
    print("Precision:", precision_score(y_true_np, best_pred_label))
    print("Recall:   ", recall_score(y_true_np, best_pred_label))
    print("Accuracy: ", accuracy_score(y_true_np, best_pred_label))
    # 可視覺化 ROC
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f'AUC={roc_auc_score(y_true_np, pred):.3f}')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"ROC Curve (Batch {start_idx//BATCH_SIZE+1})")
    # 讓檔案用存的 而不是show
    plt.savefig(f"roc_batch_{start_idx//BATCH_SIZE+1}.png")
    plt.close()

    # 跳至下一批
    start_idx = end_idx
