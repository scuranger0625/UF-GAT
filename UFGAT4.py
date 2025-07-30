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
import os
import random

# === 參數區 ===
file_path = r"C:\Users\Leon\Desktop\程式語言資料\python\TD-UF\SAML-D.parquet"
WINDOW_SIZE = 100000      # 滑動窗口大小
STRIDE = 20000            # 每次滑動步長
epochs_per_window = 20    # 每窗口訓練 epoch 數
EMBED_DIM = 16            # 群組嵌入維度
GAT_HIDDEN = 64           # GAT 隱藏層維度
TRIPLET_SAMPLES = 64      # 每 window 每 epoch 三元組樣本數
TRIPLET_MARGIN = 1.0      # Triplet Loss 閾值
ALPHA = 0.1               # 對比學習 loss 權重

# === UF-FAE核心（不動） ===
class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x]) # 路徑壓縮
        return self.parent[x]
    def union(self, x, y):
        xr, yr = self.find(x), self.find(y)
        merged = (xr != yr)
        if merged:
            self.parent[yr] = xr
        return merged

# === 動態群組嵌入表 ===
group_embeddings = {}
def get_group_embedding(gid):
    if gid not in group_embeddings:
        group_embeddings[gid] = np.random.normal(size=EMBED_DIM)
    return group_embeddings[gid]
def set_group_embedding(gid, emb):
    group_embeddings[gid] = emb

# === GAT 多模態模型（node embedding 輸出 for contrastive）===
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
        return self.mlp(edge_input).squeeze(), h  # 輸出邊分類與 node embedding

# === Triplet Loss (Contrastive for GAT node embedding) ===
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

# === 讀檔與全域資料初始化 ===
print("掃描帳戶全集...")
df = pl.read_parquet(file_path).sort(["Date", "Time"])
account_set = set()
for row in df.iter_rows(named=True):
    account_set.add(row["Sender_account"])
    account_set.add(row["Receiver_account"])
node_idx_map = {acc: i for i, acc in enumerate(sorted(account_set))}
node_count = len(node_idx_map)
print(f"Total nodes (accounts): {node_count}")

# === 分批滑動視窗訓練 ===
total_samples = df.height
all_pred, all_y_true = [], []
model, optimizer, loss_fn = None, None, None
global_epoch = 1

for start_idx in range(0, total_samples - WINDOW_SIZE + 1, STRIDE):
    end_idx = start_idx + WINDOW_SIZE
    print(f"\n=== 處理第 {start_idx+1} ~ {end_idx} 筆資料（window size: {WINDOW_SIZE}, stride: {STRIDE}） ===")
    window_df = df[start_idx:end_idx]
    uf = UnionFind()
    group_graphs = defaultdict(nx.DiGraph)
    edge_records = []
    node_features_cache = np.zeros((node_count, 4))
    merge_edges, nonmerge_edges = [], []
    # === 建立 batch 特徵 ===
    for idx, row in enumerate(window_df.iter_rows(named=True)):
        s, r = row["Sender_account"], row["Receiver_account"]
        amount = float(row["Amount"])
        paytype = row["Payment_type"]
        is_laundering = int(row["Is_laundering"])
        merged = uf.union(s, r)
        gid_s, gid_r = uf.find(s), uf.find(r)
        emb_s, emb_r = get_group_embedding(gid_s), get_group_embedding(gid_r)
        # 若合併，做embedding融合（平均，可換成MLP）
        if merged:
            new_emb = (emb_s + emb_r) / 2
            set_group_embedding(gid_s, new_emb)
            set_group_embedding(gid_r, new_emb)
            merge_flag = 1
            merge_edges.append((node_idx_map[s], node_idx_map[r]))
        else:
            merge_flag = 0
            nonmerge_edges.append((node_idx_map[s], node_idx_map[r]))
        group_graphs[gid_s].add_edge(s, r, weight=amount)
        G = group_graphs[gid_s]
        group_size = G.number_of_nodes()
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
        node_features_cache[node_idx_map[s], 1] = merge_flag
        node_features_cache[node_idx_map[r], 1] = merge_flag
        # 新增：群組嵌入相似度/合併標誌
        group_sim = np.linalg.norm(emb_s - emb_r)
        edge_feat = [
            np.log1p(amount),
            1 if paytype == "Cash Deposit" else 0,
            1 if paytype == "Credit card" else 0,
            1 if paytype == "Cross-border" else 0,
            1 if paytype == "Cheque" else 0,
            avg_closeness, avg_betweenness, group_size,
            merge_flag,        # 合併事件特徵
            group_sim          # 群組相似度特徵
        ]
        edge_records.append([node_idx_map[s], node_idx_map[r], edge_feat, is_laundering])
    print(f"本窗口樣本數: {len(edge_records)}，正樣本: {np.sum([rec[3] for rec in edge_records])}")
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
    pos_weight = torch.tensor([n_neg / n_pos]) if n_pos > 0 else torch.tensor([1.0])
    print(f"正負樣本分布：正樣本={n_pos}，負樣本={n_neg}，pos_weight={pos_weight.item():.2f}")
    if model is None:
        model = UFGAT(in_node=node_features.shape[1], in_edge=edge_features.shape[1], hidden=GAT_HIDDEN)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    # === Contrastive Triplet Index 準備 ===
    def get_triplet_samples(merge_edges, nonmerge_edges, k=TRIPLET_SAMPLES):
        triplets = []
        if len(merge_edges) == 0 or len(nonmerge_edges) == 0:
            return triplets
        sample_merges = random.choices(merge_edges, k=min(k, len(merge_edges)))
        sample_nonmerges = random.choices(nonmerge_edges, k=min(k, len(nonmerge_edges)))
        for (a, p), (n1, n2) in zip(sample_merges, sample_nonmerges):
            triplets.append((a, p, n1))  # anchor=s, positive=r, negative=n1 (可自行改組合)
        return triplets

    # === 訓練 ===
    for epoch in range(epochs_per_window):
        model.train()
        optimizer.zero_grad()
        out, node_emb = model(data.x, data.edge_index, data.edge_attr)
        loss = loss_fn(out, data.y)
        # Contrastive loss (triplet)
        triplet_idx = get_triplet_samples(merge_edges, nonmerge_edges, TRIPLET_SAMPLES)
        contrastive = triplet_contrastive_loss(node_emb, triplet_idx, margin=TRIPLET_MARGIN)
        total_loss = loss + ALPHA * contrastive
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            pred_prob = torch.sigmoid(model(data.x, data.edge_index, data.edge_attr)[0]).cpu().numpy()
            pred_label = (pred_prob > 0.5).astype(int)
            y_true_np = data.y.cpu().numpy()
            try:
                auc_val = roc_auc_score(y_true_np, pred_prob)
            except:
                auc_val = float('nan')
            f1_val = f1_score(y_true_np, pred_label, zero_division=0)
            precision_val = precision_score(y_true_np, pred_label, zero_division=0)
            recall_val = recall_score(y_true_np, pred_label, zero_division=0)
            acc_val = accuracy_score(y_true_np, pred_label)
        print(f"Win {start_idx//STRIDE+1} | Epoch {global_epoch} - Loss: {loss.item():.4f} | Contrast: {contrastive.item():.4f} | "
            f"AUC: {auc_val:.4f} | F1: {f1_val:.4f} | Pre: {precision_val:.4f} | Rec: {recall_val:.4f} | Acc: {acc_val:.4f}")
        global_epoch += 1

    # === 評估 ===
    with torch.no_grad():
        pred = torch.sigmoid(model(data.x, data.edge_index, data.edge_attr)[0]).cpu().numpy()
        y_true_np = data.y.cpu().numpy()
    all_pred.append(pred)
    all_y_true.append(y_true_np)
    fpr, tpr, thresholds = roc_curve(y_true_np, pred)
    f1s = [f1_score(y_true_np, (pred > thr).astype(int)) for thr in thresholds]
    best_idx = int(np.argmax(f1s))
    best_thr = thresholds[best_idx]
    best_pred_label = (pred > best_thr).astype(int)
    auc_val = roc_auc_score(y_true_np, pred)
    f1_val = f1_score(y_true_np, best_pred_label)
    precision_val = precision_score(y_true_np, best_pred_label, zero_division=0)
    recall_val = recall_score(y_true_np, best_pred_label, zero_division=0)
    acc_val = accuracy_score(y_true_np, best_pred_label)
    print(f"窗口ROC -- Best Threshold (max F1): {best_thr:.4f}")
    print("AUC:      ", auc_val)
    print("F1:       ", f1_val)
    print("Precision:", precision_val)
    print("Recall:   ", recall_val)
    print("Accuracy: ", acc_val)
    plt.figure(figsize=(7,5))
    plt.plot(fpr, tpr, label=f'ROC curve')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (Win {start_idx//STRIDE+1})")
    plt.legend(loc='lower right')
    metrics_text = f"AUC={auc_val:.3f}\nF1={f1_val:.3f}\nPre={precision_val:.3f}\nRec={recall_val:.3f}\nThr={best_thr:.3f}"
    plt.gca().text(0.02, 0.98, metrics_text, fontsize=11, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7))
    plt.savefig(f"roc_win_{start_idx//STRIDE+1}.png")
    plt.close()

# === 合併全體/計算 micro (global) 指標 ===
all_pred = np.concatenate(all_pred)
all_y_true = np.concatenate(all_y_true)
np.save("all_pred.npy", all_pred)
np.save("all_y_true.npy", all_y_true)
fpr, tpr, thresholds = roc_curve(all_y_true, all_pred)
f1s = [f1_score(all_y_true, (all_pred > thr).astype(int)) for thr in thresholds]
best_idx = int(np.argmax(f1s))
best_thr = thresholds[best_idx]
best_pred_label = (all_pred > best_thr).astype(int)
auc_val = roc_auc_score(all_y_true, all_pred)
f1_val = f1_score(all_y_true, best_pred_label)
precision_val = precision_score(all_y_true, best_pred_label, zero_division=0)
recall_val = recall_score(all_y_true, best_pred_label, zero_division=0)
acc_val = accuracy_score(all_y_true, best_pred_label)
print("\n--- 全體微平均(Global)最佳指標 ---")
print(f"Best Threshold (max F1): {best_thr:.4f}")
print("AUC:      ", auc_val)
print("F1:       ", f1_val)
print("Precision:", precision_val)
print("Recall:   ", recall_val)
print("Accuracy: ", acc_val)
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'Global ROC')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"Global ROC Curve (Micro Avg.)")
plt.legend(loc='lower right')
metrics_text = f"AUC={auc_val:.3f}\nF1={f1_val:.3f}\nPre={precision_val:.3f}\nRec={recall_val:.3f}\nThr={best_thr:.3f}"
plt.gca().text(0.02, 0.98, metrics_text, fontsize=13, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7))
plt.savefig("roc_global.png")
plt.close()
