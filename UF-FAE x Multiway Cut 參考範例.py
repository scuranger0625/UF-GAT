# -*- coding: utf-8 -*-
"""
UF-FAE x Multiway Cut 參考範例
說明：
  - 模擬真實社會網絡的群組演化過程
  - 每條交易邊具有權重（例如金額）
  - 使用 Union-Find 建立群組
  - 同時計算：
      1. 群組之間的跨群交易邊集合 (cut)
      2. cut 總成本 S
      3. 重疊率 overlap
      4. 拓撲健康指標 BDI

你可以想像這是「UF-FAE 的動態監測模組」，
幫助你觀察系統拓撲在群組合併過程中的健康度。
"""

import random
from collections import defaultdict

# === 1. Union-Find 基本骨架 ===
class UnionFind:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return False  # 已在同一群
        # union by rank
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1
        return True

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0

# === 2. 模擬真實社會網絡 ===
# 節點（帳戶、人物、組織）
nodes = ['A', 'B', 'C', 'D', 'E', 'F']
# 隨機生成一些交易邊與金額
edges = [
    ('A','B', 5),
    ('B','C', 8),
    ('C','D', 3),
    ('D','E', 10),
    ('E','F', 2),
    ('A','F', 7),
    ('B','E', 6),
]

# === 3. 初始化 Union-Find ===
uf = UnionFind()
for n in nodes:
    uf.add(n)

# === 4. 動態演化模擬 ===
cut_log = []  # 記錄所有跨群邊事件
S_total = []  # cut 成本曲線
BDI_total = []  # 拓撲健康指標曲線

for t, (u, v, w) in enumerate(edges, start=1):
    # 查看當前群組情況
    group_u, group_v = uf.find(u), uf.find(v)
    merged = False

    # 如果不同群，記錄這條邊作為合併事件
    if group_u != group_v:
        merged = uf.union(u, v)
        cut_log.append({
            'time': t,
            'edge': (u, v),
            'weight': w,
            'before': (group_u, group_v)
        })

    # === 計算跨群邊 (cut edges) ===
    # 此時的所有邊中，只要連接不同群，就算是 cut 邊
    cut_edges = []
    for (x, y, cost) in edges[:t]:
        if uf.find(x) != uf.find(y):
            cut_edges.append((x, y, cost))

    # S = 所有 cut 邊的總權重
    S = sum(c for (_, _, c) in cut_edges)

    # overlap = 不同群組共享 cut 邊的重疊率（簡化計算）
    # 群組間的 pair 數作為分母
    group_pairs = defaultdict(int)
    for (x, y, _) in cut_edges:
        gu, gv = uf.find(x), uf.find(y)
        group_pairs[frozenset([gu, gv])] += 1
    overlap = 1 - len(group_pairs) / (len(cut_edges) + 1e-9)

    # BDI = α·overlap + β·ΔS
    alpha, beta = 0.7, 0.3
    prev_S = S_total[-1] if S_total else S
    delta_S = S - prev_S
    BDI = alpha * overlap + beta * (delta_S / (S + 1e-9))

    S_total.append(S)
    BDI_total.append(BDI)

    # 輸出當前狀態
    print(f"\n時間 {t}: 邊 ({u},{v}) w={w}")
    if merged:
        print(f"→ 群組合併: {group_u} + {group_v} → {uf.find(u)}")
    print(f"cut 總成本 S = {S:.2f}")
    print(f"重疊率 overlap = {overlap:.3f}")
    print(f"拓撲健康指標 BDI = {BDI:.3f}")

# === 5. 模擬結束，摘要 ===
print("\n=== 群組演化摘要 ===")
groups = defaultdict(list)
for n in nodes:
    groups[uf.find(n)].append(n)
for gid, members in groups.items():
    print(f"群組 {gid}: {members}")

print("\ncut 成本曲線 S(t):", S_total)
print("BDI 曲線:", BDI_total)
