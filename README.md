# UF-GAT
UF-GAT-test

# UF-GAT: Union-Find Enhanced Graph Attention for AML Detection

![Vertex AI](https://img.shields.io/badge/GCP-Vertex%20AI-blue.svg)
![PyTorch Geometric](https://img.shields.io/badge/PyG-2.5.0-brightgreen.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 簡介

UF-GAT (Union-Find Graph Attention Network) 是專為大規模反洗錢（AML）交易網路資料設計的圖神經網路系統。  
結合 Union-Find 動態群組分群 + GAT 深度學習，讓你在雲端用 GCP Vertex AI 快速訓練、批次處理，並自動計算各種圖中心性指標與異常交易偵測指標。

**專案特色：**
- 支援 SAML-D 這類高維、動態金融交易網資料。
- 批次化訓練支援大資料量流式處理（可指定每批處理筆數）。
- 每筆交易自動融合節點中心性、群組結構、金額型態等多模態特徵。
- 適合部署於 GCP Vertex AI，搭配 PyTorch Geometric 執行。

## 架構亮點

- **Union-Find 分群**：動態建群，有新交易自動合併群組並即時計算特徵。
- **Graph Attention Network (GAT)**：用於學習帳戶之間複雜互動關係。
- **多模態特徵融合**：節點中心性 + 邊屬性（如金額/支付型態）全數入模。
- **自動調權重**：根據正負樣本比例自動設計損失函數權重。

## 執行環境建議

- Google Cloud Vertex AI Notebooks or VM
- Python 3.10+
- 主要套件：polars, torch, torch_geometric, networkx, matplotlib, scikit-learn

## 基本用法

1. 上傳你的 Parquet 格式資料（如 SAML-D.parquet）至 GCP VM 或 Vertex AI Notebook。
2. 直接執行 `UF-GAT.py`，預設每批 100,000 筆，批次訓練+即時評估。
3. 訓練結果每一批會自動顯示指標（AUC, F1, Precision, Recall, Accuracy），並繪出 ROC 曲線。

```bash
python UF-GAT.py
