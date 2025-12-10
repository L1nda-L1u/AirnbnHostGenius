# -*- coding: utf-8 -*-
import sys
import io
# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm
import pickle
import warnings
warnings.filterwarnings("ignore")

# =============================================
# KNN 辅助函数（k=10）
# =============================================
def get_knn_prediction(target_embedding, train_embeddings, train_prices, k=10):
    """基于 embedding 找 KNN，返回加权平均价格"""
    dists = norm(train_embeddings - target_embedding, axis=1)
    topk_idx = np.argsort(dists)[:k]
    eps = 1e-6
    weights = 1.0 / (dists[topk_idx] + eps)
    weighted_price = np.sum(weights * train_prices[topk_idx]) / np.sum(weights)
    return weighted_price

# =============================================
# 1. 加载数据和模型（复用之前的代码）
# =============================================
print("="*80)
print("分层分析：NN 和 KNN 在不同条件下的表现")
print("="*80)

print("\nLoading data and models...")
df_original = pd.read_csv("nn_price_training_v4.csv")
target_col = "price_num"
feature_cols = [c for c in df_original.columns if c != target_col]

# 数据清理
df = df_original.copy()
df = df[~((df["accommodates"] <= 2) & (df["price_num"] > 400))]
df = df[~((df["accommodates"] <= 4) & (df["price_num"] > 600))]
df = df[~((df["accommodates"] <= 6) & (df["price_num"] > 800))]
upper = df["price_num"].quantile(0.995)
df = df[df["price_num"] < upper]
df = df.reset_index(drop=True)

X = df[feature_cols].values.astype(np.float32)
y_raw = df[target_col].values.astype(np.float32)
y_log = np.log1p(y_raw)

bins = [0, 50, 100, 150, 200, 300, 500, 800, 1200, np.inf]
y_bins = pd.cut(y_raw, bins=bins, labels=False, include_lowest=True)

X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
    X, y_log, y_raw,
    test_size=0.10,
    random_state=42,
    stratify=y_bins
)

# 加载模型
print("  Loading XGBoost...")
with open("best_xgb_log_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("scaler_xgb.pkl", "rb") as f:
    scaler_xgb = pickle.load(f)

print("  Loading Neural Network...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PriceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

with open("scaler_price.pkl", "rb") as f:
    scaler_nn = pickle.load(f)
model = PriceMLP(input_dim=X_train.shape[1]).to(device)
model.load_state_dict(torch.load("best_price_A2_log.pth", map_location=device))
model.eval()

print("  Loading Autoencoder for KNN...")
try:
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=16):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, input_dim)
            )
        def forward(self, x):
            z = self.encoder(x)
            out = self.decoder(z)
            return out, z
    
    with open("ae_scaler.pkl", "rb") as f:
        scaler_ae = pickle.load(f)
    ae_model = Autoencoder(input_dim=X_train.shape[1], latent_dim=16).to(device)
    ae_model.load_state_dict(torch.load("autoencoder_model.pth", map_location=device))
    ae_model.eval()
    
    X_train_ae = scaler_ae.transform(X_train)
    X_test_ae = scaler_ae.transform(X_test)
    X_train_ae_t = torch.tensor(X_train_ae, dtype=torch.float32, device=device)
    X_test_ae_t = torch.tensor(X_test_ae, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        _, train_embeddings = ae_model(X_train_ae_t)
        train_embeddings = train_embeddings.cpu().numpy()
        _, test_embeddings = ae_model(X_test_ae_t)
        test_embeddings = test_embeddings.cpu().numpy()
    
    knn_pred = np.array([
        get_knn_prediction(test_emb, train_embeddings, y_train_raw, k=10)
        for test_emb in test_embeddings
    ])
    use_knn = True
    print("  [OK] KNN loaded successfully")
except Exception as e:
    print(f"  [WARNING] KNN not available: {e}")
    use_knn = False
    knn_pred = None

# 生成预测
X_test_xgb = scaler_xgb.transform(X_test)
xgb_pred_log = xgb_model.predict(X_test_xgb)
xgb_pred_real = np.expm1(xgb_pred_log)

X_test_nn = scaler_nn.transform(X_test)
X_test_t = torch.tensor(X_test_nn, dtype=torch.float32, device=device)
with torch.no_grad():
    nn_pred_log = model(X_test_t).cpu().numpy()
nn_pred_real = np.expm1(nn_pred_log)

# =============================================
# 2. 准备分层数据（从特征矩阵中提取 accommodates）
# =============================================
# 尝试从特征列中找到 accommodates
if 'accommodates' in feature_cols:
    acc_idx = feature_cols.index('accommodates')
    test_accommodates = X_test[:, acc_idx]
    print(f"  [OK] Found accommodates at feature index {acc_idx}")
else:
    # 尝试从原始数据中获取（需要重新划分以匹配）
    print("  [WARNING] accommodates not in feature_cols, trying to extract from original data...")
    # 重新划分以获取测试集的原始数据
    df_clean = df.copy()
    X_full = df_clean[feature_cols].values.astype(np.float32)
    y_full_raw = df_clean[target_col].values.astype(np.float32)
    y_full_log = np.log1p(y_full_raw)
    y_full_bins = pd.cut(y_full_raw, bins=bins, labels=False, include_lowest=True)
    
    _, _, _, _, _, y_test_raw_check = train_test_split(
        X_full, y_full_log, y_full_raw,
        test_size=0.10,
        random_state=42,
        stratify=y_full_bins
    )
    
    # 如果原始数据有 accommodates 列
    if 'accommodates' in df_clean.columns:
        _, _, _, _, _, test_accommodates = train_test_split(
            X_full, y_full_log, df_clean['accommodates'].values,
            test_size=0.10,
            random_state=42,
            stratify=y_full_bins
        )
        print(f"  [OK] Extracted accommodates from original data")
    else:
        test_accommodates = None
        print(f"  [WARNING] Could not find accommodates")

# =============================================
# 3. 分层分析函数
# =============================================
def analyze_segment(mask, segment_name, y_true, xgb_pred, nn_pred, knn_pred=None):
    """分析特定分段的模型表现"""
    if np.sum(mask) == 0:
        return None
    
    y_seg = y_true[mask]
    xgb_seg = xgb_pred[mask]
    nn_seg = nn_pred[mask]
    
    results = {
        'segment': segment_name,
        'n_samples': len(y_seg),
        'xgb': {
            'r2': r2_score(y_seg, xgb_seg),
            'mae': mean_absolute_error(y_seg, xgb_seg),
            'rmse': np.sqrt(mean_squared_error(y_seg, xgb_seg))
        },
        'nn': {
            'r2': r2_score(y_seg, nn_seg),
            'mae': mean_absolute_error(y_seg, nn_seg),
            'rmse': np.sqrt(mean_squared_error(y_seg, nn_seg))
        }
    }
    
    if knn_pred is not None:
        knn_seg = knn_pred[mask]
        results['knn'] = {
            'r2': r2_score(y_seg, knn_seg),
            'mae': mean_absolute_error(y_seg, knn_seg),
            'rmse': np.sqrt(mean_squared_error(y_seg, knn_seg))
        }
    
    return results

# =============================================
# 4. 按价格区间分层
# =============================================
print("\n" + "="*80)
print("按价格区间分层分析")
print("="*80)

price_segments = [
    (0, 50, "£0-50 (超低价)"),
    (50, 100, "£50-100 (低价)"),
    (100, 150, "£100-150 (中低价)"),
    (150, 200, "£150-200 (中价)"),
    (200, 300, "£200-300 (中高价)"),
    (300, 500, "£300-500 (高价)"),
    (500, 800, "£500-800 (超高价)"),
    (800, np.inf, "£800+ (极高价)")
]

# 定义 y_test_real 为 y_test_raw（真实价格）
y_test_real = y_test_raw

price_results = []
for low, high, name in price_segments:
    mask = (y_test_real >= low) & (y_test_real < high)
    result = analyze_segment(mask, name, y_test_real, xgb_pred_real, nn_pred_real, knn_pred)
    if result:
        price_results.append(result)

# 打印价格区间结果
print(f"\n{'价格区间':<20} {'样本数':<10} {'模型':<8} {'R²':<10} {'MAE':<10} {'RMSE':<10}")
print("-" * 80)
for r in price_results:
    print(f"\n{r['segment']:<20} {r['n_samples']:<10}")
    print(f"{'':20} {'XGBoost':<10} {r['xgb']['r2']:<10.4f} {r['xgb']['mae']:<10.2f} {r['xgb']['rmse']:<10.2f}")
    print(f"{'':20} {'NN':<10} {r['nn']['r2']:<10.4f} {r['nn']['mae']:<10.2f} {r['nn']['rmse']:<10.2f}")
    if 'knn' in r:
        print(f"{'':20} {'KNN':<10} {r['knn']['r2']:<10.4f} {r['knn']['mae']:<10.2f} {r['knn']['rmse']:<10.2f}")

# =============================================
# 5. 按房间数（accommodates）分层
# =============================================
if test_accommodates is not None:
    print("\n" + "="*80)
    print("按房间数（accommodates）分层分析")
    print("="*80)
    
    acc_segments = [
        (1, 2, "1-2人"),
        (3, 4, "3-4人"),
        (5, 6, "5-6人"),
        (7, 10, "7-10人"),
        (11, np.inf, "11+人")
    ]
    
    acc_results = []
    for low, high, name in acc_segments:
        mask = (test_accommodates >= low) & (test_accommodates <= high)
        result = analyze_segment(mask, name, y_test_real, xgb_pred_real, nn_pred_real, knn_pred)
        if result:
            acc_results.append(result)
    
    print(f"\n{'房间数':<15} {'样本数':<10} {'模型':<8} {'R²':<10} {'MAE':<10} {'RMSE':<10}")
    print("-" * 80)
    for r in acc_results:
        print(f"\n{r['segment']:<15} {r['n_samples']:<10}")
        print(f"{'':15} {'XGBoost':<10} {r['xgb']['r2']:<10.4f} {r['xgb']['mae']:<10.2f} {r['xgb']['rmse']:<10.2f}")
        print(f"{'':15} {'NN':<10} {r['nn']['r2']:<10.4f} {r['nn']['mae']:<10.2f} {r['nn']['rmse']:<10.2f}")
        if 'knn' in r:
            print(f"{'':15} {'KNN':<10} {r['knn']['r2']:<10.4f} {r['knn']['mae']:<10.2f} {r['knn']['rmse']:<10.2f}")

# =============================================
# 6. 找出 NN 和 KNN 表现最好的分段
# =============================================
print("\n" + "="*80)
print("找出 NN 和 KNN 表现最好的分段")
print("="*80)

# 找出 NN 表现最好的价格区间
nn_price_best = max(price_results, key=lambda x: x['nn']['r2'])
print(f"\n[+] NN 表现最好的价格区间: {nn_price_best['segment']}")
print(f"   R²: {nn_price_best['nn']['r2']:.4f} (vs XGBoost: {nn_price_best['xgb']['r2']:.4f})")
print(f"   MAE: {nn_price_best['nn']['mae']:.2f} (vs XGBoost: {nn_price_best['xgb']['mae']:.2f})")
print(f"   样本数: {nn_price_best['n_samples']}")

# 找出 NN 表现最差的价格区间
nn_price_worst = min(price_results, key=lambda x: x['nn']['r2'])
print(f"\n[-] NN 表现最差的价格区间: {nn_price_worst['segment']}")
print(f"   R²: {nn_price_worst['nn']['r2']:.4f} (vs XGBoost: {nn_price_worst['xgb']['r2']:.4f})")
print(f"   MAE: {nn_price_worst['nn']['mae']:.2f} (vs XGBoost: {nn_price_worst['xgb']['mae']:.2f})")
print(f"   样本数: {nn_price_worst['n_samples']}")

if use_knn:
    # 找出 KNN 表现最好的价格区间
    knn_price_best = max(price_results, key=lambda x: x['knn']['r2'])
    print(f"\n[+] KNN 表现最好的价格区间: {knn_price_best['segment']}")
    print(f"   R²: {knn_price_best['knn']['r2']:.4f} (vs XGBoost: {knn_price_best['xgb']['r2']:.4f})")
    print(f"   MAE: {knn_price_best['knn']['mae']:.2f} (vs XGBoost: {knn_price_best['xgb']['mae']:.2f})")
    print(f"   样本数: {knn_price_best['n_samples']}")
    
    # 找出 KNN 表现最差的价格区间
    knn_price_worst = min(price_results, key=lambda x: x['knn']['r2'])
    print(f"\n[-] KNN 表现最差的价格区间: {knn_price_worst['segment']}")
    print(f"   R²: {knn_price_worst['knn']['r2']:.4f} (vs XGBoost: {knn_price_worst['xgb']['r2']:.4f})")
    print(f"   MAE: {knn_price_worst['knn']['mae']:.2f} (vs XGBoost: {knn_price_worst['xgb']['mae']:.2f})")
    print(f"   样本数: {knn_price_worst['n_samples']}")

# =============================================
# 7. 找出 NN/KNN 比 XGBoost 表现更好的分段
# =============================================
print("\n" + "="*80)
print("NN/KNN 比 XGBoost 表现更好的分段")
print("="*80)

nn_better = []
for r in price_results:
    if r['nn']['r2'] > r['xgb']['r2']:
        improvement = r['nn']['r2'] - r['xgb']['r2']
        nn_better.append((r['segment'], improvement, r['n_samples']))

if nn_better:
    print(f"\n[+] NN 表现更好的分段 ({len(nn_better)} 个):")
    for seg, imp, n in sorted(nn_better, key=lambda x: x[1], reverse=True):
        print(f"   * {seg:<20} R²提升: {imp:+.4f} (样本数: {n})")
else:
    print("\n[-] NN 在所有价格区间都不如 XGBoost")

if use_knn:
    knn_better = []
    for r in price_results:
        if r['knn']['r2'] > r['xgb']['r2']:
            improvement = r['knn']['r2'] - r['xgb']['r2']
            knn_better.append((r['segment'], improvement, r['n_samples']))
    
    if knn_better:
        print(f"\n[+] KNN 表现更好的分段 ({len(knn_better)} 个):")
        for seg, imp, n in sorted(knn_better, key=lambda x: x[1], reverse=True):
            print(f"   * {seg:<20} R²提升: {imp:+.4f} (样本数: {n})")
    else:
        print("\n[-] KNN 在所有价格区间都不如 XGBoost")

# =============================================
# 8. 按价格区间计算准确率
# =============================================
def calculate_accuracy_by_segment(mask, y_true, y_pred, tolerance):
    if np.sum(mask) == 0:
        return None
    errors = np.abs(y_true[mask] - y_pred[mask])
    within = np.sum(errors <= tolerance)
    return within / len(errors) * 100

print("\n" + "="*80)
print("按价格区间的准确率分析 (±15£, ±25£)")
print("="*80)

print(f"\n{'价格区间':<20} {'样本数':<10} {'模型':<8} {'±15£':<10} {'±25£':<10}")
print("-" * 80)
for low, high, name in price_segments:
    mask = (y_test_real >= low) & (y_test_real < high)
    n = np.sum(mask)
    if n == 0:
        continue
    
    acc_xgb_15 = calculate_accuracy_by_segment(mask, y_test_real, xgb_pred_real, 15)
    acc_xgb_25 = calculate_accuracy_by_segment(mask, y_test_real, xgb_pred_real, 25)
    acc_nn_15 = calculate_accuracy_by_segment(mask, y_test_real, nn_pred_real, 15)
    acc_nn_25 = calculate_accuracy_by_segment(mask, y_test_real, nn_pred_real, 25)
    
    print(f"\n{name:<20} {n:<10}")
    print(f"{'':20} {'XGBoost':<10} {acc_xgb_15:<10.2f}% {acc_xgb_25:<10.2f}%")
    print(f"{'':20} {'NN':<10} {acc_nn_15:<10.2f}% {acc_nn_25:<10.2f}%")
    
    if use_knn:
        acc_knn_15 = calculate_accuracy_by_segment(mask, y_test_real, knn_pred, 15)
        acc_knn_25 = calculate_accuracy_by_segment(mask, y_test_real, knn_pred, 25)
        print(f"{'':20} {'KNN':<10} {acc_knn_15:<10.2f}% {acc_knn_25:<10.2f}%")

print("\n" + "="*80)
print("[OK] 分析完成！")
print("="*80)

