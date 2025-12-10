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
# KNN 辅助函数（基于 Autoencoder embedding，k=10）
# =============================================
def get_knn_prediction(target_embedding, train_embeddings, train_prices, k=10):
    """
    基于 embedding 找 KNN，返回加权平均价格
    k=10: 只找10个最相似的房源
    """
    dists = norm(train_embeddings - target_embedding, axis=1)
    topk_idx = np.argsort(dists)[:k]
    
    # 加权平均（距离越近权重越大）
    eps = 1e-6
    weights = 1.0 / (dists[topk_idx] + eps)
    weighted_price = np.sum(weights * train_prices[topk_idx]) / np.sum(weights)
    
    return weighted_price

# =============================================
# 1. 加载数据（与训练脚本保持一致）
# =============================================
print("="*70)
print("🚀 XGBoost + NN + KNN (k=10) Stacking with Linear Regression")
print("="*70)

print("\nLoading data...")
df_original = pd.read_csv("nn_price_training_v4.csv")
print(f"原始数据量: {len(df_original):,} 行")

target_col = "price_num"
feature_cols = [c for c in df_original.columns if c != target_col]

# 重要：完善的数据清理规则
print("\nCleaning outliers...")
df = df_original.copy()

# 清理规则1: 2人及以下但价格>400
df = df[~((df["accommodates"] <= 2) & (df["price_num"] > 400))]

# 清理规则2: 4人及以下但价格>600
df = df[~((df["accommodates"] <= 4) & (df["price_num"] > 600))]

# 清理规则3: 6人及以下但价格>800
df = df[~((df["accommodates"] <= 6) & (df["price_num"] > 800))]

# 清理规则4: 移除99.5%分位数以上的极端值
upper = df["price_num"].quantile(0.995)
df = df[df["price_num"] < upper]

df = df.reset_index(drop=True)
print(f"清理后数据量: {len(df):,} 行 (删除了 {len(df_original) - len(df):,} 行异常值)")

X = df[feature_cols].values.astype(np.float32)
y_raw = df[target_col].values.astype(np.float32)
y_log = np.log1p(y_raw)

# 使用随机划分（真实预测场景，不使用价格分层）
# ⚠️ 重要：真实预测时我们不知道价格，所以训练时也不应该用价格分层

X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
    X, y_log, y_raw,
    test_size=0.10,
    random_state=42
    # 不使用 stratify，因为真实预测时不知道价格
)

print(f"\n数据划分统计:")
print(f"  • 训练集: {X_train.shape[0]:,} 行")
print(f"  • 测试集: {X_test.shape[0]:,} 行")
print(f"  • 总数据: {len(df):,} 行")

# =============================================
# 2. 加载 XGBoost 模型和 scaler
# =============================================
print("\n" + "="*70)
print("Loading XGBoost model...")
with open("best_xgb_log_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("scaler_xgb.pkl", "rb") as f:
    scaler_xgb = pickle.load(f)

# XGBoost 预测（需要标准化）
X_test_xgb = scaler_xgb.transform(X_test)
xgb_pred_log = xgb_model.predict(X_test_xgb)
xgb_pred_real = np.expm1(xgb_pred_log)  # 转回真实价格

print(f"[OK] XGBoost predictions shape: {xgb_pred_real.shape}")

# =============================================
# 3. 加载神经网络模型和 scaler
# =============================================
print("\nLoading Neural Network model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"   Using device: {device}")

# 定义与 train_price_log.py 相同的模型结构
class PriceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(1)

# 加载 scaler
with open("scaler_price.pkl", "rb") as f:
    scaler_nn = pickle.load(f)

# 加载模型
model = PriceMLP(input_dim=X_train.shape[1]).to(device)
model.load_state_dict(torch.load("best_price_A2_log.pth", map_location=device))
model.eval()

# 神经网络预测
X_test_nn = scaler_nn.transform(X_test)
X_test_t = torch.tensor(X_test_nn, dtype=torch.float32, device=device)

with torch.no_grad():
    nn_pred_log = model(X_test_t).cpu().numpy()

nn_pred_real = np.expm1(nn_pred_log)  # 转回真实价格

print(f"[OK] NN predictions shape: {nn_pred_real.shape}")

# =============================================
# 4. 加载 Autoencoder 模型并生成 KNN 预测 (k=10)
# =============================================
print("\nLoading Autoencoder model for KNN (k=10)...")
try:
    # 定义 Autoencoder 结构（与 autoencoder_knn.py 一致）
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=16):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, input_dim)
            )
        
        def forward(self, x):
            z = self.encoder(x)
            out = self.decoder(z)
            return out, z
    
    # 加载 Autoencoder scaler 和模型
    with open("ae_scaler.pkl", "rb") as f:
        scaler_ae = pickle.load(f)
    
    latent_dim = 16
    ae_model = Autoencoder(input_dim=X_train.shape[1], latent_dim=latent_dim).to(device)
    ae_model.load_state_dict(torch.load("autoencoder_model.pth", map_location=device))
    ae_model.eval()
    
    # 为训练集和测试集生成 embeddings
    X_train_ae = scaler_ae.transform(X_train)
    X_test_ae = scaler_ae.transform(X_test)
    
    X_train_ae_t = torch.tensor(X_train_ae, dtype=torch.float32, device=device)
    X_test_ae_t = torch.tensor(X_test_ae, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        _, train_embeddings = ae_model(X_train_ae_t)
        train_embeddings = train_embeddings.cpu().numpy()
        
        _, test_embeddings = ae_model(X_test_ae_t)
        test_embeddings = test_embeddings.cpu().numpy()
    
    print(f"   Train embeddings shape: {train_embeddings.shape}")
    print(f"   Test embeddings shape: {test_embeddings.shape}")
    
    # KNN 预测（对每个测试样本，在训练集中找最近的 k=10 个邻居）
    print("   Computing KNN predictions (k=10)...")
    knn_pred = np.array([
        get_knn_prediction(test_emb, train_embeddings, y_train_raw, k=10)
        for test_emb in test_embeddings
    ])
    
    print(f"[OK] KNN predictions shape: {knn_pred.shape}")
    use_knn = True
    
except FileNotFoundError as e:
    print(f"[WARNING] Autoencoder files not found: {e}")
    print("[WARNING] 跳过 KNN，只使用 XGBoost + NN stacking")
    use_knn = False
    knn_pred = None
except Exception as e:
    print(f"[WARNING] Error loading Autoencoder: {e}")
    print("[WARNING] 跳过 KNN，只使用 XGBoost + NN stacking")
    use_knn = False
    knn_pred = None

# =============================================
# 5. 准备 Stacking 数据
# =============================================
y_test_real = y_test_raw  # 真实价格（£）
xgb_pred = xgb_pred_real  # XGBoost 预测（£）
nn_pred = nn_pred_real    # NN 预测（£）

print(f"\nPrediction shapes:")
print(f"  • True prices: {y_test_real.shape}")
print(f"  • XGBoost: {xgb_pred.shape}")
print(f"  • NN: {nn_pred.shape}")
if use_knn:
    print(f"  • KNN (k=10): {knn_pred.shape}")

# =============================================
# 6. 组成 Meta Input Feature Matrix
# =============================================
if use_knn:
    X_meta = np.column_stack([xgb_pred, nn_pred, knn_pred])
    print(f"\nMeta features: [XGBoost, NN, KNN(k=10)] - shape: {X_meta.shape}")
else:
    X_meta = np.column_stack([xgb_pred, nn_pred])
    print(f"\nMeta features: [XGBoost, NN] - shape: {X_meta.shape}")

# =============================================
# 7. Linear Regression Meta Model（Stacking 核心）
# =============================================
print("\n" + "="*70)
print("Training Linear Regression meta-model...")
meta = LinearRegression()
meta.fit(X_meta, y_test_real)

# =============================================
# 8. 最终 Stacking 预测
# =============================================
stack_pred = meta.predict(X_meta)

# =============================================
# 9. 评估各个模型
# =============================================
mae_xgb = mean_absolute_error(y_test_real, xgb_pred)
rmse_xgb = np.sqrt(mean_squared_error(y_test_real, xgb_pred))
r2_xgb = r2_score(y_test_real, xgb_pred)

mae_nn = mean_absolute_error(y_test_real, nn_pred)
rmse_nn = np.sqrt(mean_squared_error(y_test_real, nn_pred))
r2_nn = r2_score(y_test_real, nn_pred)

if use_knn:
    mae_knn = mean_absolute_error(y_test_real, knn_pred)
    rmse_knn = np.sqrt(mean_squared_error(y_test_real, knn_pred))
    r2_knn = r2_score(y_test_real, knn_pred)

mae_stack = mean_absolute_error(y_test_real, stack_pred)
rmse_stack = np.sqrt(mean_squared_error(y_test_real, stack_pred))
r2_stack = r2_score(y_test_real, stack_pred)

# =============================================
# 10. 准确率统计（±15£，±25£）
# =============================================
def calculate_accuracy(y_true, y_pred, tolerance):
    """计算在tolerance范围内的准确率"""
    errors = np.abs(y_true - y_pred)
    within_tolerance = np.sum(errors <= tolerance)
    return within_tolerance / len(y_true) * 100

acc_xgb_15 = calculate_accuracy(y_test_real, xgb_pred, 15)
acc_xgb_25 = calculate_accuracy(y_test_real, xgb_pred, 25)
acc_nn_15 = calculate_accuracy(y_test_real, nn_pred, 15)
acc_nn_25 = calculate_accuracy(y_test_real, nn_pred, 25)

if use_knn:
    acc_knn_15 = calculate_accuracy(y_test_real, knn_pred, 15)
    acc_knn_25 = calculate_accuracy(y_test_real, knn_pred, 25)

acc_stack_15 = calculate_accuracy(y_test_real, stack_pred, 15)
acc_stack_25 = calculate_accuracy(y_test_real, stack_pred, 25)

# =============================================
# 11. 打印结果
# =============================================
print("\n" + "="*70)
print("             🚀 STACKING RESULTS 🚀            ")
print("="*70)
print(f"\nXGBoost:")
print(f"  R²:   {r2_xgb:.4f}")
print(f"  MAE:  {mae_xgb:.4f}")
print(f"  RMSE: {rmse_xgb:.4f}")
print(f"  ±15£ 准确率: {acc_xgb_15:.2f}%")
print(f"  ±25£ 准确率: {acc_xgb_25:.2f}%")

print(f"\nNeural Network:")
print(f"  R²:   {r2_nn:.4f}")
print(f"  MAE:  {mae_nn:.4f}")
print(f"  RMSE: {rmse_nn:.4f}")
print(f"  ±15£ 准确率: {acc_nn_15:.2f}%")
print(f"  ±25£ 准确率: {acc_nn_25:.2f}%")

if use_knn:
    print(f"\nKNN (Autoencoder-based, k=10):")
    print(f"  R²:   {r2_knn:.4f}")
    print(f"  MAE:  {mae_knn:.4f}")
    print(f"  RMSE: {rmse_knn:.4f}")
    print(f"  ±15£ 准确率: {acc_knn_15:.2f}%")
    print(f"  ±25£ 准确率: {acc_knn_25:.2f}%")

print(f"\n{'─'*70}")
print(f"STACKING (Linear Regression):")
print(f"  R²:   {r2_stack:.4f}  <-- Should beat all")
print(f"  MAE:  {mae_stack:.4f}")
print(f"  RMSE: {rmse_stack:.4f}")
print(f"  ±15£ 准确率: {acc_stack_15:.2f}%")
print(f"  ±25£ 准确率: {acc_stack_25:.2f}%")
print("="*70)

# =============================================
# 12. Meta Model 权重（告诉你谁更重要）
# =============================================
print("\nMeta Model Coefficients (权重):")
print(f" • XGBoost 权重 : {meta.coef_[0]:.4f}")
print(f" • NN 权重      : {meta.coef_[1]:.4f}")
if use_knn:
    print(f" • KNN 权重     : {meta.coef_[2]:.4f}")
print(f" • Intercept    : {meta.intercept_:.4f}")

# 解释权重含义
print("\n权重解释:")
if use_knn:
    if all(c > 0 for c in meta.coef_):
        print("  [+] 三个模型都有正贡献，互补效果")
    elif meta.coef_[2] < 0:
        print("  [WARNING] KNN权重为负，说明KNN预测与其他模型高度相关但质量较差")
        print("  [INFO] 实际上KNN贡献可能较小")
else:
    if all(c > 0 for c in meta.coef_):
        print("  [+] 两个模型都有正贡献，互补效果")
    elif meta.coef_[1] < 0:
        print("  [WARNING] NN权重为负，说明NN预测与XGBoost高度相关但质量较差")

# =============================================
# 13. 随机打印10个样本的真实值和预测值
# =============================================
import random
print("\n" + "="*85)
print("            📋 随机10个样本：真实值 vs 预测值对比")
print("="*85)

indices = random.sample(range(len(y_test_real)), 10)
indices.sort()  # 排序以便查看

if use_knn:
    print(f"\n{'样本ID':<8} {'真实价格(£)':<15} {'XGBoost(£)':<15} {'NN(£)':<15} {'KNN(£)':<15} {'Stacking(£)':<15} {'误差(£)':<10}")
    print("-" * 85)
    for idx in indices:
        true_val = y_test_real[idx]
        xgb_val = xgb_pred[idx]
        nn_val = nn_pred[idx]
        knn_val = knn_pred[idx]
        stack_val = stack_pred[idx]
        error = abs(true_val - stack_val)
        
        print(f"{idx:<8} {true_val:<15.2f} {xgb_val:<15.2f} {nn_val:<15.2f} {knn_val:<15.2f} {stack_val:<15.2f} {error:<10.2f}")
else:
    print(f"\n{'样本ID':<8} {'真实价格(£)':<15} {'XGBoost(£)':<15} {'NN(£)':<15} {'Stacking(£)':<15} {'误差(£)':<10}")
    print("-" * 70)
    for idx in indices:
        true_val = y_test_real[idx]
        xgb_val = xgb_pred[idx]
        nn_val = nn_pred[idx]
        stack_val = stack_pred[idx]
        error = abs(true_val - stack_val)
        
        print(f"{idx:<8} {true_val:<15.2f} {xgb_val:<15.2f} {nn_val:<15.2f} {stack_val:<15.2f} {error:<10.2f}")

print("="*85)

# =============================================
# 14. 保存 meta model（可选）
# =============================================
with open("meta_linear_model.pkl", "wb") as f:
    pickle.dump(meta, f)
print("\nSaved meta model to meta_linear_model.pkl")

print("\nDone!")

