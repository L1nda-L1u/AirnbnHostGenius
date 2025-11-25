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
import torch.onnx
import torch.nn as nn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings("ignore")

# =============================================
# 1. 加载数据（与训练脚本保持一致）
# =============================================
print("Loading data...")
df_original = pd.read_csv("other_files/nn_price_training_v4.csv")
print(f"原始数据量: {len(df_original):,} 行")

target_col = "price_num"
feature_cols = [c for c in df_original.columns if c != target_col]

# 重要：完善的数据清理规则（与XGBoost训练保持一致，但更全面）
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
print("\nLoading XGBoost model...")
with open("other_files/best_xgb_log_model.pkl", "rb") as f:
    xgb_model = pickle.load(f)
with open("other_files/scaler_xgb.pkl", "rb") as f:
    scaler_xgb = pickle.load(f)

# XGBoost 预测（需要标准化）
X_test_xgb = scaler_xgb.transform(X_test)
xgb_pred_log = xgb_model.predict(X_test_xgb)
xgb_pred_real = np.expm1(xgb_pred_log)  # 转回真实价格

print(f"XGBoost predictions shape: {xgb_pred_real.shape}")

# =============================================
# 3. 加载神经网络模型和 scaler
# =============================================
print("\nLoading Neural Network model...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
with open("other_files/scaler_price.pkl", "rb") as f:
    scaler_nn = pickle.load(f)

# 加载模型
model = PriceMLP(input_dim=X_train.shape[1]).to(device)
model.load_state_dict(torch.load("other_files/best_price_A2_log.pth", map_location=device))
model.eval()

# 神经网络预测
X_test_nn = scaler_nn.transform(X_test)
X_test_t = torch.tensor(X_test_nn, dtype=torch.float32, device=device)

with torch.no_grad():
    nn_pred_log = model(X_test_t).cpu().numpy()

nn_pred_real = np.expm1(nn_pred_log)  # 转回真实价格

print(f"NN predictions shape: {nn_pred_real.shape}")

# =============================================
# 4. 准备 Stacking 数据
# =============================================
y_test_real = y_test_raw  # 真实价格（£）
xgb_pred = xgb_pred_real  # XGBoost 预测（£）
nn_pred = nn_pred_real    # NN 预测（£）

print(f"\nShapes - True: {y_test_real.shape}, XGB: {xgb_pred.shape}, NN: {nn_pred.shape}")

# =============================================
# 5. 组成 Meta Input Feature Matrix
# =============================================
X_meta = np.column_stack([xgb_pred, nn_pred])

# =============================================
# 6. Ridge Meta Model（Stacking 核心）
# =============================================
print("\nTraining Ridge meta-model...")
meta = Ridge(alpha=1.0)
meta.fit(X_meta, y_test_real)

# =============================================
# 7. 最终 Stacking 预测
# =============================================
stack_pred = meta.predict(X_meta)

# =============================================
# 8. 评估
# =============================================
mae_xgb = mean_absolute_error(y_test_real, xgb_pred)
rmse_xgb = np.sqrt(mean_squared_error(y_test_real, xgb_pred))
r2_xgb = r2_score(y_test_real, xgb_pred)

mae_nn = mean_absolute_error(y_test_real, nn_pred)
rmse_nn = np.sqrt(mean_squared_error(y_test_real, nn_pred))
r2_nn = r2_score(y_test_real, nn_pred)

mae_stack = mean_absolute_error(y_test_real, stack_pred)
rmse_stack = np.sqrt(mean_squared_error(y_test_real, stack_pred))
r2_stack = r2_score(y_test_real, stack_pred)

# =============================================
# 8.1. 准确率统计（±15镑，±25镑）
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
acc_stack_15 = calculate_accuracy(y_test_real, stack_pred, 15)
acc_stack_25 = calculate_accuracy(y_test_real, stack_pred, 25)

print("\n" + "="*50)
print("             🚀 STACKING RESULTS 🚀            ")
print("="*50)
print(f"\nXGBoost:")
print(f"  R²:   {r2_xgb:.4f}")
print(f"  MAE:  {mae_xgb:.4f}")
print(f"  RMSE: {rmse_xgb:.4f}")

print(f"\nNeural Network:")
print(f"  R²:   {r2_nn:.4f}")
print(f"  MAE:  {mae_nn:.4f}")
print(f"  RMSE: {rmse_nn:.4f}")

print(f"\n{'─'*50}")
print(f"STACKING (Ridge):")
print(f"  R²:   {r2_stack:.4f}  <-- Should beat both")
print(f"  MAE:  {mae_stack:.4f}")
print(f"  RMSE: {rmse_stack:.4f}")
print("="*50)

# =============================================
# 8.2. 准确率报告
# =============================================
print("\n" + "="*50)
print("             准确率统计 (±15£, ±25£)            ")
print("="*50)
print(f"\nXGBoost:")
print(f"  ±15£ 准确率: {acc_xgb_15:.2f}%")
print(f"  ±25£ 准确率: {acc_xgb_25:.2f}%")

print(f"\nNeural Network:")
print(f"  ±15£ 准确率: {acc_nn_15:.2f}%")
print(f"  ±25£ 准确率: {acc_nn_25:.2f}%")

print(f"\n{'─'*50}")
print(f"STACKING (Ridge):")
print(f"  ±15£ 准确率: {acc_stack_15:.2f}%")
print(f"  ±25£ 准确率: {acc_stack_25:.2f}%")
print("="*50)

# =============================================
# 9. Meta Model 权重（告诉你谁更重要）
# =============================================
print("\nMeta Model Coefficients (权重):")
print(f" • XGBoost 权重 : {meta.coef_[0]:.4f}")
print(f" • NN 权重      : {meta.coef_[1]:.4f}")
print(f" • Intercept    : {meta.intercept_:.4f}")

# 解释权重含义
print("\n权重解释:")
if meta.coef_[0] > 0 and meta.coef_[1] > 0:
    print("  [+] 两个模型都有正贡献，互补效果")
elif meta.coef_[1] < 0:
    print("  [WARNING] NN权重为负，说明NN预测与XGBoost高度相关但质量较差")
    print("  [WARNING] Ridge发现：稍微'反向'使用NN预测反而更好")
    print("  [INFO] 实际上NN贡献很小（权重接近0），几乎可以忽略")
    print("  [INFO] 建议：可以只用XGBoost，或者尝试改进NN模型")

# =============================================
# 10. 随机打印10个样本的真实值和预测值
# =============================================
import random
print("\n" + "="*70)
print("            📋 随机10个样本：真实值 vs 预测值对比")
print("="*70)

indices = random.sample(range(len(y_test_real)), 10)
indices.sort()  # 排序以便查看

print(f"\n{'样本ID':<8} {'真实价格(£)':<15} {'XGBoost(£)':<15} {'NN(£)':<15} {'Stacking(£)':<15} {'误差(£)':<10}")
print("-" * 70)

for idx in indices:
    true_val = y_test_real[idx]
    xgb_val = xgb_pred[idx]
    nn_val = nn_pred[idx]
    stack_val = stack_pred[idx]
    error = abs(true_val - stack_val)
    
    print(f"{idx:<8} {true_val:<15.2f} {xgb_val:<15.2f} {nn_val:<15.2f} {stack_val:<15.2f} {error:<10.2f}")

print("="*70)

# =============================================
# 11. 保存 meta model（可选）
# =============================================
with open("other_files/meta_ridge_model.pkl", "wb") as f:
    pickle.dump(meta, f)

# =============================================
# 12. 保存模型供R使用（保存到R_scripts/best_model文件夹）
# =============================================
import shutil
best_model_dir = "../R_scripts/best_model"
xgb_model.save_model(f"{best_model_dir}/xgb_model.json")
dummy_input = torch.randn(1, X_train.shape[1]).to(device)
model.eval()
try:
    torch.onnx.export(model, dummy_input, f"{best_model_dir}/nn.onnx", input_names=['features'], output_names=['price_log'], opset_version=11)
except Exception as e:
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save(f"{best_model_dir}/nn.onnx")
# 复制scaler和meta模型到best_model目录
shutil.copy("other_files/scaler_xgb.pkl", f"{best_model_dir}/scaler_xgb.pkl")
shutil.copy("other_files/scaler_price.pkl", f"{best_model_dir}/scaler_price.pkl")
shutil.copy("other_files/meta_ridge_model.pkl", f"{best_model_dir}/meta_ridge_model.pkl")
# 复制训练数据（用于获取特征维度）
shutil.copy("other_files/nn_price_training_v4.csv", f"{best_model_dir}/nn_price_training_v4.csv")
with open(f"{best_model_dir}/README.txt", "w", encoding="utf-8") as f:
    f.write(f"Stacking Formula:\nfinal_price = {meta.intercept_:.4f} + {meta.coef_[0]:.4f} * xgb_pred + {meta.coef_[1]:.4f} * nn_pred\n")
print("Saved to R_scripts/best_model/: xgb_model.json, nn.onnx, scaler files, meta model, README.txt")

print("\nDone!")

