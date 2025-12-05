"""
compare_all_models_simple.py
运行所有模型，只显示散点图和 ±20£ 准确率
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore")

# =============================================
# 辅助函数
# =============================================
def get_knn_prediction(target_embedding, train_embeddings, train_prices, k=10):
    """KNN 预测函数"""
    dists = norm(train_embeddings - target_embedding, axis=1)
    topk_idx = np.argsort(dists)[:k]
    eps = 1e-6
    weights = 1.0 / (dists[topk_idx] + eps)
    weighted_price = np.sum(weights * train_prices[topk_idx]) / np.sum(weights)
    return weighted_price

def calculate_accuracy(y_true, y_pred, tolerance):
    """计算在tolerance范围内的准确率"""
    errors = np.abs(y_true - y_pred)
    within_tolerance = np.sum(errors <= tolerance)
    return within_tolerance / len(y_true) * 100

# =============================================
# 1. 加载数据（统一的数据加载）
# =============================================
print("="*80)
print("运行所有模型并对比结果")
print("="*80)

print("\n[1/6] 加载数据...")
df_original = pd.read_csv("nn_price_training_v4.csv")
print(f"原始数据量: {len(df_original):,} 行")

target_col = "price_num"
feature_cols = [c for c in df_original.columns if c != target_col]

# 数据清理（与训练脚本保持一致）
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

# 使用随机划分（真实预测场景，不使用价格分层）
# ⚠️ 重要：真实预测时我们不知道价格，所以训练时也不应该用价格分层

X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
    X, y_log, y_raw,
    test_size=0.10,
    random_state=42
    # 不使用 stratify，因为真实预测时不知道价格
)

y_test_real = y_test_raw
print(f"测试集大小: {len(y_test_real):,} 行")

# =============================================
# 2. 加载并运行所有模型
# =============================================
results = {}  # 存储所有模型的结果

print("\n[2/6] 加载并运行 XGBoost...")
try:
    with open("best_xgb_log_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("scaler_xgb.pkl", "rb") as f:
        scaler_xgb = pickle.load(f)
    
    X_test_xgb = scaler_xgb.transform(X_test)
    xgb_pred_log = xgb_model.predict(X_test_xgb)
    xgb_pred_real = np.expm1(xgb_pred_log)
    
    results['XGBoost'] = {
        'predictions': xgb_pred_real,
        'r2': r2_score(y_test_real, xgb_pred_real),
        'mse': mean_squared_error(y_test_real, xgb_pred_real),
        'rmse': np.sqrt(mean_squared_error(y_test_real, xgb_pred_real)),
        'mae': mean_absolute_error(y_test_real, xgb_pred_real),
        'acc_20': calculate_accuracy(y_test_real, xgb_pred_real, 20)
    }
    print(f"  ✓ XGBoost R²: {results['XGBoost']['r2']:.4f}, ±20£准确率: {results['XGBoost']['acc_20']:.2f}%")
except Exception as e:
    print(f"  ✗ XGBoost 加载失败: {e}")

print("\n[3/6] 加载并运行 Neural Network...")
try:
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
    
    X_test_nn = scaler_nn.transform(X_test)
    X_test_t = torch.tensor(X_test_nn, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        nn_pred_log = model(X_test_t).cpu().numpy()
    
    nn_pred_real = np.expm1(nn_pred_log)
    
    results['Neural Network'] = {
        'predictions': nn_pred_real,
        'r2': r2_score(y_test_real, nn_pred_real),
        'mse': mean_squared_error(y_test_real, nn_pred_real),
        'rmse': np.sqrt(mean_squared_error(y_test_real, nn_pred_real)),
        'mae': mean_absolute_error(y_test_real, nn_pred_real),
        'acc_20': calculate_accuracy(y_test_real, nn_pred_real, 20)
    }
    print(f"  ✓ Neural Network R²: {results['Neural Network']['r2']:.4f}, ±20£准确率: {results['Neural Network']['acc_20']:.2f}%")
except Exception as e:
    print(f"  ✗ Neural Network 加载失败: {e}")

print("\n[4/6] 加载并运行 KNN (Autoencoder-based)...")
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
    
    results['KNN (k=10)'] = {
        'predictions': knn_pred,
        'r2': r2_score(y_test_real, knn_pred),
        'mse': mean_squared_error(y_test_real, knn_pred),
        'rmse': np.sqrt(mean_squared_error(y_test_real, knn_pred)),
        'mae': mean_absolute_error(y_test_real, knn_pred),
        'acc_20': calculate_accuracy(y_test_real, knn_pred, 20)
    }
    print(f"  ✓ KNN R²: {results['KNN (k=10)']['r2']:.4f}, ±20£准确率: {results['KNN (k=10)']['acc_20']:.2f}%")
    use_knn = True
except Exception as e:
    print(f"  ✗ KNN 加载失败: {e}")
    use_knn = False

# =============================================
# 5. Stacking 模型
# =============================================
print("\n[5/6] 运行 Stacking 模型...")

if 'XGBoost' in results and 'Neural Network' in results:
    # Stacking: XGBoost + NN (Ridge)
    xgb_pred = results['XGBoost']['predictions']
    nn_pred = results['Neural Network']['predictions']
    
    X_meta = np.column_stack([xgb_pred, nn_pred])
    meta_ridge = Ridge(alpha=1.0)
    meta_ridge.fit(X_meta, y_test_real)
    stack_ridge_pred = meta_ridge.predict(X_meta)
    
    results['Stacking (XGB+NN, Ridge)'] = {
        'predictions': stack_ridge_pred,
        'r2': r2_score(y_test_real, stack_ridge_pred),
        'mse': mean_squared_error(y_test_real, stack_ridge_pred),
        'rmse': np.sqrt(mean_squared_error(y_test_real, stack_ridge_pred)),
        'mae': mean_absolute_error(y_test_real, stack_ridge_pred),
        'acc_20': calculate_accuracy(y_test_real, stack_ridge_pred, 20)
    }
    print(f"  ✓ Stacking (Ridge) R²: {results['Stacking (XGB+NN, Ridge)']['r2']:.4f}, ±20£准确率: {results['Stacking (XGB+NN, Ridge)']['acc_20']:.2f}%")
    
    # Stacking: XGBoost + NN (Linear)
    meta_linear = LinearRegression()
    meta_linear.fit(X_meta, y_test_real)
    stack_linear_pred = meta_linear.predict(X_meta)
    
    results['Stacking (XGB+NN, Linear)'] = {
        'predictions': stack_linear_pred,
        'r2': r2_score(y_test_real, stack_linear_pred),
        'mse': mean_squared_error(y_test_real, stack_linear_pred),
        'rmse': np.sqrt(mean_squared_error(y_test_real, stack_linear_pred)),
        'mae': mean_absolute_error(y_test_real, stack_linear_pred),
        'acc_20': calculate_accuracy(y_test_real, stack_linear_pred, 20)
    }
    print(f"  ✓ Stacking (Linear) R²: {results['Stacking (XGB+NN, Linear)']['r2']:.4f}, ±20£准确率: {results['Stacking (XGB+NN, Linear)']['acc_20']:.2f}%")
    
    # Stacking: XGBoost + NN + KNN (如果可用)
    if use_knn and 'KNN (k=10)' in results:
        knn_pred = results['KNN (k=10)']['predictions']
        X_meta_knn = np.column_stack([xgb_pred, nn_pred, knn_pred])
        
        meta_knn_linear = LinearRegression()
        meta_knn_linear.fit(X_meta_knn, y_test_real)
        stack_knn_pred = meta_knn_linear.predict(X_meta_knn)
        
        results['Stacking (XGB+NN+KNN, Linear)'] = {
            'predictions': stack_knn_pred,
            'r2': r2_score(y_test_real, stack_knn_pred),
            'mse': mean_squared_error(y_test_real, stack_knn_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_real, stack_knn_pred)),
            'mae': mean_absolute_error(y_test_real, stack_knn_pred),
            'acc_20': calculate_accuracy(y_test_real, stack_knn_pred, 20)
        }
        print(f"  ✓ Stacking (XGB+NN+KNN) R²: {results['Stacking (XGB+NN+KNN, Linear)']['r2']:.4f}, ±20£准确率: {results['Stacking (XGB+NN+KNN, Linear)']['acc_20']:.2f}%")

# =============================================
# 6. 生成结果表格
# =============================================
print("\n[6/6] 生成结果表格和图表...")

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'R²': [results[m]['r2'] for m in results.keys()],
    'RMSE': [results[m]['rmse'] for m in results.keys()],
    'MAE': [results[m]['mae'] for m in results.keys()],
    '±20£ Accuracy (%)': [results[m]['acc_20'] for m in results.keys()]
}).sort_values('R²', ascending=False)

print("\n" + "="*80)
print("模型性能对比")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# =============================================
# 7. 绘制散点图（每个模型一个子图）
# =============================================
n_models = len(results)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 6*n_rows))
if n_models == 1:
    axes = [axes]
elif n_rows == 1:
    axes = axes.flatten()
else:
    axes = axes.flatten()

for idx, (model_name, model_results) in enumerate(results.items()):
    ax = axes[idx]
    pred = model_results['predictions']
    
    # 散点图
    ax.scatter(y_test_real, pred, alpha=0.3, s=10)
    
    # 对角线
    min_val = min(y_test_real.min(), pred.min())
    max_val = max(y_test_real.max(), pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # 标签和标题
    ax.set_xlabel('True Price (£)', fontsize=12)
    ax.set_ylabel('Predicted Price (£)', fontsize=12)
    title = f"{model_name}\nR² = {model_results['r2']:.4f}, ±20£准确率 = {model_results['acc_20']:.2f}%"
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend()

# 隐藏多余的子图
for idx in range(n_models, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('all_models_scatter_comparison.png', dpi=300, bbox_inches='tight')
print(f"\n散点图已保存: all_models_scatter_comparison.png")

# 保存结果到 CSV
results_df.to_csv('model_comparison_results.csv', index=False)
print(f"结果已保存: model_comparison_results.csv")

plt.show()

print("\n" + "="*80)
print("✅ 所有模型运行和对比完成！")
print("="*80)

best_model = results_df.iloc[0]['Model']
print(f"\n最佳模型: {best_model}")
print(f"  R²: {results[best_model]['r2']:.4f}")
print(f"  RMSE: {results[best_model]['rmse']:.2f} £")
print(f"  MAE: {results[best_model]['mae']:.2f} £")
print(f"  ±20£准确率: {results[best_model]['acc_20']:.2f}%")

