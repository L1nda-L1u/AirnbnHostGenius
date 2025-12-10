import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import matplotlib.pyplot as plt
import random
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
df_original = pd.read_csv("nn_price_training_v4.csv")
print(f"原始数据量: {len(df_original):,} 行")

# =====================================================
# 1. Print all features you are using
# =====================================================
target_col = "price_num"
feature_cols = [c for c in df_original.columns if c != target_col]

print("\n===== ALL FEATURES USED =====")
for i, f in enumerate(feature_cols):
    print(f"{i+1}. {f}")
print(f"\nTotal features = {len(feature_cols)}\n")

# =====================================================
# 2. 完善的数据清理规则（与 stacking 保持一致）
# =====================================================
print("Cleaning outliers (完善版)...")
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

# =====================================================
# 3. LOG-transform price → improves model stability
# =====================================================
X = df[feature_cols].values.astype(np.float32)
y_raw = df[target_col].values.astype(np.float32)
y_log = np.log1p(y_raw)

# =====================================================
# 4. Train/Test split（随机划分，真实预测场景）
# =====================================================
# ⚠️ 重要：真实预测时我们不知道价格，所以训练时也不应该用价格分层
# 使用随机划分，更接近真实预测场景

print("\n使用随机划分（真实预测场景，不使用价格分层）...")
X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
    X, y_log, y_raw,
    test_size=0.10,
    random_state=42
    # 不使用 stratify，因为真实预测时不知道价格
)

# 为了兼容后续代码，创建 y_train 和 y_test
y_train = y_train_log
y_test = y_test_log

# 获取 df_train 和 df_test（用于后续显示）
# 注意：由于 train_test_split 会打乱顺序，我们需要重新匹配
# 这里简化处理，直接使用索引
train_size = len(X_train)
df_train = df.iloc[:train_size].reset_index(drop=True) if train_size <= len(df) else df.iloc[:len(df)//10*9].reset_index(drop=True)
df_test = df.iloc[train_size:].reset_index(drop=True) if train_size < len(df) else df.iloc[len(df)//10*9:].reset_index(drop=True)

print(f"\n数据划分统计:")
print(f"  训练集: {X_train.shape[0]:,} 行")
print(f"  测试集: {X_test.shape[0]:,} 行")
print(f"  注意: 使用随机划分（真实预测场景，不使用价格分层）")

# =====================================================
# 5. Standardize features
# =====================================================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =====================================================
# 6. Train XGBoost (GPU accelerated)
# =====================================================
print("\nTraining XGBoost (GPU)...")

xgb_model = xgb.XGBRegressor(
    n_estimators=800,
    max_depth=10,
    learning_rate=0.025,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=1.2,

    # GPU acceleration
    tree_method="gpu_hist",
    predictor="gpu_predictor",

    eval_metric="rmse",   # must be here for XGBoost>=2.0
    random_state=42,
)

xgb_model.fit(X_train, y_train)

# =====================================================
# 7. Evaluation
# =====================================================
log_pred = xgb_model.predict(X_test)
pred_real = np.expm1(log_pred)    # convert back to £
true_real = np.expm1(y_test)

mae = mean_absolute_error(true_real, pred_real)
rmse = np.sqrt(mean_squared_error(true_real, pred_real))
r2 = r2_score(true_real, pred_real)

print("\n===== FINAL XGBOOST RESULTS (REAL £) =====")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# =====================================================
# 8. Scatter Plot
# =====================================================
plt.figure(figsize=(7,7))
plt.scatter(true_real, pred_real, alpha=0.3)
min_v = min(true_real.min(), pred_real.min())
max_v = max(true_real.max(), pred_real.max())
plt.plot([min_v, max_v], [min_v, max_v], 'r--')
plt.xlabel("True Price (£)")
plt.ylabel("Predicted Price (£)")
plt.title("XGBoost Predictions vs True Prices")
plt.tight_layout()
plt.savefig("xgb_price_scatter.png", dpi=300)
plt.show()

print("\nScatter plot saved: xgb_price_scatter.png")

# =====================================================
# 9. Print 10 random samples (true vs predicted vs features)
# =====================================================
print("\n===== RANDOM 10 TEST SAMPLES =====")
indices = random.sample(range(len(true_real)), 10)

for idx in indices:
    print("\n-------------------------------")
    print(f"Sample #{idx}")
    print(f"True price: £{true_real[idx]:.2f}")
    print(f"Predicted:  £{pred_real[idx]:.2f}")
    print("------ Feature Values ------")
    print(df_test.iloc[idx][feature_cols].head(20))  # print first 20 features

# =====================================================
# 10. Save Model
# =====================================================
import pickle
with open("best_xgb_log_model.pkl", "wb") as f:
    pickle.dump(xgb_model, f)
with open("scaler_xgb.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nSaved: best_xgb_log_model.pkl + scaler_xgb.pkl")
print("Done.")

