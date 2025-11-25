import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle

# =========================
# 0. 基础设置
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)


# =========================
# 1. 读数据
# =========================
df = pd.read_csv("nn_price_training_v4.csv")   # 确保文件在同一文件夹

target_col = "price_num"
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].values.astype(np.float32)
y_raw = df[target_col].values.astype(np.float32)      # 原始价格
y_log = np.log1p(y_raw)                               # log(price + 1)

print("Total samples:", X.shape[0])
print("Num features:", X.shape[1])


# =========================
# 2. 随机划分 train / test（真实预测场景，不使用价格分层）
# =========================
# ⚠️ 重要：真实预测时我们不知道价格，所以训练时也不应该用价格分层
# 使用随机划分，更接近真实预测场景

X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
    X, y_log, y_raw,
    test_size=0.10,          # 10% test
    random_state=42
    # 不使用 stratify，因为真实预测时不知道价格
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])


# =========================
# 3. 标准化
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
X_test_t  = torch.tensor(X_test_scaled,  dtype=torch.float32, device=device)

y_train_log_t = torch.tensor(y_train_log, dtype=torch.float32, device=device)
y_test_log_t  = torch.tensor(y_test_log,  dtype=torch.float32, device=device)

# 为 loss 做权重（贵价房权重大一点）
y_train_raw_t = torch.tensor(y_train_raw, dtype=torch.float32, device=device)
price_mean = y_train_raw_t.mean()
# weight = 1 + 0.5 * (price / mean)，最多放大到 3 倍
sample_weights = 1.0 + 0.5 * (y_train_raw_t / price_mean)
sample_weights = torch.clamp(sample_weights, max=3.0)


# =========================
# 4. Dataloader
# =========================
train_dataset = TensorDataset(X_train_t, y_train_log_t, sample_weights)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)


# =========================
# 5. 模型定义（强化版 MLP）
# =========================
class PriceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),          # 比 ReLU 平滑一点
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


model = PriceMLP(input_dim=X_train.shape[1]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 对 log(price) 做加权 MSE
def weighted_mse_loss(pred, target, weight):
    return torch.mean(weight * (pred - target) ** 2)


# =========================
# 6. EarlyStopping
# =========================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False   # 不停
        else:
            self.counter += 1
            return self.counter >= self.patience   # True = 该停了


early = EarlyStopping(patience=20, min_delta=1e-4)


# =========================
# 7. 训练循环
# =========================
epochs = 150
best_state = None
best_val = float("inf")

print("\nStart training Log-Price MLP (A2)...")

for epoch in range(epochs):
    model.train()
    total_train_loss = 0.0

    for xb, yb, wb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = weighted_mse_loss(pred, yb, wb)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item() * xb.size(0)

    train_loss = total_train_loss / len(train_dataset)

    # validation（用 log-price 验证）
    model.eval()
    with torch.no_grad():
        val_pred_log = model(X_test_t)
        val_loss = weighted_mse_loss(val_pred_log, y_test_log_t, torch.ones_like(y_test_log_t))

    if val_loss.item() < best_val:
        best_val = val_loss.item()
        best_state = model.state_dict()

    if epoch % 5 == 0:
        print(f"Epoch {epoch:3d}/{epochs}  Train={train_loss:.4f}  Val={val_loss.item():.4f}")

    if early.step(val_loss.item()):
        print(f"EARLY STOP at epoch {epoch}")
        break

# 使用最好的一次参数
if best_state is not None:
    model.load_state_dict(best_state)


# =========================
# 8. 在真实价格空间上做评估
# =========================
model.eval()
with torch.no_grad():
    pred_log = model(X_test_t).cpu().numpy()

pred_price = np.expm1(pred_log)      # 反 log1p
true_price = y_test_raw              # numpy array

mae = mean_absolute_error(true_price, pred_price)
rmse = np.sqrt(mean_squared_error(true_price, pred_price))
r2 = r2_score(true_price, pred_price)

print("\n===== FINAL LOG-A2 RESULTS (in REAL £) =====")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")


# =========================
# 9. 随机抽 10 个样本看看 True vs Pred
# =========================
import random
print("\n===== RANDOM 10 TEST SAMPLES (True vs Pred) =====")
indices = random.sample(range(len(true_price)), 10)

for i in indices:
    print(f"Sample {i:4d}: True = {true_price[i]:7.1f}  |  Pred = {pred_price[i]:7.1f}")


# =========================
# 10. 画散点图
# =========================
plt.figure(figsize=(6, 6))
plt.scatter(true_price, pred_price, alpha=0.35)
axis_min = min(true_price.min(), pred_price.min())
axis_max = max(true_price.max(), pred_price.max())
plt.plot([axis_min, axis_max], [axis_min, axis_max], 'r--')
plt.xlabel("True price (£)")
plt.ylabel("Predicted price (£)")
plt.title("A2 Log-Price MLP — True vs Predicted")
plt.tight_layout()
plt.show()


# =========================
# 11. 保存模型和 scaler
# =========================
torch.save(model.state_dict(), "best_price_A2_log.pth")
with open("scaler_price.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nSaved model to best_price_A2_log.pth and scaler to scaler_price.pkl")

# =========================
# 12. 检查特定样本（可选）
# =========================
def inspect_sample(idx):
    """检查测试集中的特定样本"""
    if idx < 0 or idx >= len(true_price):
        print(f"索引 {idx} 超出范围 (0-{len(true_price)-1})")
        return
    
    print(f"\n===== Inspecting Sample #{idx} =====")
    print(f"True Price: £{true_price[idx]:.2f}")
    print(f"Predicted Price: £{pred_price[idx]:.2f}")
    print(f"Error: £{abs(true_price[idx] - pred_price[idx]):.2f}")
    
    print("\n--- Features ---")
    # 获取原始特征（未标准化的）
    X_test_original = scaler.inverse_transform(X_test_nn[idx:idx+1])[0]
    for n, v in zip(feature_cols, X_test_original):
        print(f"{n:25s} : {v:.4f}")

# 取消注释下面这行来检查特定样本
# inspect_sample(3691)
