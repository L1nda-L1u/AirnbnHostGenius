import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# 1. GPU 或 CPU
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 2. 加载数据
# ============================================================
df = pd.read_csv("nn_price_training_v4.csv")

numeric_features = [
    "latitude", "longitude",
    "accommodates", "bathrooms",
    "bedrooms", "beds",
    "review_scores_cleanliness",
    "review_scores_location",
    "bath_num",
    "location_cluster",   # ⬅ 新加这一行
]


amenity_features = [c for c in df.columns if c.startswith("amenity_")]

X = df[numeric_features + amenity_features].values.astype(np.float32)
y = df["price_num"].values.astype(np.float32).reshape(-1, 1)

# ============================================================
# 3. Train-test split
# ============================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ============================================================
# 4. 标准化（神经网络非常需要）
# ============================================================
scaler = StandardScaler()
X_train[:, :len(numeric_features)] = scaler.fit_transform(X_train[:, :len(numeric_features)])
X_test[:, :len(numeric_features)] = scaler.transform(X_test[:, :len(numeric_features)])

# 转成 tensor
X_train_t = torch.tensor(X_train).to(device)
y_train_t = torch.tensor(y_train).to(device)
X_test_t = torch.tensor(X_test).to(device)
y_test_t = torch.tensor(y_test).to(device)

# ============================================================
# 5. 定义最简单的神经网络
# ============================================================
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.layers(x)


model = SimpleNN(input_dim=X_train.shape[1]).to(device)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ============================================================
# 6. 训练（最简单版本）
# ============================================================
epochs = 40
for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(X_train_t)
    loss = loss_fn(pred, y_train_t)
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}  Loss={loss.item():.4f}")

# ============================================================
# 7. 测试并输出 MAE / RMSE / R²
# ============================================================
with torch.no_grad():
    preds = model(X_test_t).cpu().numpy().flatten()

mae = mean_absolute_error(y_test, preds)

# 旧版 sklearn 没有 squared 参数 → 先算 MSE 再开根号
mse = mean_squared_error(y_test, preds)
rmse = np.sqrt(mse)

r2 = r2_score(y_test, preds)

print("\n===== RESULTS =====")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

mean_price = y_train.mean()
baseline_preds = np.full_like(y_test, mean_price)

baseline_mae = mean_absolute_error(y_test, baseline_preds)
baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))
baseline_r2 = r2_score(y_test, baseline_preds)

print("\n===== BASELINE (Always predict mean price) =====")
print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)
print("Baseline R²:", baseline_r2)
