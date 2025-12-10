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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)

# =========================
# 1. Load data (NO manual weighting)
# =========================
df = pd.read_csv("nn_price_training_v4.csv")

target_col = "price_num"
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.float32)

# =========================
# 2. Train/Test split (20%)
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 3. Standardize
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# =========================
# 4. A2 Model
# =========================
class End2EndRegressor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

input_dim = X_train.shape[1]
model = End2EndRegressor(input_dim).to(device)

criterion = nn.L1Loss()  # MAE loss
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# =========================
# 5. Training Loop
# =========================
train_loader = DataLoader(
    TensorDataset(X_train_tensor, torch.tensor(y_train).float().to(device)),
    batch_size=256,
    shuffle=True
)

epochs = 80
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for bx, by in train_loader:
        optimizer.zero_grad()
        preds = model(bx)
        loss = criterion(preds, by)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * bx.size(0)

    if epoch % 5 == 0:
        print(f"Epoch {epoch}/{epochs}  Train MAE={total_loss/len(X_train):.4f}")

# =========================
# 6. Evaluate
# =========================
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor).cpu().numpy()

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)

print("\n===== FINAL A2 RESULTS =====")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# =========================
# 7. Plot
# =========================
plt.scatter(y_test, preds, alpha=0.3)
axis_min = min(y_test.min(), preds.min())
axis_max = max(y_test.max(), preds.max())
plt.plot([axis_min, axis_max], [axis_min, axis_max], 'r--')
plt.xlabel("True Price")
plt.ylabel("Predicted Price")
plt.title("A2: End-to-End MLP")
plt.tight_layout()
plt.show()

# =========================
# 8. Save model & scaler
# =========================
torch.save(model.state_dict(), "best_price_A2.pth")
with open("scaler_price.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Saved model best_price_A2.pth and scaler_price.pkl")
# ============================================================
# 10. 随机抽 10 个测试样本，打印真实 vs 预测
# ============================================================

import random

print("\n===== RANDOM 10 SAMPLES COMPARISON =====")
indices = random.sample(range(len(y_test)), 10)

for i in indices:
    print(f"Sample {i}: True={y_test[i]:.1f}, Pred={preds[i]:.1f}")
