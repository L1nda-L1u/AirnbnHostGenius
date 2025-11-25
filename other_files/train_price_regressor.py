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
# 0. 一些基础设置
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

torch.manual_seed(42)
np.random.seed(42)

# =========================
# 1. 读数据
# =========================
df = pd.read_csv("nn_price_training_v4.csv")

# y = price, X = 除 price 以外的全部特征
target_col = "price_num"
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.float32)

print("Num samples:", X.shape[0])
print("Num features:", X.shape[1])

# =========================
# 2. train / test 划分
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# =========================
# 3. 标准化（只在 train 上 fit）
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

# =========================
# 4. 定义 Autoencoder
# =========================
input_dim = X_train.shape[1]
latent_dim = 16  # embedding 维度，可以改 32 试试

class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim):
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

ae_model = Autoencoder(input_dim, latent_dim).to(device)
ae_criterion = nn.MSELoss()
ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)

# =========================
# 5. 训练 Autoencoder（只用 train）
# =========================
batch_size = 512
train_dataset_ae = TensorDataset(X_train_tensor)
train_loader_ae = DataLoader(train_dataset_ae, batch_size=batch_size, shuffle=True)

epochs_ae = 40
print("Start training autoencoder...")

for epoch in range(epochs_ae):
    ae_model.train()
    total_loss = 0.0

    for (batch_x,) in train_loader_ae:
        ae_optimizer.zero_grad()
        recon, z = ae_model(batch_x)
        loss = ae_criterion(recon, batch_x)
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item() * batch_x.size(0)

    avg_loss = total_loss / len(train_dataset_ae)
    print(f"AE Epoch {epoch+1}/{epochs_ae}  Recon Loss={avg_loss:.4f}")

# =========================
# 6. 得到 train / test 的 embedding
# =========================
ae_model.eval()
with torch.no_grad():
    _, Z_train = ae_model(X_train_tensor)
    _, Z_test = ae_model(X_test_tensor)

emb_train = Z_train.cpu().numpy()
emb_test = Z_test.cpu().numpy()

print("Embedding shapes:", emb_train.shape, emb_test.shape)

# =========================
# 7. 定义 Deep Price Regressor（MLP）
# =========================
class PriceRegressor(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, z):
        out = self.net(z)
        return out.squeeze(1)  # [batch] 而不是 [batch,1]

reg_model = PriceRegressor(latent_dim).to(device)
reg_criterion = nn.L1Loss()  # MAE 作为训练目标，更抗 outlier
reg_optimizer = torch.optim.Adam(reg_model.parameters(), lr=1e-3)

# 准备 DataLoader
train_dataset_reg = TensorDataset(
    torch.tensor(emb_train, dtype=torch.float32).to(device),
    torch.tensor(y_train, dtype=torch.float32).to(device)
)
train_loader_reg = DataLoader(train_dataset_reg, batch_size=256, shuffle=True)

# =========================
# 8. 训练回归器
# =========================
epochs_reg = 60
print("\nStart training price regressor...")

for epoch in range(epochs_reg):
    reg_model.train()
    total_loss = 0.0

    for batch_z, batch_y in train_loader_reg:
        reg_optimizer.zero_grad()
        preds = reg_model(batch_z)
        loss = reg_criterion(preds, batch_y)
        loss.backward()
        reg_optimizer.step()
        total_loss += loss.item() * batch_z.size(0)

    avg_loss = total_loss / len(train_dataset_reg)
    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Reg Epoch {epoch+1}/{epochs_reg}  Train MAE Loss={avg_loss:.4f}")

# =========================
# 9. 在 test 集上评估
# =========================
reg_model.eval()
with torch.no_grad():
    test_preds = reg_model(
        torch.tensor(emb_test, dtype=torch.float32).to(device)
    ).cpu().numpy()

mae = mean_absolute_error(y_test, test_preds)
rmse = np.sqrt(mean_squared_error(y_test, test_preds))
r2 = r2_score(y_test, test_preds)

print("\n===== PRICE REGRESSOR RESULTS (on test set) =====")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")

# =========================
# 10. 画 True vs Predicted
# =========================
plt.figure(figsize=(6, 6))
plt.scatter(y_test, test_preds, alpha=0.4)
min_axis = min(y_test.min(), test_preds.min())
max_axis = max(y_test.max(), test_preds.max())
plt.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--")
plt.xlabel("True price")
plt.ylabel("Predicted price (deep regressor)")
plt.title("Deep regression price prediction vs true price")
plt.tight_layout()
plt.show()

# =========================
# 11. 保存模型 & scaler 方便之后 ensemble 使用
# =========================
torch.save(ae_model.state_dict(), "ae_price_model.pth")
torch.save(reg_model.state_dict(), "price_regressor.pth")
with open("scaler_price.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nSaved ae_price_model.pth, price_regressor.pth and scaler_price.pkl")
