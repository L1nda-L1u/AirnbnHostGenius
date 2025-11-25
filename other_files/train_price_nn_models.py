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
# EarlyStopping
# =========================
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False


# =========================
# 1. 读数据 + 增强地理权重
# =========================
df = pd.read_csv("nn_price_training_v4.csv")

# ★★★ 地理位置权重增强（位置影响价格非常大）★★★
df['latitude'] *= 3
df['longitude'] *= 3

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
# 3. 标准化
# =========================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)


# =========================
# 4. Autoencoder
# =========================
input_dim = X_train.shape[1]
latent_dim = 16

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


# =========================
# 5. A1: Autoencoder + Regressor
# =========================
print("\n===== A1: Autoencoder + Regressor =====")

ae_model = Autoencoder(input_dim, latent_dim).to(device)
ae_criterion = nn.MSELoss()
ae_optimizer = torch.optim.Adam(ae_model.parameters(), lr=1e-3)

batch_size_ae = 512
train_dataset_ae = TensorDataset(X_train_tensor)
train_loader_ae = DataLoader(train_dataset_ae, batch_size=batch_size_ae, shuffle=True)

epochs_ae = 60
print("Start training Autoencoder...")

for epoch in range(epochs_ae):
    ae_model.train()
    total_loss = 0.0

    for (batch_x,) in train_loader_ae:
        ae_optimizer.zero_grad()
        recon, _ = ae_model(batch_x)
        loss = ae_criterion(recon, batch_x)
        loss.backward()
        ae_optimizer.step()
        total_loss += loss.item() * batch_x.size(0)

    if (epoch + 1) % 5 == 0:
        print(f"AE Epoch {epoch+1}/{epochs_ae}  Loss={total_loss/len(train_dataset_ae):.4f}")

# 得 embedding
ae_model.eval()
with torch.no_grad():
    _, Z_train = ae_model(X_train_tensor)
    _, Z_test = ae_model(X_test_tensor)

emb_train = Z_train.cpu().numpy()
emb_test = Z_test.cpu().numpy()


# A1 回归器
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

    def forward(self, x):
        return self.net(x).squeeze(1)


reg_model_a1 = PriceRegressor(latent_dim).to(device)
reg_opt = torch.optim.Adam(reg_model_a1.parameters(), lr=1e-3)
reg_loss_fn = nn.L1Loss()

train_loader_reg = DataLoader(
    TensorDataset(
        torch.tensor(emb_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.float32).to(device)
    ),
    batch_size=256,
    shuffle=True
)

epochs_reg_a1 = 70
print("\nTraining A1 regressor...")

for epoch in range(epochs_reg_a1):
    reg_model_a1.train()
    total_loss = 0.0
    for bz, by in train_loader_reg:
        reg_opt.zero_grad()
        pred = reg_model_a1(bz)
        loss = reg_loss_fn(pred, by)
        loss.backward()
        reg_opt.step()
        total_loss += loss.item() * bz.size(0)

    if (epoch + 1) % 5 == 0:
        print(f"A1 Epoch {epoch+1}/{epochs_reg_a1}  MAE={total_loss/len(train_loader_reg.dataset):.4f}")

reg_model_a1.eval()
with torch.no_grad():
    pred_a1 = reg_model_a1(torch.tensor(emb_test, dtype=torch.float32).to(device)).cpu().numpy()

mae_a1 = mean_absolute_error(y_test, pred_a1)
rmse_a1 = np.sqrt(mean_squared_error(y_test, pred_a1))
r2_a1 = r2_score(y_test, pred_a1)

print("\n===== A1 RESULTS =====")
print(f"A1 MAE:  {mae_a1:.4f}")
print(f"A1 RMSE: {rmse_a1:.4f}")
print(f"A1 R²:   {r2_a1:.4f}")


# =========================
# 6. A2 End-to-End MLP + EarlyStopping + 120 epochs
# =========================
print("\n===== A2: End-to-End MLP (+ EarlyStopping) =====")

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

model_a2 = End2EndRegressor(input_dim).to(device)
opt_a2 = torch.optim.Adam(model_a2.parameters(), lr=1e-3)
loss_fn_a2 = nn.L1Loss()

train_loader_a2 = DataLoader(
    TensorDataset(X_train_tensor, torch.tensor(y_train, dtype=torch.float32).to(device)),
    batch_size=256,
    shuffle=True
)

epochs_a2 = 120
earlystop = EarlyStopping(patience=12)

print("Training A2...")

best_state = None
best_loss = float("inf")

for epoch in range(epochs_a2):
    model_a2.train()
    total_loss = 0.0

    for bx, by in train_loader_a2:
        opt_a2.zero_grad()
        pred = model_a2(bx)
        loss = loss_fn_a2(pred, by)
        loss.backward()
        opt_a2.step()
        total_loss += loss.item() * bx.size(0)

    train_loss = total_loss / len(train_loader_a2.dataset)

    # validation
    model_a2.eval()
    with torch.no_grad():
        val_pred = model_a2(X_test_tensor)
        val_loss = loss_fn_a2(val_pred, torch.tensor(y_test, dtype=torch.float32).to(device)).item()

    if val_loss < best_loss:
        best_loss = val_loss
        best_state = model_a2.state_dict()

    if epoch % 5 == 0:
        print(f"A2 Epoch {epoch}/{epochs_a2}  Train={train_loss:.4f}  Val={val_loss:.4f}")

    if earlystop.step(val_loss):
        print(f"Early stopping at epoch {epoch}")
        break

model_a2.load_state_dict(best_state)

model_a2.eval()
with torch.no_grad():
    pred_a2 = model_a2(X_test_tensor).cpu().numpy()

mae_a2 = mean_absolute_error(y_test, pred_a2)
rmse_a2 = np.sqrt(mean_squared_error(y_test, pred_a2))
r2_a2 = r2_score(y_test, pred_a2)

print("\n===== A2 RESULTS =====")
print(f"A2 MAE:  {mae_a2:.4f}")
print(f"A2 RMSE: {rmse_a2:.4f}")
print(f"A2 R²:   {r2_a2:.4f}")


# =========================
# 7. baseline
# =========================
baseline_pred = np.full_like(y_test, y_train.mean())
mae_base = mean_absolute_error(y_test, baseline_pred)
rmse_base = np.sqrt(mean_squared_error(y_test, baseline_pred))
r2_base = r2_score(y_test, baseline_pred)

print("\n===== BASELINE =====")
print(f"Base MAE:  {mae_base:.4f}")
print(f"Base RMSE: {rmse_base:.4f}")
print(f"Base R²:   {r2_base:.4f}")


# =========================
# 8. 图：A1 vs A2
# =========================
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.scatter(y_test, pred_a1, alpha=0.35)
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
plt.title("A1: AE + Regressor")
plt.xlabel("True")
plt.ylabel("Predicted")

plt.subplot(1,2,2)
plt.scatter(y_test, pred_a2, alpha=0.35)
plt.plot([0, max(y_test)], [0, max(y_test)], 'r--')
plt.title("A2: End-to-End MLP")
plt.xlabel("True")
plt.ylabel("Predicted")

plt.tight_layout()
plt.show()


# =========================
# 9. 保存最好的模型
# =========================
if mae_a2 <= mae_a1:
    best_name = "A2 (End-to-End MLP)"
    torch.save(model_a2.state_dict(), "best_price_model.pth")
else:
    best_name = "A1 (AE + Regressor)"
    torch.save(ae_model.state_dict(), "best_ae_model.pth")
    torch.save(reg_model_a1.state_dict(), "best_price_regressor_a1.pth")

with open("scaler_price.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"\nSaved best model: {best_name}")

