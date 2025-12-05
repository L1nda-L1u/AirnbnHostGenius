import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from numpy.linalg import norm
import matplotlib.pyplot as plt

# ============================================================
# 1. 设备（GPU / CPU）
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ============================================================
# 2. 读数据 & 划分 train / predict 集
# ============================================================
df = pd.read_csv("nn_price_training_v4.csv")

# 如果行数 > 1000，就固定拿 1000 行做“预测集”；否则按 10% 拿
test_size = 1000 if len(df) > 1000 else 0.1
train_df, predict_df = train_test_split(df, test_size=test_size, random_state=42)

train_df = train_df.reset_index(drop=True)
predict_df = predict_df.reset_index(drop=True)

print("Train samples:", len(train_df))
print("Predict samples:", len(predict_df))

# 特征：不把价格喂进去，因为我们想学“房子的特征空间”
feature_cols = [c for c in df.columns if c != "price_num"]

X_train = train_df[feature_cols].values.astype(np.float32)
X_pred = predict_df[feature_cols].values.astype(np.float32)

print("Num train samples:", X_train.shape[0])
print("Num features:", X_train.shape[1])

# ============================================================
# 3. 标准化（只在 train 上 fit）
# ============================================================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
dataset = TensorDataset(X_train_tensor)
loader = DataLoader(dataset, batch_size=512, shuffle=True)

input_dim = X_train.shape[1]
latent_dim = 16  # embedding 维度，可以改大一点比如 32

# ============================================================
# 4. 定义 Autoencoder（神经网络部分）
# ============================================================
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

model = Autoencoder(input_dim, latent_dim).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============================================================
# 5. 训练 Autoencoder（只用 train_df）
# ============================================================
epochs = 50
print("Start training autoencoder...")

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for (batch_x,) in loader:
        optimizer.zero_grad()
        recon, z = model(batch_x)
        loss = criterion(recon, batch_x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch_x.size(0)

    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{epochs}  Recon Loss={avg_loss:.4f}")

# ============================================================
# 6. 得到 train / predict 的 embedding
# ============================================================
model.eval()
with torch.no_grad():
    # train embeddings
    _, Z_train = model(X_train_tensor)
    embeddings_train = Z_train.cpu().numpy()

    # predict embeddings（模拟“新 listing”，模型没见过）
    X_pred_scaled = scaler.transform(X_pred)
    X_pred_tensor = torch.tensor(X_pred_scaled, dtype=torch.float32).to(device)
    _, Z_pred = model(X_pred_tensor)
    embeddings_pred = Z_pred.cpu().numpy()

print("Train embedding shape:", embeddings_train.shape)
print("Predict embedding shape:", embeddings_pred.shape)

# 保存一下，之后可以直接加载用
np.save("listing_embeddings_train.npy", embeddings_train)
np.save("listing_embeddings_predict.npy", embeddings_pred)

import pickle
with open("ae_scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

torch.save(model.state_dict(), "autoencoder_model.pth")
print("Saved embeddings, scaler, and model.")

# ============================================================
# 7. KNN：给一个 embedding 找 comps（在 train_df 里）
# ============================================================
def get_comps_from_embedding(target_embedding, k=30):
    """
    给定一个 target embedding（长度 latent_dim 的向量），
    在训练集 embedding 中找最近的 k 个邻居，
    返回：comps_df（包含 distance 列）。
    """
    dists = norm(embeddings_train - target_embedding, axis=1)
    topk_idx = np.argsort(dists)[:k]

    cols_to_show = [
        "price_num",
        "latitude", "longitude",
        "location_cluster",
        "neighbourhood_id",
        "room_type_id"
    ]
    cols_to_show = [c for c in cols_to_show if c in train_df.columns]

    comps_df = train_df.iloc[topk_idx][cols_to_show].copy()
    comps_df["distance"] = dists[topk_idx]

    return comps_df

def summarize_prices(comps_df):
    """
    对 comps 的价格做统计：
    - median / p25 / p75 / min / max
    - weighted_price（按距离倒数加权的加权平均）
    """
    prices = comps_df["price_num"].values
    dists = comps_df["distance"].values

    eps = 1e-6
    weights = 1.0 / (dists + eps)
    weighted_price = float(np.sum(weights * prices) / np.sum(weights))

    summary = {
        "k": len(prices),
        "median_price": float(np.median(prices)),
        "p25_price": float(np.percentile(prices, 25)),
        "p75_price": float(np.percentile(prices, 75)),
        "weighted_price": weighted_price,
        "min_price": float(prices.min()),
        "max_price": float(prices.max()),
    }
    return summary

def get_comps_for_train_index(idx, k=30):
    """
    给定 train_df 里的 index，找它的 comps + 价格 summary。
    """
    if idx < 0 or idx >= len(train_df):
        raise ValueError("train idx out of range")

    target_emb = embeddings_train[idx]
    comps_df = get_comps_from_embedding(target_emb, k=k)
    summary = summarize_prices(comps_df)
    return comps_df, summary

def get_comps_for_predict_index(idx, k=30):
    """
    给定 predict_df 里的 index，视为“新 listing”，
    在 train_df 里找 comps + 价格 summary。
    """
    if idx < 0 or idx >= len(predict_df):
        raise ValueError("predict idx out of range")

    target_emb = embeddings_pred[idx]
    comps_df = get_comps_from_embedding(target_emb, k=k)
    summary = summarize_prices(comps_df)
    return comps_df, summary

def get_comps_for_new_listing(features_dict, k=30):
    """
    完全不在 df 里的新 listing：
    - 输入一个字典（key 和 feature_cols 对齐）
    - 用 scaler + encoder 得到 embedding
    - 在 train_df 里找 comps + 价格 summary
    """
    new_df = pd.DataFrame([features_dict])
    new_X = new_df[feature_cols].values.astype(np.float32)
    new_scaled = scaler.transform(new_X)
    new_tensor = torch.tensor(new_scaled, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        z = model.encoder(new_tensor)
    new_emb = z.cpu().numpy()[0]

    comps_df = get_comps_from_embedding(new_emb, k=k)
    summary = summarize_prices(comps_df)
    return comps_df, summary

# ============================================================
# 8. 示例 1：看一个 predict_df 里的 listing 的 comps + 建议价格
# ============================================================
example_idx = 0
print(f"\nExample: predict_df index {example_idx}")

example_true_price = float(predict_df.iloc[example_idx]["price_num"])
comps_ex, summary_ex = get_comps_for_predict_index(example_idx, k=30)

print("\nTop 10 comps from train_df:")
print(comps_ex.head(10))

print("\nPrice suggestion based on comps (median & weighted):")
print(summary_ex)
print(f"True price of this listing: {example_true_price}")

# ============================================================
# 9. 在 predict_df 上评估：median vs weighted KNN 的“准确度”
# ============================================================
n_eval = min(200, len(predict_df))  # 随便评估前 200 个，够看趋势了

true_list = []
median_list = []
weighted_list = []

for i in range(n_eval):
    true_price = float(predict_df.iloc[i]["price_num"])
    _, s = get_comps_for_predict_index(i, k=30)

    true_list.append(true_price)
    median_list.append(s["median_price"])
    weighted_list.append(s["weighted_price"])

true_arr = np.array(true_list)
median_arr = np.array(median_list)
weighted_arr = np.array(weighted_list)

mae_median = float(np.mean(np.abs(true_arr - median_arr)))
mae_weighted = float(np.mean(np.abs(true_arr - weighted_arr)))

print(f"\nEvaluate on {n_eval} predict listings:")
print(f"MAE (median comps):   {mae_median:.4f}")
print(f"MAE (weighted KNN):   {mae_weighted:.4f}")

# 画图：True price vs Weighted KNN price
plt.figure(figsize=(6, 6))
plt.scatter(true_arr, weighted_arr, alpha=0.4)
min_axis = min(true_list + weighted_list)
max_axis = max(true_list + weighted_list)
plt.plot([min_axis, max_axis], [min_axis, max_axis], linestyle="--")
plt.xlabel("True price")
plt.ylabel("Weighted KNN suggested price")
plt.title("Weighted KNN price suggestion vs true price")
plt.tight_layout()
plt.show()
