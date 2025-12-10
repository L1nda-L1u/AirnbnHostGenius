# =============================================
# Neural Network 模型训练 - 改进版
# 包含多项优化以提升性能
# =============================================

library(caret)
library(dplyr)
library(reticulate)

cat("========================================\n")
cat("Improved Neural Network Training (PyTorch)\n")
cat("========================================\n\n")

# 检查 reticulate
if (!require(reticulate, quietly = TRUE)) {
  install.packages("reticulate")
}
library(reticulate)

if (!py_available()) {
  stop("Python not available. Please configure Python first.")
}

if (!py_module_available("torch")) {
  cat("Installing torch...\n")
  py_install("torch", pip = TRUE)
}

# 1. 加载数据
cat("Loading data...\n")
data_file <- "nn_price_training_v4.csv"
if (!file.exists(data_file)) {
  data_file <- "../best_model/nn_price_training_v4.csv"
  if (!file.exists(data_file)) {
    data_file <- "../../other_files/nn_price_training_v4.csv"
  }
}
if (!file.exists(data_file)) {
  stop("Cannot find nn_price_training_v4.csv")
}
df_original <- read.csv(data_file, stringsAsFactors = FALSE)
cat(sprintf("原始数据量: %s 行\n", format(nrow(df_original), big.mark = ",")))

target_col <- "price_num"
feature_cols <- setdiff(colnames(df_original), target_col)

# 2. 数据清理
cat("\nCleaning outliers...\n")
df <- df_original
df <- df[!((df$accommodates <= 2) & (df$price_num > 400)), ]
df <- df[!((df$accommodates <= 4) & (df$price_num > 600)), ]
df <- df[!((df$accommodates <= 6) & (df$price_num > 800)), ]
upper <- quantile(df$price_num, 0.995)
df <- df[df$price_num < upper, ]
rownames(df) <- NULL
cat(sprintf("清理后数据量: %s 行\n", format(nrow(df), big.mark = ",")))

# 3. 准备特征和目标
X_df <- df[, feature_cols, drop = FALSE]
for (col in feature_cols) {
  X_df[[col]] <- as.numeric(X_df[[col]])
}
y_raw <- as.numeric(df[[target_col]])
y_log <- log1p(y_raw)  # ✅ 使用 log 变换

# 4. 划分数据
cat("\nSplitting data...\n")
set.seed(42)
train_idx <- createDataPartition(y_log, p = 0.90, list = FALSE)

X_train_df <- X_df[train_idx, , drop = FALSE]
X_test_df <- X_df[-train_idx, , drop = FALSE]
y_train_log <- y_log[train_idx]
y_test_log <- y_log[-train_idx]
y_train_raw <- y_raw[train_idx]
y_test_raw <- y_raw[-train_idx]

cat(sprintf("训练集: %d, 测试集: %d\n", nrow(X_train_df), nrow(X_test_df)))

# 5. 标准化
cat("\nScaling features...\n")
preProc <- preProcess(X_train_df, method = c("center", "scale"))
X_train_scaled <- as.matrix(predict(preProc, X_train_df))
X_test_scaled <- as.matrix(predict(preProc, X_test_df))

# 6. 传递给 Python
py_run_string("
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
")
py$X_train_scaled <- X_train_scaled
py$X_test_scaled <- X_test_scaled
py$y_train_log <- as.numeric(y_train_log)
py$device <- "cpu"

# 7. 在 Python 中训练改进的模型
cat("\nTraining Improved Neural Network...\n")
cat("改进点：\n")
cat("  1. 更深的网络（增加残差连接）\n")
cat("  2. 更强的正则化（Dropout 0.2-0.15-0.1）\n")
cat("  3. 学习率调度器（自适应降低学习率）\n")
cat("  4. 权重衰减（L2 正则化）\n")
cat("  5. 更多训练轮次（200 epochs）\n")
cat("  6. 更大的 patience（30）\n\n")

py_run_string("
# 转换为 numpy 数组
X_train_scaled = np.array(X_train_scaled, dtype=np.float32)
X_test_scaled = np.array(X_test_scaled, dtype=np.float32)
y_train_log = np.array(y_train_log, dtype=np.float32)

# 转换为 tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_log).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_scaled)

# 定义改进的模型（带残差连接）
class ImprovedPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(ImprovedPricePredictor, self).__init__()
        # 第一层
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        
        # 第二层
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.15)
        
        # 第三层
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.1)
        
        # 输出层
        self.fc4 = nn.Linear(64, 1)
        
        # 残差连接的投影层（如果需要）
        self.proj1 = nn.Linear(256, 128) if input_size != 256 else nn.Identity()
        self.proj2 = nn.Linear(128, 64) if 256 != 128 else nn.Identity()
    
    def forward(self, x):
        # 第一层
        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.nn.functional.silu(out)
        out = self.dropout1(out)
        
        # 第二层（带残差连接）
        identity = self.proj1(out) if hasattr(self, 'proj1') and self.proj1 is not None else None
        out = self.fc2(out)
        out = self.bn2(out)
        out = torch.nn.functional.silu(out)
        if identity is not None and out.shape == identity.shape:
            out = out + identity  # 残差连接
        out = self.dropout2(out)
        
        # 第三层
        out = self.fc3(out)
        out = self.bn3(out)
        out = torch.nn.functional.silu(out)
        out = self.dropout3(out)
        
        # 输出层
        out = self.fc4(out)
        return out

model = ImprovedPricePredictor(X_train_scaled.shape[1])
# 使用 AdamW（带权重衰减的 Adam）
optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
# 学习率调度器（移除 verbose 参数，因为某些版本不支持）
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
criterion = nn.MSELoss()

# 训练参数
batch_size = 256
n_epochs = 200  # 增加训练轮次
n_samples = len(X_train_scaled)
n_batches = (n_samples + batch_size - 1) // batch_size

# 验证集
val_size = int(n_samples * 0.1)
np.random.seed(42)
val_idx = np.random.choice(n_samples, size=val_size, replace=False)
val_idx = np.sort(val_idx)
train_idx_final = np.setdiff1d(np.arange(n_samples), val_idx)

X_train_final = X_train_scaled[train_idx_final]
X_val = X_train_scaled[val_idx]
y_train_final = y_train_log[train_idx_final]
y_val = y_train_log[val_idx]

X_train_final_tensor = torch.FloatTensor(X_train_final)
y_train_final_tensor = torch.FloatTensor(y_train_final).unsqueeze(1)
X_val_tensor = torch.FloatTensor(X_val)
y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)

best_val_loss = float('inf')
patience = 30  # 增加 patience
patience_counter = 0

print(f'模型架构: {X_train_scaled.shape[1]} -> 256 -> 128 -> 64 -> 1 (带残差连接)')
print(f'优化器: AdamW (lr=5e-4, weight_decay=1e-4)')
print(f'学习率调度: ReduceLROnPlateau')
print('开始训练...')

for epoch in range(1, n_epochs + 1):
    model.train()
    epoch_loss = 0
    
    perm = np.random.permutation(len(X_train_final))
    X_train_shuffled = X_train_final[perm]
    y_train_shuffled = y_train_final[perm]
    
    for batch_idx in range(n_batches):
        start_idx = int(batch_idx * batch_size)
        end_idx = int(min((batch_idx + 1) * batch_size, len(X_train_final)))
        
        X_batch = torch.FloatTensor(X_train_shuffled[start_idx:end_idx])
        y_batch = torch.FloatTensor(y_train_shuffled[start_idx:end_idx]).unsqueeze(1)
        
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        # 梯度裁剪（防止梯度爆炸）
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor)
        val_loss = criterion(val_pred, y_val_tensor).item()
    
    # 更新学习率
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    
    avg_train_loss = epoch_loss / n_batches
    
    # 如果学习率改变了，打印出来
    lr_changed = old_lr != new_lr
    if epoch % 10 == 0 or epoch == 1 or lr_changed:
        lr_msg = f', LR = {new_lr:.6f}' + (' (reduced!)' if lr_changed else '')
        print(f'Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}{lr_msg}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        best_model_state = model.state_dict()
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f'\\nEarly stopping at epoch {epoch}')
            model.load_state_dict(best_model_state)
            break

print('\\n训练完成!')

# 保存模型
torch.save(model.state_dict(), 'best_price_A2_log_improved.pt')
torch.save({
    'input_dim': X_train_scaled.shape[1],
    'model_class': 'ImprovedPricePredictor'
}, 'best_price_A2_log_improved_meta.pt')
print('改进的模型已保存')
")

# 8. 评估
cat("\nEvaluating improved model...\n")
py_run_string("
model.eval()
with torch.no_grad():
    pred_log_tensor = model(X_test_tensor)
    pred_log = pred_log_tensor.numpy().flatten()
")

nn_pred_log <- py$pred_log
nn_pred <- expm1(nn_pred_log)  # ✅ 转回真实价格
true_price <- y_test_raw

mae <- mean(abs(true_price - nn_pred))
rmse <- sqrt(mean((true_price - nn_pred)^2))
r2 <- cor(true_price, nn_pred)^2

cat("\n===== IMPROVED NEURAL NETWORK RESULTS =====\n")
cat(sprintf("MAE:  %.4f\n", mae))
cat(sprintf("RMSE: %.4f\n", rmse))
cat(sprintf("R²:   %.4f\n", r2))

# 9. 保存 scaler
saveRDS(preProc, "scaler_price_improved.rds")

cat("\n✓ Saved: best_price_A2_log_improved.pt, best_price_A2_log_improved_meta.pt, scaler_price_improved.rds\n")
cat("\n改进总结：\n")
cat("  1. ✅ 使用 log 变换价格（log1p/expm1）\n")
cat("  2. ✅ 残差连接（提升梯度流动）\n")
cat("  3. ✅ 更强的正则化（Dropout 0.2/0.15/0.1）\n")
cat("  4. ✅ AdamW 优化器（带权重衰减）\n")
cat("  5. ✅ 学习率调度器（自适应降低）\n")
cat("  6. ✅ 梯度裁剪（防止梯度爆炸）\n")
cat("  7. ✅ 更多训练轮次和更大的 patience\n")
cat("\nDone!\n")

