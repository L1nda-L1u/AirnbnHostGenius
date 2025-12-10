# =============================================
# Neural Network Model Training (Using Python PyTorch)
# =============================================

library(caret)
library(dplyr)
library(reticulate)

cat("========================================\n")
cat("Neural Network Model Training (PyTorch)\n")
cat("========================================\n\n")

# Check reticulate
if (!require(reticulate, quietly = TRUE)) {
  install.packages("reticulate")
}
library(reticulate)

if (!py_available()) {
  stop("Python not available. Please configure Python first.")
}

# Install torch (if needed)
if (!py_module_available("torch")) {
  cat("Installing torch...\n")
  py_install("torch", pip = TRUE)
}

# 1. Load Data
cat("Loading data...\n")
data_file <- "nn_price_training_v4.csv"
if (!file.exists(data_file)) {
  data_file <- "other_files/nn_price_training_v4.csv"
  if (!file.exists(data_file)) {
    data_file <- "R_scripts/best_model/nn_price_training_v4.csv"
  }
}
if (!file.exists(data_file)) {
  stop("Cannot find nn_price_training_v4.csv")
}
df_original <- read.csv(data_file, stringsAsFactors = FALSE)
cat(sprintf("Original data: %s rows\n", format(nrow(df_original), big.mark = ",")))

target_col <- "price_num"
feature_cols <- setdiff(colnames(df_original), target_col)

# 2. Clean Data
cat("\nCleaning outliers...\n")
df <- df_original
df <- df[!((df$accommodates <= 2) & (df$price_num > 400)), ]
df <- df[!((df$accommodates <= 4) & (df$price_num > 600)), ]
df <- df[!((df$accommodates <= 6) & (df$price_num > 800)), ]
upper <- quantile(df$price_num, 0.995)
df <- df[df$price_num < upper, ]
rownames(df) <- NULL
cat(sprintf("After cleaning: %s rows\n", format(nrow(df), big.mark = ",")))

# 3. Prepare Features and Target
X_df <- df[, feature_cols, drop = FALSE]
for (col in feature_cols) {
  X_df[[col]] <- as.numeric(X_df[[col]])
}
y_raw <- as.numeric(df[[target_col]])
y_log <- log1p(y_raw)

# 4. Split Data
cat("\nSplitting data...\n")
set.seed(42)
train_idx <- createDataPartition(y_log, p = 0.90, list = FALSE)

X_train_df <- X_df[train_idx, , drop = FALSE]
X_test_df <- X_df[-train_idx, , drop = FALSE]
y_train_log <- y_log[train_idx]
y_test_log <- y_log[-train_idx]
y_train_raw <- y_raw[train_idx]
y_test_raw <- y_raw[-train_idx]

cat(sprintf("Training set: %d, Test set: %d\n", nrow(X_train_df), nrow(X_test_df)))

# 5. Standardize
cat("\nScaling features...\n")
preProc <- preProcess(X_train_df, method = c("center", "scale"))
X_train_scaled <- as.matrix(predict(preProc, X_train_df))
X_test_scaled <- as.matrix(predict(preProc, X_test_df))

# 6. Pass to Python
py_run_string("
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
")
py$X_train_scaled <- X_train_scaled
py$X_test_scaled <- X_test_scaled
py$y_train_log <- as.numeric(y_train_log)
py$device <- "cpu"

# 7. Train in Python
cat("\nTraining Neural Network in Python...\n")
py_run_string("
# Convert to numpy array
X_train_scaled = np.array(X_train_scaled, dtype=np.float32)
X_test_scaled = np.array(X_test_scaled, dtype=np.float32)
y_train_log = np.array(y_train_log, dtype=np.float32)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train_scaled)
y_train_tensor = torch.FloatTensor(y_train_log).unsqueeze(1)
X_test_tensor = torch.FloatTensor(X_test_scaled)

# Define model
class PricePredictor(nn.Module):
    def __init__(self, input_size):
        super(PricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.nn.functional.silu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.nn.functional.silu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = torch.nn.functional.silu(x)
        x = self.fc4(x)
        return x

model = PricePredictor(X_train_scaled.shape[1])
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# Training parameters
batch_size = 256
n_epochs = 150
n_samples = len(X_train_scaled)
n_batches = (n_samples + batch_size - 1) // batch_size

# Validation set
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
patience = 20
patience_counter = 0

print(f'Model architecture: {X_train_scaled.shape[1]} -> 256 -> 128 -> 64 -> 1')
print('Starting training...')

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
        optimizer.step()
        
        epoch_loss += loss.item()
    
    model.eval()
    with torch.no_grad():
        val_pred = model(X_val_tensor)
        val_loss = criterion(val_pred, y_val_tensor).item()
    
    avg_train_loss = epoch_loss / n_batches
    
    if epoch % 10 == 0 or epoch == 1:
        print(f'Epoch {epoch:3d}: Train Loss = {avg_train_loss:.6f}, Val Loss = {val_loss:.6f}')
    
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

print('\\nTraining complete!')

# Save model
torch.save(model.state_dict(), 'best_price_A2_log_pytorch.pt')
torch.save({
    'input_dim': X_train_scaled.shape[1],
    'model_class': 'PricePredictor'
}, 'best_price_A2_log_pytorch_meta.pt')
print('模型已保存')
")

# 8. 评估
cat("\nEvaluating...\n")
py_run_string("
model.eval()
with torch.no_grad():
    pred_log_tensor = model(X_test_tensor)
    pred_log = pred_log_tensor.numpy().flatten()
")

nn_pred_log <- py$pred_log
nn_pred <- expm1(nn_pred_log)
true_price <- y_test_raw

mae <- mean(abs(true_price - nn_pred))
rmse <- sqrt(mean((true_price - nn_pred)^2))
r2 <- cor(true_price, nn_pred)^2

cat("\n===== NEURAL NETWORK RESULTS =====\n")
cat(sprintf("MAE:  %.4f\n", mae))
cat(sprintf("RMSE: %.4f\n", rmse))
cat(sprintf("R²:   %.4f\n", r2))

# 9. Save scaler
saveRDS(preProc, "scaler_price_pytorch.rds")

cat("\n✓ Saved: best_price_A2_log_pytorch.pt, best_price_A2_log_pytorch_meta.pt, scaler_price_pytorch.rds\n")
cat("Done!\n")

