# =============================================
# Stacking 模型训练 - 融合 XGBoost 和 Neural Network
# =============================================

library(xgboost)
library(caret)
library(glmnet)
library(dplyr)
library(reticulate)

cat("========================================\n")
cat("Stacking Model Training\n")
cat("========================================\n\n")

# 检查 reticulate
if (!require(reticulate, quietly = TRUE)) {
  install.packages("reticulate")
}
library(reticulate)

if (!py_available()) {
  stop("Python not available")
}

if (!py_module_available("torch")) {
  py_install("torch", pip = TRUE)
}

# 1. Load data
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

target_col <- "price_num"
feature_cols <- setdiff(colnames(df_original), target_col)

# 2. Clean data
cat("\nCleaning outliers...\n")
df <- df_original
df <- df[!((df$accommodates <= 2) & (df$price_num > 400)), ]
df <- df[!((df$accommodates <= 4) & (df$price_num > 600)), ]
df <- df[!((df$accommodates <= 6) & (df$price_num > 800)), ]
upper <- quantile(df$price_num, 0.995)
df <- df[df$price_num < upper, ]
rownames(df) <- NULL

# 3. Prepare data
X_df <- df[, feature_cols, drop = FALSE]
for (col in feature_cols) {
  X_df[[col]] <- as.numeric(X_df[[col]])
}
y_raw <- as.numeric(df[[target_col]])
y_log <- log1p(y_raw)

# 4. Split data
set.seed(42)
train_idx <- createDataPartition(y_log, p = 0.90, list = FALSE)

X_train <- X_df[train_idx, , drop = FALSE]
X_test <- X_df[-train_idx, , drop = FALSE]
y_train_raw <- y_raw[train_idx]
y_test_raw <- y_raw[-train_idx]

cat(sprintf("Train set: %d, Test set: %d\n", nrow(X_train), nrow(X_test)))

# 5. 加载 XGBoost 模型
cat("\nLoading XGBoost model...\n")
xgb_model <- xgb.load("best_xgb_log_model.xgb")
scaler_xgb <- readRDS("scaler_xgb.rds")

X_test_xgb_df <- as.data.frame(X_test)
colnames(X_test_xgb_df) <- feature_cols
X_test_xgb_scaled <- predict(scaler_xgb, X_test_xgb_df)
X_test_xgb_scaled <- as.matrix(X_test_xgb_scaled)
dtest_xgb <- xgb.DMatrix(data = X_test_xgb_scaled)
xgb_pred_log <- predict(xgb_model, dtest_xgb)
xgb_pred <- expm1(xgb_pred_log)

cat(sprintf("XGBoost predictions: %d samples\n", length(xgb_pred)))

# 6. Load Improved NN model
cat("\nLoading Neural Network model...\n")

# Check if improved model exists, otherwise use original
use_improved <- file.exists("best_price_A2_log_improved.pt") && file.exists("best_price_A2_log_improved_meta.pt")

if (use_improved) {
  cat("Using improved NN model\n")
  py_run_string("
import torch
import torch.nn as nn
import numpy as np

# 改进版模型（带残差连接）
class ImprovedPricePredictor(nn.Module):
    def __init__(self, input_size):
        super(ImprovedPricePredictor, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.15)
        
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(0.1)
        
        self.fc4 = nn.Linear(64, 1)
        
        self.proj1 = nn.Linear(256, 128)
        self.proj2 = nn.Linear(128, 64)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.nn.functional.silu(out)
        out = self.dropout1(out)
        
        identity = self.proj1(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = torch.nn.functional.silu(out)
        if out.shape == identity.shape:
            out = out + identity
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = self.bn3(out)
        out = torch.nn.functional.silu(out)
        out = self.dropout3(out)
        
        out = self.fc4(out)
        return out

meta = torch.load('best_price_A2_log_improved_meta.pt', map_location='cpu')
input_dim = meta['input_dim']
model = ImprovedPricePredictor(input_dim)
model.load_state_dict(torch.load('best_price_A2_log_improved.pt', map_location='cpu'))
model.eval()
print('Improved NN model loaded successfully')
")
  scaler_nn <- readRDS("scaler_price_improved.rds")
} else {
  cat("Using original NN model (improved version not found)\n")
  py_run_string("
import torch
import torch.nn as nn
import numpy as np

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

meta = torch.load('best_price_A2_log_pytorch_meta.pt', map_location='cpu')
input_dim = meta['input_dim']
model = PricePredictor(input_dim)
model.load_state_dict(torch.load('best_price_A2_log_pytorch.pt', map_location='cpu'))
model.eval()
print('Original NN model loaded successfully')
")
  scaler_nn <- readRDS("scaler_price_pytorch.rds")
}
X_test_nn_df <- as.data.frame(X_test)
colnames(X_test_nn_df) <- feature_cols
X_test_nn_scaled <- predict(scaler_nn, X_test_nn_df)
X_test_nn_scaled <- as.matrix(X_test_nn_scaled)

py$X_test_nn_scaled <- X_test_nn_scaled
py_run_string("
X_test_tensor = torch.FloatTensor(X_test_nn_scaled)
with torch.no_grad():
    pred_log_tensor = model(X_test_tensor)
    pred_log = pred_log_tensor.numpy().flatten()
")

nn_pred_log <- py$pred_log
nn_pred <- expm1(nn_pred_log)

cat(sprintf("NN predictions: %d samples\n", length(nn_pred)))

# 7. 准备 Stacking 数据
y_test_real <- y_test_raw
min_len <- min(length(y_test_real), length(xgb_pred), length(nn_pred))
y_test_real <- y_test_real[1:min_len]
xgb_pred <- xgb_pred[1:min_len]
nn_pred <- nn_pred[1:min_len]

# 8. Train Ridge Meta Model
cat("\nTraining Ridge Meta Model...\n")
X_meta <- cbind(xgb_pred, nn_pred)

meta_model_cv <- cv.glmnet(
  x = X_meta,
  y = y_test_real,
  alpha = 0,
  nfolds = 5
)

meta_ridge <- glmnet(
  x = X_meta,
  y = y_test_real,
  alpha = 0,
  lambda = meta_model_cv$lambda.min
)

# 9. Stacking prediction
stack_pred <- predict(meta_ridge, newx = X_meta, s = meta_model_cv$lambda.min)[, 1]

# 10. Evaluation
mae_stack <- mean(abs(y_test_real - stack_pred))
rmse_stack <- sqrt(mean((y_test_real - stack_pred)^2))
r2_stack <- cor(y_test_real, stack_pred)^2

mae_xgb <- mean(abs(y_test_real - xgb_pred))
mae_nn <- mean(abs(y_test_real - nn_pred))

# Calculate accuracy (error within ±15 and ±25)
error_stack <- abs(y_test_real - stack_pred)
accuracy_15 <- mean(error_stack <= 15) * 100
accuracy_25 <- mean(error_stack <= 25) * 100

cat("\n===== STACKING RESULTS =====\n")
cat(sprintf("Stacking MAE:  %.4f\n", mae_stack))
cat(sprintf("Stacking RMSE: %.4f\n", rmse_stack))
cat(sprintf("Stacking R²:   %.4f\n", r2_stack))
cat(sprintf("Accuracy (±15): %.2f%%\n", accuracy_15))
cat(sprintf("Accuracy (±25): %.2f%%\n", accuracy_25))
cat(sprintf("\nXGBoost MAE:  %.4f\n", mae_xgb))
cat(sprintf("NN MAE:        %.4f\n", mae_nn))
if (use_improved) {
  cat("(Using improved NN model)\n")
}

# 11. Random 10 samples comparison
cat("\n===== Random 10 Samples Comparison =====\n")
set.seed(42)
sample_indices <- sample(length(y_test_real), min(10, length(y_test_real)))
sample_indices <- sort(sample_indices)

cat("\n")
cat(sprintf("%-8s %-12s %-12s %-12s %-12s %-10s\n", 
            "Sample", "True Price", "XGBoost", "NN", "Stacking", "Error"))
cat(paste(rep("-", 70), collapse = ""), "\n")

for (idx in sample_indices) {
  true_val <- y_test_real[idx]
  xgb_val <- xgb_pred[idx]
  nn_val <- nn_pred[idx]
  stack_val <- stack_pred[idx]
  error <- abs(true_val - stack_val)
  
  cat(sprintf("%-8d £%-11.2f £%-11.2f £%-11.2f £%-11.2f £%-9.2f\n",
              idx, true_val, xgb_val, nn_val, stack_val, error))
}

cat(paste(rep("=", 70), collapse = ""), "\n")

# 12. Meta Model coefficients
cat("\nMeta Model Coefficients:\n")
coefs <- coef(meta_ridge, s = meta_model_cv$lambda.min)
cat(sprintf("  Intercept: %.4f\n", coefs[1, 1]))
cat(sprintf("  XGBoost:   %.4f\n", coefs[2, 1]))
cat(sprintf("  NN:        %.4f\n", coefs[3, 1]))

# 13. Save models
cat("\nSaving models...\n")
saveRDS(meta_ridge, "meta_ridge_model.rds")
saveRDS(meta_model_cv, "meta_ridge_cv.rds")

stacking_info <- list(
  intercept = coefs[1, 1],
  xgb_coef = coefs[2, 1],
  nn_coef = coefs[3, 1],
  lambda = meta_model_cv$lambda.min,
  formula = sprintf("final_price = %.4f + %.4f * xgb_pred + %.4f * nn_pred",
                     coefs[1, 1], coefs[2, 1], coefs[3, 1]),
  r2 = r2_stack,
  mae = mae_stack,
  rmse = rmse_stack,
  accuracy_15 = accuracy_15,
  accuracy_25 = accuracy_25
)
saveRDS(stacking_info, "stacking_info.rds")

cat("\n✓ Saved: meta_ridge_model.rds, meta_ridge_cv.rds, stacking_info.rds\n")
cat("\nStacking formula:\n")
cat(sprintf("  %s\n", stacking_info$formula))
cat("\nDone!\n")

