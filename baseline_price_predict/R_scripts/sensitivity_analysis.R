### ---------------------------------------------------------
### sensitivity_analysis.R
### 敏感性分析：测试输入特征变化对baseline定价的影响
### 例如：加/减amenity（如电视）对价格的影响
### ---------------------------------------------------------

library(xgboost)
library(caret)
library(dplyr)
library(reticulate)
library(glmnet)

cat("========================================\n")
cat("Sensitivity Analysis for Baseline Price Model\n")
cat("========================================\n\n")

# =============================================
# 1. 加载模型和数据
# =============================================

cat("Loading models and data...\n")

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

# 智能查找baseprice_model目录
original_wd <- getwd()

# 尝试多种可能的路径（按优先级）
baseprice_dir <- NULL

# 方法1: 当前目录就是baseprice_model
if (basename(getwd()) == "baseprice_model" && file.exists("nn_price_training_v4.csv")) {
  baseprice_dir <- getwd()
  cat("Detected: Currently in baseprice_model directory\n")
} 
# 方法2: 当前目录下的baseprice_model
else if (dir.exists("baseprice_model") && file.exists(file.path("baseprice_model", "nn_price_training_v4.csv"))) {
  baseprice_dir <- normalizePath("baseprice_model")
  cat("Detected: baseprice_model in current directory\n")
}
# 方法3: 上一级目录下的baseprice_model
else if (dir.exists("../baseprice_model") && file.exists("../baseprice_model/nn_price_training_v4.csv")) {
  baseprice_dir <- normalizePath("../baseprice_model")
  cat("Detected: baseprice_model in parent directory\n")
}
# 方法4: 上两级目录下的baseprice_model（如果在R_scripts中）
else if (dir.exists("../../baseprice_model") && file.exists("../../baseprice_model/nn_price_training_v4.csv")) {
  baseprice_dir <- normalizePath("../../baseprice_model")
  cat("Detected: baseprice_model in grandparent directory\n")
}

if (is.null(baseprice_dir) || !dir.exists(baseprice_dir)) {
  cat(sprintf("\nError: Cannot find baseprice_model directory.\n"))
  cat(sprintf("Current directory: %s\n", getwd()))
  cat("\nPlease either:\n")
  cat("  1. Change to project root directory: setwd('..')\n")
  cat("  2. Or change to R_scripts directory: setwd('../R_scripts')\n")
  cat("  3. Or use absolute path to source the script\n")
  stop("baseprice_model directory not found")
}

cat(sprintf("Using baseprice_model: %s\n\n", baseprice_dir))

# 加载训练数据以获取特征列
data_file <- file.path(baseprice_dir, "nn_price_training_v4.csv")
if (!file.exists(data_file)) {
  stop(sprintf("Cannot find %s", data_file))
}

df_train <- read.csv(data_file, stringsAsFactors = FALSE)
target_col <- "price_num"
feature_cols <- setdiff(colnames(df_train), target_col)

cat(sprintf("Loaded training data: %d rows, %d features\n", 
            nrow(df_train), length(feature_cols)))

# 加载XGBoost模型
cat("Loading XGBoost model...\n")
xgb_model_path <- file.path(baseprice_dir, "best_xgb_log_model.xgb")
scaler_xgb_path <- file.path(baseprice_dir, "scaler_xgb.rds")

if (!file.exists(xgb_model_path) || !file.exists(scaler_xgb_path)) {
  stop("XGBoost model files not found")
}

xgb_model <- xgb.load(xgb_model_path)
scaler_xgb <- readRDS(scaler_xgb_path)

# 加载Neural Network模型
cat("Loading Neural Network model...\n")
nn_model_path <- file.path(baseprice_dir, "best_price_A2_log_pytorch.pt")
nn_meta_path <- file.path(baseprice_dir, "best_price_A2_log_pytorch_meta.pt")
scaler_nn_path <- file.path(baseprice_dir, "scaler_price_pytorch.rds")

if (!file.exists(nn_model_path) || !file.exists(nn_meta_path) || !file.exists(scaler_nn_path)) {
  stop("Neural Network model files not found")
}

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
")

# 加载NN模型
# 使用Python的os.path来处理路径（更安全，跨平台）
py_run_string("
import os
")

py$nn_meta_path <- normalizePath(nn_meta_path, winslash = "/")
py$nn_model_path <- normalizePath(nn_model_path, winslash = "/")

py_run_string("
meta = torch.load(nn_meta_path, map_location='cpu')
input_dim = meta['input_dim']
model = PricePredictor(input_dim)
model.load_state_dict(torch.load(nn_model_path, map_location='cpu'))
model.eval()
print('Neural Network model loaded successfully')
")

scaler_nn <- readRDS(scaler_nn_path)

# 加载Stacking模型
cat("Loading Stacking meta model...\n")
meta_model_path <- file.path(baseprice_dir, "meta_ridge_model.rds")
meta_cv_path <- file.path(baseprice_dir, "meta_ridge_cv.rds")

if (!file.exists(meta_model_path)) {
  stop("Meta model not found")
}

meta_model <- readRDS(meta_model_path)
meta_cv <- if (file.exists(meta_cv_path)) readRDS(meta_cv_path) else NULL

# 加载stacking信息
stacking_info_path <- file.path(baseprice_dir, "stacking_info.rds")
if (file.exists(stacking_info_path)) {
  stacking_info <- readRDS(stacking_info_path)
  cat(sprintf("Stacking formula: %s\n", stacking_info$formula))
} else {
  # 如果没有stacking_info，从meta_model中提取
  coefs <- coef(meta_model, s = if (!is.null(meta_cv)) meta_cv$lambda.min else NULL)
  stacking_info <- list(
    intercept = coefs[1, 1],
    xgb_coef = coefs[2, 1],
    nn_coef = coefs[3, 1]
  )
}

cat("All models loaded successfully!\n\n")

# =============================================
# 2. 预测函数（使用stacking模型）
# =============================================

predict_price_stacking <- function(feature_vector) {
  # feature_vector应该是一个命名向量或数据框，包含所有特征列
  
  # 确保特征顺序正确
  if (is.data.frame(feature_vector)) {
    feature_vector <- feature_vector[1, , drop = FALSE]
  }
  
  # 转换为数据框并确保所有特征都存在
  X_df <- data.frame(matrix(0, nrow = 1, ncol = length(feature_cols)))
  colnames(X_df) <- feature_cols
  
  # 填充提供的特征值
  for (col in names(feature_vector)) {
    if (col %in% feature_cols) {
      X_df[[col]] <- as.numeric(feature_vector[[col]])
    }
  }
  
  # 确保所有列都是数值型
  for (col in feature_cols) {
    X_df[[col]] <- as.numeric(X_df[[col]])
  }
  
  # XGBoost预测
  X_xgb_scaled <- predict(scaler_xgb, X_df)
  X_xgb_scaled <- as.matrix(X_xgb_scaled)
  dtest_xgb <- xgb.DMatrix(data = X_xgb_scaled)
  xgb_pred_log <- predict(xgb_model, dtest_xgb)
  xgb_pred <- expm1(xgb_pred_log)
  
  # Neural Network预测
  X_nn_scaled <- predict(scaler_nn, X_df)
  X_nn_scaled <- as.matrix(X_nn_scaled)
  
  py$X_nn_scaled <- X_nn_scaled
  py_run_string("
X_test_tensor = torch.FloatTensor(X_nn_scaled)
with torch.no_grad():
    pred_log_tensor = model(X_test_tensor)
    pred_log = pred_log_tensor.numpy().flatten()
")
  
  nn_pred_log <- py$pred_log
  nn_pred <- expm1(nn_pred_log)
  
  # Stacking预测
  X_meta <- cbind(xgb_pred, nn_pred)
  lambda_val <- if (!is.null(meta_cv)) meta_cv$lambda.min else NULL
  stack_pred <- predict(meta_model, newx = X_meta, s = lambda_val)[, 1]
  
  return(list(
    xgb_pred = as.numeric(xgb_pred),
    nn_pred = as.numeric(nn_pred),
    final_price = as.numeric(stack_pred)
  ))
}

# =============================================
# 3. 敏感性分析函数
# =============================================

#' 测试特征变化对价格的影响
#' 
#' @param base_features 基础特征向量（命名向量或数据框的一行）
#' @param feature_name 要测试的特征名称
#' @param values 要测试的特征值（向量），默认为c(0, 1)用于二元特征
#' @param verbose 是否显示详细信息
#' 
#' @return 数据框，包含特征值和对应的价格预测
sensitivity_test <- function(base_features, feature_name, values = c(0, 1), verbose = TRUE) {
  
  if (!feature_name %in% feature_cols) {
    stop(sprintf("Feature '%s' not found in model features", feature_name))
  }
  
  results <- data.frame()
  
  for (val in values) {
    # 复制基础特征
    test_features <- base_features
    test_features[[feature_name]] <- val
    
    # 预测价格
    pred <- predict_price_stacking(test_features)
    
    results <- rbind(results, data.frame(
      feature_name = feature_name,
      feature_value = val,
      xgb_pred = pred$xgb_pred,
      nn_pred = pred$nn_pred,
      final_price = pred$final_price
    ))
  }
  
  # 计算价格变化
  if (nrow(results) > 1) {
    base_price <- results$final_price[1]
    results$price_change <- results$final_price - base_price
    results$price_change_pct <- (results$final_price - base_price) / base_price * 100
  }
  
  if (verbose) {
    cat("\n========================================\n")
    cat(sprintf("Sensitivity Analysis: %s\n", feature_name))
    cat("========================================\n")
    print(results)
    cat("\n")
  }
  
  return(results)
}

# =============================================
# 4. 批量测试多个特征
# =============================================

#' 批量测试多个特征变化对价格的影响
#' 
#' @param base_features 基础特征向量
#' @param feature_names 要测试的特征名称向量
#' @param feature_values 每个特征要测试的值（列表），如果为NULL则使用c(0, 1)
#' 
#' @return 数据框，包含所有测试结果
batch_sensitivity_test <- function(base_features, feature_names, feature_values = NULL) {
  
  all_results <- data.frame()
  
  for (i in seq_along(feature_names)) {
    feat_name <- feature_names[i]
    
    if (!feat_name %in% feature_cols) {
      cat(sprintf("Warning: Feature '%s' not found, skipping...\n", feat_name))
      next
    }
    
    values <- if (!is.null(feature_values) && i <= length(feature_values)) {
      feature_values[[i]]
    } else {
      c(0, 1)
    }
    
    results <- sensitivity_test(base_features, feat_name, values, verbose = FALSE)
    all_results <- rbind(all_results, results)
  }
  
  return(all_results)
}

# =============================================
# 5. 从训练数据中获取示例
# =============================================

#' 从训练数据中获取一个示例作为基础特征
#' 
#' @param row_idx 行索引（如果为NULL则随机选择）
#' @return 特征向量
get_example_features <- function(row_idx = NULL) {
  if (is.null(row_idx)) {
    row_idx <- sample(nrow(df_train), 1)
  }
  
  example <- df_train[row_idx, feature_cols, drop = FALSE]
  cat(sprintf("Using example from training data (row %d)\n", row_idx))
  cat(sprintf("Original price: £%.2f\n", df_train$price_num[row_idx]))
  
  return(example)
}

# =============================================
# 6. 查找amenity相关特征
# =============================================

get_amenity_features <- function() {
  amenity_cols <- grep("^amenity_", feature_cols, value = TRUE)
  return(amenity_cols)
}

# =============================================
# 示例使用
# =============================================

cat("\n========================================\n")
cat("Example Usage:\n")
cat("========================================\n\n")

cat("1. Get an example from training data:\n")
cat("   example <- get_example_features()\n\n")

cat("2. Test a single feature (e.g., TV amenity):\n")
cat("   # First, find amenity features:\n")
cat("   amenity_cols <- get_amenity_features()\n")
cat("   # Then test:\n")
cat("   results <- sensitivity_test(example, 'amenity_TV', c(0, 1))\n\n")

cat("3. Test multiple features:\n")
cat("   results <- batch_sensitivity_test(example, c('amenity_TV', 'amenity_WiFi', 'accommodates'),\n")
cat("                                     list(c(0, 1), c(0, 1), c(2, 4, 6)))\n\n")

cat("4. Test any feature change:\n")
cat("   results <- sensitivity_test(example, 'bedrooms', c(1, 2, 3, 4))\n\n")

cat("========================================\n")
cat("Ready to use! Functions loaded:\n")
cat("  - predict_price_stacking()\n")
cat("  - sensitivity_test()\n")
cat("  - batch_sensitivity_test()\n")
cat("  - get_example_features()\n")
cat("  - get_amenity_features()\n")
cat("========================================\n\n")

