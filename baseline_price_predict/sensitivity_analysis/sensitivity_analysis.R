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

# 检查 reticulate 和 Python（可选，用于Neural Network模型）
use_nn_model <- TRUE
if (!require(reticulate, quietly = TRUE)) {
  cat("Warning: reticulate package not available. Neural Network model will be skipped.\n")
  cat("Only XGBoost model will be used for prediction.\n")
  use_nn_model <- FALSE
} else {
  library(reticulate)
  
  # 尝试配置Python（如果还没配置）
  if (!py_available()) {
    # 尝试使用已知的Python路径
    python_paths <- c(
      Sys.which("python3"),
      Sys.which("python"),
      "/usr/bin/python3",
      "/usr/local/bin/python3"
    )
    
    python_configured <- FALSE
    for (py_path in python_paths) {
      if (file.exists(py_path)) {
        tryCatch({
          use_python(py_path, required = FALSE)
          # 尝试运行一个简单的Python命令来激活
          py_run_string("import sys")
          if (py_available()) {
            cat(sprintf("✓ Python configured: %s\n", py_path))
            python_configured <- TRUE
            break
          }
        }, error = function(e) {
          # 继续尝试下一个路径
        })
      }
    }
    
    if (!python_configured) {
      cat("Warning: Python not available. Neural Network model will be skipped.\n")
      cat("Only XGBoost model will be used for prediction.\n")
      use_nn_model <- FALSE
    }
  }
  
  # 检查PyTorch
  if (use_nn_model && !py_module_available("torch")) {
    cat("Warning: PyTorch not available. Neural Network model will be skipped.\n")
    cat("Only XGBoost model will be used for prediction.\n")
    use_nn_model <- FALSE
  }
}

# 智能查找baseprice_model目录
original_wd <- getwd()

# 尝试多种可能的路径（按优先级）
baseprice_dir <- NULL

# 方法1: 当前目录就是baseprice_model
if (basename(getwd()) == "baseprice_model" && file.exists("nn_price_training_v4.csv")) {
  baseprice_dir <- getwd()
  cat("Detected: Currently in baseprice_model directory\n")
} else if (dir.exists("baseprice_model") && file.exists(file.path("baseprice_model", "nn_price_training_v4.csv"))) {
  # 方法2: 当前目录下的baseprice_model
  baseprice_dir <- normalizePath("baseprice_model")
  cat("Detected: baseprice_model in current directory\n")
} else if (dir.exists("../baseprice_model") && file.exists("../baseprice_model/nn_price_training_v4.csv")) {
  # 方法3: 上一级目录下的baseprice_model（从sensitivity_analysis文件夹运行）
  baseprice_dir <- normalizePath("../baseprice_model")
  cat("Detected: baseprice_model in parent directory\n")
} else if (dir.exists("../../baseprice_model") && file.exists("../../baseprice_model/nn_price_training_v4.csv")) {
  # 方法4: 上两级目录下的baseprice_model（如果在R_scripts中）
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

# 加载Neural Network模型（如果Python可用）
nn_model <- NULL
scaler_nn <- NULL

if (use_nn_model) {
  cat("Loading Neural Network model...\n")
  nn_model_path <- file.path(baseprice_dir, "best_price_A2_log_pytorch.pt")
  nn_meta_path <- file.path(baseprice_dir, "best_price_A2_log_pytorch_meta.pt")
  scaler_nn_path <- file.path(baseprice_dir, "scaler_price_pytorch.rds")
  
  if (!file.exists(nn_model_path) || !file.exists(nn_meta_path) || !file.exists(scaler_nn_path)) {
    cat("Warning: Neural Network model files not found. Skipping NN model.\n")
    use_nn_model <- FALSE
  } else {
    tryCatch({
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
      nn_model <- "loaded"  # 标记为已加载
    }, error = function(e) {
      cat(sprintf("Error loading Neural Network model: %s\n", e$message))
      cat("Falling back to XGBoost-only mode.\n")
      use_nn_model <<- FALSE
    })
  }
} else {
  cat("Skipping Neural Network model (Python/PyTorch not available).\n")
}

# 加载Stacking模型（如果NN可用）或准备仅使用XGBoost
meta_model <- NULL
meta_cv <- NULL
stacking_info <- NULL

if (use_nn_model) {
  cat("Loading Stacking meta model...\n")
  meta_model_path <- file.path(baseprice_dir, "meta_ridge_model.rds")
  meta_cv_path <- file.path(baseprice_dir, "meta_ridge_cv.rds")
  
  if (!file.exists(meta_model_path)) {
    cat("Warning: Stacking meta model not found. Will use XGBoost predictions directly.\n")
  } else {
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
  }
} else {
  cat("Using XGBoost-only mode (no stacking, no Neural Network).\n")
}

cat("All available models loaded successfully!\n\n")

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
  
  # Neural Network预测（如果可用）
  nn_pred <- NULL
  if (use_nn_model && !is.null(scaler_nn)) {
    tryCatch({
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
    }, error = function(e) {
      cat("Warning: Neural Network prediction failed, using XGBoost only.\n")
      nn_pred <<- NULL
    })
  }
  
  # 最终价格预测
  if (!is.null(nn_pred) && !is.null(meta_model)) {
    # 使用Stacking（XGBoost + NN）
    X_meta <- cbind(xgb_pred, nn_pred)
    lambda_val <- if (!is.null(meta_cv)) meta_cv$lambda.min else NULL
    final_price <- predict(meta_model, newx = X_meta, s = lambda_val)[, 1]
  } else {
    # 仅使用XGBoost
    final_price <- xgb_pred
    if (is.null(nn_pred)) nn_pred <- NA
  }
  
  return(list(
    xgb_pred = as.numeric(xgb_pred),
    nn_pred = if (!is.null(nn_pred)) as.numeric(nn_pred) else NA,
    final_price = as.numeric(final_price)
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
# 7. Amenity推荐功能：找出缺失的amenity并推荐前三个
# =============================================

#' 分析房源缺失的amenity，并推荐前三个最有价值的
#' 
#' @param property_features 房源特征向量（如果为NULL则随机选择一个）
#' @param top_n 推荐数量，默认3
#' @param verbose 是否显示详细信息
#' 
#' @return 数据框，包含推荐的amenity及其价格影响
recommend_amenities <- function(property_features = NULL, top_n = 3, verbose = TRUE) {
  
  # 如果没有提供房源，随机选择一个
  if (is.null(property_features)) {
    property_features <- get_example_features()
  } else {
    # 确保是数据框格式
    if (is.data.frame(property_features)) {
      property_features <- property_features[1, , drop = FALSE]
    }
  }
  
  # 获取所有amenity特征
  amenity_cols <- get_amenity_features()
  
  # 找出当前房源没有的amenity（值为0的）
  missing_amenities <- c()
  for (amenity in amenity_cols) {
    if (amenity %in% names(property_features)) {
      if (as.numeric(property_features[[amenity]]) == 0) {
        missing_amenities <- c(missing_amenities, amenity)
      }
    }
  }
  
  if (length(missing_amenities) == 0) {
    cat("这个房源已经有所有amenity了！\n")
    return(data.frame())
  }
  
  if (verbose) {
    cat("\n========================================\n")
    cat("Amenity 推荐分析\n")
    cat("========================================\n\n")
    
    # 显示当前已有的amenity
    existing_amenities <- c()
    for (amenity in amenity_cols) {
      if (amenity %in% names(property_features)) {
        if (as.numeric(property_features[[amenity]]) == 1) {
          existing_amenities <- c(existing_amenities, amenity)
        }
      }
    }
    
    cat(sprintf("当前房源已有的amenity (%d个):\n", length(existing_amenities)))
    if (length(existing_amenities) > 0) {
      for (i in seq_along(existing_amenities)) {
        amenity_name <- gsub("amenity_", "", existing_amenities[i])
        amenity_name <- gsub("\\.", " ", amenity_name)
        cat(sprintf("  %d. %s\n", i, amenity_name))
      }
    } else {
      cat("  (无)\n")
    }
    
    cat(sprintf("\n缺失的amenity (%d个):\n", length(missing_amenities)))
    cat("正在分析每个缺失amenity的价格影响...\n\n")
  }
  
  # 对每个缺失的amenity进行敏感性分析
  amenity_impacts <- data.frame()
  
  for (i in seq_along(missing_amenities)) {
    amenity <- missing_amenities[i]
    
    if (verbose && i %% 10 == 0) {
      cat(sprintf("  分析进度: %d/%d\n", i, length(missing_amenities)))
    }
    
    # 测试添加这个amenity的影响（从0到1）
    results <- sensitivity_test(property_features, amenity, c(0, 1), verbose = FALSE)
    
    # 提取价格变化（添加amenity后的影响）
    if (nrow(results) >= 2) {
      impact <- results$price_change[2]  # 从0到1的变化
      impact_pct <- results$price_change_pct[2]
      final_price_with <- results$final_price[2]
      
      amenity_impacts <- rbind(amenity_impacts, data.frame(
        amenity = amenity,
        amenity_name = gsub("amenity_", "", amenity),
        price_impact = impact,
        price_impact_pct = impact_pct,
        new_price = final_price_with,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # 只保留正面影响（价格提升）的amenity
  positive_impacts <- amenity_impacts[amenity_impacts$price_impact > 0, ]
  
  if (nrow(positive_impacts) == 0) {
    if (verbose) {
      cat("\n⚠ 警告: 没有找到能提升价格的amenity\n")
      cat("所有缺失的amenity都会降低价格或没有影响\n")
    }
    return(data.frame())
  }
  
  # 按价格影响降序排序
  positive_impacts <- positive_impacts[order(-positive_impacts$price_impact), ]
  
  # 选择前top_n个
  top_recommendations <- head(positive_impacts, min(top_n, nrow(positive_impacts)))
  
  if (verbose) {
    cat("\n========================================\n")
    cat(sprintf("推荐的前 %d 个 Amenity（仅显示能提升价格的）\n", nrow(top_recommendations)))
    cat("========================================\n\n")
    
    # 获取基础价格
    base_pred <- predict_price_stacking(property_features)
    base_price <- base_pred$final_price
    
    cat(sprintf("当前预测价格: £%.2f\n\n", base_price))
    
    if (nrow(top_recommendations) > 0) {
      for (i in 1:nrow(top_recommendations)) {
        rec <- top_recommendations[i, ]
        amenity_display <- gsub("\\.", " ", rec$amenity_name)
        amenity_display <- tools::toTitleCase(amenity_display)
        
        cat(sprintf("推荐 %d: %s\n", i, amenity_display))
        cat(sprintf("  价格提升: £%.2f (%.2f%%)\n", rec$price_impact, rec$price_impact_pct))
        cat(sprintf("  新价格: £%.2f\n", rec$new_price))
        cat("\n")
      }
    } else {
      cat("没有找到能提升价格的amenity推荐\n")
    }
    
    cat("========================================\n")
  }
  
  return(top_recommendations)
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
cat("  - recommend_amenities()  # NEW: 推荐缺失的amenity\n")
cat("========================================\n\n")

