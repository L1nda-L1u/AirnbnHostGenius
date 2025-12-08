# =============================================
# 评估所有基线模型并生成对比图
# =============================================
# 
# 评估五个基线模型：
# 1. Linear Regression
# 2. Random Forest (ranger, 500 trees, mtry=3)
# 3. KNN Regression (k=5)
# 4. KNN Regression (k=10)
# 5. XGBoost (eta=0.05, max_depth=8, subsample=0.8, 500 rounds)
# 6. Neural Network (4-layer MLP, 256-128-64 units, 150 epochs)
#
# =============================================

library(dplyr)
library(caret)
library(ranger)
library(xgboost)
library(glmnet)
library(ggplot2)
library(gridExtra)
library(reticulate)
library(tidyr)

# 设置随机种子
set.seed(42)

# =============================================
# 1. 加载和准备数据
# =============================================
cat("========================================\n")
cat("Loading and preparing data...\n")
cat("========================================\n\n")

# 尝试找到 nn_price_training_v4.csv 文件
# 构建所有可能的路径
current_dir <- getwd()

# 首先尝试最常见的路径（基于用户的工作目录结构）
data_paths <- c(
  # 最可能的路径：从 AirbnbHostGeniusR 到 baseline_price_predict
  file.path(current_dir, "baseline_price_predict", "baseprice_model", "nn_price_training_v4.csv"),
  # 当前目录及子目录
  file.path(current_dir, "nn_price_training_v4.csv"),
  file.path(current_dir, "baseprice_model", "nn_price_training_v4.csv"),
  file.path(current_dir, "R_scripts", "best_model", "nn_price_training_v4.csv"),
  file.path(current_dir, "baseline_price_predict", "nn_price_training_v4.csv"),
  # 父目录及子目录
  file.path(dirname(current_dir), "baseline_price_predict", "baseprice_model", "nn_price_training_v4.csv"),
  file.path(dirname(current_dir), "nn_price_training_v4.csv"),
  # 相对路径
  "baseline_price_predict/baseprice_model/nn_price_training_v4.csv",
  "nn_price_training_v4.csv",
  "../baseprice_model/nn_price_training_v4.csv",
  "../nn_price_training_v4.csv",
  "../../baseline_price_predict/baseprice_model/nn_price_training_v4.csv"
)

# 尝试找到脚本所在目录（通过检查调用栈）
script_path <- tryCatch({
  # 方法1: 检查 sys.frame
  frames <- sys.frames()
  for (frame in frames) {
    if (exists("ofile", envir = frame)) {
      script_dir <- dirname(get("ofile", envir = frame))
      if (file.exists(script_dir)) {
        return(script_dir)
      }
    }
  }
  # 方法2: 如果脚本在 R_scripts 目录下
  if (grepl("R_scripts", current_dir) || file.exists(file.path(current_dir, "R_scripts"))) {
    return(file.path(current_dir, "R_scripts"))
  }
  NULL
}, error = function(e) NULL)

if (!is.null(script_path)) {
  # 添加脚本目录相关的路径
  data_paths <- c(
    file.path(script_path, "..", "baseprice_model", "nn_price_training_v4.csv"),
    file.path(script_path, "..", "..", "baseprice_model", "nn_price_training_v4.csv"),
    file.path(script_path, "best_model", "nn_price_training_v4.csv"),
    data_paths
  )
}

# 添加从 AirbnbHostGeniusR 到 baseline_price_predict 的路径
if (grepl("AirbnbHostGeniusR", current_dir)) {
  baseline_path <- file.path(current_dir, "baseline_price_predict", "baseprice_model", "nn_price_training_v4.csv")
  if (!baseline_path %in% data_paths) {
    data_paths <- c(baseline_path, data_paths)
  }
}

# 去重并尝试每个路径
data_paths <- unique(data_paths)

# 如果仍然找不到，尝试递归搜索（最多搜索2层深度）
if (length(data_paths) > 0) {
  # 添加一些常见的项目结构路径
  possible_base_dirs <- c(
    current_dir,
    dirname(current_dir),
    file.path(current_dir, "baseline_price_predict"),
    file.path(dirname(current_dir), "baseline_price_predict")
  )
  
  for (base_dir in possible_base_dirs) {
    if (dir.exists(base_dir)) {
      baseprice_path <- file.path(base_dir, "baseprice_model", "nn_price_training_v4.csv")
      if (!baseprice_path %in% data_paths) {
        data_paths <- c(data_paths, baseprice_path)
      }
    }
  }
}

data_paths <- unique(data_paths)
data_file <- NULL

cat("Searching for nn_price_training_v4.csv...\n")
cat(sprintf("Current directory: %s\n", current_dir))

# 优先检查最可能的路径（从当前目录到 baseline_price_predict）
most_likely_path <- file.path(current_dir, "baseline_price_predict", "baseprice_model", "nn_price_training_v4.csv")
if (file.exists(most_likely_path)) {
  data_file <- normalizePath(most_likely_path)
  cat(sprintf("Found data file: %s\n", data_file))
} else {
  # 如果最可能的路径不存在，尝试所有其他路径
  for (path in data_paths) {
    # 尝试直接检查（不展开路径，因为相对路径可能无法normalize）
    if (file.exists(path)) {
      data_file <- normalizePath(path)
      cat(sprintf("Found data file: %s\n", data_file))
      break
    }
    
    # 也尝试展开路径
    expanded_path <- tryCatch({
      normalizePath(path, mustWork = FALSE)
    }, error = function(e) {
      NULL
    })
    
    if (!is.null(expanded_path) && file.exists(expanded_path)) {
      data_file <- expanded_path
      cat(sprintf("Found data file: %s\n", expanded_path))
      break
    }
  }
}

if (is.null(data_file)) {
  # 提供更详细的错误信息
  cat("\nTried the following paths:\n")
  for (path in data_paths) {
    expanded_path <- tryCatch({
      normalizePath(path, mustWork = FALSE)
    }, error = function(e) {
      path
    })
    exists_status <- ifelse(file.exists(expanded_path), "[EXISTS]", "[NOT FOUND]")
    cat(sprintf("  %s %s\n", exists_status, expanded_path))
  }
  
  stop("\nCannot find nn_price_training_v4.csv.\n",
       "Current working directory: ", getwd(), "\n",
       "Please ensure the file exists or run the script from the correct directory.\n",
       "Expected location: baseline_price_predict/baseprice_model/nn_price_training_v4.csv")
}

# 加载数据
cat("Loading data...\n")
df_original <- read.csv(data_file, stringsAsFactors = FALSE)
cat(sprintf("Original data: %d rows, %d columns\n", nrow(df_original), ncol(df_original)))

# 检查目标变量
target_col <- "price_num"
if (!target_col %in% colnames(df_original)) {
  stop(sprintf("Target variable '%s' not found in data. Available columns: %s\n",
               target_col, paste(colnames(df_original), collapse = ", ")))
}

# 获取特征列
feature_cols <- setdiff(colnames(df_original), target_col)
cat(sprintf("Features: %d\n", length(feature_cols)))

# 数据清理（与 train_xgb.R 和 train_nn.R 一致）
cat("\nCleaning outliers...\n")
df <- df_original

# 清理规则
df <- df[!((df$accommodates <= 2) & (df$price_num > 400)), ]
df <- df[!((df$accommodates <= 4) & (df$price_num > 600)), ]
df <- df[!((df$accommodates <= 6) & (df$price_num > 800)), ]
upper <- quantile(df$price_num, 0.995, na.rm = TRUE)
df <- df[df$price_num < upper, ]

rownames(df) <- NULL
cat(sprintf("After cleaning: %d rows (removed %d rows)\n", 
            nrow(df), nrow(df_original) - nrow(df)))

# 确保所有特征都是数值型
X_df <- df[, feature_cols, drop = FALSE]
for (col in feature_cols) {
  X_df[[col]] <- as.numeric(X_df[[col]])
}

y_raw <- as.numeric(df[[target_col]])

# 移除包含NA的行
complete_cases <- complete.cases(X_df) & !is.na(y_raw)
X_df <- X_df[complete_cases, , drop = FALSE]
y_raw <- y_raw[complete_cases]

cat(sprintf("After removing NA: %d rows\n", nrow(X_df)))

# 划分训练集和测试集（使用相同的随机种子确保一致性）
set.seed(42)
train_index <- createDataPartition(y_raw, p = 0.8, list = FALSE)

X_train <- X_df[train_index, , drop = FALSE]
X_test <- X_df[-train_index, , drop = FALSE]
y_train <- y_raw[train_index]
y_test <- y_raw[-train_index]

cat(sprintf("\nTrain size: %d\n", nrow(X_train)))
cat(sprintf("Test size: %d\n", nrow(X_test)))
cat("\n")

# 存储所有模型的预测结果
results <- list()

# 为Linear Regression和Random Forest准备数据框（包含目标变量）
train_df <- cbind(X_train, price_num = y_train)
test_df <- cbind(X_test, price_num = y_test)

# =============================================
# 2. Linear Regression
# =============================================
cat("Training Linear Regression...\n")
model_lm <- lm(price_num ~ ., data = train_df)
pred_lm <- predict(model_lm, newdata = test_df)

results[["Linear Regression"]] <- list(
  predictions = pred_lm,
  r2 = cor(pred_lm, y_test)^2,
  rmse = sqrt(mean((pred_lm - y_test)^2)),
  mae = mean(abs(pred_lm - y_test))
)
cat(sprintf("  R²: %.4f, RMSE: %.2f, MAE: %.2f\n", 
            results[["Linear Regression"]]$r2,
            results[["Linear Regression"]]$rmse,
            results[["Linear Regression"]]$mae))
cat("\n")

# =============================================
# 3. Random Forest (ranger)
# =============================================
cat("Training Random Forest (ranger, 500 trees, mtry=3)...\n")
model_rf <- ranger(
  formula = price_num ~ .,
  data = train_df,
  num.trees = 500,
  mtry = 3,
  importance = "impurity",
  verbose = FALSE
)
pred_rf <- predict(model_rf, data = test_df)$predictions

results[["Random Forest"]] <- list(
  predictions = pred_rf,
  r2 = cor(pred_rf, y_test)^2,
  rmse = sqrt(mean((pred_rf - y_test)^2)),
  mae = mean(abs(pred_rf - y_test))
)
cat(sprintf("  R²: %.4f, RMSE: %.2f, MAE: %.2f\n", 
            results[["Random Forest"]]$r2,
            results[["Random Forest"]]$rmse,
            results[["Random Forest"]]$mae))
cat("\n")

# =============================================
# 4. KNN Regression (k=5)
# =============================================
cat("Training KNN Regression (k=5)...\n")

# KNN需要标准化后的数值矩阵
if (!require(FNN, quietly = TRUE)) {
  install.packages("FNN")
}
library(FNN)

# 标准化
preProc_knn <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preProc_knn, X_train)
X_test_scaled <- predict(preProc_knn, X_test)

# 转换为矩阵
X_train_scaled <- as.matrix(X_train_scaled)
X_test_scaled <- as.matrix(X_test_scaled)

knn_pred_5 <- knn.reg(
  train = X_train_scaled,
  test = X_test_scaled,
  y = y_train,
  k = 5
)$pred

results[["KNN (k=5)"]] <- list(
  predictions = knn_pred_5,
  r2 = cor(knn_pred_5, y_test)^2,
  rmse = sqrt(mean((knn_pred_5 - y_test)^2)),
  mae = mean(abs(knn_pred_5 - y_test))
)
cat(sprintf("  R²: %.4f, RMSE: %.2f, MAE: %.2f\n", 
            results[["KNN (k=5)"]]$r2,
            results[["KNN (k=5)"]]$rmse,
            results[["KNN (k=5)"]]$mae))
cat("\n")

# =============================================
# 5. KNN Regression (k=10)
# =============================================
cat("Training KNN Regression (k=10)...\n")

knn_pred_10 <- knn.reg(
  train = X_train_scaled,
  test = X_test_scaled,
  y = y_train,
  k = 10
)$pred

results[["KNN (k=10)"]] <- list(
  predictions = knn_pred_10,
  r2 = cor(knn_pred_10, y_test)^2,
  rmse = sqrt(mean((knn_pred_10 - y_test)^2)),
  mae = mean(abs(knn_pred_10 - y_test))
)
cat(sprintf("  R²: %.4f, RMSE: %.2f, MAE: %.2f\n", 
            results[["KNN (k=10)"]]$r2,
            results[["KNN (k=10)"]]$rmse,
            results[["KNN (k=10)"]]$mae))
cat("\n")

# =============================================
# 6. XGBoost
# =============================================
cat("Training XGBoost (eta=0.05, max_depth=8, subsample=0.8, 500 rounds)...\n")

# XGBoost需要矩阵格式
X_train_matrix <- as.matrix(X_train)
X_test_matrix <- as.matrix(X_test)

dtrain <- xgb.DMatrix(data = X_train_matrix, label = y_train)
dtest <- xgb.DMatrix(data = X_test_matrix, label = y_test)

params_xgb <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eta = 0.05,
  max_depth = 8,
  subsample = 0.8,
  colsample_bytree = 0.8,
  min_child_weight = 3
)

model_xgb <- xgb.train(
  params = params_xgb,
  data = dtrain,
  nrounds = 500,
  watchlist = list(train = dtrain, test = dtest),
  print_every_n = 50,
  early_stopping_rounds = 50,
  verbose = 0
)

pred_xgb <- predict(model_xgb, newdata = dtest)

results[["XGBoost"]] <- list(
  predictions = pred_xgb,
  r2 = cor(pred_xgb, y_test)^2,
  rmse = sqrt(mean((pred_xgb - y_test)^2)),
  mae = mean(abs(pred_xgb - y_test))
)
cat(sprintf("  R²: %.4f, RMSE: %.2f, MAE: %.2f\n", 
            results[["XGBoost"]]$r2,
            results[["XGBoost"]]$rmse,
            results[["XGBoost"]]$mae))
cat("\n")

# =============================================
# 7. Neural Network (PyTorch via reticulate)
# =============================================
cat("Training Neural Network (4-layer MLP, 256-128-64 units, 150 epochs)...\n")

# 检查Python环境
if (!py_available()) {
  cat("  Python not available, attempting to configure...\n")
  
  # 尝试自动配置 Python
  config_script <- file.path(getwd(), "R_scripts", "configure_python_simple.R")
  if (!file.exists(config_script)) {
    # 尝试其他路径
    config_script <- file.path(dirname(getwd()), "R_scripts", "configure_python_simple.R")
    if (!file.exists(config_script)) {
      config_script <- "configure_python_simple.R"
    }
  }
  
  if (file.exists(config_script)) {
    cat("  Loading Python configuration script...\n")
    tryCatch({
      source(config_script)
    }, error = function(e) {
      cat(sprintf("  Configuration script error: %s\n", e$message))
    })
  }
  
  # 再次检查
  if (!py_available()) {
    cat("  Warning: Python not available, skipping Neural Network\n")
    cat("  (Neural Network requires Python and PyTorch)\n")
    cat("  To configure Python, run: source('R_scripts/configure_python_simple.R')\n\n")
  } else {
    cat("  ✓ Python configured successfully\n")
  }
}

# 如果 Python 可用，继续训练
if (py_available()) {
  tryCatch({
    # 准备数据（使用标准化后的特征，复用KNN的标准化器）
    X_train_nn <- X_train_scaled
    X_test_nn <- X_test_scaled
    y_train_nn <- y_train
    y_test_nn <- y_test
    
    # 传递给Python
    py_run_string("
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
")
    
    py$X_train_nn <- X_train_nn
    py$X_test_nn <- X_test_nn
    py$y_train_nn <- as.numeric(y_train_nn)
    py$y_test_nn <- as.numeric(y_test_nn)
    
    # 在Python中定义和训练模型
    py_run_string("
# 定义神经网络
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.net(x).squeeze()

device = torch.device('cpu')
input_dim = X_train_nn.shape[1]

model_nn = MLP(input_dim).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model_nn.parameters(), lr=0.001)

# 转换为tensor
X_train_tensor = torch.FloatTensor(X_train_nn).to(device)
y_train_tensor = torch.FloatTensor(y_train_nn).to(device)
X_test_tensor = torch.FloatTensor(X_test_nn).to(device)

# 训练
model_nn.train()
for epoch in range(150):
    optimizer.zero_grad()
    outputs = model_nn(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 30 == 0:
        print(f'Epoch [{epoch+1}/150], Loss: {loss.item():.4f}')

# 预测
model_nn.eval()
with torch.no_grad():
    pred_nn = model_nn(X_test_tensor).cpu().numpy()
")
    
    pred_nn <- py$pred_nn
    
    results[["Neural Network"]] <- list(
      predictions = pred_nn,
      r2 = cor(pred_nn, y_test)^2,
      rmse = sqrt(mean((pred_nn - y_test)^2)),
      mae = mean(abs(pred_nn - y_test))
    )
    cat(sprintf("  R²: %.4f, RMSE: %.2f, MAE: %.2f\n", 
                results[["Neural Network"]]$r2,
                results[["Neural Network"]]$rmse,
                results[["Neural Network"]]$mae))
    cat("\n")
  }, error = function(e) {
    cat(sprintf("  Error training Neural Network: %s\n", e$message))
    cat("  Skipping Neural Network\n\n")
  })
} else {
  cat("  Skipping Neural Network (Python not available)\n\n")
}

# =============================================
# 8. 汇总结果并生成对比图
# =============================================
cat("========================================\n")
cat("Generating comparison plots...\n")
cat("========================================\n\n")

# 创建结果汇总表
summary_df <- data.frame(
  Model = names(results),
  R2 = sapply(results, function(x) x$r2),
  RMSE = sapply(results, function(x) x$rmse),
  MAE = sapply(results, function(x) x$mae)
)

# 按R²排序
summary_df <- summary_df[order(-summary_df$R2), ]

print(summary_df)

# =============================================
# 9. 生成可视化图表
# =============================================

# 9.1 R²对比柱状图
p1 <- ggplot(summary_df, aes(x = reorder(Model, R2), y = R2, fill = R2)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_gradient(low = "lightblue", high = "darkblue") +
  labs(
    title = "Model Comparison: R² Score",
    x = "Model",
    y = "R² Score",
    fill = "R²"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.text = element_text(size = 10),
    legend.position = "none"
  ) +
  geom_text(aes(label = sprintf("%.4f", R2)), hjust = -0.1, size = 3.5)

# 9.2 RMSE对比柱状图
p2 <- ggplot(summary_df, aes(x = reorder(Model, -RMSE), y = RMSE, fill = RMSE)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_gradient(low = "darkred", high = "lightcoral") +
  labs(
    title = "Model Comparison: RMSE",
    x = "Model",
    y = "RMSE (£)",
    fill = "RMSE"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.text = element_text(size = 10),
    legend.position = "none"
  ) +
  geom_text(aes(label = sprintf("%.2f", RMSE)), hjust = -0.1, size = 3.5)

# 9.3 MAE对比柱状图
p3 <- ggplot(summary_df, aes(x = reorder(Model, -MAE), y = MAE, fill = MAE)) +
  geom_bar(stat = "identity") +
  coord_flip() +
  scale_fill_gradient(low = "darkgreen", high = "lightgreen") +
  labs(
    title = "Model Comparison: MAE",
    x = "Model",
    y = "MAE (£)",
    fill = "MAE"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.text = element_text(size = 10),
    legend.position = "none"
  ) +
  geom_text(aes(label = sprintf("%.2f", MAE)), hjust = -0.1, size = 3.5)

# 9.4 综合对比图（三个指标）
summary_long <- summary_df %>%
  tidyr::pivot_longer(cols = c(R2, RMSE, MAE), 
                      names_to = "Metric", 
                      values_to = "Value") %>%
  mutate(Metric = factor(Metric, levels = c("R2", "RMSE", "MAE")))

# 标准化值用于对比（R²越大越好，RMSE和MAE越小越好）
summary_long_norm <- summary_long %>%
  group_by(Metric) %>%
  mutate(
    ValueNorm = ifelse(Metric == "R2", 
                       Value,  # R²直接使用
                       max(Value) - Value)  # RMSE和MAE取反
  ) %>%
  ungroup()

p4 <- ggplot(summary_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  facet_wrap(~Metric, scales = "free_y", ncol = 3) +
  labs(
    title = "Model Performance Comparison: All Metrics",
    x = "Model",
    y = "Value",
    fill = "Metric"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5, size = 14, face = "bold"),
    axis.text.x = element_text(angle = 45, hjust = 1, size = 9),
    strip.text = element_text(size = 11, face = "bold")
  )

# 9.5 散点图：预测 vs 真实值（每个模型）
scatter_plots <- list()
for (i in seq_along(results)) {
  model_name <- names(results)[i]
  pred <- results[[model_name]]$predictions
  r2_val <- results[[model_name]]$r2
  
  scatter_data <- data.frame(
    True = y_test,
    Predicted = pred
  )
  
  scatter_plots[[i]] <- ggplot(scatter_data, aes(x = True, y = Predicted)) +
    geom_point(alpha = 0.5, color = "steelblue") +
    geom_abline(intercept = 0, slope = 1, color = "red", linetype = "dashed") +
    labs(
      title = paste0(model_name, "\nR² = ", sprintf("%.4f", r2_val)),
      x = "True Price (£)",
      y = "Predicted Price (£)"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, size = 10))
}

# 保存所有图表
cat("Saving plots...\n")

# 保存R²对比图
ggsave("baseline_models_r2_comparison.png", p1, 
       width = 10, height = 6, dpi = 300)
cat("  Saved: baseline_models_r2_comparison.png\n")

# 保存RMSE对比图
ggsave("baseline_models_rmse_comparison.png", p2, 
       width = 10, height = 6, dpi = 300)
cat("  Saved: baseline_models_rmse_comparison.png\n")

# 保存MAE对比图
ggsave("baseline_models_mae_comparison.png", p3, 
       width = 10, height = 6, dpi = 300)
cat("  Saved: baseline_models_mae_comparison.png\n")

# 保存综合对比图
ggsave("baseline_models_all_metrics.png", p4, 
       width = 14, height = 5, dpi = 300)
cat("  Saved: baseline_models_all_metrics.png\n")

# 保存散点图网格
if (length(scatter_plots) > 0) {
  png("baseline_models_scatter_plots.png", width = 15, height = 10, 
      units = "in", res = 300)
  do.call(gridExtra::grid.arrange, c(scatter_plots, ncol = 3))
  dev.off()
  cat("  Saved: baseline_models_scatter_plots.png\n")
}

# 保存结果汇总表
write.csv(summary_df, "baseline_models_summary.csv", row.names = FALSE)
cat("  Saved: baseline_models_summary.csv\n")

cat("\n========================================\n")
cat("All plots saved successfully!\n")
cat("========================================\n")

# 打印最终汇总
cat("\nFinal Summary:\n")
cat("==============\n")
print(summary_df)

