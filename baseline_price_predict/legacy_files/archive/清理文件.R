# 清理脚本 - 只保留原始模型、stacking 和数据集

cat("========================================\n")
cat("清理 R_pure_models 文件夹\n")
cat("========================================\n\n")

# 要保留的文件
keep_files <- c(
  # 数据集
  "nn_price_training_v4.csv",
  
  # XGBoost 原始模型
  "best_xgb_log_model.xgb",
  "scaler_xgb.rds",
  
  # NN 原始模型（PyTorch）
  "best_price_A2_log_pytorch.pt",
  "best_price_A2_log_pytorch_meta.pt",
  "scaler_price_pytorch.rds",
  
  # Stacking 模型
  "meta_ridge_model.rds",
  "meta_ridge_cv.rds",
  "stacking_info.rds",
  
  # 项目文件
  "R_pure_models.Rproj",
  "README.md"
)

# 获取所有文件
all_files <- list.files(".", full.names = FALSE)

# 找出要删除的文件
files_to_delete <- setdiff(all_files, keep_files)

cat(sprintf("保留 %d 个文件:\n", length(keep_files)))
for (f in keep_files) {
  if (file.exists(f)) {
    cat(sprintf("  ✓ %s\n", f))
  }
}

cat(sprintf("\n删除 %d 个文件:\n", length(files_to_delete)))
for (f in files_to_delete) {
  if (file.exists(f)) {
    cat(sprintf("  ✗ %s\n", f))
    file.remove(f)
  }
}

cat("\n清理完成！\n")
cat("========================================\n")

