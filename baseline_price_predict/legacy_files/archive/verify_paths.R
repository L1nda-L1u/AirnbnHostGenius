# =============================================
# verify_paths.R
# 验证所有路径是否正确
# =============================================

cat(paste0(rep("=", 80), collapse = ""), "\n")
cat("Path Verification\n")
cat(paste0(rep("=", 80), collapse = ""), "\n\n")

# 检查best_model文件夹
cat("[1] Checking best_model folder...\n")
best_model_files <- c(
  "nn_price_training_v4.csv",
  "xgb_model.json",
  "nn.onnx",
  "README.txt",
  "scaler_xgb.pkl",
  "scaler_price.pkl",
  "meta_ridge_model.pkl"
)

all_ok <- TRUE
for (file in best_model_files) {
  path <- file.path("best_model", file)
  if (file.exists(path)) {
    cat("  ✓", file, "\n")
  } else {
    cat("  ✗", file, "MISSING\n")
    all_ok <- FALSE
  }
}

# 检查R_scripts
cat("\n[2] Checking R_scripts folder...\n")
r_files <- c("Dataclean.R", "get_comps.R", "model_baseline.R", "model_rf.R", "predict_price.R")
for (file in r_files) {
  path <- file.path("R_scripts", file)
  if (file.exists(path)) {
    cat("  ✓", file, "\n")
  } else {
    cat("  ✗", file, "MISSING\n")
    all_ok <- FALSE
  }
}

# 检查关键数据文件
cat("\n[3] Checking data files...\n")
if (file.exists("../listings.csv.gz")) {
  cat("  ✓ listings.csv.gz found (for Dataclean.R)\n")
} else {
  cat("  ⚠ listings.csv.gz not found at ../listings.csv.gz\n")
  cat("     Dataclean.R may need path adjustment\n")
}

cat("\n", paste0(rep("=", 80), collapse = ""), "\n")
if (all_ok) {
  cat("All paths verified!\n")
} else {
  cat("Some files missing\n")
}
cat(paste0(rep("=", 80), collapse = ""), "\n")

