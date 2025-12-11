# =============================================
# Verify Paths - Check if all files are accessible
# =============================================

cat("========================================\n")
cat("Path Verification\n")
cat("========================================\n\n")

# Get current working directory
current_dir <- getwd()
cat(sprintf("Current directory: %s\n\n", current_dir))

# Check data file
cat("1. Checking data file...\n")
data_files <- c(
  "nn_price_training_v4.csv",
  "other_files/nn_price_training_v4.csv",
  "R_scripts/best_model/nn_price_training_v4.csv"
)
data_found <- FALSE
for (f in data_files) {
  if (file.exists(f)) {
    cat(sprintf("   ✓ Found: %s\n", f))
    data_found <- TRUE
    break
  }
}
if (!data_found) {
  cat("   ✗ Data file not found!\n")
}

# Check model files
cat("\n2. Checking model files...\n")
model_files <- c(
  "best_xgb_log_model.xgb",
  "scaler_xgb.rds"
)

all_models_exist <- TRUE
for (f in model_files) {
  if (file.exists(f)) {
    cat(sprintf("   ✓ %s\n", f))
  } else {
    cat(sprintf("   ✗ Missing: %s\n", f))
    all_models_exist <- FALSE
  }
}

# Check training scripts
cat("\n3. Checking training scripts...\n")
scripts <- c("train_xgb.R")
all_scripts_exist <- TRUE
for (s in scripts) {
  if (file.exists(s)) {
    cat(sprintf("   ✓ %s\n", s))
  } else {
    cat(sprintf("   ✗ Missing: %s\n", s))
    all_scripts_exist <- FALSE
  }
}

# Summary
cat("\n========================================\n")
cat("Summary:\n")
cat(sprintf("  Data file: %s\n", ifelse(data_found, "✓ Found", "✗ Not found")))
cat(sprintf("  Model files: %s\n", ifelse(all_models_exist, "✓ All present", "✗ Some missing")))
cat(sprintf("  Training scripts: %s\n", ifelse(all_scripts_exist, "✓ All present", "✗ Some missing")))

if (data_found && all_models_exist && all_scripts_exist) {
  cat("\n✓ All paths verified! Ready to train.\n")
} else {
  cat("\n⚠ Some files are missing. Please check above.\n")
}
cat("========================================\n")

