# =============================================
# XGBoost Model Training - Pure R Version
# Fully consistent with Python version
# =============================================

library(xgboost)
library(caret)
library(dplyr)
library(Matrix)

cat("========================================\n")
cat("XGBoost Model Training (Pure R)\n")
cat("========================================\n\n")

# =============================================
# 1. Load Data
# =============================================
cat("Loading data...\n")
# Data file in current directory
data_file <- "nn_price_training_v4.csv"
if (!file.exists(data_file)) {
  # Try other locations if not in current directory
  data_file <- "other_files/nn_price_training_v4.csv"
  if (!file.exists(data_file)) {
    data_file <- "R_scripts/best_model/nn_price_training_v4.csv"
  }
}
if (!file.exists(data_file)) {
  stop("Cannot find nn_price_training_v4.csv. Please ensure the file exists.")
}
cat(sprintf("Using data file: %s\n", data_file))
df_original <- read.csv(data_file, stringsAsFactors = FALSE)
cat(sprintf("Original data: %s rows\n", format(nrow(df_original), big.mark = ",")))

target_col <- "price_num"
feature_cols <- setdiff(colnames(df_original), target_col)

cat("\n===== ALL FEATURES USED =====\n")
for (i in seq_along(feature_cols)) {
  cat(sprintf("%d. %s\n", i, feature_cols[i]))
}
cat(sprintf("\nTotal features = %d\n\n", length(feature_cols)))

# =============================================
# 2. Data Cleaning (Fully consistent with Python version)
# =============================================
cat("Cleaning outliers...\n")
df <- df_original

# Cleaning rule 1: 2 or fewer people but price > 400
df <- df[!((df$accommodates <= 2) & (df$price_num > 400)), ]

# Cleaning rule 2: 4 or fewer people but price > 600
df <- df[!((df$accommodates <= 4) & (df$price_num > 600)), ]

# Cleaning rule 3: 6 or fewer people but price > 800
df <- df[!((df$accommodates <= 6) & (df$price_num > 800)), ]

# Cleaning rule 4: Remove extreme values above 99.5% quantile
upper <- quantile(df$price_num, 0.995)
df <- df[df$price_num < upper, ]

rownames(df) <- NULL
cat(sprintf("After cleaning: %s rows (removed %s outliers)\n", 
            format(nrow(df), big.mark = ","),
            format(nrow(df_original) - nrow(df), big.mark = ",")))

# =============================================
# 3. LOG-transform price → improves model stability
# =============================================
# Ensure column names are preserved
X <- as.matrix(df[, feature_cols, drop = FALSE])
colnames(X) <- feature_cols  # Ensure column names
X <- matrix(as.numeric(X), nrow = nrow(X), ncol = ncol(X))
colnames(X) <- feature_cols  # Set column names again (matrix() may lose them)
y_raw <- as.numeric(df[[target_col]])
y_log <- log1p(y_raw)  # log(price + 1)

# =============================================
# 4. Train/Test Split (Random split, real prediction scenario)
# =============================================
cat("\nUsing random split (real prediction scenario, no price stratification)...\n")
set.seed(42)
train_idx <- createDataPartition(y_log, p = 0.90, list = FALSE)

X_train <- X[train_idx, , drop = FALSE]
X_test <- X[-train_idx, , drop = FALSE]
y_train_log <- y_log[train_idx]
y_test_log <- y_log[-train_idx]
y_train_raw <- y_raw[train_idx]
y_test_raw <- y_raw[-train_idx]

cat(sprintf("\nData split statistics:\n"))
cat(sprintf("  Training set: %s rows\n", format(nrow(X_train), big.mark = ",")))
cat(sprintf("  Test set: %s rows\n", format(nrow(X_test), big.mark = ",")))
cat(sprintf("  Note: Using random split (real prediction scenario, no price stratification)\n"))

# =============================================
# 5. Standardize features
# =============================================
# Convert to data frame (required by preProcess)
X_train_df <- as.data.frame(X_train)
X_test_df <- as.data.frame(X_test)
colnames(X_train_df) <- feature_cols
colnames(X_test_df) <- feature_cols

preProc <- preProcess(X_train_df, method = c("center", "scale"))
X_train_scaled <- predict(preProc, X_train_df)
X_test_scaled <- predict(preProc, X_test_df)

# Ensure conversion to numeric matrix (required by XGBoost)
X_train_scaled <- as.matrix(X_train_scaled)
X_test_scaled <- as.matrix(X_test_scaled)
# Ensure numeric type
storage.mode(X_train_scaled) <- "numeric"
storage.mode(X_test_scaled) <- "numeric"

# =============================================
# 6. Train XGBoost
# =============================================
cat("\nTraining XGBoost...\n")

# Convert to DMatrix (XGBoost format)
# Ensure data is in matrix format
dtrain <- xgb.DMatrix(data = X_train_scaled, label = y_train_log)
dtest <- xgb.DMatrix(data = X_test_scaled, label = y_test_log)

# XGBoost parameters (consistent with Python version)
params <- list(
  objective = "reg:squarederror",
  eval_metric = "rmse",
  max_depth = 10,
  eta = 0.025,  # learning_rate
  subsample = 0.85,
  colsample_bytree = 0.85,
  lambda = 1.2,  # reg_lambda
  nthread = parallel::detectCores() - 1
)

# Train model
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 800,  # n_estimators
  watchlist = list(train = dtrain, test = dtest),
  early_stopping_rounds = 50,
  verbose = 1,
  print_every_n = 100
)

# =============================================
# 7. Evaluation
# =============================================
log_pred <- predict(xgb_model, dtest)
pred_real <- expm1(log_pred)  # Convert back to real price (£)
true_real <- y_test_raw

mae <- mean(abs(true_real - pred_real))
rmse <- sqrt(mean((true_real - pred_real)^2))
r2 <- cor(true_real, pred_real)^2

cat("\n===== FINAL XGBOOST RESULTS (REAL £) =====\n")
cat(sprintf("MAE:  %.4f\n", mae))
cat(sprintf("RMSE: %.4f\n", rmse))
cat(sprintf("R²:   %.4f\n", r2))

# =============================================
# 8. Scatter Plot
# =============================================
png("xgb_price_scatter.png", width = 700, height = 700, res = 300)
plot(true_real, pred_real, 
     xlab = "True Price (£)", 
     ylab = "Predicted Price (£)",
     main = "XGBoost Predictions vs True Prices",
     pch = 19, cex = 0.3, col = rgb(0, 0, 0, 0.3))
min_v <- min(c(true_real, pred_real))
max_v <- max(c(true_real, pred_real))
abline(a = 0, b = 1, col = "red", lty = 2, lwd = 2)
dev.off()

cat("\nScatter plot saved: xgb_price_scatter.png\n")

# =============================================
# 9. Print 10 random samples
# =============================================
cat("\n===== RANDOM 10 TEST SAMPLES =====\n")
set.seed(42)
indices <- sample(length(true_real), 10)

for (idx in indices) {
  cat("\n-------------------------------\n")
  cat(sprintf("Sample #%d\n", idx))
  cat(sprintf("True price: £%.2f\n", true_real[idx]))
  cat(sprintf("Predicted:  £%.2f\n", pred_real[idx]))
  cat("------ Feature Values ------\n")
  # Display top 20 features
  test_row <- df[-train_idx, ][idx, feature_cols[1:min(20, length(feature_cols))]]
  print(test_row)
}

# =============================================
# 10. Save Model
# =============================================
xgb.save(xgb_model, "best_xgb_log_model.xgb")
saveRDS(preProc, "scaler_xgb.rds")

cat("\nSaved: best_xgb_log_model.xgb + scaler_xgb.rds\n")
cat("Done.\n")



