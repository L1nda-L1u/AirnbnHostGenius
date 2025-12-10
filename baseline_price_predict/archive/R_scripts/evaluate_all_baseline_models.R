# =============================================
# Evaluate All Baseline Models and Generate Comparison Plots
# =============================================
# 
# Evaluates baseline models:
# 1. Linear Regression
# 2. Random Forest (ranger, 500 trees, mtry=3)
# 3. KNN Regression (k=5)
# 4. KNN Regression (k=10)
# 5. XGBoost (pre-trained model)
# 6. Neural Network (pre-trained model)
# 7. Stacking (XGBoost + Neural Network, pre-trained model)
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

# Set random seed
set.seed(42)

# =============================================
# 1. Load and Prepare Data
# =============================================
cat("========================================\n")
cat("Loading and preparing data...\n")
cat("========================================\n\n")

# Try to find nn_price_training_v4.csv file
# Build all possible paths
current_dir <- getwd()

# First try the most likely paths (based on user's working directory structure)
data_paths <- c(
  # Most likely path: from AirbnbHostGeniusR to baseline_price_predict
  file.path(current_dir, "baseline_price_predict", "baseprice_model", "nn_price_training_v4.csv"),
  # Current directory and subdirectories
  file.path(current_dir, "nn_price_training_v4.csv"),
  file.path(current_dir, "baseprice_model", "nn_price_training_v4.csv"),
  file.path(current_dir, "R_scripts", "best_model", "nn_price_training_v4.csv"),
  file.path(current_dir, "baseline_price_predict", "nn_price_training_v4.csv"),
  # Parent directory and subdirectories
  file.path(dirname(current_dir), "baseline_price_predict", "baseprice_model", "nn_price_training_v4.csv"),
  file.path(dirname(current_dir), "nn_price_training_v4.csv"),
  # Relative paths
  "baseline_price_predict/baseprice_model/nn_price_training_v4.csv",
  "nn_price_training_v4.csv",
  "../baseprice_model/nn_price_training_v4.csv",
  "../nn_price_training_v4.csv",
  "../../baseline_price_predict/baseprice_model/nn_price_training_v4.csv"
)

# Try to find script directory (by checking call stack)
script_path <- tryCatch({
  # Method 1: Check sys.frame
  frames <- sys.frames()
  for (frame in frames) {
    if (exists("ofile", envir = frame)) {
      script_dir <- dirname(get("ofile", envir = frame))
      if (file.exists(script_dir)) {
        return(script_dir)
      }
    }
  }
  # Method 2: If script is in R_scripts directory
  if (grepl("R_scripts", current_dir) || file.exists(file.path(current_dir, "R_scripts"))) {
    return(file.path(current_dir, "R_scripts"))
  }
  NULL
}, error = function(e) NULL)

if (!is.null(script_path)) {
  # Add script directory related paths
  data_paths <- c(
    file.path(script_path, "..", "baseprice_model", "nn_price_training_v4.csv"),
    file.path(script_path, "..", "..", "baseprice_model", "nn_price_training_v4.csv"),
    file.path(script_path, "best_model", "nn_price_training_v4.csv"),
    data_paths
  )
}

# Add path from AirbnbHostGeniusR to baseline_price_predict
if (grepl("AirbnbHostGeniusR", current_dir)) {
  baseline_path <- file.path(current_dir, "baseline_price_predict", "baseprice_model", "nn_price_training_v4.csv")
  if (!baseline_path %in% data_paths) {
    data_paths <- c(baseline_path, data_paths)
  }
}

# Remove duplicates and try each path
data_paths <- unique(data_paths)

# If still not found, try recursive search (max 2 levels deep)
if (length(data_paths) > 0) {
  # Add some common project structure paths
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

# Prioritize checking the most likely path (from current directory to baseline_price_predict)
most_likely_path <- file.path(current_dir, "baseline_price_predict", "baseprice_model", "nn_price_training_v4.csv")
if (file.exists(most_likely_path)) {
  data_file <- normalizePath(most_likely_path)
  cat(sprintf("Found data file: %s\n", data_file))
} else {
  # If the most likely path doesn't exist, try all other paths
  for (path in data_paths) {
    # Try direct check (without expanding path, as relative paths may not normalize)
    if (file.exists(path)) {
      data_file <- normalizePath(path)
      cat(sprintf("Found data file: %s\n", data_file))
      break
    }
    
    # Also try expanding path
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
  # Provide more detailed error information
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

# Load data
cat("Loading data...\n")
df_original <- read.csv(data_file, stringsAsFactors = FALSE)
cat(sprintf("Original data: %d rows, %d columns\n", nrow(df_original), ncol(df_original)))

# Check target variable
target_col <- "price_num"
if (!target_col %in% colnames(df_original)) {
  stop(sprintf("Target variable '%s' not found in data. Available columns: %s\n",
               target_col, paste(colnames(df_original), collapse = ", ")))
}

# Get feature columns
feature_cols <- setdiff(colnames(df_original), target_col)
cat(sprintf("Features: %d\n", length(feature_cols)))

# Data cleaning (consistent with train_xgb.R and train_nn.R)
cat("\nCleaning outliers...\n")
df <- df_original

# Cleaning rules
df <- df[!((df$accommodates <= 2) & (df$price_num > 400)), ]
df <- df[!((df$accommodates <= 4) & (df$price_num > 600)), ]
df <- df[!((df$accommodates <= 6) & (df$price_num > 800)), ]
upper <- quantile(df$price_num, 0.995, na.rm = TRUE)
df <- df[df$price_num < upper, ]

rownames(df) <- NULL
cat(sprintf("After cleaning: %d rows (removed %d rows)\n", 
            nrow(df), nrow(df_original) - nrow(df)))

# Ensure all features are numeric
X_df <- df[, feature_cols, drop = FALSE]
for (col in feature_cols) {
  X_df[[col]] <- as.numeric(X_df[[col]])
}

y_raw <- as.numeric(df[[target_col]])

# Remove rows with NA
complete_cases <- complete.cases(X_df) & !is.na(y_raw)
X_df <- X_df[complete_cases, , drop = FALSE]
y_raw <- y_raw[complete_cases]

cat(sprintf("After removing NA: %d rows\n", nrow(X_df)))

# Split training and test sets
# IMPORTANT: Use the same splitting method as train_xgb.R and train_nn.R
# - Use log-transformed price for splitting (y_log)
# - Use 90% training set (consistent with training scripts)
# - This ensures the test set matches the one used during training
set.seed(42)
y_log <- log1p(y_raw)  # Log transformation (consistent with training scripts)
train_index <- createDataPartition(y_log, p = 0.90, list = FALSE)  # 90% train (consistent with training scripts)

X_train <- X_df[train_index, , drop = FALSE]
X_test <- X_df[-train_index, , drop = FALSE]
y_train <- y_raw[train_index]
y_test <- y_raw[-train_index]

cat(sprintf("\nTrain size: %d\n", nrow(X_train)))
cat(sprintf("Test size: %d\n", nrow(X_test)))
cat("\n")

# Store predictions from all models
results <- list()

# Prepare data frames for Linear Regression and Random Forest (with target variable)
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

# KNN requires standardized numeric matrix
if (!require(FNN, quietly = TRUE)) {
  install.packages("FNN")
}
library(FNN)

# Standardize
preProc_knn <- preProcess(X_train, method = c("center", "scale"))
X_train_scaled <- predict(preProc_knn, X_train)
X_test_scaled <- predict(preProc_knn, X_test)

# Convert to matrix
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
# 6. XGBoost (Load Pre-trained Model)
# =============================================
cat("Loading pre-trained XGBoost model...\n")

tryCatch({
  # Find XGBoost model file
  xgb_paths <- c(
    file.path(getwd(), "baseprice_model", "best_xgb_log_model.xgb"),
    file.path(getwd(), "baseline_price_predict", "baseprice_model", "best_xgb_log_model.xgb"),
    file.path(dirname(getwd()), "baseprice_model", "best_xgb_log_model.xgb"),
    "baseprice_model/best_xgb_log_model.xgb",
    "../baseprice_model/best_xgb_log_model.xgb"
  )
  
  xgb_file <- NULL
  for (path in xgb_paths) {
    if (file.exists(path)) {
      xgb_file <- normalizePath(path)
      break
    }
  }
  
  if (is.null(xgb_file)) {
    stop("Cannot find best_xgb_log_model.xgb. Please train the model first using train_xgb.R")
  }
  
  # Find scaler file
  scaler_xgb_paths <- c(
    file.path(getwd(), "baseprice_model", "scaler_xgb.rds"),
    file.path(getwd(), "baseline_price_predict", "baseprice_model", "scaler_xgb.rds"),
    file.path(dirname(getwd()), "baseprice_model", "scaler_xgb.rds"),
    "baseprice_model/scaler_xgb.rds",
    "../baseprice_model/scaler_xgb.rds"
  )
  
  scaler_xgb_file <- NULL
  for (path in scaler_xgb_paths) {
    if (file.exists(path)) {
      scaler_xgb_file <- normalizePath(path)
      break
    }
  }
  
  if (is.null(scaler_xgb_file)) {
    stop("Cannot find scaler_xgb.rds. Please train the model first using train_xgb.R")
  }
  
  cat(sprintf("  Model file: %s\n", xgb_file))
  cat(sprintf("  Scaler file: %s\n", scaler_xgb_file))
  
  # Load model and scaler
  xgb_model <- xgb.load(xgb_file)
  scaler_xgb <- readRDS(scaler_xgb_file)
  
  # Prepare test data (standardize using scaler)
  X_test_df <- as.data.frame(X_test)
  colnames(X_test_df) <- feature_cols
  X_test_scaled <- predict(scaler_xgb, X_test_df)
  X_test_scaled <- as.matrix(X_test_scaled)
  
  # Create DMatrix and predict
  dtest <- xgb.DMatrix(data = X_test_scaled)
  pred_xgb_log <- predict(xgb_model, dtest)
  
  # XGBoost model was trained in log space, need to convert back to original space
  pred_xgb <- expm1(pred_xgb_log)
  
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
}, error = function(e) {
  cat(sprintf("  Error loading XGBoost model: %s\n", e$message))
  cat("  Skipping XGBoost\n\n")
})

# =============================================
# 7. Neural Network (Load Pre-trained Model)
# =============================================
cat("Loading pre-trained Neural Network model...\n")

# Check Python environment (don't need py_available(), just need to import torch)
if (!py_module_available("torch")) {
  cat("  Python not available, attempting to configure...\n")
  
  # Try to automatically configure Python
  config_script <- file.path(getwd(), "R_scripts", "configure_python_simple.R")
  if (!file.exists(config_script)) {
    # Try other paths
    config_script <- file.path(dirname(getwd()), "R_scripts", "configure_python_simple.R")
    if (!file.exists(config_script)) {
      config_script <- file.path(getwd(), "baseline_price_predict", "R_scripts", "configure_python_simple.R")
      if (!file.exists(config_script)) {
        config_script <- "configure_python_simple.R"
      }
    }
  }
  
  if (file.exists(config_script)) {
    cat("  Loading Python configuration script...\n")
    tryCatch({
      source(config_script)
    }, error = function(e) {
      cat(sprintf("  Configuration script error: %s\n", e$message))
    })
  } else {
    cat("  Configuration script not found, trying manual configuration...\n")
    # Try to manually find Python
    python_paths <- c(
      Sys.which("python"),
      Sys.which("python3"),
      "C:/Python39/python.exe",
      "C:/Python310/python.exe",
      "C:/Python311/python.exe",
      "C:/Python312/python.exe"
    )
    python_paths <- python_paths[python_paths != ""]
    
    for (py_path in python_paths) {
      if (file.exists(py_path)) {
        tryCatch({
          use_python(py_path, required = FALSE)
          if (py_available()) {
            cat(sprintf("  ✓ Found Python at: %s\n", py_path))
            break
          }
        }, error = function(e) {
          # Continue to next
        })
      }
    }
  }
  
  # Check again
  if (!py_module_available("torch")) {
    cat("  Warning: PyTorch not available, skipping Neural Network\n")
    cat("  (Neural Network requires Python and PyTorch)\n")
    cat("  To configure Python, run: source('R_scripts/configure_python_simple.R')\n\n")
  } else {
    cat("  ✓ PyTorch is available\n")
  }
}

# =============================================
# 7. Neural Network (Load Pre-trained Model)
# =============================================
# If Python is available, load pre-trained model
if (py_module_available("torch")) {
  tryCatch({
    cat("  Loading pre-trained Neural Network model...\n")
    
    # Find model file paths
    model_paths <- c(
      file.path(getwd(), "baseprice_model", "best_price_A2_log_pytorch.pt"),
      file.path(getwd(), "baseline_price_predict", "baseprice_model", "best_price_A2_log_pytorch.pt"),
      file.path(dirname(getwd()), "baseprice_model", "best_price_A2_log_pytorch.pt"),
      "baseprice_model/best_price_A2_log_pytorch.pt",
      "../baseprice_model/best_price_A2_log_pytorch.pt"
    )
    
    model_file <- NULL
    for (path in model_paths) {
      if (file.exists(path)) {
        model_file <- normalizePath(path)
        break
      }
    }
    
    if (is.null(model_file)) {
      stop("Cannot find best_price_A2_log_pytorch.pt. Please train the model first using train_nn.R")
    }
    
    # Find scaler file
    scaler_paths <- c(
      file.path(getwd(), "baseprice_model", "scaler_price_pytorch.rds"),
      file.path(getwd(), "baseline_price_predict", "baseprice_model", "scaler_price_pytorch.rds"),
      file.path(dirname(getwd()), "baseprice_model", "scaler_price_pytorch.rds"),
      "baseprice_model/scaler_price_pytorch.rds",
      "../baseprice_model/scaler_price_pytorch.rds"
    )
    
    scaler_file <- NULL
    for (path in scaler_paths) {
      if (file.exists(path)) {
        scaler_file <- normalizePath(path)
        break
      }
    }
    
    if (is.null(scaler_file)) {
      stop("Cannot find scaler_price_pytorch.rds. Please train the model first using train_nn.R")
    }
    
    # Find meta file
    meta_paths <- c(
      file.path(getwd(), "baseprice_model", "best_price_A2_log_pytorch_meta.pt"),
      file.path(getwd(), "baseline_price_predict", "baseprice_model", "best_price_A2_log_pytorch_meta.pt"),
      file.path(dirname(getwd()), "baseprice_model", "best_price_A2_log_pytorch_meta.pt"),
      "baseprice_model/best_price_A2_log_pytorch_meta.pt",
      "../baseprice_model/best_price_A2_log_pytorch_meta.pt"
    )
    
    meta_file <- NULL
    for (path in meta_paths) {
      if (file.exists(path)) {
        meta_file <- normalizePath(path)
        break
      }
    }
    
    cat(sprintf("  Model file: %s\n", model_file))
    cat(sprintf("  Scaler file: %s\n", scaler_file))
    if (!is.null(meta_file)) {
      cat(sprintf("  Meta file: %s\n", meta_file))
    }
    
    # Load scaler (for standardizing test data)
    scaler <- readRDS(scaler_file)
    # Ensure X_test is data.frame format with correct column names
    X_test_df <- as.data.frame(X_test)
    colnames(X_test_df) <- feature_cols
    X_test_scaled_nn <- as.matrix(predict(scaler, X_test_df))
    
    # Pass to Python
    py_run_string("
import numpy as np
import torch
import torch.nn as nn
import os

# Define model architecture (consistent with train_nn.R)
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
    
    # Load model metadata to get input_dim
    py$meta_file <- meta_file
    py$model_file <- model_file
    py$X_test_scaled_nn <- X_test_scaled_nn
    
    py_run_string("
# Load metadata
if meta_file and os.path.exists(meta_file):
    meta = torch.load(meta_file, map_location='cpu')
    input_dim = meta['input_dim']
    print(f'Loaded input_dim from meta: {input_dim}')
else:
    # If no meta file, use test data dimension
    input_dim = X_test_scaled_nn.shape[1]
    print(f'Using input_dim from test data: {input_dim}')

# Create model and load weights
model_nn = PricePredictor(input_dim)
model_nn.load_state_dict(torch.load(model_file, map_location='cpu'))
model_nn.eval()
print('Model loaded successfully')
")
    
    # Make predictions
    py_run_string("
# Convert to tensor and predict
X_test_tensor = torch.FloatTensor(X_test_scaled_nn)
with torch.no_grad():
    pred_log_tensor = model_nn(X_test_tensor)
    pred_log = pred_log_tensor.numpy().flatten()

# Convert predictions from log space back to original space
pred_nn = np.expm1(pred_log)

print(f'Prediction shape: {pred_nn.shape}')
print(f'Prediction range: [{pred_nn.min():.2f}, {pred_nn.max():.2f}]')
")
    
    pred_nn <- py$pred_nn
    
    # Ensure prediction result is a 1D vector
    if (length(dim(pred_nn)) > 1) {
      pred_nn <- as.vector(pred_nn)
    }
    
    # Calculate metrics
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
    cat(sprintf("  Error loading Neural Network: %s\n", e$message))
    cat("  Full error details:\n")
    if (exists("py_last_error", envir = asNamespace("reticulate"))) {
      tryCatch({
        py_error <- reticulate::py_last_error()
        cat(sprintf("    Python error: %s\n", py_error$message))
      }, error = function(e2) {
        # Ignore
      })
    }
    cat("  Skipping Neural Network\n\n")
  })
} else {
  cat("  Skipping Neural Network (PyTorch not available)\n\n")
}

# =============================================
# 8. Stacking Model (Load Pre-trained Model)
# =============================================
cat("Loading Stacking Model (XGBoost + Neural Network)...\n")

tryCatch({
  # Find stacking model files
  stacking_paths <- c(
    file.path(getwd(), "baseprice_model", "meta_ridge_model.rds"),
    file.path(getwd(), "baseline_price_predict", "baseprice_model", "meta_ridge_model.rds"),
    file.path(dirname(getwd()), "baseprice_model", "meta_ridge_model.rds"),
    "baseprice_model/meta_ridge_model.rds",
    "../baseprice_model/meta_ridge_model.rds"
  )
  
  stacking_file <- NULL
  for (path in stacking_paths) {
    if (file.exists(path)) {
      stacking_file <- normalizePath(path)
      break
    }
  }
  
  if (is.null(stacking_file)) {
    cat("  ⚠ Stacking model not found, skipping...\n")
    cat("  (Run train_stacking.R first to create the stacking model)\n\n")
  } else {
    # Load stacking model
    meta_ridge <- readRDS(stacking_file)
    
    # Find CV model (for lambda.min)
    cv_paths <- c(
      file.path(getwd(), "baseprice_model", "meta_ridge_cv.rds"),
      file.path(getwd(), "baseline_price_predict", "baseprice_model", "meta_ridge_cv.rds"),
      file.path(dirname(getwd()), "baseprice_model", "meta_ridge_cv.rds"),
      "baseprice_model/meta_ridge_cv.rds",
      "../baseprice_model/meta_ridge_cv.rds"
    )
    
    meta_ridge_cv <- NULL
    for (path in cv_paths) {
      if (file.exists(path)) {
        meta_ridge_cv <- readRDS(path)
        break
      }
    }
    
    # Get XGBoost and NN predictions (must be already calculated)
    if ("XGBoost" %in% names(results) && "Neural Network" %in% names(results)) {
      xgb_pred <- results[["XGBoost"]]$predictions
      nn_pred <- results[["Neural Network"]]$predictions
      
      # Ensure consistent length
      min_len <- min(length(y_test), length(xgb_pred), length(nn_pred))
      y_test_stack <- y_test[1:min_len]
      xgb_pred <- xgb_pred[1:min_len]
      nn_pred <- nn_pred[1:min_len]
      
      # Prepare stacking input (predictions from both models)
      X_meta <- cbind(
        xgb_pred = xgb_pred,
        nn_pred = nn_pred
      )
      
      # Stacking prediction
      if (!is.null(meta_ridge_cv)) {
        stack_pred <- predict(meta_ridge, newx = X_meta, s = meta_ridge_cv$lambda.min)[, 1]
      } else {
        # If no CV model, use default lambda
        stack_pred <- predict(meta_ridge, newx = X_meta)[, 1]
      }
      
      results[["Stacking (XGB+NN)"]] <- list(
        predictions = stack_pred,
        r2 = cor(stack_pred, y_test_stack)^2,
        rmse = sqrt(mean((stack_pred - y_test_stack)^2)),
        mae = mean(abs(stack_pred - y_test_stack))
      )
      cat(sprintf("  R²: %.4f, RMSE: %.2f, MAE: %.2f\n", 
                  results[["Stacking (XGB+NN)"]]$r2,
                  results[["Stacking (XGB+NN)"]]$rmse,
                  results[["Stacking (XGB+NN)"]]$mae))
      cat("\n")
    } else {
      cat("  ⚠ Cannot create stacking predictions: XGBoost or Neural Network predictions not available\n\n")
    }
  }
}, error = function(e) {
  cat(sprintf("  Error loading Stacking model: %s\n", e$message))
  cat("  Skipping Stacking\n\n")
})

# =============================================
# 9. Summarize Results and Generate Comparison Plots
# =============================================
cat("========================================\n")
cat("Generating comparison plots...\n")
cat("========================================\n\n")

# Create results summary table
summary_df <- data.frame(
  Model = names(results),
  R2 = sapply(results, function(x) x$r2),
  RMSE = sapply(results, function(x) x$rmse),
  MAE = sapply(results, function(x) x$mae)
)

# Sort by R²
summary_df <- summary_df[order(-summary_df$R2), ]

print(summary_df)

# =============================================
# 10. Generate Visualization Charts
# =============================================

# 10.1 R² comparison bar chart
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

# 10.2 RMSE comparison bar chart
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

# 10.3 MAE comparison bar chart
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

# 10.4 Comprehensive comparison chart (all three metrics)
summary_long <- summary_df %>%
  tidyr::pivot_longer(cols = c(R2, RMSE, MAE), 
                      names_to = "Metric", 
                      values_to = "Value") %>%
  mutate(Metric = factor(Metric, levels = c("R2", "RMSE", "MAE")))

# Normalize values for comparison (R² higher is better, RMSE and MAE lower is better)
summary_long_norm <- summary_long %>%
  group_by(Metric) %>%
  mutate(
    ValueNorm = ifelse(Metric == "R2", 
                       Value,  # R² use directly
                       max(Value) - Value)  # RMSE and MAE invert
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

# 10.5 Scatter plots: Predicted vs True values (for each model)
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

# Save all charts
cat("Saving plots...\n")

# Save R² comparison chart
ggsave("baseline_models_r2_comparison.png", p1, 
       width = 10, height = 6, dpi = 300)
cat("  Saved: baseline_models_r2_comparison.png\n")

# Save RMSE comparison chart
ggsave("baseline_models_rmse_comparison.png", p2, 
       width = 10, height = 6, dpi = 300)
cat("  Saved: baseline_models_rmse_comparison.png\n")

# Save MAE comparison chart
ggsave("baseline_models_mae_comparison.png", p3, 
       width = 10, height = 6, dpi = 300)
cat("  Saved: baseline_models_mae_comparison.png\n")

# Save comprehensive comparison chart
ggsave("baseline_models_all_metrics.png", p4, 
       width = 14, height = 5, dpi = 300)
cat("  Saved: baseline_models_all_metrics.png\n")

# Save scatter plot grid
if (length(scatter_plots) > 0) {
  png("baseline_models_scatter_plots.png", width = 15, height = 10, 
      units = "in", res = 300)
  do.call(gridExtra::grid.arrange, c(scatter_plots, ncol = 3))
  dev.off()
  cat("  Saved: baseline_models_scatter_plots.png\n")
}

# 10.6 Comprehensive large plot (all metrics + scatter plots)
cat("\nGenerating comprehensive comparison plot...\n")

# Adjust font size for large plot
p1_large <- p1 + theme(
  plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
  axis.text = element_text(size = 11),
  axis.title = element_text(size = 12)
)

p2_large <- p2 + theme(
  plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
  axis.text = element_text(size = 11),
  axis.title = element_text(size = 12)
)

p3_large <- p3 + theme(
  plot.title = element_text(hjust = 0.5, size = 16, face = "bold"),
  axis.text = element_text(size = 11),
  axis.title = element_text(size = 12)
)

# Adjust scatter plot fonts
if (length(scatter_plots) > 0) {
  scatter_plots_large <- lapply(scatter_plots, function(p) {
    p + theme(
      plot.title = element_text(hjust = 0.5, size = 12, face = "bold"),
      axis.text = element_text(size = 10),
      axis.title = element_text(size = 11)
    )
  })
  
  # Calculate layout: top row 3 metric charts, bottom rows scatter plots
  n_cols <- 3
  n_scatter <- length(scatter_plots_large)
  n_scatter_rows <- ceiling(n_scatter / n_cols)
  
  # Calculate height
  plot_height <- 6 + 5 * n_scatter_rows
  
  # Create comprehensive large plot
  png("baseline_models_comprehensive.png", width = 20, height = plot_height, 
      units = "in", res = 300)
  
  # Method: arrange top and bottom rows separately, then combine
  # Top row: 3 metric charts
  top_plots <- list(p1_large, p2_large, p3_large)
  
  # Bottom rows: scatter plots, if not multiple of 3, add blank plots
  n_scatter_needed <- n_scatter_rows * n_cols
  n_empty <- n_scatter_needed - n_scatter
  
  if (n_empty > 0) {
    # Create blank plot
    empty_plot <- ggplot() + 
      theme_void() + 
      theme(plot.margin = margin(0, 0, 0, 0))
    scatter_plots_filled <- c(scatter_plots_large, rep(list(empty_plot), n_empty))
  } else {
    scatter_plots_filled <- scatter_plots_large
  }
  
  # Combine all plots: first top row 3, then all scatter plots
  all_plots <- c(top_plots, scatter_plots_filled)
  total_rows <- 1 + n_scatter_rows
  
  # Use grid.arrange
  gridExtra::grid.arrange(
    grobs = all_plots,
    ncol = n_cols,
    nrow = total_rows,
    heights = c(1, rep(1, n_scatter_rows))
  )
  
  dev.off()
  cat(sprintf("  Saved: baseline_models_comprehensive.png (20x%.1f inches, 300 DPI)\n", 
              plot_height))
} else {
  # If no scatter plots, only save three metric charts
  png("baseline_models_comprehensive.png", width = 20, height = 8, 
      units = "in", res = 300)
  gridExtra::grid.arrange(p1_large, p2_large, p3_large, ncol = 3)
  dev.off()
  cat("  Saved: baseline_models_comprehensive.png (20x8 inches)\n")
}

# Save results summary table
write.csv(summary_df, "baseline_models_summary.csv", row.names = FALSE)
cat("  Saved: baseline_models_summary.csv\n")

cat("\n========================================\n")
cat("All plots saved successfully!\n")
cat("========================================\n")

# Print final summary
cat("\nFinal Summary:\n")
cat("==============\n")
print(summary_df)

