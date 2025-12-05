# =============================================
# Model Loader - Load Price Prediction Models
# =============================================

library(xgboost)
library(reticulate)
library(glmnet)
library(caret)  # Required for preProcess scaler

# Global variable to store models
models <- list(
  xgb_model = NULL,
  nn_model = NULL,
  meta_model = NULL,
  scaler_xgb = NULL,
  scaler_nn = NULL,
  feature_cols = NULL,
  loaded = FALSE,
  use_nn = FALSE,
  meta_cv = NULL
)

# Load models
load_models <- function() {
  if (models$loaded) {
    cat("Models already loaded\n")
    return(TRUE)
  }
  
  cat("Loading models...\n")
  
  # Set working directory to baseprice_model
  original_wd <- getwd()
  
  # Try multiple possible paths
  possible_dirs <- c(
    file.path(getwd(), "..", "baseprice_model"),
    file.path(getwd(), "baseprice_model"),
    file.path(dirname(dirname(getwd())), "baseline_price_predict", "baseprice_model"),
    file.path(dirname(getwd()), "baseprice_model")
  )
  
  model_dir <- NULL
  for (dir in possible_dirs) {
    if (dir.exists(dir)) {
      model_dir <- dir
      break
    }
  }
  
  if (is.null(model_dir)) {
    stop("Cannot find baseprice_model directory. Tried:", paste(possible_dirs, collapse = ", "))
  }
  
  setwd(model_dir)
  on.exit(setwd(original_wd))
  
  # 1. Load feature columns
  training_file <- "nn_price_training_v4.csv"
  if (!file.exists(training_file)) {
    stop("Cannot find training data file: ", training_file)
  }
  
  df_sample <- read.csv(training_file, nrows = 1)
  models$feature_cols <<- setdiff(colnames(df_sample), "price_num")
  
  cat("Feature columns loaded:", length(models$feature_cols), "\n")
  
  # 2. Load XGBoost model
  cat("Loading XGBoost model...\n")
  if (!file.exists("best_xgb_log_model.xgb")) {
    stop("Cannot find XGBoost model file")
  }
  models$xgb_model <<- xgb.load("best_xgb_log_model.xgb")
  models$scaler_xgb <<- readRDS("scaler_xgb.rds")
  cat("XGBoost model loaded\n")
  
  # 3. Try to load Neural Network model (optional, requires Python)
  models$use_nn <<- FALSE
  
  # Try to discover and configure Python first
  if (!py_available()) {
    tryCatch({
      python_paths <- c(
        "C:/Users/linda/AppData/Local/Programs/Python/Python310/python.exe",
        Sys.which("python"),
        Sys.which("python3")
      )
      
      for (py_path in python_paths) {
        if (py_path != "" && file.exists(py_path)) {
          tryCatch({
            use_python(py_path, required = FALSE)
            py_discover_config()
            if (py_available()) break
          }, error = function(e) {})
        }
      }
    }, error = function(e) {})
  }
  
  if (py_available()) {
    cat("Python available, attempting to load Neural Network model...\n")
    
    if (py_module_available("torch")) {
      tryCatch({
        # Load PyTorch model
        py$feature_cols <- models$feature_cols
        
        py_run_string("
import torch
import torch.nn as nn
import numpy as np
import os

# Define model structure (must match training structure)
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

# Load model metadata to get input dimension
meta = torch.load('best_price_A2_log_pytorch_meta.pt', map_location='cpu', weights_only=False)
input_dim = meta['input_dim']

# Create model and load weights
model = PricePredictor(input_dim).to('cpu')
model.load_state_dict(torch.load('best_price_A2_log_pytorch.pt', map_location='cpu', weights_only=False))
model.eval()
print('Neural Network model loaded successfully')
")
        
        # Load scaler (R format)
        if (file.exists("scaler_price_pytorch.rds")) {
          models$scaler_nn <<- readRDS("scaler_price_pytorch.rds")
          models$use_nn <<- TRUE
          cat("Neural Network model loaded successfully\n")
        } else {
          cat("Warning: scaler_price_pytorch.rds not found. Using XGBoost only.\n")
        }
      }, error = function(e) {
        cat("Warning: Failed to load Neural Network model:", e$message, "\n")
        cat("Falling back to XGBoost-only mode.\n")
        models$use_nn <<- FALSE
      })
    } else {
      cat("Warning: PyTorch not installed. Using XGBoost only.\n")
      cat("To install PyTorch, run: py_install('torch', pip = TRUE)\n")
    }
  } else {
    cat("Python not available. Using XGBoost-only mode.\n")
    cat("To enable Neural Network model, configure Python first:\n")
    cat("  source('../sensitivity_analysis/configure_python.R')\n")
  }
  
  # 4. Load Meta model (Stacking, only if NN is available)
  if (models$use_nn) {
    cat("Loading Meta model (Stacking)...\n")
    if (file.exists("meta_ridge_model.rds")) {
      models$meta_model <<- readRDS("meta_ridge_model.rds")
      models$meta_cv <<- readRDS("meta_ridge_cv.rds")
      cat("Meta model loaded\n")
    } else {
      cat("Warning: meta_ridge_model.rds not found. Using XGBoost only.\n")
      models$use_nn <<- FALSE
    }
  } else {
    cat("Skipping Meta model (using XGBoost-only mode)\n")
  }
  
  models$loaded <<- TRUE
  
  if (models$use_nn) {
    cat("All models loaded successfully! (XGBoost + Neural Network + Stacking)\n")
  } else {
    cat("XGBoost model loaded successfully! (XGBoost-only mode)\n")
  }
  
  # Verify models are actually loaded
  if (!models$loaded) {
    stop("Model loading verification failed")
  }
  
  return(TRUE)
}

# Predict price
predict_baseline_price <- function(feature_vector) {
  cat("predict_baseline_price called\n")
  cat("models$loaded:", models$loaded, "\n")
  
  if (!models$loaded) {
    stop("Models not loaded. Call load_models() first.")
  }
  
  # Ensure feature vector is correct length
  cat("Feature vector length:", length(feature_vector), "\n")
  cat("Expected length:", length(models$feature_cols), "\n")
  
  if (length(feature_vector) != length(models$feature_cols)) {
    stop(sprintf("Feature vector length (%d) does not match expected (%d)", 
                 length(feature_vector), length(models$feature_cols)))
  }
  
  # Convert to data frame
  X_df <- data.frame(matrix(feature_vector, nrow = 1))
  colnames(X_df) <- models$feature_cols
  
  # Ensure all columns are numeric
  for (col in models$feature_cols) {
    X_df[[col]] <- as.numeric(X_df[[col]])
  }
  
  # XGBoost prediction
  cat("Running XGBoost prediction...\n")
  tryCatch({
    X_xgb_scaled <- predict(models$scaler_xgb, X_df)
    X_xgb_scaled <- as.matrix(X_xgb_scaled)
    dtest_xgb <- xgb.DMatrix(data = X_xgb_scaled)
    xgb_pred_log <- predict(models$xgb_model, dtest_xgb)
    xgb_pred <- expm1(xgb_pred_log[1])
    cat("XGBoost prediction:", xgb_pred, "\n")
  }, error = function(e) {
    cat("XGBoost prediction error:", e$message, "\n")
    stop("XGBoost prediction failed: ", e$message)
  })
  
  # If NN model is available, use Stacking; otherwise use XGBoost only
  if (models$use_nn && !is.null(models$scaler_nn) && !is.null(models$meta_model)) {
    # Neural Network prediction
    tryCatch({
      cat("Running Neural Network prediction...\n")
      X_nn_scaled <- predict(models$scaler_nn, X_df)
      X_nn_scaled <- as.matrix(X_nn_scaled)
      
      py$X_nn_scaled <- X_nn_scaled
      py_run_string("
X_test_tensor = torch.FloatTensor(X_nn_scaled)
with torch.no_grad():
    pred_log_tensor = model(X_test_tensor)
    pred_log = pred_log_tensor.numpy().flatten()
")
      
      nn_pred_log <- py$pred_log[1]
      nn_pred <- expm1(nn_pred_log)
      cat("NN prediction:", nn_pred, "\n")
      
      # Stacking prediction
      cat("Running Stacking prediction...\n")
      stack_input <- matrix(c(xgb_pred, nn_pred), nrow = 1)
      stack_pred <- predict(models$meta_model, newx = stack_input, 
                           s = models$meta_cv$lambda.min)[1, 1]
      cat("Stacking prediction:", stack_pred, "\n")
      
      return(stack_pred)
    }, error = function(e) {
      # If NN prediction fails, fall back to XGBoost
      cat("Warning: NN prediction failed, using XGBoost only:", e$message, "\n")
      return(xgb_pred)
    })
  } else {
    # Use XGBoost only
    cat("Using XGBoost-only mode (NN not available)\n")
    return(xgb_pred)
  }
}
