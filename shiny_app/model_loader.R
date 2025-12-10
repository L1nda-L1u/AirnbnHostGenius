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
  
  # Try multiple possible paths (from root directory)
  possible_dirs <- c(
    file.path(getwd(), "baseline_price_predict", "baseprice_model"),
    file.path(dirname(getwd()), "baseline_price_predict", "baseprice_model"),
    file.path(getwd(), "..", "baseline_price_predict", "baseprice_model"),
    file.path(getwd(), "baseprice_model")  # Fallback: if baseprice_model is directly in root
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
        Sys.which("python"),
        Sys.which("python3"),
        "/usr/bin/python3",
        "/usr/local/bin/python3"
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
    cat("  source('baseline_price_predict/sensitivity_analysis/configure_python.R')\n")
  }
  
  # 4. Skip Meta model (Stacking) - using XGBoost only
  cat("Skipping Meta model (using XGBoost-only mode)\n")
  models$use_nn <<- FALSE  # Disable NN to ensure XGBoost-only mode
  
  models$loaded <<- TRUE
  cat("XGBoost model loaded successfully! (XGBoost-only mode)\n")
  
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
  
  # Use XGBoost only (no Stacking)
  cat("Using XGBoost-only mode\n")
  return(xgb_pred)
}

# Load competitor data for map
load_competitor_data <- function() {
  # Set working directory to baseprice_model
  original_wd <- getwd()
  
  # Try multiple possible paths (from root directory)
  possible_dirs <- c(
    file.path(getwd(), "baseline_price_predict", "baseprice_model"),
    file.path(dirname(getwd()), "baseline_price_predict", "baseprice_model"),
    file.path(getwd(), "..", "baseline_price_predict", "baseprice_model"),
    file.path(getwd(), "baseprice_model")
  )
  
  model_dir <- NULL
  for (dir in possible_dirs) {
    if (dir.exists(dir)) {
      model_dir <- dir
      break
    }
  }
  
  if (is.null(model_dir)) {
    warning("Cannot find baseprice_model directory for competitor data.")
    return(NULL)
  }
  
  setwd(model_dir)
  on.exit(setwd(original_wd))
  
  training_file <- "nn_price_training_v4.csv"
  if (!file.exists(training_file)) {
    warning("Cannot find training data file: ", training_file)
    return(NULL)
  }
  
  # Load only necessary columns to save memory
  # latitude, longitude, price_num, bedrooms, bathrooms, accommodates
  tryCatch({
    df <- fread(training_file, select = c("latitude", "longitude", "price_num", "bedrooms", "accommodates", "room_type_id"))
    return(df)
  }, error = function(e) {
    # Fallback if fread fails or is not available (though data.table is loaded in app.R)
    df <- read.csv(training_file)
    return(df[, c("latitude", "longitude", "price_num", "bedrooms", "accommodates", "room_type_id")])
  })
}