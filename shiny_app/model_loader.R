# =============================================
# Model Loader - XGBoost Only (no Python/stacking)
# =============================================

library(xgboost)
library(caret)  # For preProcess scaler predict

MODEL_FILES <- list(
  training = "nn_price_training_v4.csv",
  xgb = "best_xgb_log_model.xgb",
  scaler = "scaler_xgb.rds"
)

models <- list(
  xgb_model = NULL,
  scaler_xgb = NULL,
  feature_cols = NULL,
  loaded = FALSE
)

find_model_dir <- function() {
  candidates <- c(
    file.path(getwd(), "baseprice_model"),
    file.path(getwd(), "shiny_app", "baseprice_model"),
    file.path(dirname(getwd()), "shiny_app", "baseprice_model")
  )
  
  for (dir in candidates) {
    if (dir.exists(dir)) {
      return(dir)
    }
  }
  
  stop("Cannot find baseprice_model directory. Tried: ", paste(candidates, collapse = ", "))
}

load_models <- function() {
  if (models$loaded) {
    message("Models already loaded")
    return(TRUE)
  }
  
  model_dir <- find_model_dir()
  training_file <- file.path(model_dir, MODEL_FILES$training)
  xgb_file <- file.path(model_dir, MODEL_FILES$xgb)
  scaler_file <- file.path(model_dir, MODEL_FILES$scaler)
  
  if (!file.exists(training_file)) stop("Missing training data file: ", training_file)
  if (!file.exists(xgb_file)) stop("Missing XGBoost model file: ", xgb_file)
  if (!file.exists(scaler_file)) stop("Missing scaler file: ", scaler_file)
  
  df_sample <- read.csv(training_file, nrows = 1)
  models$feature_cols <<- setdiff(colnames(df_sample), "price_num")
  
  message("Loading XGBoost model...")
  models$xgb_model <<- xgb.load(xgb_file)
  models$scaler_xgb <<- readRDS(scaler_file)
  
  models$loaded <<- TRUE
  message("✓ XGBoost model ready (XGBoost-only mode)")
  TRUE
}

predict_baseline_price <- function(feature_vector) {
  if (!models$loaded) stop("Models not loaded. Call load_models() first.")
  
  if (length(feature_vector) != length(models$feature_cols)) {
    stop(
      sprintf(
        "Feature vector length (%d) does not match expected (%d)",
        length(feature_vector), length(models$feature_cols)
      )
    )
  }
  
  X_df <- data.frame(matrix(feature_vector, nrow = 1))
  colnames(X_df) <- models$feature_cols
  
  # Ensure numeric for scaler
  for (col in models$feature_cols) {
    X_df[[col]] <- as.numeric(X_df[[col]])
  }
  
  X_xgb_scaled <- predict(models$scaler_xgb, X_df)
  dtest_xgb <- xgb.DMatrix(data = as.matrix(X_xgb_scaled))
  xgb_pred_log <- predict(models$xgb_model, dtest_xgb)
  
  unname(expm1(xgb_pred_log[1]))
}

load_competitor_data <- function() {
  model_dir <- tryCatch(find_model_dir(), error = function(e) NULL)
  if (is.null(model_dir)) {
    warning("Cannot find baseprice_model directory for competitor data.")
    return(NULL)
  }
  
  training_file <- file.path(model_dir, MODEL_FILES$training)
  if (!file.exists(training_file)) {
    warning("Cannot find training data file: ", training_file)
    return(NULL)
  }
  
  tryCatch({
    data.table::fread(
      training_file,
      select = c("latitude", "longitude", "price_num", "bedrooms", "accommodates", "room_type_id", "neighbourhood_id")
    )
  }, error = function(e) {
    df <- read.csv(training_file)
    df[, c("latitude", "longitude", "price_num", "bedrooms", "accommodates", "room_type_id", "neighbourhood_id")]
  })
}