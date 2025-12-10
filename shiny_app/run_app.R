# =============================================
# Launch Shiny App - Airbnb Baseline Price Predictor
# =============================================
# 
# Usage:
#   Run from project root: source("shiny_app/run_app.R")
#   App will automatically open in browser: http://localhost:3838
#
# =============================================

# Check and install required R packages
required_packages <- c(
  "shiny", "shinydashboard", "DT", "leaflet", "plotly",
  "dplyr", "geosphere", "xgboost", "reticulate", "glmnet",
  "httr", "jsonlite", "caret", "zoo", "data.table", "sf",
  "lubridate", "scales", "ggplot2"
)

missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if (length(missing_packages) > 0) {
  cat("Installing missing R packages...\n")
  install.packages(missing_packages)
  cat("Installation complete!\n\n")
}

# Check Python and PyTorch (optional, for Neural Network model)
# Note: App uses XGBoost-only mode, Python is not required
if (requireNamespace("reticulate", quietly = TRUE)) {
  library(reticulate)
  
  if (!py_available()) {
    cat("Info: Python not configured. App will use XGBoost-only mode (this is normal).\n")
    cat("To enable Neural Network model, run: source('baseline_price_predict/sensitivity_analysis/configure_python.R')\n\n")
  } else {
    if (!py_module_available("torch")) {
      cat("PyTorch not installed. Installing...\n")
      py_install("torch", pip = TRUE)
      cat("PyTorch installation complete!\n\n")
    }
  }
}

# Launch app
cat("========================================\n")
cat("Launching Airbnb Baseline Price Predictor\n")
cat("========================================\n\n")

# Ensure we're in the correct directory
# If not in shiny_app directory and shiny_app exists, switch to it
if (basename(getwd()) != "shiny_app") {
  if (dir.exists("shiny_app")) {
    setwd("shiny_app")
    cat("Switched to shiny_app directory\n\n")
  } else {
    cat("Warning: shiny_app directory not found. Please run this script from project root.\n")
    cat("Current directory:", getwd(), "\n\n")
  }
}

# Launch Shiny app
# host = "0.0.0.0" allows access from any network interface
# port = 3838 is Shiny's default port
cat("App is starting...\n")
cat("Browser will open automatically, or visit: http://localhost:3838\n")
cat("Press Ctrl+C or Esc to stop the app\n\n")

shiny::runApp("app.R", host = "0.0.0.0", port = 3838)

