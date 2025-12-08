# =============================================
# Launch Shiny App - Airbnb Baseline Price Predictor
# =============================================

# Check and install required packages
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
if (requireNamespace("reticulate", quietly = TRUE)) {
  library(reticulate)
  
  if (!py_available()) {
    cat("Warning: Python not configured. Neural Network model will be skipped.\n")
    cat("To enable Neural Network model, run: source('../sensitivity_analysis/configure_python.R')\n\n")
  } else {
    if (!py_module_available("torch")) {
      cat("PyTorch not installed. Installing...\n")
      py_install("torch", pip = TRUE)
      cat("PyTorch installation complete!\n\n")
    }
  }
}

# Run the app
cat("========================================\n")
cat("Launching Airbnb Baseline Price Predictor\n")
cat("========================================\n\n")

# Ensure we're in the correct directory
if (basename(getwd()) != "shiny_app") {
  if (dir.exists("shiny_app")) {
    setwd("shiny_app")
  }
}

# Launch Shiny app
shiny::runApp("app.R", host = "0.0.0.0", port = 3838)

