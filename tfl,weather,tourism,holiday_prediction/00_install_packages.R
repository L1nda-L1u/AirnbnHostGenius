# ==================================================================================
# Install Required Packages for Foot Traffic Prediction
# ==================================================================================
# Run this script once before running the modeling pipeline

cat("==========================================================\n")
cat("INSTALLING REQUIRED PACKAGES\n")
cat("==========================================================\n\n")

# List of required packages
packages <- c(
  # Core data manipulation
  "tidyverse",
  "data.table",
  "lubridate",
  
  # Visualization
  "corrplot",
  "gridExtra",
  "scales",
  
  # Time series & forecasting
  "forecast",
  "zoo",
  "prophet",
  
  # Machine learning
  "xgboost",
  "caret",
  "SHAPforxgboost",
  
  # API & data handling
  "httr",
  "jsonlite"
)

cat("Packages to check/install:\n")
cat(paste(" -", packages, collapse = "\n"), "\n\n")

# Check which packages are already installed
installed <- rownames(installed.packages())
to_install <- packages[!packages %in% installed]

if (length(to_install) > 0) {
  cat("Installing", length(to_install), "missing packages...\n\n")
  
  for (pkg in to_install) {
    cat("Installing:", pkg, "...")
    tryCatch({
      install.packages(pkg, repos = "https://cloud.r-project.org/", quiet = TRUE)
      cat(" ✓\n")
    }, error = function(e) {
      cat(" ✗ FAILED\n")
      cat("  Error:", e$message, "\n")
    })
  }
} else {
  cat("All required packages are already installed!\n")
}

# Verify installation
cat("\n==========================================================\n")
cat("VERIFICATION\n")
cat("==========================================================\n\n")

installed_now <- rownames(installed.packages())
success <- 0
failed <- 0

for (pkg in packages) {
  if (pkg %in% installed_now) {
    cat("✓", pkg, "\n")
    success <- success + 1
  } else {
    cat("✗", pkg, "MISSING\n")
    failed <- failed + 1
  }
}

cat("\n==========================================================\n")
cat("Summary:", success, "installed,", failed, "failed\n")
cat("==========================================================\n\n")

if (failed > 0) {
  cat("Some packages failed to install. Try installing them manually:\n")
  cat("install.packages(c(", paste0("'", packages[!packages %in% installed_now], "'", collapse = ", "), "))\n\n")
} else {
  cat("All packages installed successfully!\n")
  cat("You can now run the modeling pipeline.\n\n")
}

