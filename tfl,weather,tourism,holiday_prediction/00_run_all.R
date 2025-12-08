# ==================================================================================
# MASTER SCRIPT - Complete Foot Traffic Prediction Pipeline
# ==================================================================================
# 
# This script runs the entire pipeline from data cleaning to model training
# Run this file to execute all steps in correct order
#
# ==================================================================================

rm(list = ls())

# Auto-detect script directory (works when sourced or run directly)
if (interactive()) {
  # If running interactively, use current working directory
  script_dir <- getwd()
} else {
  # If running via source() or Rscript
  script_dir <- dirname(sys.frame(1)$ofile)
}

# Ensure we're in the correct directory
if (!file.exists(file.path(script_dir, "00_run_all.R"))) {
  # Try to find the script directory
  if (file.exists("00_run_all.R")) {
    script_dir <- getwd()
  } else if (file.exists("tfl,weather,tourism,holiday_prediction/00_run_all.R")) {
    script_dir <- file.path(getwd(), "tfl,weather,tourism,holiday_prediction")
  }
}

setwd(script_dir)

message("==========================================================")
message("COMPONENT PREDICTION PIPELINE")
message("==========================================================")
message("Predicts REAL components: TfL, Tourism, Weather")
message("(No synthetic foot_traffic_score)\n")

# ==================================================================================
# STEP 1: Data Cleaning & Merging
# ==================================================================================

message("\n========== STEP 1: Data Cleaning & Merging ==========\n")
source("01a_clean_tfl.R")
source("01b_clean_tourism.R")
source("01c_clean_weather.R")
source("01d_clean_holidays.R")
source("02_merge_to_daily.R")

# ==================================================================================
# STEP 3: Predict & Validate Data Components
# ==================================================================================

message("\n========== STEP 3: Predict & Validate Data Components ==========\n")
message("Each component prediction includes train/validation split for accuracy assessment\n")
source("03a_predict_tfl.R")        # TfL transport (ARIMA with validation)
source("03b_predict_tourism.R")   # International tourism (with validation)
source("03c_predict_weather.R")   # Weather forecast (Open-Meteo API + validation)

# ==================================================================================
# STEP 4: Fill Missing Historical Data with Predictions
# ==================================================================================

message("\n========== STEP 4: Fill Missing Historical Data ==========\n")
source("04_fill_missing_data.R")

# ==================================================================================
# STEP 5: Descriptive Analysis
# ==================================================================================

message("\n========== STEP 5: Descriptive Analysis ==========\n")
source("05_descriptive_analysis.R")

# ==================================================================================
# STEP 6: Component Validation Report
# ==================================================================================

message("\n========== STEP 6: Component Validation Report ==========\n")
source("06b_component_validation.R")

# ==================================================================================
# STEP 9: Export for Shiny App
# ==================================================================================

message("\n========== STEP 9: Export for Shiny App ==========\n")
source("09_export_for_shiny.R")

# ==================================================================================
# Pipeline Complete
# ==================================================================================

message("\n==========================================================")
message("✅ COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
message("==========================================================\n")
message("Results available in:")
message("  📊 Plots: outputs/plots/")
message("  🤖 Component Models: outputs/models/ (TfL, Tourism)")
message("  📈 Component Predictions: outputs/*_daily_forecast.csv")
message("  📦 Shiny exports: shiny_export/")
message("\n⚠️  Note: No foot_traffic_score - only REAL component predictions")
message("   Use components directly in your pricing logic\n")

