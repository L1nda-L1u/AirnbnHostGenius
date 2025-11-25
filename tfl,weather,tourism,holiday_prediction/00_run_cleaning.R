# ==================================================================================
# MASTER SCRIPT - Run All Data Cleaning (with Optional API Integration)
# ==================================================================================

rm(list = ls())
setwd("/Users/xiongyi/Desktop/Airbnb/AirbnbHostGenius/foot_traffic_prediction")

# ==================================================================================
# Configuration
# ==================================================================================

# Set to TRUE to fetch fresh data from APIs (weather forecast, future holidays)
# Set to FALSE to use historical data only
USE_APIS <- TRUE

message("==========================================================")
message("Starting Data Cleaning Pipeline")
message("API Integration: ", if(USE_APIS) "ENABLED" else "DISABLED")
message("==========================================================\n")

# ==================================================================================
# Phase 1: API Data Fetching (if enabled)
# ==================================================================================

if (USE_APIS) {
  message("\n[1/2] Fetching data from external APIs...")
  
  # Create cache directory
  dir.create("foot_traffic_data/api_cache", recursive = TRUE, showWarnings = FALSE)
  
  # Note: API scripts include testing code, skip it during sourcing
  skip_test <- TRUE
  
  # Source and call API integration functions
  source("apis/weather_api.R")
  save_weather_forecast_cache()  # Fetch and save weather forecast
  
  source("apis/holidays_api.R")
  # holidays API will be called within 01d_clean_holidays.R
  
  message("\nAPI data fetched and cached successfully.\n")
}

# ==================================================================================
# Phase 2: Clean Individual Data Sources
# ==================================================================================

message("\n[2/2] Cleaning data sources...")

source("01a_clean_tfl.R")
source("01b_clean_tourism.R")
source("01c_clean_weather.R")
source("01d_clean_holidays.R")

# ==================================================================================
# Phase 3: Merge to Daily Framework
# ==================================================================================

message("\nMerging all data sources to daily framework...")
source("02_merge_to_daily.R")

# ==================================================================================
# Summary
# ==================================================================================

message("\n==========================================================")
message("Data Cleaning Pipeline Complete!")
message("==========================================================")
message("\nCleaned files available in: foot_traffic_data/cleaned/")
message("  - tfl_monthly.csv")
message("  - tourism_quarterly.csv")
message("  - weather_daily.csv")
message("  - holidays_daily.csv")
message("  - foot_traffic_daily.csv (MERGED)\n")
