# ==================================================================================
# 04 - Fill Missing Historical Data
# Fill missing TfL and Tourism data with predictions
# ==================================================================================

library(tidyverse)
library(data.table)
library(lubridate)

# Note: Working directory should be set by the master script (00_run_all.R)
# or manually before running this script

# Load the daily data with missing values
foot_traffic_daily <- fread("foot_traffic_data/cleaned/foot_traffic_daily.csv") %>%
  mutate(date = as.Date(date))

# Check for missing data before filling
missing_before <- foot_traffic_daily %>%
  summarise(
    missing_tfl = sum(is.na(tfl_daily_avg_m)),
    missing_tourism = sum(is.na(tourism_quarterly_visits_k))
  )

message("Missing data before filling:")
message("  TfL: ", missing_before$missing_tfl, " rows")
message("  Tourism: ", missing_before$missing_tourism, " rows")

# Load TfL predictions (DAILY forecast from 03a)
tfl_predictions <- tryCatch({
  df <- fread("outputs/tfl_daily_forecast.csv")
  df <- df %>%
    mutate(date = as.Date(date))
  
  # Check if column exists
  if (!"tfl_daily_avg_m" %in% names(df)) {
    stop("Column 'tfl_daily_avg_m' not found. Available: ", paste(names(df), collapse = ", "))
  }
  
  df %>% select(date, tfl_daily_avg_m)
}, error = function(e) {
  message("⚠️  Error loading TfL predictions: ", e$message)
  return(NULL)
})

# Load Tourism predictions (DAILY forecast from 03b)
tourism_predictions <- tryCatch({
  df <- fread("outputs/tourism_daily_forecast.csv")
  df <- df %>%
    mutate(date = as.Date(date))
  
  # Check column names and rename from prediction file format
  col_mapping <- c(
    "tourism_quarterly_visits_k" = "tourism_quarterly_visits_k",
    "tourism_quarterly_spend_m" = "tourism_quarterly_spend_m",
    "tourism_avg_spend" = "tourism_avg_spend",
    "tourism_avg_nights" = "tourism_avg_nights"
  )
  
  available_cols <- names(df)
  message("📊 Tourism file columns: ", paste(available_cols, collapse = ", "))
  
  df %>% select(date, 
                tourism_quarterly_visits_k,
                tourism_quarterly_spend_m, 
                tourism_avg_spend,
                tourism_avg_nights)
}, error = function(e) {
  message("⚠️  Error loading Tourism predictions: ", e$message)
  return(NULL)
})

# Merge predictions - fill missing values ONLY
if (!is.null(tfl_predictions)) {
  foot_traffic_daily <- foot_traffic_daily %>%
    left_join(
      tfl_predictions %>% rename(tfl_pred = tfl_daily_avg_m),
      by = "date"
    ) %>%
    mutate(
      tfl_daily_avg_m = coalesce(tfl_daily_avg_m, tfl_pred)
    ) %>%
    select(-tfl_pred)
  
  message("✅ Filled missing TfL data with predictions")
}

if (!is.null(tourism_predictions)) {
  foot_traffic_daily <- foot_traffic_daily %>%
    left_join(
      tourism_predictions %>% 
        rename(tourism_visits_pred = tourism_quarterly_visits_k,
               tourism_spend_pred = tourism_quarterly_spend_m,
               tourism_avg_spend_pred = tourism_avg_spend,
               tourism_avg_nights_pred = tourism_avg_nights),
      by = "date"
    ) %>%
    mutate(
      tourism_quarterly_visits_k = coalesce(tourism_quarterly_visits_k, tourism_visits_pred),
      tourism_quarterly_spend_m = coalesce(tourism_quarterly_spend_m, tourism_spend_pred),
      tourism_avg_spend = coalesce(tourism_avg_spend, tourism_avg_spend_pred),
      tourism_avg_nights = coalesce(tourism_avg_nights, tourism_avg_nights_pred)
    ) %>%
    select(-ends_with("_pred"))
  
  message("✅ Filled missing Tourism data with predictions")
}

# Recalculate normalized indices with complete data (for consistency)
foot_traffic_daily <- foot_traffic_daily %>%
  mutate(
    # Recalculate TfL index (normalized 0-1)
    tfl_index = scales::rescale(tfl_daily_avg_m, to = c(0, 1), na.rm = TRUE),
    
    # Recalculate Tourism index (normalized 0-1)
    tourism_index = scales::rescale(tourism_quarterly_visits_k, to = c(0, 1), na.rm = TRUE),
    
    # Ensure weather and holiday indices are set
    weather_index = replace_na(weather_quality, 0),
    holiday_index = case_when(
      is_major_holiday ~ 1.0,
      is_holiday ~ 0.7,
      is_holiday_period ~ 0.5,
      is_weekend ~ 0.3,
      TRUE ~ 0.0
    )
  )

# Check missing data after filling
missing_after <- foot_traffic_daily %>%
  summarise(
    missing_tfl = sum(is.na(tfl_daily_avg_m)),
    missing_tourism = sum(is.na(tourism_quarterly_visits_k))
  )

message("\nMissing data after filling:")
message("  TfL: ", missing_after$missing_tfl, " rows")
message("  Tourism: ", missing_after$missing_tourism, " rows")

# Save updated data
fwrite(foot_traffic_daily, "foot_traffic_data/cleaned/foot_traffic_daily.csv")

message("\n✅ Successfully filled missing data")
message("   Total rows: ", nrow(foot_traffic_daily))
message("   Date range: ", min(foot_traffic_daily$date), " to ", max(foot_traffic_daily$date))
message("   Components: TfL, Tourism, Weather, Holidays (indices recalculated)")

