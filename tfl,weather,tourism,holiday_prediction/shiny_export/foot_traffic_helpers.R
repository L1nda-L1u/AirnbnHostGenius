
# ==================================================================================
# Component Prediction Helper Functions for Shiny
# ==================================================================================
# Source this file in your Shiny app
# Provides functions to access TfL, Tourism, and Weather predictions

library(data.table)
library(lubridate)

# Load component predictions lookup table (call once in server)
load_component_predictions <- function() {
  readRDS("foot_traffic_prediction/shiny_export/component_predictions_lookup.rds")
}

# Get component predictions for a specific date
get_components <- function(date, lookup_table) {
  date <- as.Date(date)
  
  result <- lookup_table %>%
    filter(date == date)
  
  if (nrow(result) == 0) {
    # Date not in forecast - return averages
    return(list(
      tfl = mean(lookup_table$tfl_daily_avg_m, na.rm = TRUE),
      tourism = mean(lookup_table$tourism_quarterly_visits_k, na.rm = TRUE),
      temp = mean(lookup_table$temp_c, na.rm = TRUE),
      weather_quality = mean(lookup_table$weather_quality, na.rm = TRUE),
      available = FALSE
    ))
  }
  
  return(list(
    tfl = result$tfl_daily_avg_m,
    tourism = result$tourism_quarterly_visits_k,
    temp = result$temp_c,
    weather_quality = result$weather_quality,
    is_good_weather = result$is_good_weather,
    available = TRUE
  ))
}

# Get components for date range (e.g., booking period)
get_components_range <- function(start_date, end_date, lookup_table) {
  start_date <- as.Date(start_date)
  end_date <- as.Date(end_date)
  
  result <- lookup_table %>%
    filter(date >= start_date & date <= end_date)
  
  if (nrow(result) == 0) {
    return(list(
      mean_tfl = mean(lookup_table$tfl_daily_avg_m, na.rm = TRUE),
      mean_tourism = mean(lookup_table$tourism_quarterly_visits_k, na.rm = TRUE),
      mean_temp = mean(lookup_table$temp_c, na.rm = TRUE),
      dates_available = 0
    ))
  }
  
  return(list(
    mean_tfl = mean(result$tfl_daily_avg_m, na.rm = TRUE),
    mean_tourism = mean(result$tourism_quarterly_visits_k, na.rm = TRUE),
    mean_temp = mean(result$temp_c, na.rm = TRUE),
    mean_weather_quality = mean(result$weather_quality, na.rm = TRUE),
    good_weather_days = sum(result$is_good_weather, na.rm = TRUE),
    dates_available = nrow(result)
  ))
}

# Example usage in Shiny:
#
# server <- function(input, output, session) {
#   
#   # Load data once
#   component_lookup <- load_component_predictions()
#   
#   # Get predictions for checkin date
#   checkin_date <- input$checkin_date
#   components <- get_components(checkin_date, component_lookup)
#   
#   # Use components for pricing logic
#   # (You can create your own adjustment function based on these real values)
# }

