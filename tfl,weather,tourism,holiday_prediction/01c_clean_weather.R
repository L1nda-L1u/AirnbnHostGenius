# ==================================================================================
# 01c - Clean Weather Data
# ==================================================================================

rm(list = ls())
library(tidyverse)
library(data.table)
library(lubridate)

# Note: Working directory should be set by the master script (00_run_all.R)
# or manually before running this script

# Load historical data (skip metadata rows)
weather_raw <- fread("foot_traffic_data/raw/weather/london_weather.csv", skip = 3)

# Clean historical weather data
weather_daily <- weather_raw %>%
  rename_with(tolower) %>%
  rename(
    date = time,
    temp_c = `temperature_2m_mean (°c)`,
    wind_kmh = `wind_speed_10m_max (km/h)`,
    precip_mm = `precipitation_sum (mm)`
  ) %>%
  mutate(
    date = as.Date(date),
    year = year(date),
    month = month(date),
    day_of_week = lubridate::wday(date, label = TRUE),
    is_weekend = day_of_week %in% c("Sat", "Sun")
  ) %>%
  filter(!is.na(date)) %>%
  mutate(
    temp_c = replace_na(temp_c, mean(temp_c, na.rm = TRUE)),
    wind_kmh = replace_na(wind_kmh, mean(wind_kmh, na.rm = TRUE)),
    precip_mm = replace_na(precip_mm, 0),
    temp_comfort = case_when(
      temp_c >= 15 & temp_c <= 25 ~ 1.0,
      temp_c >= 10 & temp_c < 15 ~ 0.8,
      temp_c > 25 & temp_c <= 30 ~ 0.7,
      TRUE ~ 0.5
    ),
    rain_score = case_when(
      precip_mm == 0 ~ 1.0,
      precip_mm < 2 ~ 0.8,
      precip_mm < 5 ~ 0.6,
      TRUE ~ 0.4
    ),
    weather_quality = (temp_comfort * 0.6 + rain_score * 0.4),
    is_good_weather = weather_quality >= 0.7
  ) %>%
  select(date, year, month, day_of_week, is_weekend,
         temp_c, wind_kmh, precip_mm, 
         weather_quality, is_good_weather) %>%
  arrange(date)

# Check for API forecast data (from Open-Meteo)
api_cache_file <- "foot_traffic_data/api_cache/weather_forecast.csv"

if (file.exists(api_cache_file)) {
  tryCatch({
    api_forecast <- fread(api_cache_file) %>%
      mutate(date = as.Date(date))
    
    # Only keep forecast dates beyond historical data
    max_historical_date <- max(weather_daily$date)
    api_forecast_new <- api_forecast %>%
      filter(date > max_historical_date) %>%
      mutate(
        year = year(date),
        month = month(date),
        day_of_week = lubridate::wday(date, label = TRUE),
        is_weekend = day_of_week %in% c("Sat", "Sun")
      )
    
    if (nrow(api_forecast_new) > 0) {
      weather_daily <- bind_rows(weather_daily, api_forecast_new) %>%
        arrange(date)
      message("✅ Added ", nrow(api_forecast_new), " days of weather forecast from Open-Meteo API")
    }
  }, error = function(e) {
    message("ℹ️  Could not load API forecast, using historical only: ", e$message)
  })
} else {
  message("ℹ️  No API forecast cache found, using historical data only")
}

# Save
fwrite(weather_daily, "foot_traffic_data/cleaned/weather_daily.csv")
