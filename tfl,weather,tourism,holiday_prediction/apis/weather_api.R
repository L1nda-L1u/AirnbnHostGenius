# ==================================================================================
# Weather API Integration - Open-Meteo
# ==================================================================================
# Uses Open-Meteo API for future weather forecasts
# Free tier: 10,000 calls/day, 16-day forecast available, NO API KEY REQUIRED
# Website: https://open-meteo.com
# License: CC BY 4.0 (attribution required)

library(httr)
library(jsonlite)
library(tidyverse)
library(data.table)
library(lubridate)

# ==================================================================================
# Configuration
# ==================================================================================

LONDON_LAT <- 51.5074
LONDON_LON <- -0.1278
CACHE_DIR <- "foot_traffic_data/api_cache"

# ==================================================================================
# Helper Functions
# ==================================================================================

create_cache_dir <- function() {
  if (!dir.exists(CACHE_DIR)) {
    dir.create(CACHE_DIR, recursive = TRUE)
  }
}

get_cache_file <- function(api_name) {
  file.path(CACHE_DIR, paste0(api_name, "_", Sys.Date(), ".rds"))
}

load_from_cache <- function(api_name, max_age_hours = 24) {
  cache_file <- get_cache_file(api_name)
  
  if (file.exists(cache_file)) {
    cache_time <- file.info(cache_file)$mtime
    age_hours <- as.numeric(difftime(Sys.time(), cache_time, units = "hours"))
    
    if (age_hours < max_age_hours) {
      message("Loading from cache (", round(age_hours, 1), " hours old)")
      return(readRDS(cache_file))
    }
  }
  return(NULL)
}

save_to_cache <- function(data, api_name) {
  create_cache_dir()
  cache_file <- get_cache_file(api_name)
  saveRDS(data, cache_file)
  message("Saved to cache: ", cache_file)
}

# ==================================================================================
# API Functions
# ==================================================================================

#' Fetch 16-day weather forecast from Open-Meteo
#' @param forecast_days Number of days to forecast (max 16)
#' @return tibble with date, temp_c, wind_kmh, precip_mm, weather_quality
fetch_weather_forecast <- function(forecast_days = 16) {
  
  # Check cache first
  cached <- load_from_cache("weather_forecast")
  if (!is.null(cached)) return(cached)
  
  # Open-Meteo API endpoint (NO API KEY NEEDED!)
  url <- paste0(
    "https://api.open-meteo.com/v1/forecast?",
    "latitude=", LONDON_LAT,
    "&longitude=", LONDON_LON,
    "&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,windspeed_10m_max",
    "&timezone=Europe/London",
    "&forecast_days=", forecast_days
  )
  
  message("Fetching ", forecast_days, "-day forecast from Open-Meteo API...")
  
  response <- GET(url)
  
  if (status_code(response) != 200) {
    stop("Open-Meteo API returned status ", status_code(response))
  }
  
  data <- content(response, "parsed")
  
  # Parse daily forecast data safely - handle variable length arrays
  dates <- as.Date(sapply(data$daily$time, function(x) if(is.null(x)) NA else x))
  n_days <- length(dates)
  
  # Extract each field with proper length handling
  extract_field <- function(field_data, n) {
    vals <- sapply(field_data, function(x) if(is.null(x)) NA else as.numeric(x))
    if (length(vals) < n) {
      vals <- c(vals, rep(NA, n - length(vals)))
    }
    vals[1:n]
  }
  
  forecast_df <- tibble(
    date = dates,
    temp_max_c = extract_field(data$daily$temperature_2m_max, n_days),
    temp_min_c = extract_field(data$daily$temperature_2m_min, n_days),
    precip_mm = extract_field(data$daily$precipitation_sum, n_days),
    wind_kmh = extract_field(data$daily$windspeed_10m_max, n_days)
  ) %>%
    filter(!is.na(date)) %>%
    mutate(
      # Calculate average temperature
      temp_c = (temp_max_c + temp_min_c) / 2,
      
      # Calculate weather quality index (0-1)
      weather_quality = case_when(
        temp_c >= 15 & temp_c <= 25 & precip_mm < 1 & wind_kmh < 20 ~ 1.0,
        temp_c >= 10 & temp_c < 15 & precip_mm < 2 ~ 0.8,
        temp_c >= 25 | precip_mm >= 2 ~ 0.6,
        temp_c < 10 | wind_kmh >= 20 ~ 0.4,
        TRUE ~ 0.5
      ),
      is_good_weather = weather_quality >= 0.8
    ) %>%
    select(date, temp_c, wind_kmh, precip_mm, weather_quality, is_good_weather)
  
  # Save to cache
  save_to_cache(forecast_df, "weather_forecast")
  
  message("✅ Fetched ", nrow(forecast_df), " days of weather forecast from Open-Meteo")
  message("   Date range: ", min(forecast_df$date), " to ", max(forecast_df$date))
  
  return(forecast_df)
}

#' Fallback: Use historical weather averages for forecast
#' @param days Number of days to forecast (default 365 for long-term predictions)
get_weather_fallback <- function(days = 365) {
  
  weather_file <- "foot_traffic_data/cleaned/weather_daily.csv"
  
  if (!file.exists(weather_file)) {
    stop("No historical weather data available for fallback")
  }
  
  historical <- fread(weather_file) %>%
    mutate(date = as.Date(date))
  
  # Calculate averages by day of year
  weather_avg <- historical %>%
    mutate(doy = yday(date)) %>%
    group_by(doy) %>%
    summarise(
      temp_c = mean(temp_c, na.rm = TRUE),
      wind_kmh = mean(wind_kmh, na.rm = TRUE),
      precip_mm = mean(precip_mm, na.rm = TRUE),
      weather_quality = mean(weather_quality, na.rm = TRUE),
      .groups = "drop"
    )
  
  # Generate forecast for next N days using historical averages
  future_dates <- seq(Sys.Date(), by = "day", length.out = days)
  
  forecast <- tibble(date = future_dates) %>%
    mutate(doy = yday(date)) %>%
    left_join(weather_avg, by = "doy") %>%
    mutate(
      is_good_weather = weather_quality >= 0.8
    ) %>%
    select(-doy)
  
  message("ℹ️  Using historical weather averages for ", nrow(forecast), " days (fallback mode)")
  return(forecast)
}

#' Get extended weather forecast (16 days API + historical averages for longer periods)
#' @param use_api Attempt to fetch from API (TRUE) or use fallback only (FALSE)
#' @param total_days Total days to forecast (API for first 16, historical for rest)
get_extended_weather_forecast <- function(use_api = TRUE, total_days = 730) {
  
  if (use_api) {
    # Get 16-day API forecast
    api_forecast <- fetch_weather_forecast(forecast_days = 16)
    
    if (nrow(api_forecast) > 0) {
      # If we need more than 16 days, supplement with historical averages
      if (total_days > 16) {
        start_date <- max(api_forecast$date) + 1
        remaining_days <- total_days - nrow(api_forecast)
        
        # Generate dates for remaining period
        future_dates <- seq(start_date, by = "day", length.out = remaining_days)
        
        # Load historical averages
        weather_file <- "foot_traffic_data/cleaned/weather_daily.csv"
        historical <- fread(weather_file) %>%
          mutate(date = as.Date(date))
        
        weather_avg <- historical %>%
          mutate(doy = yday(date)) %>%
          group_by(doy) %>%
          summarise(
            temp_c = mean(temp_c, na.rm = TRUE),
            wind_kmh = mean(wind_kmh, na.rm = TRUE),
            precip_mm = mean(precip_mm, na.rm = TRUE),
            weather_quality = mean(weather_quality, na.rm = TRUE),
            .groups = "drop"
          )
        
        # Create long-term forecast using historical averages
        longterm_forecast <- tibble(date = future_dates) %>%
          mutate(doy = yday(date)) %>%
          left_join(weather_avg, by = "doy") %>%
          mutate(
            is_good_weather = weather_quality >= 0.8
          ) %>%
          select(-doy)
        
        # Combine API forecast with historical averages
        combined <- bind_rows(api_forecast, longterm_forecast) %>%
          arrange(date)
        
        message("✅ Combined 16-day API forecast with ", remaining_days, " days of historical averages")
        message("   Total: ", nrow(combined), " days (", min(combined$date), " to ", max(combined$date), ")")
        
        return(combined)
      } else {
        return(api_forecast)
      }
    }
  }
  
  # Fallback: use only historical averages
  message("ℹ️  API disabled or unavailable, using historical averages only")
  return(get_weather_fallback(days = total_days))
}

# ==================================================================================
# Main Export Function (called by cleaning scripts)
# ==================================================================================

#' Save weather forecast to cache for use in cleaning pipeline
#' This function is called by 00_run_cleaning.R
save_weather_forecast_cache <- function() {
  create_cache_dir()
  
  # Fetch 16-day forecast from Open-Meteo
  forecast <- fetch_weather_forecast(forecast_days = 16)
  
  # Save to CSV cache for cleaning scripts
  cache_file <- file.path(CACHE_DIR, "weather_forecast.csv")
  fwrite(forecast, cache_file)
  
  message("✅ Weather forecast saved to: ", cache_file)
  return(forecast)
}

# ==================================================================================
# Testing & Execution
# ==================================================================================

if (interactive() || !exists("skip_test")) {
  message("\n========================================")
  message("Testing Open-Meteo Weather API")
  message("========================================\n")
  
  # Test 16-day API forecast
  message("\n--- Test 1: 16-day API forecast ---")
  weather_16d <- fetch_weather_forecast(forecast_days = 16)
  print(head(weather_16d, 5))
  print(tail(weather_16d, 5))
  
  # Test extended forecast (16 days API + historical for rest)
  message("\n--- Test 2: Extended forecast (730 days) ---")
  weather_extended <- get_extended_weather_forecast(use_api = TRUE, total_days = 730)
  print(head(weather_extended, 5))
  print(tail(weather_extended, 5))
  
  # Test fallback only
  message("\n--- Test 3: Fallback mode (historical averages only) ---")
  weather_fallback <- get_extended_weather_forecast(use_api = FALSE, total_days = 365)
  print(head(weather_fallback, 5))
  
  message("\n========================================")
  message("All tests completed!")
  message("========================================")
  
  # Attribution reminder
  message("\n📝 Remember to include attribution in your report:")
  message("   'Weather data by Open-Meteo.com'")
  message("   https://open-meteo.com")
}
