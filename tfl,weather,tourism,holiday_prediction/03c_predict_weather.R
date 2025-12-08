# ==================================================================================
# 03c - Weather Prediction/Forecast
# ==================================================================================
# Obtain weather forecasts from Open-Meteo API for future periods
# For consistency with 03a (TfL) and 03b (Tourism) prediction scripts

rm(list = ls())
library(tidyverse)
library(data.table)
library(lubridate)

# Note: Working directory should be set by the master script (00_run_all.R)
# or manually before running this script

# Create output directories
dir.create("outputs/plots", recursive = TRUE, showWarnings = FALSE)

message("\n==========================================================")
message("WEATHER FORECAST - OPEN-METEO API")
message("==========================================================\n")

# ==================================================================================
# 1. LOAD HISTORICAL WEATHER DATA
# ==================================================================================

message("[1/5] Loading historical weather data...")

weather_historical <- fread("foot_traffic_data/cleaned/weather_daily.csv") %>%
  mutate(date = as.Date(date))

message("Historical data: ", nrow(weather_historical), " days")
message("Date range: ", min(weather_historical$date), " to ", max(weather_historical$date))

# ==================================================================================
# 2. EVALUATE HISTORICAL SEASONAL PATTERNS (for fallback)
# ==================================================================================

message("\n[2/5] Analyzing seasonal patterns...")

# Calculate seasonal averages by day of year
seasonal_avg <- weather_historical %>%
  mutate(doy = yday(date)) %>%
  group_by(doy) %>%
  summarise(
    temp_c_avg = mean(temp_c, na.rm = TRUE),
    temp_c_sd = sd(temp_c, na.rm = TRUE),
    wind_kmh_avg = mean(wind_kmh, na.rm = TRUE),
    precip_mm_avg = mean(precip_mm, na.rm = TRUE),
    weather_quality_avg = mean(weather_quality, na.rm = TRUE),
    n_years = n_distinct(year(date)),
    .groups = "drop"
  )

message("Calculated seasonal averages across ", 
        max(seasonal_avg$n_years, na.rm = TRUE), " years")

# ==================================================================================
# 3. VALIDATE SEASONAL AVERAGE METHOD (Backtesting)
# ==================================================================================

message("\n[3/5] Validating seasonal average method (backtesting)...")

# Use last year as test set
test_year <- max(year(weather_historical$date))
train_data <- weather_historical %>% filter(year(date) < test_year)
test_data <- weather_historical %>% filter(year(date) == test_year)

# Calculate seasonal averages from training data only
train_seasonal <- train_data %>%
  mutate(doy = yday(date)) %>%
  group_by(doy) %>%
  summarise(
    temp_c_pred = mean(temp_c, na.rm = TRUE),
    weather_quality_pred = mean(weather_quality, na.rm = TRUE),
    .groups = "drop"
  )

# Make predictions for test year
test_predictions <- test_data %>%
  mutate(doy = yday(date)) %>%
  left_join(train_seasonal, by = "doy") %>%
  mutate(
    temp_error = abs(temp_c - temp_c_pred),
    weather_error = abs(weather_quality - weather_quality_pred)
  )

# Calculate validation metrics
temp_mae <- mean(test_predictions$temp_error, na.rm = TRUE)
temp_rmse <- sqrt(mean((test_predictions$temp_c - test_predictions$temp_c_pred)^2, na.rm = TRUE))
weather_mae <- mean(test_predictions$weather_error, na.rm = TRUE)

message("\n--- Validation Results (Seasonal Average Method) ---")
message("Test Year: ", test_year)
message("Temperature MAE: ", round(temp_mae, 2), "°C")
message("Temperature RMSE: ", round(temp_rmse, 2), "°C")
message("Weather Quality MAE: ", round(weather_mae, 3))

# Visualization: Actual vs Predicted
test_predictions <- test_predictions %>%
  mutate(date = as.Date(date))

p_validation <- ggplot(test_predictions, aes(x = as.Date(date))) +
  geom_line(aes(y = temp_c, color = "Actual"), size = 0.8) +
  geom_line(aes(y = temp_c_pred, color = "Seasonal Average"), size = 0.8, linetype = "dashed") +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "1 month") +
  labs(title = paste0("Weather Validation: Seasonal Average vs Actual (", test_year, ")"),
       subtitle = paste0("Temperature MAE: ", round(temp_mae, 2), "°C, RMSE: ", round(temp_rmse, 2), "°C"),
       x = "Date", y = "Temperature (°C)", color = "Data") +
  scale_color_manual(values = c("Actual" = "steelblue", "Seasonal Average" = "coral")) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/plots/17_weather_validation.png", p_validation, width = 12, height = 6, dpi = 300)

# ==================================================================================
# 4. FETCH FUTURE FORECAST FROM OPEN-METEO API
# ==================================================================================

message("\n[4/5] Fetching weather forecast from Open-Meteo API...")

# Source API functions
skip_test <- TRUE
source("apis/weather_api.R")

# Fetch 16-day forecast
api_forecast <- fetch_weather_forecast(forecast_days = 16)

message("✅ API forecast: ", nrow(api_forecast), " days")
message("   Date range: ", min(api_forecast$date), " to ", max(api_forecast$date))

# Check if these are truly future dates
max_historical <- max(weather_historical$date)
future_api_dates <- api_forecast %>% filter(date > max_historical)

if (nrow(future_api_dates) > 0) {
  message("   Future dates: ", nrow(future_api_dates), " days beyond historical data")
}

# ==================================================================================
# 5. GENERATE EXTENDED FORECAST (API + Seasonal Averages)
# ==================================================================================

message("\n[5/5] Generating extended forecast (2025-2026)...")

# Create 2-year forecast framework
forecast_dates <- seq(Sys.Date(), by = "day", length.out = 730)

# For days 1-16: Use API if available
# For days 17+: Use seasonal averages
weather_forecast <- tibble(date = as.Date(forecast_dates)) %>%
  mutate(
    date = as.Date(date),
    doy = yday(date)
  )

# Join with API forecast first
if (nrow(api_forecast) > 0) {
  # Ensure api_forecast date column is Date type
  api_forecast <- api_forecast %>%
    mutate(date = as.Date(date))
  
  weather_forecast <- weather_forecast %>%
    left_join(
      api_forecast %>% 
        mutate(date = as.Date(date)) %>%
        select(date, temp_c, wind_kmh, precip_mm, weather_quality, is_good_weather) %>%
        rename_with(~paste0(., "_api"), -date),
      by = "date"
    )
}

# Join with seasonal averages
weather_forecast <- weather_forecast %>%
  left_join(seasonal_avg, by = "doy") %>%
  mutate(
    # Ensure date remains Date type
    date = as.Date(date),
    
    # Use API data if available, otherwise seasonal average
    temp_c = if_else(!is.na(temp_c_api), temp_c_api, temp_c_avg),
    wind_kmh = if_else(!is.na(wind_kmh_api), wind_kmh_api, wind_kmh_avg),
    precip_mm = if_else(!is.na(precip_mm_api), precip_mm_api, precip_mm_avg),
    weather_quality = if_else(!is.na(weather_quality_api), weather_quality_api, weather_quality_avg),
    is_good_weather = if_else(!is.na(is_good_weather_api), is_good_weather_api, weather_quality >= 0.7),
    
    # Add source indicator
    forecast_source = case_when(
      !is.na(temp_c_api) ~ "Open-Meteo API",
      TRUE ~ "Seasonal Average"
    )
  ) %>%
  select(date, temp_c, wind_kmh, precip_mm, weather_quality, is_good_weather, forecast_source) %>%
  mutate(date = as.Date(date))  # Final check to ensure Date type

message("Total forecast: ", nrow(weather_forecast), " days")
message("  API forecast: ", sum(weather_forecast$forecast_source == "Open-Meteo API"), " days")
message("  Seasonal avg: ", sum(weather_forecast$forecast_source == "Seasonal Average"), " days")

# ==================================================================================
# 6. VISUALIZE FORECAST
# ==================================================================================

# Plot: Historical + Forecast
# Ensure date columns are Date type before binding
historical_plot_data <- weather_historical %>%
  mutate(
    date = as.Date(date),
    forecast_source = "Historical"
  ) %>%
  select(date, temp_c, weather_quality, forecast_source)

forecast_plot_data <- weather_forecast %>%
  mutate(date = as.Date(date)) %>%
  select(date, temp_c, weather_quality, forecast_source)

combined_plot_data <- bind_rows(historical_plot_data, forecast_plot_data) %>%
  filter(is.finite(temp_c), !is.na(temp_c), !is.na(date))

today_date <- as.Date(Sys.Date())

# Ensure combined_plot_data date is Date type
combined_plot_data <- combined_plot_data %>%
  mutate(date = as.Date(date))

p_forecast <- ggplot(combined_plot_data, aes(x = date, y = temp_c, color = forecast_source)) +
  geom_line(size = 0.7, alpha = 0.8) +
  geom_vline(xintercept = today_date, linetype = "dashed", color = "gray30") +
  annotate("text", x = today_date, y = max(combined_plot_data$temp_c, na.rm = TRUE), 
           label = "Today", vjust = -0.5, hjust = 1.1) +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "3 months") +
  labs(title = "Weather Forecast: Historical + Future (2025-2026)",
       subtitle = "16-day API forecast + Seasonal averages",
       x = "Date", y = "Temperature (°C)", color = "Source") +
  scale_color_manual(values = c("Historical" = "steelblue", 
                                 "Open-Meteo API" = "coral", 
                                 "Seasonal Average" = "lightblue")) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/plots/18_weather_forecast_complete.png", p_forecast, width = 14, height = 6, dpi = 300)

# ==================================================================================
# 7. SAVE OUTPUTS
# ==================================================================================

# Save complete forecast
fwrite(weather_forecast, "outputs/weather_predictions_complete.csv")

message("\n✅ Weather forecast complete!")
message("   Saved: outputs/weather_predictions_complete.csv")
message("   Plots: outputs/plots/17_weather_validation.png")
message("          outputs/plots/18_weather_forecast_complete.png")

message("\n📝 Attribution: Weather data by Open-Meteo.com")

