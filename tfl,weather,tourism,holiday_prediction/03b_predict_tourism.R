# ==================================================================================
# 05b - Tourism Prediction Model
# ==================================================================================
# Predict future international visitor numbers using seasonal decomposition
# Account for COVID-19 impact and recovery

rm(list = ls())
library(tidyverse)
library(data.table)
library(lubridate)
library(forecast)
library(zoo)

setwd("/Users/xiongyi/Desktop/Airbnb/AirbnbHostGenius/foot_traffic_prediction")

message("\n==========================================================")
message("TOURISM PREDICTION MODEL")
message("==========================================================\n")

# Load tourism quarterly data
tourism_data <- fread("foot_traffic_data/cleaned/tourism_quarterly.csv") %>%
  mutate(date = as.Date(date)) %>%
  arrange(date)

message("Loaded tourism data: ", nrow(tourism_data), " quarters")
message("Date range: ", min(tourism_data$date), " to ", max(tourism_data$date))

# ==================================================================================
# 1. EXPLORATORY ANALYSIS & COVID IMPACT
# ==================================================================================

message("\n[1/5] Analyzing tourism patterns and COVID impact...")

# Identify pre-COVID and COVID periods
tourism_data <- tourism_data %>%
  mutate(
    period = case_when(
      year < 2020 ~ "Pre-COVID",
      year >= 2020 & year <= 2021 ~ "COVID",
      year >= 2022 ~ "Recovery"
    )
  )

# Plot historical data
p1 <- ggplot(tourism_data, aes(x = as.Date(date), y = total_visits_k, color = period)) +
  geom_line(size = 1) +
  geom_point(size = 2.5) +
  geom_vline(xintercept = as.Date("2020-03-01"), linetype = "dashed", color = "red") +
  annotate("text", x = as.Date("2020-03-01"), y = max(tourism_data$total_visits_k) * 0.9,
           label = "COVID-19", hjust = -0.1, color = "red") +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "1 year") +
  scale_color_manual(values = c("Pre-COVID" = "steelblue", 
                                 "COVID" = "orange", 
                                 "Recovery" = "darkgreen")) +
  labs(title = "International Visitors to London: Historical Data",
       subtitle = "Quarterly visitor numbers with COVID-19 impact",
       x = "Date", y = "Visitors (thousands)", color = "Period") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/plots/18_tourism_historical.png", 
       p1, width = 12, height = 6, dpi = 300)

# ==================================================================================
# 2. MODEL PRE-COVID DATA
# ==================================================================================

message("\n[2/5] Building forecast model using pre-COVID data...")

# Use only pre-COVID data for training (more reliable pattern)
tourism_pre_covid <- tourism_data %>%
  filter(year < 2020)

message("Training on ", nrow(tourism_pre_covid), " pre-COVID quarters")

# Convert to time series
tourism_ts <- ts(tourism_pre_covid$total_visits_k,
                 start = c(lubridate::year(min(tourism_pre_covid$date)), 
                           lubridate::quarter(min(tourism_pre_covid$date))),
                 frequency = 4)

# Fit ARIMA model
tourism_model <- auto.arima(tourism_ts,
                            seasonal = TRUE,
                            stepwise = FALSE,
                            approximation = FALSE)

message("Best ARIMA model: ", arimaorder(tourism_model))
print(summary(tourism_model))

# Model diagnostics
png("outputs/plots/19_tourism_model_diagnostics.png",
    width = 12, height = 10, units = "in", res = 300)
par(mfrow = c(2, 2))
plot(tourism_model$residuals, main = "Residuals", ylab = "Residual", xlab = "Time")
acf(tourism_model$residuals, main = "ACF of Residuals")
pacf(tourism_model$residuals, main = "PACF of Residuals")
hist(tourism_model$residuals, main = "Histogram of Residuals", xlab = "Residual", col = "lightblue")
dev.off()

# ==================================================================================
# 3. DATA LIMITATION NOTICE
# ==================================================================================

message("\n[3/5] Checking for post-COVID validation data...")

# Check if we have any data after 2019 for validation
recovery_data <- tourism_data %>%
  filter(year >= 2020)

if (nrow(recovery_data) == 0) {
  message("⚠️  WARNING: No post-2019 tourism data available")
  message("   Predictions are based solely on 2002-2019 pre-COVID trends")
  message("   COVID impact and recovery patterns are NOT incorporated")
  message("   Future predictions may not reflect actual post-pandemic tourism levels")
  recovery_ratio <- 1.0  # No adjustment - use raw pre-COVID forecast
} else {
  message("✓ Found ", nrow(recovery_data), " quarters of post-2019 data for validation")
  
  # Calculate recovery ratio from actual data
  recovery_forecast <- forecast(tourism_model, h = nrow(recovery_data))
  recovery_ratio <- mean(recovery_data$total_visits_k / as.numeric(recovery_forecast$mean), na.rm = TRUE)
  message("   Data-based recovery ratio: ", round(recovery_ratio, 3))
}

# ==================================================================================
# 4. FORECAST 2024 Q2 - 2026 Q4
# ==================================================================================

message("\n[4/5] Forecasting tourism for 2024-2026...")

# Forecast from 2024 Q2 (to cover missing 2024 data) through 2026
forecast_start_date <- as.Date("2024-04-01")  # Q2 2024
forecast_end_date <- as.Date("2026-12-31")  # Q4 2026
forecast_horizon <- length(seq(forecast_start_date, forecast_end_date, by = "quarter"))

last_date <- max(tourism_pre_covid$date)
quarters_to_start <- as.numeric(interval(last_date, forecast_start_date) / months(3))

tourism_forecast_raw <- forecast(tourism_model, h = quarters_to_start + forecast_horizon)

# Extract forecasts from 2024 Q2 onwards
future_start_idx <- quarters_to_start + 1
future_end_idx <- future_start_idx + forecast_horizon - 1

future_dates <- seq(forecast_start_date, by = "quarter", length.out = forecast_horizon)

# Apply recovery adjustment to future forecasts
tourism_predictions <- tibble(
  date = future_dates,
  year = lubridate::year(date),
  quarter_num = lubridate::quarter(date),
  total_visits_k = as.numeric(tourism_forecast_raw$mean)[future_start_idx:future_end_idx] * recovery_ratio,
  lower_80 = as.numeric(tourism_forecast_raw$lower[future_start_idx:future_end_idx, 1]) * recovery_ratio,
  upper_80 = as.numeric(tourism_forecast_raw$upper[future_start_idx:future_end_idx, 1]) * recovery_ratio,
  lower_95 = as.numeric(tourism_forecast_raw$lower[future_start_idx:future_end_idx, 2]) * recovery_ratio,
  upper_95 = as.numeric(tourism_forecast_raw$upper[future_start_idx:future_end_idx, 2]) * recovery_ratio,
  total_spend_m = total_visits_k * mean(tourism_pre_covid$avg_spend_per_visit, na.rm = TRUE) / 1000,
  avg_spend_per_visit = mean(tourism_pre_covid$avg_spend_per_visit, na.rm = TRUE),
  avg_nights_per_visit = mean(tourism_pre_covid$avg_nights_per_visit, na.rm = TRUE),
  is_forecast = TRUE
)

message("Forecasted ", forecast_horizon, " quarters: ", min(future_dates), " to ", max(future_dates))
message("  Coverage: 2024 Q2-Q4 + 2025-2026 (to fill missing 2024 data)")

# ==================================================================================
# 5. VISUALIZATION & SAVE
# ==================================================================================

message("\n[5/5] Creating visualizations and saving results...")

# Combine with historical
tourism_complete <- tourism_data %>%
  mutate(
    lower_80 = total_visits_k,
    upper_80 = total_visits_k,
    lower_95 = total_visits_k,
    upper_95 = total_visits_k,
    is_forecast = FALSE
  ) %>%
  bind_rows(tourism_predictions) %>%
  arrange(date)

# Plot forecast
tourism_complete <- tourism_complete %>%
  mutate(date = as.Date(date))

p2 <- ggplot(tourism_complete, aes(x = as.Date(date))) +
  geom_line(aes(y = total_visits_k, color = is_forecast), size = 1) +
  geom_point(aes(y = total_visits_k, color = is_forecast), size = 2) +
  geom_ribbon(data = filter(tourism_complete, is_forecast),
              aes(ymin = lower_80, ymax = upper_80), alpha = 0.2, fill = "darkgreen") +
  geom_ribbon(data = filter(tourism_complete, is_forecast),
              aes(ymin = lower_95, ymax = upper_95), alpha = 0.1, fill = "darkgreen") +
  geom_vline(xintercept = as.Date("2020-03-01"), linetype = "dashed", color = "red", alpha = 0.5) +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "6 months") +
  scale_color_manual(values = c("TRUE" = "darkgreen", "FALSE" = "steelblue"),
                     labels = c("Historical", "Forecast")) +
  labs(title = "International Visitors: Historical + Forecast (2024-2026)",
       subtitle = if (recovery_ratio == 1.0) {
         paste0(forecast_horizon, "-quarter forecast (Pre-COVID trend only - no COVID adjustment)")
       } else {
         paste0(forecast_horizon, "-quarter forecast with ", round(recovery_ratio * 100, 0), 
                "% COVID recovery adjustment")
       },
       x = "Date", y = "Visitors (thousands)",
       color = "Data Type") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "top")

ggsave("outputs/plots/20_tourism_forecast.png",
       p2, width = 12, height = 6, dpi = 300)

# Save model and predictions
saveRDS(tourism_model, "outputs/models/tourism_arima_model.rds")
saveRDS(list(recovery_ratio = recovery_ratio), 
        "outputs/models/tourism_recovery_params.rds")
fwrite(tourism_complete, "outputs/tourism_predictions_complete.csv")

message("\nSaved:")
message("  - Model: outputs/models/tourism_arima_model.rds")
message("  - Recovery params: outputs/models/tourism_recovery_params.rds")
message("  - Predictions: outputs/tourism_predictions_complete.csv")

# ==================================================================================
# 6. EXPAND TO DAILY LEVEL
# ==================================================================================

message("\n[6/6] Expanding quarterly predictions to daily level...")

# For daily, distribute evenly across quarter with seasonal within-quarter pattern
tourism_daily_forecast <- tourism_predictions %>%
  rowwise() %>%
  do({
    quarter_row <- .
    quarter_end <- quarter_row$date + months(3) - days(1)
    quarter_dates <- seq(quarter_row$date, quarter_end, by = "day")
    days_in_quarter <- length(quarter_dates)
    
    # Simple even distribution (could be enhanced with within-quarter patterns)
    tibble(
      date = quarter_dates,
      year = lubridate::year(date),
      quarter_num = lubridate::quarter(date),
      tourism_quarterly_visits_k = quarter_row$total_visits_k,
      tourism_quarterly_spend_m = quarter_row$total_spend_m,
      tourism_avg_spend = quarter_row$avg_spend_per_visit,
      tourism_avg_nights = quarter_row$avg_nights_per_visit,
      is_forecast = TRUE
    )
  }) %>%
  ungroup()

# Save daily forecast
fwrite(tourism_daily_forecast, "outputs/tourism_daily_forecast.csv")
message("\nSaved daily forecast: outputs/tourism_daily_forecast.csv")

# Summary statistics
message("\n==========================================================")
message("TOURISM PREDICTION COMPLETE!")
message("==========================================================")
message("\nForecast Summary:")
message("  Quarters forecasted: ", forecast_horizon)
message("  Date range: ", min(tourism_predictions$date), " to ", max(tourism_predictions$date))
message("  Mean prediction: ", round(mean(tourism_predictions$total_visits_k), 0), " thousand visitors/quarter")
message("  Prediction range: ", round(min(tourism_predictions$total_visits_k), 0), " - ",
        round(max(tourism_predictions$total_visits_k), 0), " thousand")
if (recovery_ratio == 1.0) {
  message("  ⚠️  No COVID adjustment applied (no post-2019 data available)")
} else {
  message("  Recovery adjustment: ", round(recovery_ratio * 100, 1), "% (data-based)")
}

# Show sample predictions
message("\nSample predictions:")
print(head(tourism_predictions %>% select(date, total_visits_k, lower_80, upper_80), 4))
message("")

