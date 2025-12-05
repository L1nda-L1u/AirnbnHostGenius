# ==================================================================================
# 05a - TfL Transport Prediction Model
# ==================================================================================
# Predict future TfL monthly journeys using time series forecasting

rm(list = ls())
library(tidyverse)
library(data.table)
library(lubridate)
library(forecast)
library(zoo)

# Note: Working directory should be set by the master script (00_run_all.R)
# or manually before running this script

# Create output directories (force create)
if (!dir.exists("outputs")) dir.create("outputs", recursive = TRUE)
if (!dir.exists("outputs/models")) dir.create("outputs/models", recursive = TRUE)
if (!dir.exists("outputs/plots")) dir.create("outputs/plots", recursive = TRUE)

message("\n==========================================================")
message("TFL TRANSPORT PREDICTION MODEL")
message("==========================================================\n")

# Load TfL monthly data
tfl_data <- fread("foot_traffic_data/cleaned/tfl_monthly.csv") %>%
  mutate(date = as.Date(date)) %>%
  arrange(date)

message("Loaded TfL data: ", nrow(tfl_data), " months")
message("Date range: ", min(tfl_data$date), " to ", max(tfl_data$date))

# ==================================================================================
# 1. EXPLORATORY ANALYSIS
# ==================================================================================

message("\n[1/4] Exploring TfL data patterns...")

# Plot historical data
p1 <- ggplot(tfl_data, aes(x = as.Date(date), y = avg_daily_journeys_m)) +
  geom_line(color = "steelblue", size = 1) +
  geom_point(color = "steelblue", size = 2, alpha = 0.6) +
  geom_smooth(method = "loess", color = "red", se = TRUE, span = 0.2, na.rm = TRUE) +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "1 year") +
  labs(title = "TfL Daily Journeys: Historical Data",
       subtitle = paste0("Monthly averages from ", min(tfl_data$date), " to ", max(tfl_data$date)),
       x = "Date", y = "Average Daily Journeys (millions)") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14))

ggsave("outputs/plots/15_tfl_historical.png", 
       p1, width = 12, height = 6, dpi = 300)

# ==================================================================================
# 2. TRAIN/VALIDATION SPLIT
# ==================================================================================

message("\n[2/5] Creating train/validation split...")

# Use last 12 months as validation set
n_validation <- 12
n_total <- nrow(tfl_data)
train_data <- tfl_data[1:(n_total - n_validation), ]
validation_data <- tfl_data[(n_total - n_validation + 1):n_total, ]

message("Training set: ", nrow(train_data), " months (", 
        min(train_data$date), " to ", max(train_data$date), ")")
message("Validation set: ", nrow(validation_data), " months (", 
        min(validation_data$date), " to ", max(validation_data$date), ")")

# ==================================================================================
# 3. TIME SERIES MODELING (on training data)
# ==================================================================================

message("\n[3/5] Building time series forecast model (training data only)...")

# Convert training data to time series object
train_ts <- ts(train_data$avg_daily_journeys_m, 
               start = c(year(min(train_data$date)), month(min(train_data$date))),
               frequency = 12)

# Fit ARIMA model (auto.arima finds best parameters)
tfl_model <- auto.arima(train_ts, 
                        seasonal = TRUE,
                        stepwise = FALSE,
                        approximation = FALSE)

message("Best ARIMA model: ", arimaorder(tfl_model))
print(summary(tfl_model))

# ==================================================================================
# 4. MODEL VALIDATION
# ==================================================================================

message("\n[4/5] Validating model on holdout data...")

# Forecast validation period
validation_forecast <- forecast(tfl_model, h = n_validation)

# Calculate validation metrics
validation_pred <- as.numeric(validation_forecast$mean)
validation_actual <- validation_data$avg_daily_journeys_m

mae_val <- mean(abs(validation_actual - validation_pred), na.rm = TRUE)
rmse_val <- sqrt(mean((validation_actual - validation_pred)^2, na.rm = TRUE))
mape_val <- mean(abs((validation_actual - validation_pred) / validation_actual) * 100, na.rm = TRUE)

# R-squared for validation
ss_res <- sum((validation_actual - validation_pred)^2, na.rm = TRUE)
ss_tot <- sum((validation_actual - mean(validation_actual, na.rm = TRUE))^2, na.rm = TRUE)
r2_val <- 1 - (ss_res / ss_tot)

message("\n--- Validation Metrics ---")
message("MAE: ", round(mae_val, 4), " million journeys")
message("RMSE: ", round(rmse_val, 4), " million journeys")
message("MAPE: ", round(mape_val, 2), "%")
message("R²: ", round(r2_val, 4))

# Plot validation: Actual vs Predicted
validation_plot_data <- tibble(
  date = validation_data$date,
  actual = validation_actual,
  predicted = validation_pred
)

validation_plot_data <- validation_plot_data %>%
  mutate(date = as.Date(date))

p_validation <- ggplot(validation_plot_data, aes(x = as.Date(date))) +
  geom_line(aes(y = actual, color = "Actual"), size = 1) +
  geom_line(aes(y = predicted, color = "Predicted"), size = 1, linetype = "dashed") +
  geom_ribbon(aes(ymin = as.numeric(validation_forecast$lower[,2]), 
                  ymax = as.numeric(validation_forecast$upper[,2])),
              alpha = 0.2, fill = "coral") +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "3 months") +
  labs(title = "TfL Prediction Validation (Last 12 Months)",
       subtitle = paste0("MAE: ", round(mae_val, 3), "M journeys, MAPE: ", round(mape_val, 1), "%, R²: ", round(r2_val, 3)),
       x = "Date", y = "Avg Daily Journeys (millions)", color = "Data") +
  scale_color_manual(values = c("Actual" = "steelblue", "Predicted" = "coral")) +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold"))

ggsave("outputs/plots/15_tfl_validation.png", p_validation, width = 12, height = 6, dpi = 300)

# ==================================================================================
# 5. RE-TRAIN ON FULL DATA & FORECAST FUTURE
# ==================================================================================

message("\n[5/5] Re-training on full dataset and forecasting 2025-2026...")

# Convert full dataset to time series object
tfl_ts <- ts(tfl_data$avg_daily_journeys_m, 
             start = c(year(min(tfl_data$date)), month(min(tfl_data$date))),
             frequency = 12)

# Re-fit ARIMA model on full data using same order as before
tfl_model_full <- Arima(tfl_ts, order = arimaorder(tfl_model)[1:3], 
                         seasonal = list(order = arimaorder(tfl_model)[4:6], period = 12))

message("Final model: ", arimaorder(tfl_model_full))
print(summary(tfl_model_full))

# Model diagnostics
png("outputs/plots/16_tfl_model_diagnostics.png", 
    width = 12, height = 10, units = "in", res = 300)
par(mfrow = c(2, 2))
plot(tfl_model_full$residuals, main = "Residuals", ylab = "Residual", xlab = "Time")
acf(tfl_model_full$residuals, main = "ACF of Residuals")
pacf(tfl_model_full$residuals, main = "PACF of Residuals")
hist(tfl_model_full$residuals, main = "Histogram of Residuals", xlab = "Residual", col = "lightblue")
dev.off()

# ==================================================================================
# 6. FORECAST FUTURE VALUES (2025-2026)
# ==================================================================================

message("\n[6/6] Generating 2025-2026 forecasts...")

# Forecast 24 months ahead (2025-2026)
forecast_horizon <- 24
tfl_forecast <- forecast(tfl_model_full, h = forecast_horizon)

# Extract forecasts
last_date <- max(tfl_data$date)
future_dates <- seq(last_date + months(1), by = "month", length.out = forecast_horizon)

tfl_predictions <- tibble(
  date = future_dates,
  year = year(date),
  month = month(date),
  avg_daily_journeys_m = as.numeric(tfl_forecast$mean),
  lower_80 = as.numeric(tfl_forecast$lower[,1]),
  upper_80 = as.numeric(tfl_forecast$upper[,1]),
  lower_95 = as.numeric(tfl_forecast$lower[,2]),
  upper_95 = as.numeric(tfl_forecast$upper[,2]),
  is_forecast = TRUE
)

# Combine with historical
tfl_complete <- tfl_data %>%
  mutate(
    lower_80 = avg_daily_journeys_m,
    upper_80 = avg_daily_journeys_m,
    lower_95 = avg_daily_journeys_m,
    upper_95 = avg_daily_journeys_m,
    is_forecast = FALSE
  ) %>%
  bind_rows(tfl_predictions) %>%
  arrange(date)

message("Forecasted ", forecast_horizon, " months: ", min(future_dates), " to ", max(future_dates))

# ==================================================================================
# 4. VISUALIZATION & SAVE
# ==================================================================================

message("\n[4/4] Creating visualizations and saving results...")

# Plot forecast
tfl_complete <- tfl_complete %>%
  mutate(date = as.Date(date))

p2 <- ggplot(tfl_complete, aes(x = as.Date(date))) +
  geom_line(aes(y = avg_daily_journeys_m, color = is_forecast), size = 1) +
  geom_ribbon(data = filter(tfl_complete, is_forecast),
              aes(ymin = lower_80, ymax = upper_80), alpha = 0.2, fill = "steelblue") +
  geom_ribbon(data = filter(tfl_complete, is_forecast),
              aes(ymin = lower_95, ymax = upper_95), alpha = 0.1, fill = "steelblue") +
  scale_x_date(date_labels = "%Y-%m", date_breaks = "6 months") +
  scale_color_manual(values = c("TRUE" = "red", "FALSE" = "steelblue"),
                     labels = c("Historical", "Forecast")) +
  labs(title = "TfL Daily Journeys: Historical + Forecast",
       subtitle = "24-month forecast with 80% and 95% confidence intervals",
       x = "Date", y = "Average Daily Journeys (millions)",
       color = "Data Type") +
  theme_minimal() +
  theme(plot.title = element_text(face = "bold", size = 14),
        legend.position = "top")

ggsave("outputs/plots/17_tfl_forecast.png", 
       p2, width = 12, height = 6, dpi = 300)

# Save model and predictions
saveRDS(tfl_model, "outputs/models/tfl_arima_model.rds")
fwrite(tfl_complete, "outputs/tfl_predictions_complete.csv")

message("\nSaved:")
message("  - Model: outputs/models/tfl_arima_model.rds")
message("  - Predictions: outputs/tfl_predictions_complete.csv")

# ==================================================================================
# 5. EXPAND TO DAILY LEVEL
# ==================================================================================

message("\n[5/5] Expanding monthly predictions to daily level...")

# For daily predictions, use historical daily patterns within each month
# Load historical daily data to learn within-month patterns
daily_data <- fread("foot_traffic_data/cleaned/foot_traffic_daily.csv") %>%
  mutate(date = as.Date(date))

# Calculate day-of-month adjustment factors from historical data
daily_patterns <- daily_data %>%
  filter(!is.na(tfl_daily_avg_m)) %>%
  mutate(
    year_month = floor_date(date, "month"),
    day_of_month = day(date)
  ) %>%
  group_by(year_month) %>%
  mutate(
    monthly_mean = mean(tfl_daily_avg_m, na.rm = TRUE),
    daily_factor = tfl_daily_avg_m / monthly_mean
  ) %>%
  ungroup() %>%
  group_by(day_of_month, day_of_week) %>%
  summarise(
    avg_daily_factor = mean(daily_factor, na.rm = TRUE),
    .groups = "drop"
  )

# Create daily forecast for each predicted month
tfl_daily_forecast <- tfl_predictions %>%
  rowwise() %>%
  do({
    month_row <- .
    days_in_month <- days_in_month(month_row$date)
    month_dates <- seq(month_row$date, by = "day", length.out = days_in_month)
    
    tibble(
      date = month_dates,
      year = lubridate::year(date),
      month = lubridate::month(date),
      day_of_month = lubridate::day(date),
      day_of_week = lubridate::wday(date, label = TRUE),
      monthly_avg = month_row$avg_daily_journeys_m
    ) %>%
      left_join(daily_patterns, by = c("day_of_month", "day_of_week")) %>%
      mutate(
        avg_daily_factor = replace_na(avg_daily_factor, 1.0),
        tfl_daily_avg_m = monthly_avg * avg_daily_factor,
        is_forecast = TRUE
      )
  }) %>%
  ungroup()

# Save daily forecast
fwrite(tfl_daily_forecast, "outputs/tfl_daily_forecast.csv")
message("\nSaved daily forecast: outputs/tfl_daily_forecast.csv")

# Summary statistics
message("\n==========================================================")
message("TFL PREDICTION COMPLETE!")
message("==========================================================")
message("\nForecast Summary:")
message("  Months forecasted: ", forecast_horizon)
message("  Date range: ", min(tfl_predictions$date), " to ", max(tfl_predictions$date))
message("  Mean prediction: ", round(mean(tfl_predictions$avg_daily_journeys_m), 2), " million journeys/day")
message("  Prediction range: ", round(min(tfl_predictions$avg_daily_journeys_m), 2), " - ", 
        round(max(tfl_predictions$avg_daily_journeys_m), 2), " million")

# Show sample predictions
message("\nSample predictions:")
print(head(tfl_predictions %>% select(date, avg_daily_journeys_m, lower_80, upper_80), 6))
message("")

