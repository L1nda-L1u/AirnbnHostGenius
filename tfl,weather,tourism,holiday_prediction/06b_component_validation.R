# ==================================================================================
# 07a - Component-Level Validation Report
# ==================================================================================
# Summarize validation results for TfL, Tourism, and Weather predictions
# This shows how well each individual component predicts before combining them

rm(list = ls())
library(tidyverse)
library(data.table)
library(lubridate)
library(gridExtra)

setwd("/Users/xiongyi/Desktop/Airbnb/AirbnbHostGenius/foot_traffic_prediction")

message("\n==========================================================")
message("COMPONENT-LEVEL VALIDATION REPORT")
message("==========================================================\n")

# ==================================================================================
# 1. TFL VALIDATION RESULTS
# ==================================================================================

message("[1/3] Loading TfL validation results...")

# TfL validation was done in 03a_predict_tfl.R
# Validation period: Last 12 months of training data
if (file.exists("outputs/tfl_predictions_complete.csv")) {
  tfl_predictions <- fread("outputs/tfl_predictions_complete.csv") %>%
    mutate(date = as.Date(date))
  
  # Load historical for validation comparison
  tfl_monthly <- fread("foot_traffic_data/cleaned/tfl_monthly.csv") %>%
    mutate(date = as.Date(date))
  
  # Validation period: 2023-06-01 to 2024-05-01 (last 12 months)
  validation_start <- as.Date("2023-06-01")
  validation_end <- as.Date("2024-05-01")
  
  tfl_val <- tfl_monthly %>%
    filter(date >= validation_start & date <= validation_end) %>%
    select(date, actual = avg_daily_journeys_m)
  
  # Note: TfL predictions start from 2024-06-01, so we need to load the model
  # and backtest on validation period
  tfl_model <- readRDS("outputs/models/tfl_arima_model.rds")
  
  message("✓ TfL validation period: ", validation_start, " to ", validation_end)
  message("  Validation method: ARIMA holdout (last 12 months)")
  message("  See 03a_predict_tfl.R for detailed metrics")
  message("  Validation R²: -2.61 (ARIMA struggles with recent volatility)")
  message("  Validation MAPE: 10.1%")
  
} else {
  message("⚠️  TfL predictions not found. Run 03a_predict_tfl.R first.")
}

# ==================================================================================
# 2. TOURISM VALIDATION RESULTS
# ==================================================================================

message("\n[2/3] Loading Tourism validation results...")

if (file.exists("outputs/tourism_predictions_complete.csv")) {
  tourism_predictions <- fread("outputs/tourism_predictions_complete.csv") %>%
    mutate(date = as.Date(date))
  
  message("⚠️  LIMITATION: Tourism validation not possible")
  message("   Reason: Historical data only available until 2019 Q4")
  message("   COVID impact (2020-2023) cannot be validated")
  message("   Predictions are based on pre-COVID trends (2002-2019)")
  message("   Model: ARIMA(1,0,0)(0,1,2)[4] trained on 72 pre-COVID quarters")
  message("   No recovery ratio applied (no post-2019 data)")
  
} else {
  message("⚠️  Tourism predictions not found. Run 03b_predict_tourism.R first.")
}

# ==================================================================================
# 3. WEATHER VALIDATION RESULTS
# ==================================================================================

message("\n[3/3] Loading Weather validation results...")

if (file.exists("outputs/weather_predictions_complete.csv")) {
  weather_predictions <- fread("outputs/weather_predictions_complete.csv") %>%
    mutate(date = as.Date(date))
  
  message("✓ Weather validation: Seasonal average backtesting")
  message("  Test year: 2024 (held out from seasonal calculation)")
  message("  Temperature MAE: 3.14°C")
  message("  Temperature RMSE: 3.47°C")
  message("  Weather Quality MAE: 0.163")
  message("  Method: 7-year seasonal averages (2019-2023 excluding test year)")
  message("  + Open-Meteo API for 16-day forecasts")
  
} else {
  message("⚠️  Weather predictions not found. Run 03c_predict_weather.R first.")
}

# ==================================================================================
# 4. CREATE SUMMARY VISUALIZATION
# ==================================================================================

message("\n[4/4] Creating validation summary visualization...")

# Create validation summary table
validation_summary <- tibble(
  Component = c("TfL Transport", "Tourism", "Weather (Temp)", "Weather (Quality)"),
  `Validation Period` = c(
    "2023-06 to 2024-05",
    "N/A (No post-2019 data)",
    "2024 (backtesting)",
    "2024 (backtesting)"
  ),
  `Method` = c(
    "ARIMA holdout",
    "None (pre-COVID model only)",
    "Seasonal average",
    "Seasonal average"
  ),
  `MAE` = c(
    "0.99M journeys",
    "N/A",
    "3.14°C",
    "0.163"
  ),
  `MAPE` = c(
    "10.1%",
    "N/A",
    "N/A",
    "N/A"
  ),
  `R²` = c(
    "-2.61",
    "N/A",
    "N/A",
    "N/A"
  ),
  `Data Quality` = c(
    "✓ Complete",
    "⚠️ Ends 2019",
    "✓ Complete",
    "✓ Complete"
  )
)

# Save summary
fwrite(validation_summary, "outputs/component_validation_summary.csv")

# Create text summary visualization
png("outputs/plots/35_component_validation_summary.png",
    width = 14, height = 8, units = "in", res = 300)

par(mar = c(1, 1, 3, 1))
plot(1, type = "n", axes = FALSE, xlab = "", ylab = "", 
     xlim = c(0, 10), ylim = c(0, 10))

title("Component-Level Validation Summary", 
      cex.main = 2, font.main = 2, line = 1)

# Component 1: TfL
text(1, 8.5, "TfL Transport Journeys", pos = 4, cex = 1.4, font = 2, col = "darkblue")
text(1, 8, "✓ Validation Period: 2023-06 to 2024-05 (12 months)", pos = 4, cex = 1.1)
text(1, 7.5, "✓ Method: ARIMA(3,1,0)(2,0,0)[12] with holdout validation", pos = 4, cex = 1.1)
text(1, 7, "✓ Performance: MAPE = 10.1%, MAE = 0.99M journeys", pos = 4, cex = 1.1)
text(1, 6.5, "⚠  R² = -2.61 (model struggles with 2023-2024 volatility)", pos = 4, cex = 1.1, col = "orange")

# Component 2: Tourism
text(1, 5.5, "International Tourism", pos = 4, cex = 1.4, font = 2, col = "darkgreen")
text(1, 5, "⚠  Data Limitation: Only available until 2019 Q4", pos = 4, cex = 1.1, col = "red")
text(1, 4.5, "✓ Model: ARIMA(1,0,0)(0,1,2)[4] on 2002-2019 data", pos = 4, cex = 1.1)
text(1, 4, "⚠  No validation possible (no post-COVID data)", pos = 4, cex = 1.1, col = "red")
text(1, 3.5, "⚠  Predictions assume pre-COVID trends continue", pos = 4, cex = 1.1, col = "orange")

# Component 3: Weather
text(1, 2.5, "Weather (Open-Meteo API + Seasonal)", pos = 4, cex = 1.4, font = 2, col = "darkmagenta")
text(1, 2, "✓ Validation: 2024 backtesting (held out from training)", pos = 4, cex = 1.1)
text(1, 1.5, "✓ Temperature MAE: 3.14°C, RMSE: 3.47°C", pos = 4, cex = 1.1)
text(1, 1, "✓ Weather Quality MAE: 0.163 (on 0-1 scale)", pos = 4, cex = 1.1)

dev.off()

message("\n==========================================================")
message("COMPONENT VALIDATION REPORT COMPLETE!")
message("==========================================================")
message("\nKey Findings:")
message("  ✓ TfL: Validated on 12-month holdout (decent MAPE, poor R²)")
message("  ⚠  Tourism: Cannot validate (no post-2019 data)")
message("  ✓ Weather: Validated via backtesting (acceptable errors)")
message("\nImplication for Foot Traffic Score:")
message("  - Overall predictions are most reliable for periods with complete TfL data")
message("  - Post-2024-05 predictions rely on ARIMA forecasts")
message("  - Tourism predictions do not account for COVID recovery patterns")
message("\nSaved:")
message("  - outputs/component_validation_summary.csv")
message("  - outputs/plots/35_component_validation_summary.png\n")

