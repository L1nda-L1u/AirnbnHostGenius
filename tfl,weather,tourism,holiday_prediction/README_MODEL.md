# Component Prediction Documentation

## Overview

This pipeline predicts **REAL components** for London:
- **TfL Transport**: Daily journey counts (millions)
- **Tourism**: Quarterly visitor statistics (thousands)
- **Weather**: Daily temperature, precipitation, quality
- **Holidays**: UK bank holidays and major events

**Note**: No synthetic `foot_traffic_score` - only real, observable data components.

**Prediction Coverage:**
- Historical: 2019-2024 (with validation)
- Future: 2025-2026 (forecasts)
- Components: TfL, Tourism, Weather (Holidays from API)

## Component Prediction Models

### 1. TfL Transport (ARIMA)
- **Model**: ARIMA(3,1,0)(2,0,0)[12]
- **Validation**: Last 12 months holdout
- **Output**: Daily journey averages (millions)
- **File**: `outputs/models/tfl_arima_model.rds`

### 2. Tourism (ARIMA + Recovery)
- **Model**: ARIMA with pre-COVID trend projection
- **Validation**: Last 4 quarters holdout
- **Output**: Quarterly visitor counts (thousands)
- **File**: `outputs/models/tourism_arima_model.rds`
- **Note**: Predictions based on pre-COVID trends (no post-2019 data available)

### 3. Weather (Open-Meteo API)
- **Source**: Open-Meteo API (16-day forecast)
- **Fallback**: Historical averages for extended periods
- **Output**: Daily temperature, precipitation, quality
- **Cache**: `foot_traffic_data/api_cache/weather_forecast.csv`

### 4. Holidays (Gov.uk API)
- **Source**: UK Government API
- **Coverage**: 2018-2027
- **Output**: Bank holidays, major events
- **Cache**: `foot_traffic_data/api_cache/holidays_forecast.csv`

## Project Structure

```
foot_traffic_prediction/
├── apis/
│   ├── weather_api.R           # OpenWeatherMap integration
│   └── holidays_api.R          # Gov.uk holidays API
│
├── foot_traffic_data/
│   ├── raw/                    # Original data files
│   ├── cleaned/                # Processed data
│   └── api_cache/             # Cached API responses
│
├── outputs/
│   ├── models/                 # Trained .rds model files
│   ├── plots/                  # Visualizations (30+ charts)
│   └── *.csv                   # Predictions and metrics
│
├── predictions/
│   ├── foot_traffic_forecast_2025_2026.csv
│   ├── foot_traffic_simple_forecast.csv
│   └── forecast_2025_2026.png
│
├── 00_run_cleaning.R           # Master cleaning pipeline
├── 01a-d_clean_*.R            # Individual data cleaning
├── 02_merge_to_daily.R        # Merge to daily framework
├── 03_descriptive_analysis.R  # EDA and visualizations
├── 04_feature_engineering.R   # Advanced feature creation
├── 05a_predict_tfl.R          # TfL forecasting
├── 05b_predict_tourism.R      # Tourism forecasting
├── 06a_model_xgboost.R        # XGBoost training
├── 06b_model_timeseries.R     # Prophet training
├── 06c_model_comparison.R     # Model evaluation
├── 07_generate_predictions.R  # 2025-2026 forecasts
└── 08_export_for_shiny.R      # Shiny integration
```

## Usage Instructions

### 1. Initial Setup

```r
# Set working directory
setwd("/path/to/AirbnbHostGenius/foot_traffic_prediction")

# Optional: Set API keys as environment variables
Sys.setenv(OPENWEATHER_API_KEY = "your_key_here")

# Run complete pipeline (data cleaning + modeling)
source("00_run_cleaning.R")   # Clean data
source("03_descriptive_analysis.R")  # Optional: EDA
source("04_feature_engineering.R")
source("05a_predict_tfl.R")
source("05b_predict_tourism.R")
source("06a_model_xgboost.R")
source("06b_model_timeseries.R")
source("06c_model_comparison.R")
```

### 2. Generate New Predictions

```r
# Generates forecasts for 2025-2026 using best model
source("07_generate_predictions.R")

# Output: predictions/foot_traffic_simple_forecast.csv
```

### 3. Load and Use Component Predictions

```r
library(data.table)

# Load component predictions
components <- fread("outputs/component_predictions_lookup.csv") %>%
  mutate(date = as.Date(date))

# Get components for specific date
target_date <- as.Date("2025-07-15")
comps <- components %>%
  filter(date == target_date)

# Use REAL components in pricing model
# Example: Higher TfL + Tourism + Good weather = Higher demand
tfl_factor <- (comps$tfl_daily_avg_m / 10) - 0.9  # Normalize around 10M
tourism_factor <- (comps$tourism_quarterly_visits_k / 5000) - 0.9  # Normalize around 5000k
weather_factor <- comps$weather_quality - 0.6  # Normalize around 0.6

adjustment <- 0.3 * tfl_factor + 0.3 * tourism_factor + 0.2 * weather_factor
multiplier <- 1 + adjustment * 0.2  # Max ±20% adjustment
adjusted_price <- base_price * multiplier
```

## API Configuration

### Weather API (OpenWeatherMap)

1. Sign up at: https://openweathermap.org/api
2. Get free API key (1000 calls/day)
3. Set environment variable:
   ```r
   Sys.setenv(OPENWEATHER_API_KEY = "your_key_here")
   ```
4. Fallback: Uses historical averages if API unavailable

### Holidays API (Gov.uk)

- No API key required
- Endpoint: https://www.gov.uk/bank-holidays.json
- Automatic fallback to cached CSV data

## Model Outputs

### Files Generated

1. **Component Models** (`outputs/models/`)
   - `tfl_arima_model.rds`: TfL transport ARIMA model
   - `tourism_arima_model.rds`: Tourism ARIMA model
   - `tourism_recovery_params.rds`: Tourism recovery parameters

2. **Component Predictions** (`outputs/`)
   - `tfl_daily_forecast.csv`: TfL daily journey predictions
   - `tourism_daily_forecast.csv`: Tourism quarterly predictions (expanded to daily)
   - `prophet_predictions.csv`: Prophet train/test predictions
   - `ensemble_predictions.csv`: Combined predictions
   - `tfl_predictions_complete.csv`: TfL forecasts
   - `tourism_predictions_complete.csv`: Tourism forecasts

3. **Metrics** (`outputs/`)
   - `model_comparison_metrics.csv`: R², MAE, RMSE, MAPE
   - `xgboost_feature_importance.csv`: Feature importance scores

4. **Visualizations** (`outputs/plots/`)
   - 30+ PNG charts covering EDA, model evaluation, forecasts

## Performance Metrics

### Evaluation Metrics

- **R² (Coefficient of Determination)**: Proportion of variance explained
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **RMSE (Root Mean Square Error)**: Penalizes large errors
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error

### Typical Performance

Based on historical data (exact numbers in `model_comparison_metrics.csv`):
- R² > 0.80: Excellent predictive power
- MAPE < 10%: High accuracy for practical use
- Best features: Lagged foot traffic, TfL journeys, weather quality

## Limitations

1. **COVID-19 Impact**: Tourism data incomplete after 2019
   - Model accounts for recovery using 2022-2023 adjustment ratio
   
2. **Lagged Features**: Future predictions use historical averages
   - Cannot compute actual lags without observed foot traffic
   
3. **API Dependencies**: 
   - Weather forecast limited to 5-7 days with free tier
   - Falls back to historical patterns beyond API range

4. **Geographic Scope**: London-specific model
   - May not generalize to other UK cities without retraining

## Maintenance

### Regular Updates

**Monthly:**
- Re-run `00_run_cleaning.R` to incorporate new data
- Check API responses for changes

**Quarterly:**
- Retrain models with `06a_model_xgboost.R` and `06b_model_timeseries.R`
- Update tourism predictions as new data becomes available

**Annually:**
- Review feature engineering (`04_feature_engineering.R`)
- Update holiday data source
- Perform full model evaluation

### Troubleshooting

**Issue: API calls failing**
- Check internet connection
- Verify API key is set correctly
- Check API rate limits
- Fallback mode activates automatically

**Issue: Poor predictions**
- Check data quality in `foot_traffic_data/cleaned/`
- Verify all component models (TfL, Tourism) are working
- Review feature correlations in descriptive analysis
- Consider retraining with updated hyperparameters

**Issue: Missing data**
- Run `00_run_cleaning.R` to regenerate all cleaned files
- Check raw data files are present and correctly formatted

## Integration with Airbnb Pricing

**Use REAL components directly** - no synthetic scores:

```r
# Example pricing adjustment using REAL components
base_price <- 100  # Base nightly rate

# Get component predictions for target date
components <- get_components(target_date, component_lookup)

# Create adjustment based on REAL data
# Higher TfL + Tourism + Good weather = Higher demand
tfl_norm <- (components$tfl - 9) / 2  # Center around 9M
tourism_norm <- (components$tourism - 5000) / 1000  # Center around 5000k
weather_norm <- components$weather_quality - 0.6  # Center around 0.6

# Weighted combination (adjust weights based on your data analysis)
adjustment <- 0.3 * tfl_norm + 0.3 * tourism_norm + 0.2 * weather_norm
multiplier <- 1 + adjustment * 0.2  # Max ±20% adjustment
multiplier <- max(0.8, min(1.2, multiplier))  # Clamp to reasonable range

adjusted_price <- base_price * multiplier

# Combine with other factors (seasonality, events, occupancy)
final_price <- adjusted_price * seasonal_factor * occupancy_factor
```

**Note**: You should create your own pricing adjustment logic based on:
1. Your historical Airbnb data (price, occupancy)
2. Correlation analysis between components and demand
3. Domain knowledge of your market

## References

- **XGBoost**: https://xgboost.readthedocs.io/
- **Prophet**: https://facebook.github.io/prophet/
- **TfL Data**: https://tfl.gov.uk/corporate/publications-and-reports/travel-in-london-reports
- **Tourism Data**: https://www.visitbritain.org/
- **Weather API**: https://openweathermap.org/
- **UK Holidays**: https://www.gov.uk/bank-holidays

## Support

For questions or issues:
1. Review this documentation
2. Check log outputs from R scripts
3. Examine `outputs/plots/00_ANALYSIS_SUMMARY.txt` for data insights
4. Verify all dependencies are installed (tidyverse, xgboost, prophet, etc.)

---

**Last Updated**: [Auto-generated on script run]
**Model Version**: 1.0
**Data Coverage**: 2019-2026

