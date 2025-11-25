# Foot Traffic Prediction Pipeline - File Structure

## Complete Pipeline Order

### Step 1: Data Cleaning (01a-01d)
- `01a_clean_tfl.R` - Clean TfL transport data
- `01b_clean_tourism.R` - Clean international tourism data  
- `01c_clean_weather.R` - Clean weather data + integrate Open-Meteo API forecast
- `01d_clean_holidays.R` - Clean UK public holidays data

### Step 2: Data Merging (02)
- `02_merge_to_daily.R` - Merge all sources to daily granularity

### Step 3: Component Predictions with Validation (03a-03c) ⭐ NEW
- `03a_predict_tfl.R` - **Train/Validate/Predict** TfL data (ARIMA model)
- `03b_predict_tourism.R` - **Train/Validate/Predict** Tourism data
- `03c_predict_weather.R` - **Validate/Fetch** Weather forecasts (Open-Meteo API)

### Step 4: Fill Missing Historical Data (04)
- `04_fill_missing_data.R` - Use predictions to fill 2024 gaps

### Step 5: Descriptive Analytics (05)
- `05_descriptive_analysis.R` - Explore patterns and correlations

### Step 6: Feature Engineering (06)
- `06_feature_engineering.R` - Create advanced features for modeling

### Step 7: Model Training & Comparison (07a-07c)
- `07a_model_xgboost.R` - Train XGBoost model
- `07b_model_timeseries.R` - Train Prophet model
- `07c_model_comparison.R` - Compare models and create ensemble

### Step 8: Generate Future Predictions (08)
- `08_generate_predictions.R` - Generate 2025-2026 forecasts

### Step 9: Export for Shiny (09)
- `09_export_for_shiny.R` - Prepare data/models for web app

---

## Key Changes from Previous Version

1. **Added Validation**: 03a, 03b, 03c now include train/validation splits
2. **Unified Format**: All prediction scripts (03a-03c) follow same structure
3. **Weather API**: Switched to Open-Meteo (free, 16-day forecast, no API key)
4. **Renumbered**: Files 04-09 shifted to accommodate new 03c

---

## Master Execution Scripts

- `00_run_all.R` - Run complete pipeline (Steps 1-9)
- `00_run_cleaning.R` - Run only data cleaning (Steps 1-2)
- `00_install_packages.R` - Install required R packages

---

## API Integration

### apis/weather_api.R
- Open-Meteo API integration
- 16-day forecast (free, no key required)
- Automatic fallback to seasonal averages
- Attribution: Weather data by Open-Meteo.com

### apis/holidays_api.R
- UK Government holidays API
- Fetches future bank holidays

---

## Testing Scripts

- `test_open_meteo.R` - Test weather API integration

---

## Output Directories

- `outputs/plots/` - All visualizations
- `outputs/models/` - Trained models (.rds files)
- `outputs/` - Prediction CSVs and ensemble configs
- `foot_traffic_data/cleaned/` - Cleaned data
- `foot_traffic_data/api_cache/` - API forecast cache

