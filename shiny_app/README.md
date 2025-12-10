# Airbnb Baseline Price Predictor - Shiny App

A beautiful Shiny application for predicting Airbnb baseline prices with amenity recommendations.

## Quick Start

### For Your Friend - How to Open the App

1. **Navigate to the project root folder** (where `shiny_app` folder is located):
   ```r
   # Make sure you're in the project root (AirbnbHostGeniusR)
   getwd()
   ```

2. **Launch the app** (choose one method):

   **Method 1 - Simple (Recommended)**:
   ```r
   source("shiny_app/run_app.R")
   ```

   **Method 2 - Direct**:
   ```r
   shiny::runApp("shiny_app")
   ```

   **Method 3 - From inside shiny_app folder**:
   ```r
   setwd("shiny_app")
   shiny::runApp("app.R")
   ```

3. **The app will open in your browser automatically!**

## Features

- 🏠 **Property Information Input**: Bedrooms, bathrooms, accommodates, etc.
- 📍 **Auto Geocoding**: Automatically converts address/postcode to coordinates
- ✨ **Amenity Selection**: Rich amenity options (WiFi, Kitchen, Washer, etc.)
- 💰 **Smart Price Prediction**: Based on Stacking model (XGBoost + Neural Network)
- 📈 **Occupancy & Revenue Prediction**: Placeholders for future model integration
- 💡 **Amenity Recommendations**: Shows top 3 amenities that can increase price
- 🗺️ **Location Map**: Displays property location
- 🎨 **Beautiful UI**: Cyan/gray theme with low saturation, fresh design

## Installation Requirements

### R Packages

The `run_app.R` script will automatically install missing packages, or install manually:

```r
install.packages(c(
  "shiny", "shinydashboard", "DT", "leaflet", "plotly",
  "dplyr", "geosphere", "xgboost", "reticulate", "glmnet",
  "httr", "jsonlite", "caret"
))
```

### Python (Optional, for Neural Network model)

If you want to use the Neural Network model:

```r
library(reticulate)
py_install("torch", pip = TRUE)
```

The app will work with XGBoost-only mode if Python/PyTorch is not available.

### Model Files

Ensure these model files exist in `baseline_price_predict/baseprice_model/` directory:

- `best_xgb_log_model.xgb` - XGBoost model
- `best_price_A2_log_pytorch.pt` - Neural Network model (optional)
- `scaler_xgb.rds` - XGBoost feature scaler
- `scaler_price_pytorch.rds` - Neural Network feature scaler (optional)
- `meta_ridge_model.rds` - Stacking meta model
- `meta_ridge_cv.rds` - Stacking cross-validation results
- `nn_price_training_v4.csv` - Training data (for feature reference)

## How to Use

1. Enter address or postcode (e.g., "London, UK" or "SW1A 1AA")
2. Wait for automatic geocoding (shows location status)
3. Fill in property information:
   - Bedrooms, Bathrooms
   - Accommodates, Beds
   - Room Type
   - Select Amenities
4. Click "🚀 Predict Price"
5. View results:
   - Baseline Price (middle column)
   - Amenity Recommendations (right column)
   - Location Map (right column, bottom)
   - Occupancy & Revenue (middle column, pending model integration)

## File Structure

```
shiny_app/
├── app.R                  # Main application file
├── run_app.R              # Launch script (recommended)
├── model_loader.R         # Model loading functions
├── geocoding.R            # Address to coordinates conversion
├── feature_builder.R      # Feature construction
├── sensitivity_helper.R   # Amenity recommendation functions
├── init_python.R          # Python initialization (optional)
└── README.md              # This file
```

## Technical Details

### Model Architecture

- **XGBoost**: Gradient boosting tree model
- **Neural Network**: Multi-layer perceptron (MLP) with PyTorch
- **Stacking**: Ridge regression to combine both models

### Features

The model uses 49 features including:
- Location (latitude, longitude, neighbourhood, cluster)
- Property attributes (bedrooms, bathrooms, accommodates, etc.)
- Amenities (one-hot encoded)
- Review scores (cleanliness, location)
- Cluster price statistics

### Auto-Detection

The app automatically:
- Detects neighbourhood from coordinates
- Assigns location cluster based on nearest cluster center
- Calculates cluster price statistics
- Recommends amenities that can increase price

## Troubleshooting

### App won't start

1. Check you're in the project root directory (where `shiny_app` folder is located)
2. Make sure all R packages are installed
3. Try: `source("shiny_app/run_app.R")`

### Model loading fails

1. Check model files exist in `baseline_price_predict/baseprice_model/` directory
2. App will work with XGBoost-only mode if Python/PyTorch is unavailable
3. Check console for detailed error messages

### Address lookup fails

1. Check internet connection
2. Try a more detailed address
3. Check Nominatim API availability

### Prediction fails

1. Ensure all required fields are filled
2. Make sure coordinates are obtained (check geocoding status)
3. Check console for error details

## License

MIT License
