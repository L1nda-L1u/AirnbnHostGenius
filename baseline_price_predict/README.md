# Baseline Price Prediction

This folder contains the baseline price prediction models for Airbnb listings in London.

## Structure

```
baseline_price_predict/
├── baseprice_model/          # Core model training scripts and saved models
│   ├── train_xgb.R           # XGBoost model training
│   ├── train_nn.R            # Neural Network model training (PyTorch)
│   ├── train_stacking.R      # Stacking model (combines XGBoost + NN)
│   ├── verify_paths.R        # Path verification utility
│   ├── nn_price_training_v4.csv  # Training dataset
│   └── [model files]        # Trained model files (.xgb, .pt, .rds)
└── README.md                 # This file
```

## Models

### 1. XGBoost Model
- **Script**: `baseprice_model/train_xgb.R`
- **Model file**: `best_xgb_log_model.xgb`
- **Scaler**: `scaler_xgb.rds`
- **Features**: Property characteristics, location, amenities, etc.
- **Target**: Price (log-transformed: `log1p(price)`)

### 2. Neural Network Model
- **Script**: `baseprice_model/train_nn.R`
- **Model file**: `best_price_A2_log_pytorch.pt`
- **Scaler**: `scaler_price_pytorch.rds`
- **Architecture**: Multi-layer perceptron with BatchNorm and Dropout
- **Implementation**: PyTorch (via R `reticulate`)

### 3. Stacking Model
- **Script**: `baseprice_model/train_stacking.R`
- **Meta-model**: Ridge regression
- **Base models**: XGBoost + Neural Network
- **Model files**: `meta_ridge_model.rds`, `meta_ridge_cv.rds`, `stacking_info.rds`

## Usage

### Prerequisites

1. **R packages**:
   ```r
   install.packages(c("xgboost", "caret", "glmnet", "dplyr", "reticulate"))
   ```

2. **Python environment** (for Neural Network):
   ```r
   library(reticulate)
   # Configure Python if needed
   py_install("torch", pip = TRUE)
   ```

### Training Models

1. **Verify paths first**:
   ```r
   setwd("baseline_price_predict/baseprice_model")
   source("verify_paths.R")
   ```

2. **Train XGBoost**:
   ```r
   source("train_xgb.R")
   ```

3. **Train Neural Network**:
   ```r
   source("train_nn.R")
   ```

4. **Train Stacking** (requires XGBoost and NN models):
   ```r
   source("train_stacking.R")
   ```

### Model Performance

The stacking model typically achieves:
- **MAE**: ~35-40 £
- **RMSE**: ~50-55 £
- **R²**: ~0.75-0.80
- **Accuracy (±15)**: ~45-50%
- **Accuracy (±25)**: ~60-65%

## Data

- **Training dataset**: `nn_price_training_v4.csv`
- **Features**: Property characteristics, location clusters, amenities, etc.
- **Target**: `price_num` (price in £)
- **Preprocessing**: 
  - Log transformation: `log1p(price)` for training
  - Feature scaling: StandardScaler
  - Outlier removal based on price and accommodates

## Key Features

- **Log transformation**: All models use `log1p(price)` for training stability
- **Location clustering**: K-means clustering of latitude/longitude
- **Cluster price features**: Aggregated price statistics by location cluster
- **Automatic data cleaning**: Outlier removal and missing value handling

## Notes

- Models are saved automatically after training
- Stacking automatically uses improved NN model if available
- All paths are relative to `baseprice_model/` directory
- Large model files (`.xgb`, `.pt`, `.rds`) are excluded from git (see `.gitignore`)

## Integration

This baseline price prediction model can be integrated with:
- **TFL/Weather/Tourism/Holiday predictions**: Use predicted external factors as additional features
- **Shiny app**: Load models and make predictions for new listings
- **Occupancy prediction**: Price and occupancy are related (revenue = price × occupancy)

## Related Work

- See `tfl,weather,tourism,holiday_prediction/` for external factor predictions
- See `legacy_files/` for older model versions and experimental code
