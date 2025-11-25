# R Pure Models

This folder contains the core training scripts and models for price prediction.

## Files

### Training Scripts
- `train_xgb.R` - Train XGBoost model
- `train_nn.R` - Train Neural Network model (using Python PyTorch)
- `train_stacking.R` - Train Stacking model (combines XGBoost + NN)

### Data
- `nn_price_training_v4.csv` - Training dataset

### Models (generated after training)
- `best_xgb_log_model.xgb` - XGBoost model
- `scaler_xgb.rds` - XGBoost feature scaler
- `best_price_A2_log_pytorch.pt` - Neural Network model
- `best_price_A2_log_pytorch_meta.pt` - NN model metadata
- `scaler_price_pytorch.rds` - NN feature scaler
- `meta_ridge_model.rds` - Stacking meta model (Ridge)
- `meta_ridge_cv.rds` - Stacking cross-validation results
- `stacking_info.rds` - Stacking formula and metrics

### Utilities
- `verify_paths.R` - Verify all paths and files are accessible

## Usage

1. **Verify paths first:**
   ```r
   source("verify_paths.R")
   ```

2. **Train XGBoost:**
   ```r
   source("train_xgb.R")
   ```

3. **Train Neural Network:**
   ```r
   source("train_nn.R")
   ```

4. **Train Stacking (requires XGBoost and NN models):**
   ```r
   source("train_stacking.R")
   ```

## Notes

- All models use **log transformation** for price (`log1p` / `expm1`)
- Data is automatically cleaned (outlier removal)
- Models are saved automatically after training
- Stacking automatically uses improved NN model if available
