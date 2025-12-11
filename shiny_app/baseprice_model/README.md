# R Pure Models

This folder contains the core training scripts and models for price prediction.

## Files

### Training Scripts
- `train_xgb.R` - Train XGBoost model

### Data
- `nn_price_training_v4.csv` - Training dataset

### Models (generated after training)
- `best_xgb_log_model.xgb` - XGBoost model
- `scaler_xgb.rds` - XGBoost feature scaler

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

3. (Optional) Extend workflow: add your own scripts if you need extra models; default app uses XGBoost only.

## Notes

- All models use **log transformation** for price (`log1p` / `expm1`)
- Data is automatically cleaned (outlier removal)
- Models are saved automatically after training
*** XGBoost-only mode *** (legacy NN/stacking files removed)
