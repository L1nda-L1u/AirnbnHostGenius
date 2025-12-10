========================================
Best Model Package - For Sharing
========================================

This folder contains the best stacking model for Airbnb price prediction.

Files:
--------
1. nn_price_training_v4.csv
   - Training dataset (70 features)
   - Used to train XGBoost and Neural Network models

2. xgb_model.json
   - XGBoost model (trained on log(price))
   - Can be loaded in R using: xgb.load("xgb_model.json")
   - Or in Python using: xgb.Booster(model_file="xgb_model.json")

3. nn.onnx
   - Neural Network model (PyTorch exported to ONNX)
   - Can be loaded in R using ONNX package
   - Or in Python using onnxruntime

4. scaler_xgb.pkl
   - StandardScaler for XGBoost features
   - Used to normalize input features before XGBoost prediction

5. scaler_price.pkl
   - StandardScaler for Neural Network features
   - Used to normalize input features before NN prediction

6. meta_ridge_model.pkl
   - Ridge regression stacking model
   - Combines XGBoost and Neural Network predictions
   - Formula: final_price = intercept + xgb_weight * xgb_pred + nn_weight * nn_pred

7. README.txt
   - Stacking formula with coefficients

========================================
Model Usage:
========================================

Python:
  - Load models from this folder
  - Use scalers to preprocess features
  - Get predictions from XGBoost and NN
  - Combine using stacking formula

R:
  - Load xgb_model.json using xgboost package
  - Load nn.onnx using ONNX package (or use reticulate)
  - Apply stacking formula from README.txt

========================================

