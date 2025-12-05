Stacking Formula:
final_price = 1.7482 + 0.2381 * xgb_pred + 0.7743 * nn_pred

========================================
How to Use in R:
========================================

# Quick start (recommended):
# 在RStudio中打开 R_scripts/AirbnbHostGeniusR.Rproj
# 然后运行:
source("best_model/load_and_predict.R")
# Then use: predict_price(feature_vector)

# Or manually:
# 1. Load libraries
library(reticulate)

# 2. Load models
py_run_string("
import pickle, torch, torch.nn as nn, numpy as np, pandas as pd, xgboost as xgb
import os
script_dir = os.path.join('best_model')
xgb_model = xgb.Booster()
xgb_model.load_model(os.path.join(script_dir, 'xgb_model.json'))
df = pd.read_csv(os.path.join(script_dir, 'nn_price_training_v4.csv'), nrows=1)
input_dim = len([c for c in df.columns if c != 'price_num'])
class PriceMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.1),
                                 nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.1),
                                 nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1))
    def forward(self, x): return self.net(x).squeeze(1)
nn_model = PriceMLP(input_dim).to('cpu')
nn_model.load_state_dict(torch.load('../../other_files/best_price_A2_log.pth', map_location='cpu'))
nn_model.eval()
with open(os.path.join(script_dir, 'scaler_xgb.pkl'), 'rb') as f: scaler_xgb = pickle.load(f)
with open(os.path.join(script_dir, 'scaler_price.pkl'), 'rb') as f: scaler_nn = pickle.load(f)
with open(os.path.join(script_dir, 'meta_ridge_model.pkl'), 'rb') as f: meta_ridge = pickle.load(f)
")

# 3. Predict function
predict_price <- function(feature_vector) {
  py_run_string(sprintf("
X = np.array([%s], dtype=np.float32).reshape(1, -1)
xgb_pred = np.expm1(xgb_model.predict(scaler_xgb.transform(X))[0])
X_nn = torch.tensor(scaler_nn.transform(X), dtype=torch.float32)
with torch.no_grad(): nn_pred = np.expm1(nn_model(X_nn).cpu().numpy()[0])
final_price = meta_ridge.predict(np.array([[xgb_pred, nn_pred]]))[0]
", paste(feature_vector, collapse = ", ")))
  return(py$final_price)
}

# 4. Use it
df <- read.csv("best_model/nn_price_training_v4.csv", nrows = 1)
features <- as.numeric(df[, setdiff(colnames(df), "price_num")])
price <- predict_price(features)