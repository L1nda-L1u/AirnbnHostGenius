import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import VotingRegressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings("ignore")

print("Loading data...")
df = pd.read_csv("nn_price_training_v4.csv")

# ------------------------------
# 1. Outlier Removal
# ------------------------------
print("Cleaning outliers...")

# rule: remove price > 400 if accommodates <= 2
if "accommodates" in df.columns:
    df = df[~((df["accommodates"] <= 2) & (df["price_num"] > 400))]

# remove global outliers (top 0.5%)
upper = df["price_num"].quantile(0.995)
df = df[df["price_num"] < upper]

df = df.reset_index(drop=True)

# ------------------------------
# 2. Train/Test Split
# ------------------------------
target_col = "price_num"
feature_cols = [c for c in df.columns if c != target_col]

X = df[feature_cols].values.astype(np.float32)
y = df[target_col].values.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------
# 3. Define 3 Strong Tree Models
# ------------------------------
print("Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
)

print("Training LightGBM...")
lgb_model = lgb.LGBMRegressor(
    n_estimators=700,
    learning_rate=0.03,
    max_depth=-1,
    num_leaves=64,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    reg_lambda=1.0,
    random_state=42,
)

print("Training CatBoost...")
cat_model = CatBoostRegressor(
    iterations=700,
    learning_rate=0.03,
    depth=8,
    loss_function="MAE",
    eval_metric="MAE",
    verbose=False,
    random_seed=42,
)

# ------------------------------
# 4. Train Each Model
# ------------------------------
xgb_model.fit(X_train, y_train)
lgb_model.fit(X_train, y_train)
cat_model.fit(X_train, y_train)

# ------------------------------
# 5. Individual Model Results
# ------------------------------
def evaluate(model, name):
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    rmse = np.sqrt(((pred - y_test) ** 2).mean())
    r2 = r2_score(y_test, pred)

    print(f"\n===== {name} RESULTS =====")
    print(f"MAE:  {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²:   {r2:.4f}")

    return pred, mae, rmse, r2

pred_xgb, _, _, _ = evaluate(xgb_model, "XGBoost")
pred_lgb, _, _, _ = evaluate(lgb_model, "LightGBM")
pred_cat, _, _, _ = evaluate(cat_model, "CatBoost")

# ------------------------------
# 6. Ensemble (VotingRegressor)
# ------------------------------
print("\nTraining Ensemble (XGB + LGB + CAT)...")

ensemble = VotingRegressor(
    estimators=[
        ("xgb", xgb_model),
        ("lgb", lgb_model),
        ("cat", cat_model)
    ],
    weights=[1.0, 1.0, 1.2]  # catboost usually best on MAE
)

ensemble.fit(X_train, y_train)
pred_ens = ensemble.predict(X_test)

mae_ens = mean_absolute_error(y_test, pred_ens)
rmse_ens = np.sqrt(((pred_ens - y_test) ** 2).mean())
r2_ens = r2_score(y_test, pred_ens)

print("\n===== ENSEMBLE RESULTS (Best) =====")
print(f"MAE:  {mae_ens:.4f}")
print(f"RMSE: {rmse_ens:.4f}")
print(f"R²:   {r2_ens:.4f}")

# ------------------------------
# 7. Save best model
# ------------------------------
import pickle

with open("best_tree_ensemble.pkl", "wb") as f:
    pickle.dump(ensemble, f)

with open("scaler_tree.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nSaved: best_tree_ensemble.pkl + scaler_tree.pkl")
print("Done.")


