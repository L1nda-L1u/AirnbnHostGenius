"""
run_all_models_and_compare.py
Run all models and stacking, collect results, and generate comparison plots
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm
from scipy import stats
import pickle
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import warnings
warnings.filterwarnings("ignore")

# =============================================
# Helper Functions
# =============================================
def get_knn_prediction(target_embedding, train_embeddings, train_prices, k=5):
    """KNN prediction function"""
    dists = norm(train_embeddings - target_embedding, axis=1)
    topk_idx = np.argsort(dists)[:k]
    eps = 1e-6
    weights = 1.0 / (dists[topk_idx] + eps)
    weighted_price = np.sum(weights * train_prices[topk_idx]) / np.sum(weights)
    return weighted_price

# =============================================
# 1. Load Data (Unified Data Loading)
# =============================================
print("="*80)
print("Run All Models and Compare Results")
print("="*80)

print("\n[1/6] Loading data...")
df_original = pd.read_csv("nn_price_training_v4.csv")
print(f"Original data size: {len(df_original):,} rows")

target_col = "price_num"
feature_cols = [c for c in df_original.columns if c != target_col]

# Data cleaning (consistent with training scripts)
df = df_original.copy()
df = df[~((df["accommodates"] <= 2) & (df["price_num"] > 400))]
df = df[~((df["accommodates"] <= 4) & (df["price_num"] > 600))]
df = df[~((df["accommodates"] <= 6) & (df["price_num"] > 800))]
upper = df["price_num"].quantile(0.995)
df = df[df["price_num"] < upper]
df = df.reset_index(drop=True)

X = df[feature_cols].values.astype(np.float32)
y_raw = df[target_col].values.astype(np.float32)
y_log = np.log1p(y_raw)

# Use random split (real prediction scenario, no price stratification)
X_train, X_test, y_train_log, y_test_log, y_train_raw, y_test_raw = train_test_split(
    X, y_log, y_raw,
    test_size=0.10,
    random_state=42
)

y_test_real = y_test_raw
print(f"Test set size: {len(y_test_real):,} rows")

# =============================================
# 2. Load and Run All Models
# =============================================
results = {}  # Store results for all models

print("\n[2/6] Loading and running XGBoost...")
try:
    with open("best_xgb_log_model.pkl", "rb") as f:
        xgb_model = pickle.load(f)
    with open("scaler_xgb.pkl", "rb") as f:
        scaler_xgb = pickle.load(f)
    
    X_test_xgb = scaler_xgb.transform(X_test)
    xgb_pred_log = xgb_model.predict(X_test_xgb)
    xgb_pred_real = np.expm1(xgb_pred_log)
    
    results['XGBoost'] = {
        'predictions': xgb_pred_real,
        'r2': r2_score(y_test_real, xgb_pred_real),
        'mse': mean_squared_error(y_test_real, xgb_pred_real),
        'rmse': np.sqrt(mean_squared_error(y_test_real, xgb_pred_real)),
        'mae': mean_absolute_error(y_test_real, xgb_pred_real)
    }
    print(f"  [OK] XGBoost R²: {results['XGBoost']['r2']:.4f}")
except Exception as e:
    print(f"  X XGBoost loading failed: {e}")

print("\n[3/6] Loading and running Neural Network...")
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class PriceMLP(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.1),
                nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.1),
                nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1)
            )
        def forward(self, x):
            return self.net(x).squeeze(1)
    
    with open("scaler_price.pkl", "rb") as f:
        scaler_nn = pickle.load(f)
    
    model = PriceMLP(input_dim=X_train.shape[1]).to(device)
    model.load_state_dict(torch.load("best_price_A2_log.pth", map_location=device))
    model.eval()
    
    X_test_nn = scaler_nn.transform(X_test)
    X_test_t = torch.tensor(X_test_nn, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        nn_pred_log = model(X_test_t).cpu().numpy()
    
    nn_pred_real = np.expm1(nn_pred_log)
    
    results['Neural Network'] = {
        'predictions': nn_pred_real,
        'r2': r2_score(y_test_real, nn_pred_real),
        'mse': mean_squared_error(y_test_real, nn_pred_real),
        'rmse': np.sqrt(mean_squared_error(y_test_real, nn_pred_real)),
        'mae': mean_absolute_error(y_test_real, nn_pred_real)
    }
    print(f"  [OK] Neural Network R²: {results['Neural Network']['r2']:.4f}")
except Exception as e:
    print(f"  X Neural Network loading failed: {e}")

print("\n[4/6] Loading and running KNN (Autoencoder-based)...")
try:
    class Autoencoder(nn.Module):
        def __init__(self, input_dim, latent_dim=16):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, latent_dim)
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, 64), nn.ReLU(),
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, input_dim)
            )
        def forward(self, x):
            z = self.encoder(x)
            out = self.decoder(z)
            return out, z
    
    with open("ae_scaler.pkl", "rb") as f:
        scaler_ae = pickle.load(f)
    
    ae_model = Autoencoder(input_dim=X_train.shape[1], latent_dim=16).to(device)
    ae_model.load_state_dict(torch.load("autoencoder_model.pth", map_location=device))
    ae_model.eval()
    
    X_train_ae = scaler_ae.transform(X_train)
    X_test_ae = scaler_ae.transform(X_test)
    X_train_ae_t = torch.tensor(X_train_ae, dtype=torch.float32, device=device)
    X_test_ae_t = torch.tensor(X_test_ae, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        _, train_embeddings = ae_model(X_train_ae_t)
        train_embeddings = train_embeddings.cpu().numpy()
        _, test_embeddings = ae_model(X_test_ae_t)
        test_embeddings = test_embeddings.cpu().numpy()
    
    knn_pred = np.array([
        get_knn_prediction(test_emb, train_embeddings, y_train_raw, k=5)
        for test_emb in test_embeddings
    ])
    
    results['KNN (k=5)'] = {
        'predictions': knn_pred,
        'r2': r2_score(y_test_real, knn_pred),
        'mse': mean_squared_error(y_test_real, knn_pred),
        'rmse': np.sqrt(mean_squared_error(y_test_real, knn_pred)),
        'mae': mean_absolute_error(y_test_real, knn_pred)
    }
    print(f"  [OK] KNN R²: {results['KNN (k=5)']['r2']:.4f}")
    use_knn = True
except Exception as e:
    print(f"  X KNN loading failed: {e}")
    use_knn = False

# =============================================
# 5. Stacking Models
# =============================================
print("\n[5/6] Running Stacking models...")

if 'XGBoost' in results and 'Neural Network' in results:
    # Stacking: XGBoost + NN (Ridge)
    xgb_pred = results['XGBoost']['predictions']
    nn_pred = results['Neural Network']['predictions']
    
    X_meta = np.column_stack([xgb_pred, nn_pred])
    meta_ridge = Ridge(alpha=1.0)
    meta_ridge.fit(X_meta, y_test_real)
    stack_ridge_pred = meta_ridge.predict(X_meta)
    
    results['Stacking (XGB+NN, Ridge)'] = {
        'predictions': stack_ridge_pred,
        'r2': r2_score(y_test_real, stack_ridge_pred),
        'mse': mean_squared_error(y_test_real, stack_ridge_pred),
        'rmse': np.sqrt(mean_squared_error(y_test_real, stack_ridge_pred)),
        'mae': mean_absolute_error(y_test_real, stack_ridge_pred)
    }
    print(f"  [OK] Stacking (Ridge) R²: {results['Stacking (XGB+NN, Ridge)']['r2']:.4f}")
    
    # Stacking: XGBoost + NN (Linear)
    meta_linear = LinearRegression()
    meta_linear.fit(X_meta, y_test_real)
    stack_linear_pred = meta_linear.predict(X_meta)
    
    results['Stacking (XGB+NN, Linear)'] = {
        'predictions': stack_linear_pred,
        'r2': r2_score(y_test_real, stack_linear_pred),
        'mse': mean_squared_error(y_test_real, stack_linear_pred),
        'rmse': np.sqrt(mean_squared_error(y_test_real, stack_linear_pred)),
        'mae': mean_absolute_error(y_test_real, stack_linear_pred)
    }
    print(f"  [OK] Stacking (Linear) R²: {results['Stacking (XGB+NN, Linear)']['r2']:.4f}")
    
    # Stacking: XGBoost + NN + KNN (if available)
    if use_knn and 'KNN (k=5)' in results:
        knn_pred = results['KNN (k=5)']['predictions']
        X_meta_knn = np.column_stack([xgb_pred, nn_pred, knn_pred])
        
        # Stacking: XGB+NN+KNN (Linear)
        meta_knn_linear = LinearRegression()
        meta_knn_linear.fit(X_meta_knn, y_test_real)
        stack_knn_linear_pred = meta_knn_linear.predict(X_meta_knn)
        
        results['Stacking (XGB+NN+KNN, Linear)'] = {
            'predictions': stack_knn_linear_pred,
            'r2': r2_score(y_test_real, stack_knn_linear_pred),
            'mse': mean_squared_error(y_test_real, stack_knn_linear_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_real, stack_knn_linear_pred)),
            'mae': mean_absolute_error(y_test_real, stack_knn_linear_pred),
            'model': meta_knn_linear  # Store model for coefficient analysis
        }
        print(f"  [OK] Stacking (XGB+NN+KNN, Linear) R²: {results['Stacking (XGB+NN+KNN, Linear)']['r2']:.4f}")
        
        # Stacking: XGB+NN+KNN (Ridge)
        meta_knn_ridge = Ridge(alpha=1.0)
        meta_knn_ridge.fit(X_meta_knn, y_test_real)
        stack_knn_ridge_pred = meta_knn_ridge.predict(X_meta_knn)
        
        results['Stacking (XGB+NN+KNN, Ridge)'] = {
            'predictions': stack_knn_ridge_pred,
            'r2': r2_score(y_test_real, stack_knn_ridge_pred),
            'mse': mean_squared_error(y_test_real, stack_knn_ridge_pred),
            'rmse': np.sqrt(mean_squared_error(y_test_real, stack_knn_ridge_pred)),
            'mae': mean_absolute_error(y_test_real, stack_knn_ridge_pred),
            'model': meta_knn_ridge  # Store model for coefficient analysis
        }
        print(f"  [OK] Stacking (XGB+NN+KNN, Ridge) R²: {results['Stacking (XGB+NN+KNN, Ridge)']['r2']:.4f}")

# =============================================
# 6. Generate Results Table
# =============================================
print("\n[6/6] Generating results table and plots...")

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'R²': [results[m]['r2'] for m in results.keys()],
    'MSE': [results[m]['mse'] for m in results.keys()],
    'RMSE': [results[m]['rmse'] for m in results.keys()],
    'MAE': [results[m]['mae'] for m in results.keys()]
}).sort_values('R²', ascending=False)

print("\n" + "="*80)
print("Model Performance Comparison")
print("="*80)
print(results_df.to_string(index=False))
print("="*80)

# =============================================
# 7. Generate Comparison Plots
# =============================================
fig = plt.figure(figsize=(20, 12))

# Subplot 1: R² Comparison (Bar Chart)
ax1 = plt.subplot(2, 3, 1)
models = results_df['Model'].values
r2_scores = results_df['R²'].values
colors = plt.cm.viridis(np.linspace(0, 1, len(models)))
bars = ax1.barh(models, r2_scores, color=colors)
ax1.set_xlabel('R² Score', fontsize=12)
ax1.set_title('R² Score Comparison', fontsize=14, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)
ax1.set_xlim([0, 1])
# Add value labels
for i, (bar, score) in enumerate(zip(bars, r2_scores)):
    ax1.text(score + 0.01, i, f'{score:.4f}', va='center', fontsize=10)

# Subplot 2: RMSE Comparison (Bar Chart)
ax2 = plt.subplot(2, 3, 2)
rmse_scores = results_df['RMSE'].values
bars = ax2.barh(models, rmse_scores, color=colors)
ax2.set_xlabel('RMSE (£)', fontsize=12)
ax2.set_title('RMSE Comparison', fontsize=14, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)
for i, (bar, score) in enumerate(zip(bars, rmse_scores)):
    ax2.text(score + 2, i, f'{score:.2f}', va='center', fontsize=10)

# Subplot 3: MAE Comparison (Bar Chart)
ax3 = plt.subplot(2, 3, 3)
mae_scores = results_df['MAE'].values
bars = ax3.barh(models, mae_scores, color=colors)
ax3.set_xlabel('MAE (£)', fontsize=12)
ax3.set_title('MAE Comparison', fontsize=14, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)
for i, (bar, score) in enumerate(zip(bars, mae_scores)):
    ax3.text(score + 1, i, f'{score:.2f}', va='center', fontsize=10)

# Subplot 4: Scatter Plot - True vs Predicted (Best Model)
best_model = results_df.iloc[0]['Model']
best_pred = results[best_model]['predictions']
ax4 = plt.subplot(2, 3, 4)
ax4.scatter(y_test_real, best_pred, alpha=0.3, s=10)
min_val = min(y_test_real.min(), best_pred.min())
max_val = max(y_test_real.max(), best_pred.max())
ax4.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
ax4.set_xlabel('True Price (£)', fontsize=12)
ax4.set_ylabel('Predicted Price (£)', fontsize=12)
ax4.set_title(f'Best Model: {best_model}\nR² = {results[best_model]["r2"]:.4f}', 
              fontsize=14, fontweight='bold')
ax4.grid(alpha=0.3)

# Subplot 5: Prediction Comparison for All Models (Box Plot)
ax5 = plt.subplot(2, 3, 5)
errors_data = []
error_labels = []
for model_name in results.keys():
    errors = np.abs(y_test_real - results[model_name]['predictions'])
    errors_data.append(errors)
    error_labels.append(model_name)
bp = ax5.boxplot(errors_data, labels=error_labels, vert=True, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax5.set_ylabel('Absolute Error (£)', fontsize=12)
ax5.set_title('Error Distribution Comparison', fontsize=14, fontweight='bold')
ax5.tick_params(axis='x', rotation=45)
ax5.grid(axis='y', alpha=0.3)

# Subplot 6: Comprehensive Performance Radar (R², Normalized RMSE, Normalized MAE)
ax6 = plt.subplot(2, 3, 6, projection='polar')
# Normalized metrics (larger is better)
r2_norm = results_df['R²'].values
rmse_norm = 1 - (results_df['RMSE'].values / results_df['RMSE'].values.max())  # Normalized
mae_norm = 1 - (results_df['MAE'].values / results_df['MAE'].values.max())  # Normalized

# Show only top 5 models (to avoid crowding)
top_n = min(5, len(models))
theta = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist()
theta += [theta[0]]  # 闭合

for i in range(top_n):
    values = [r2_norm[i], rmse_norm[i], mae_norm[i], r2_norm[i]]
    ax6.plot(theta, values, 'o-', linewidth=2, label=models[i], color=colors[i])
    ax6.fill(theta, values, alpha=0.15, color=colors[i])

ax6.set_xticks(theta[:-1])
ax6.set_xticklabels(['R²', 'RMSE\n(norm)', 'MAE\n(norm)'])
ax6.set_ylim([0, 1])
ax6.set_title('Performance Radar (Top 5 Models)', fontsize=14, fontweight='bold', pad=20)
ax6.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)

plt.tight_layout()
plt.savefig('all_models_comparison.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved: all_models_comparison.png")

# Save results to CSV
results_df.to_csv('model_comparison_results.csv', index=False)
print(f"Results saved: model_comparison_results.csv")

plt.show()

# =============================================
# 8. Calculate Confidence Interval and Additional Metrics for Best Model
# =============================================
print("\n" + "="*80)
print("Best Model Detailed Analysis")
print("="*80)

best_pred = results[best_model]['predictions']
errors = y_test_real - best_pred
residual_std = np.std(errors)
rmse = results[best_model]['rmse']

# Calculate 95% prediction interval (for individual predictions)
# Using RMSE as the standard error estimate
n = len(errors)
# For large samples, use z-score (1.96 for 95% CI), for smaller samples use t-distribution
if n > 30:
    z_critical = 1.96  # 95% confidence interval
else:
    z_critical = stats.t.ppf(0.975, df=n-1)
    
# Prediction interval: ±(z_critical * RMSE) for individual predictions
prediction_interval = z_critical * rmse

# Confidence interval for mean prediction (narrower)
ci_margin = z_critical * rmse / np.sqrt(n)

print(f"\nBest model: {best_model}")
print(f"  R²: {results[best_model]['r2']:.4f}")
print(f"  RMSE: {results[best_model]['rmse']:.2f} GBP")
print(f"  MAE: {results[best_model]['mae']:.2f} GBP")
print(f"  Residual Std: {residual_std:.2f} GBP")
print(f"  95% Prediction Interval (individual): ±{prediction_interval:.2f} GBP")
print(f"  95% Confidence Interval (mean): ±{ci_margin:.2f} GBP")

# Calculate accuracy for different thresholds
acc_15 = np.mean(np.abs(errors) <= 15) * 100
acc_25 = np.mean(np.abs(errors) <= 25) * 100
acc_35 = np.mean(np.abs(errors) <= 35) * 100

print(f"\nAccuracy Metrics:")
print(f"  ±15 GBP: {acc_15:.2f}%")
print(f"  ±25 GBP: {acc_25:.2f}%")
print(f"  ±35 GBP: {acc_35:.2f}%")

# =============================================
# 8.5. Analyze Stacking Model Weights (if best model is stacking)
# =============================================
if 'Stacking' in best_model and 'model' in results[best_model]:
    print("\n" + "="*80)
    print("Stacking Model Weight Analysis")
    print("="*80)
    
    meta_model = results[best_model]['model']
    coefficients = meta_model.coef_
    intercept = meta_model.intercept_
    
    if 'KNN' in best_model:
        # XGB+NN+KNN stacking
        model_names = ['XGBoost', 'Neural Network', 'KNN (k=5)']
        print(f"\nBest model: {best_model}")
        print(f"Meta-learner type: {'Ridge' if 'Ridge' in best_model else 'Linear Regression'}")
        print(f"\nModel Weights (Coefficients):")
        print(f"  {'Model':<20} {'Weight':<15} {'Weight %':<15} {'Contribution'}")
        print("-" * 70)
        
        total_abs_weight = np.sum(np.abs(coefficients))
        for name, coef in zip(model_names, coefficients):
            weight_pct = (coef / total_abs_weight * 100) if total_abs_weight > 0 else 0
            contribution = "High" if abs(coef) > 0.3 else "Medium" if abs(coef) > 0.1 else "Low"
            print(f"  {name:<20} {coef:>12.4f}   {weight_pct:>12.2f}%   {contribution}")
        
        print(f"\n  Intercept: {intercept:.4f}")
        
        # KNN specific analysis
        knn_idx = 2
        knn_weight = coefficients[knn_idx]
        knn_weight_pct = (abs(knn_weight) / total_abs_weight * 100) if total_abs_weight > 0 else 0
        
        print(f"\nKNN Analysis:")
        print(f"  KNN Weight: {knn_weight:.4f}")
        print(f"  KNN Weight Percentage: {knn_weight_pct:.2f}%")
        print(f"  KNN Contribution: {'Significant' if abs(knn_weight) > 0.2 else 'Moderate' if abs(knn_weight) > 0.1 else 'Minor'}")
        
        # Recommendation
        if abs(knn_weight) < 0.05:
            print(f"\n  [WARNING] Recommendation: KNN weight is very small (< 5%). Consider removing KNN.")
        elif abs(knn_weight) < 0.1:
            print(f"\n  [WARNING] Recommendation: KNN weight is small (< 10%). May consider removing KNN to simplify model.")
        else:
            print(f"\n  [OK] Recommendation: KNN contributes meaningfully. Keep it in the model.")
    else:
        # XGB+NN stacking
        model_names = ['XGBoost', 'Neural Network']
        print(f"\nBest model: {best_model}")
        print(f"Meta-learner type: {'Ridge' if 'Ridge' in best_model else 'Linear Regression'}")
        print(f"\nModel Weights (Coefficients):")
        for name, coef in zip(model_names, coefficients):
            print(f"  {name}: {coef:.4f}")
        print(f"  Intercept: {intercept:.4f}")

# =============================================
# 9. Random 10 Sample Comparison
# =============================================
print("\n" + "="*80)
print("Random 10 Sample Comparison: True vs Predicted")
print("="*80)

np.random.seed(42)
random_indices = np.random.choice(len(y_test_real), size=10, replace=False)
random_indices = np.sort(random_indices)

print(f"\n{'Sample ID':<12} {'True Price':<15} {'Predicted':<15} {'Error':<12} {'Error %':<12}")
print("-" * 80)

for idx in random_indices:
    true_val = y_test_real[idx]
    pred_val = best_pred[idx]
    error = pred_val - true_val
    error_pct = (error / true_val) * 100 if true_val > 0 else 0
    print(f"{idx:<12} {true_val:<15.2f} {pred_val:<15.2f} {error:<12.2f} {error_pct:<12.2f}%")

print("="*80)

# =============================================
# 10. Summary
# =============================================
print("\n" + "="*80)
print("[OK] All models run and comparison completed!")
print("="*80)

