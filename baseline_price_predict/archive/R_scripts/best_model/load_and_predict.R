# =============================================
# load_and_predict.R
# 加载最佳Stacking模型并创建预测函数
# 使用方法: source("best_model/load_and_predict.R")
# =============================================

library(reticulate)

cat("Loading best stacking model...\n")
cat("Checking Python environment...\n")

# 检查并配置 Python 环境
if (!py_available()) {
  cat("\n========================================\n")
  cat("ERROR: Python not available!\n")
  cat("========================================\n\n")
  cat("请先配置 Python 环境:\n\n")
  cat("Python 已安装但 R 找不到，需要手动配置路径\n\n")
  cat("快速解决方法:\n\n")
  cat("方法1 (最简单):\n")
  cat("  1. 找到你的 python.exe 路径（通常在 C:\\Python39\\python.exe）\n")
  cat("  2. 在 R 中运行:\n")
  cat("     use_python('C:/Python39/python.exe')  # 替换为你的路径\n")
  cat("  3. 然后重新运行此脚本\n\n")
  cat("方法2 (自动查找):\n")
  cat("  source('best_model/直接配置Python.R')\n\n")
  cat("方法3 (详细查找):\n")
  cat("  source('best_model/配置Python环境.R')\n\n")
  stop("Python not available. Please configure Python path first.")
}

# 检查 Python 版本
py_version <- py_config()$version
cat(sprintf("Python version: %s\n", py_version))

# 检查必要的包
required_packages <- list(
  "torch" = "torch",
  "numpy" = "numpy", 
  "pandas" = "pandas",
  "xgboost" = "xgboost",
  "sklearn" = "sklearn"  # 导入时用 sklearn，但安装时是 scikit-learn
)
missing_packages <- c()

for (pkg_name in names(required_packages)) {
  import_name <- required_packages[[pkg_name]]
  tryCatch({
    py_run_string(sprintf("import %s", import_name))
    cat(sprintf("✓ %s installed\n", pkg_name))
  }, error = function(e) {
    missing_packages <<- c(missing_packages, pkg_name)
    cat(sprintf("✗ %s NOT installed\n", pkg_name))
  })
}

if (length(missing_packages) > 0) {
  cat("\n========================================\n")
  cat("Missing Python packages detected!\n")
  cat("========================================\n")
  cat("Please install missing packages:\n\n")
  cat("Option 1: In R, run:\n")
  cat("  source('best_model/安装Python依赖.R')\n\n")
  cat("Option 2: In R, manually install:\n")
  install_cmd <- ifelse(missing_packages == "sklearn", 
                       'py_install("scikit-learn")',
                       sprintf('py_install("%s")', missing_packages))
  cat(paste("  ", install_cmd, collapse = "\n"), "\n\n")
  cat("Option 3: In Python terminal, run:\n")
  pip_packages <- ifelse(missing_packages == "sklearn", "scikit-learn", missing_packages)
  cat("  pip install", paste(pip_packages, collapse = " "), "\n\n")
  cat("After installation, restart R and try again.\n")
  stop("Missing required Python packages")
}

cat("\nAll required packages are installed. Loading models...\n")

# Load models (assumes running from R_scripts/ directory)
py_run_string("
import pickle, torch, torch.nn as nn, numpy as np, pandas as pd, xgboost as xgb
import os
# Get the directory where this R script is located
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

# Predict function
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

cat("✓ Model loaded! Use predict_price(feature_vector) to predict.\n")

