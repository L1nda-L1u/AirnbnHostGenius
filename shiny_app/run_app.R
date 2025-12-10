# =============================================
# 启动 Shiny 应用 - Airbnb 基准价格预测器
# =============================================
# 
# 使用方法：
#   在项目根目录运行: source("shiny_app/run_app.R")
#   应用将在浏览器中自动打开: http://localhost:3838
#
# =============================================

# 检查并安装必需的 R 包
required_packages <- c(
  "shiny", "shinydashboard", "DT", "leaflet", "plotly",
  "dplyr", "geosphere", "xgboost", "reticulate", "glmnet",
  "httr", "jsonlite", "caret", "zoo", "data.table", "sf",
  "lubridate", "scales", "ggplot2"
)

missing_packages <- required_packages[!sapply(required_packages, requireNamespace, quietly = TRUE)]

if (length(missing_packages) > 0) {
  cat("正在安装缺失的 R 包...\n")
  install.packages(missing_packages)
  cat("安装完成！\n\n")
}

# 检查 Python 和 PyTorch（可选，用于神经网络模型）
# 注意：应用使用 XGBoost-only 模式，不需要 Python 也能运行
if (requireNamespace("reticulate", quietly = TRUE)) {
  library(reticulate)
  
  if (!py_available()) {
    cat("提示: Python 未配置。应用将使用 XGBoost-only 模式（这是正常的）。\n")
    cat("如需启用神经网络模型，运行: source('baseline_price_predict/sensitivity_analysis/configure_python.R')\n\n")
  } else {
    if (!py_module_available("torch")) {
      cat("PyTorch 未安装。正在安装...\n")
      py_install("torch", pip = TRUE)
      cat("PyTorch 安装完成！\n\n")
    }
  }
}

# 启动应用
cat("========================================\n")
cat("正在启动 Airbnb 基准价格预测器\n")
cat("========================================\n\n")

# 确保在正确的目录
# 如果不在 shiny_app 目录，且 shiny_app 目录存在，则切换到该目录
if (basename(getwd()) != "shiny_app") {
  if (dir.exists("shiny_app")) {
    setwd("shiny_app")
    cat("已切换到 shiny_app 目录\n\n")
  } else {
    cat("警告: 未找到 shiny_app 目录。请确保在项目根目录运行此脚本。\n")
    cat("当前目录:", getwd(), "\n\n")
  }
}

# 启动 Shiny 应用
# host = "0.0.0.0" 允许从任何网络接口访问
# port = 3838 是 Shiny 的默认端口
cat("应用正在启动...\n")
cat("浏览器将自动打开，或手动访问: http://localhost:3838\n")
cat("按 Ctrl+C 或 Esc 停止应用\n\n")

shiny::runApp("app.R", host = "0.0.0.0", port = 3838)

