# =============================================
# 安装Python依赖包
# 如果遇到 "ModuleNotFoundError" 错误，运行此脚本
# =============================================

library(reticulate)

cat("========================================\n")
cat("安装Python依赖包\n")
cat("========================================\n\n")

# 检查 Python 是否可用
if (!py_available()) {
  cat("Python 不可用，正在安装...\n")
  install_python()
  cat("请重启 R 后再次运行此脚本\n")
  stop("请重启 R")
}

# 显示当前 Python 路径
py_config_info <- py_config()
cat(sprintf("Python 路径: %s\n", py_config_info$python))
cat(sprintf("Python 版本: %s\n", py_config_info$version))
cat("\n")

# 需要安装的包
required_packages <- c(
  "torch",
  "numpy", 
  "pandas",
  "xgboost",
  "scikit-learn"  # sklearn
)

cat("开始安装以下包:\n")
cat(paste(required_packages, collapse = ", "), "\n\n")

# 安装包
for (pkg in required_packages) {
  cat(sprintf("正在安装 %s...\n", pkg))
  tryCatch({
    py_install(pkg, pip = TRUE)
    cat(sprintf("✓ %s 安装成功\n", pkg))
  }, error = function(e) {
    cat(sprintf("✗ %s 安装失败: %s\n", pkg, e$message))
  })
  cat("\n")
}

cat("========================================\n")
cat("安装完成！\n")
cat("如果仍有问题，请尝试在命令行运行:\n")
cat("  pip install torch numpy pandas xgboost scikit-learn\n")
cat("========================================\n")

