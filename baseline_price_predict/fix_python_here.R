# =============================================
# 快速修复 Python 配置
# 从任何目录都可以运行
# =============================================

library(reticulate)

cat("========================================\n")
cat("快速修复 Python 配置\n")
cat("========================================\n\n")

# Python 路径
python_path <- "C:/Users/linda/AppData/Local/Programs/Python/Python310/python.exe"

cat(sprintf("当前工作目录: %s\n", getwd()))
cat(sprintf("Python 路径: %s\n", python_path))
cat("\n")

# 检查文件是否存在
if (!file.exists(python_path)) {
  cat("✗ Python 文件不存在\n")
  stop("请检查 Python 路径")
}

cat("✓ Python 文件存在\n\n")

# 方法 1: 设置环境变量
cat("步骤 1: 设置环境变量...\n")
Sys.setenv(RETICULATE_PYTHON = python_path)
cat("✓ 完成\n\n")

# 方法 2: 使用 use_python
cat("步骤 2: 配置 Python...\n")
tryCatch({
  use_python(python_path, required = TRUE)
  cat("✓ 完成\n\n")
}, error = function(e) {
  cat(sprintf("✗ 失败: %s\n\n", e$message))
})

# 方法 3: 检查状态
cat("步骤 3: 检查配置...\n")
if (py_available()) {
  config <- py_config()
  cat("✓ Python 配置成功！\n")
  cat(sprintf("Python 路径: %s\n", config$python))
  cat(sprintf("Python 版本: %s\n", as.character(config$version)))
  
  # 测试 PyTorch
  cat("\n步骤 4: 检查 PyTorch...\n")
  if (py_module_available("torch")) {
    cat("✓ PyTorch 已安装\n")
    py_run_string("import torch; print(f'PyTorch version: {torch.__version__}')")
  } else {
    cat("⚠ PyTorch 未安装，正在安装...\n")
    py_install("torch", pip = TRUE)
    if (py_module_available("torch")) {
      cat("✓ PyTorch 安装成功\n")
    } else {
      cat("✗ PyTorch 安装失败\n")
    }
  }
} else {
  cat("✗ Python 仍然不可用\n")
  cat("\n请尝试:\n")
  cat("1. 重启 R 会话（Session > Restart R）\n")
  cat("2. 然后重新运行此脚本\n")
}

cat("\n========================================\n")
cat("完成\n")
cat("========================================\n")

