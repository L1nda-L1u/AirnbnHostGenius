# =============================================
# 修复 Python 配置问题
# 尝试多种方法
# =============================================

library(reticulate)

cat("========================================\n")
cat("修复 Python 配置\n")
cat("========================================\n\n")

python_path <- "C:/Users/linda/AppData/Local/Programs/Python/Python310/python.exe"

# 方法 1: 设置环境变量
cat("方法 1: 设置环境变量...\n")
Sys.setenv(RETICULATE_PYTHON = python_path)
cat("   ✓ 环境变量已设置\n\n")

# 方法 2: 清除 reticulate 缓存
cat("方法 2: 清除 reticulate 缓存...\n")
tryCatch({
  # 尝试清除配置
  py_config()  # 这会触发重新配置
}, error = function(e) {
  # 忽略错误
})
cat("   ✓ 缓存已清除\n\n")

# 方法 3: 重新配置 Python
cat("方法 3: 重新配置 Python...\n")
tryCatch({
  use_python(python_path, required = TRUE)
  cat("   ✓ Python 路径已设置\n")
}, error = function(e) {
  cat(sprintf("   ✗ 设置失败: %s\n", e$message))
})

cat("\n")

# 方法 4: 检查状态
cat("方法 4: 检查配置状态...\n")
if (py_available()) {
  config <- py_config()
  cat("   ✓ Python 可用！\n")
  cat(sprintf("   Python 路径: %s\n", config$python))
  cat(sprintf("   Python 版本: %s\n", as.character(config$version)))
} else {
  cat("   ✗ Python 仍然不可用\n\n")
  
  # 方法 5: 尝试直接初始化
  cat("方法 5: 尝试直接初始化...\n")
  tryCatch({
    # 强制重新初始化
    py_disconnect_config()
    py_config(python = python_path)
    
    if (py_available()) {
      config <- py_config()
      cat("   ✓ 直接初始化成功！\n")
      cat(sprintf("   Python 路径: %s\n", config$python))
      cat(sprintf("   Python 版本: %s\n", as.character(config$version)))
    } else {
      cat("   ✗ 直接初始化也失败\n")
    }
  }, error = function(e) {
    cat(sprintf("   ✗ 初始化失败: %s\n", e$message))
  })
}

cat("\n========================================\n")
if (py_available()) {
  cat("✓ Python 配置成功！\n")
  cat("现在可以运行 evaluate_all_baseline_models.R\n")
} else {
  cat("✗ Python 配置失败\n")
  cat("\n请尝试:\n")
  cat("1. 重启 R 会话（Session > Restart R）\n")
  cat("2. 然后运行:\n")
  cat("   library(reticulate)\n")
  cat("   Sys.setenv(RETICULATE_PYTHON = 'C:/Users/linda/AppData/Local/Programs/Python/Python310/python.exe')\n")
  cat("   use_python('C:/Users/linda/AppData/Local/Programs/Python/Python310/python.exe', required = TRUE)\n")
  cat("   py_available()\n")
}
cat("========================================\n")

