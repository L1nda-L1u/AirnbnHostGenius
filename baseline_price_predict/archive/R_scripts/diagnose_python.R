# =============================================
# Python 配置诊断脚本
# 自动检测当前目录并找到正确的路径
# =============================================

library(reticulate)

cat("========================================\n")
cat("Python 配置诊断\n")
cat("========================================\n\n")

# 自动检测脚本位置
current_dir <- getwd()
cat(sprintf("当前工作目录: %s\n", current_dir))

# 尝试找到 Python310
python_paths <- c(
  "C:/Users/linda/AppData/Local/Programs/Python/Python310/python.exe",
  "C:/Users/linda/AppData/Local/Programs/Python/Python312/python.exe"
)

# 1. 检查 Python 路径是否存在
cat("1. 检查 Python 文件是否存在...\n")
python_path <- NULL
for (path in python_paths) {
  if (file.exists(path)) {
    python_path <- path
    cat(sprintf("   ✓ 找到 Python: %s\n", python_path))
    break
  }
}

if (is.null(python_path)) {
  cat("   ✗ 未找到 Python 文件\n")
  stop("Python 文件不存在")
}

cat("\n")

# 2. 尝试直接执行 Python
cat("2. 尝试直接执行 Python...\n")
tryCatch({
  python_output <- system2(python_path, "--version", stdout = TRUE, stderr = TRUE)
  cat(sprintf("   Python 版本: %s\n", paste(python_output, collapse = " ")))
}, error = function(e) {
  cat(sprintf("   ✗ 无法执行 Python: %s\n", e$message))
})

cat("\n")

# 3. 检查 reticulate 状态
cat("3. 检查 reticulate 状态...\n")
cat(sprintf("   py_available(): %s\n", py_available()))

if (py_available()) {
  tryCatch({
    config <- py_config()
    cat(sprintf("   当前 Python 路径: %s\n", config$python))
    cat(sprintf("   当前 Python 版本: %s\n", as.character(config$version)))
  }, error = function(e) {
    cat(sprintf("   无法获取配置: %s\n", e$message))
  })
} else {
  cat("   Python 未配置\n")
}

cat("\n")

# 4. 尝试配置 Python
cat("4. 尝试配置 Python...\n")
cat(sprintf("   使用路径: %s\n", python_path))

# 方法 1: 设置环境变量
Sys.setenv(RETICULATE_PYTHON = python_path)
cat("   ✓ 环境变量已设置\n")

# 方法 2: 使用 use_python
tryCatch({
  use_python(python_path, required = FALSE)
  cat("   ✓ use_python 已调用（非强制模式）\n")
}, error = function(e) {
  cat(sprintf("   ⚠ use_python 警告: %s\n", e$message))
})

# 方法 3: 如果还是不可用，尝试强制模式
if (!py_available()) {
  cat("   尝试强制模式...\n")
  tryCatch({
    use_python(python_path, required = TRUE)
    cat("   ✓ use_python 已调用（强制模式）\n")
  }, error = function(e) {
    cat(sprintf("   ✗ 强制模式失败: %s\n", e$message))
  })
}

cat("\n")

# 5. 再次检查状态
cat("5. 配置后的状态...\n")
cat(sprintf("   py_available(): %s\n", py_available()))

if (py_available()) {
  tryCatch({
    config <- py_config()
    cat(sprintf("   ✓ Python 配置成功！\n"))
    cat(sprintf("   Python 路径: %s\n", config$python))
    cat(sprintf("   Python 版本: %s\n", as.character(config$version)))
    
    # 尝试执行简单的 Python 代码
    cat("\n6. 测试 Python 执行...\n")
    py_run_string("import sys; print(f'Python executable: {sys.executable}')")
    py_run_string("print(f'Python version: {sys.version}')")
    cat("   ✓ Python 可以正常执行\n")
  }, error = function(e) {
    cat(sprintf("   ✗ Python 执行失败: %s\n", e$message))
  })
} else {
  cat("   ✗ Python 仍然不可用\n")
}

cat("\n")

# 6. 检查可能的冲突
cat("7. 检查环境变量...\n")
env_vars <- c("PYTHON_PATH", "PYTHONHOME", "RETICULATE_PYTHON")
found_vars <- FALSE
for (var in env_vars) {
  value <- Sys.getenv(var)
  if (value != "") {
    cat(sprintf("   %s = %s\n", var, value))
    found_vars <- TRUE
  }
}
if (!found_vars) {
  cat("   未找到相关环境变量\n")
}

cat("\n")

# 7. 建议
cat("========================================\n")
cat("诊断完成\n")
cat("========================================\n\n")

if (!py_available()) {
  cat("建议:\n")
  cat("1. 重启 R 会话（Session > Restart R）\n")
  cat("2. 然后运行以下命令:\n")
  cat("   library(reticulate)\n")
  cat(sprintf("   Sys.setenv(RETICULATE_PYTHON = '%s')\n", python_path))
  cat(sprintf("   use_python('%s', required = TRUE)\n", python_path))
  cat("   py_available()\n")
  cat("\n3. 如果还是失败，检查 Python 是否损坏\n")
} else {
  cat("✓ Python 配置正常！\n")
  cat("现在可以运行 evaluate_all_baseline_models.R\n")
}
