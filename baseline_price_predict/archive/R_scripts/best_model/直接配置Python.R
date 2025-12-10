# =============================================
# 直接配置 Python - 最简单的方法
# =============================================

library(reticulate)

cat("========================================\n")
cat("直接配置 Python\n")
cat("========================================\n\n")

# ============================================
# 步骤1: 找到你的 Python 路径
# ============================================
cat("请找到你的 Python.exe 文件位置\n\n")
cat("常见位置:\n")
cat("1. C:\\Python39\\python.exe\n")
cat("2. C:\\Python310\\python.exe\n")
cat("3. C:\\Python311\\python.exe\n")
cat("4. C:\\Users\\你的用户名\\AppData\\Local\\Programs\\Python\\Python39\\python.exe\n")
cat("5. C:\\Program Files\\Python39\\python.exe\n\n")

cat("快速查找方法:\n")
cat("1. 按 Win+R，输入: python\n")
cat("2. 如果打开了 Python，输入: import sys; print(sys.executable)\n")
cat("3. 复制显示的路径\n\n")

# ============================================
# 步骤2: 在这里设置你的 Python 路径
# ============================================
# 取消注释下面这行，并替换为你的实际路径
# python_path <- "C:/Python39/python.exe"

# 或者使用这个函数让 R 帮你找
find_python_simple <- function() {
  # 检查 PATH
  python <- Sys.which("python")
  if (python != "" && file.exists(python)) {
    return(python)
  }
  
  # 检查常见位置
  common <- c(
    "C:/Python39/python.exe",
    "C:/Python310/python.exe", 
    "C:/Python311/python.exe",
    "C:/Python312/python.exe"
  )
  
  for (path in common) {
    if (file.exists(path)) {
      return(path)
    }
  }
  
  return(NULL)
}

# 自动查找
auto_path <- find_python_simple()
if (!is.null(auto_path)) {
  cat(sprintf("自动找到 Python: %s\n\n", auto_path))
  python_path <- auto_path
} else {
  cat("未自动找到 Python，请手动设置 python_path 变量\n\n")
  cat("示例:\n")
  cat('  python_path <- "C:/Python39/python.exe"\n')
  cat("  然后重新运行此脚本\n\n")
  
  # 如果用户已经设置了，使用它
  if (!exists("python_path")) {
    stop("请先设置 python_path 变量")
  }
}

# ============================================
# 步骤3: 配置 Python
# ============================================
if (exists("python_path") && file.exists(python_path)) {
  cat(sprintf("正在配置: %s\n", python_path))
  
  tryCatch({
    # 配置 Python
    use_python(python_path, required = TRUE)
    
    # 验证
    if (py_available()) {
      cat("\n✓ 配置成功！\n\n")
      config <- py_config()
      cat("Python 信息:\n")
      cat(sprintf("  路径: %s\n", config$python))
      cat(sprintf("  版本: %s\n", config$version))
      cat("\n现在可以加载模型了！\n")
    } else {
      stop("配置后仍不可用")
    }
  }, error = function(e) {
    cat("\n✗ 配置失败\n")
    cat("错误:", e$message, "\n\n")
    cat("可能的原因:\n")
    cat("1. Python 路径不正确\n")
    cat("2. Python 版本太旧（需要 3.7+）\n")
    cat("3. Python 安装不完整\n\n")
    cat("解决方法:\n")
    cat("1. 确认路径是否正确（注意使用正斜杠 / 或双反斜杠 \\\\）\n")
    cat("2. 重新安装 Python: https://www.python.org/downloads/\n")
    cat("3. 安装时勾选 'Add Python to PATH'\n")
  })
} else {
  cat("✗ Python 路径不存在或未设置\n")
  cat("请检查路径是否正确\n")
}

cat("\n========================================\n")

