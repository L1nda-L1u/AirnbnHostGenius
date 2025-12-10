# =============================================
# 快速配置 Python - 手动指定路径
# 如果你知道 Python 的安装路径，直接使用这个
# =============================================

library(reticulate)

cat("========================================\n")
cat("快速配置 Python\n")
cat("========================================\n\n")

# 如果你知道 Python 路径，直接在这里修改
# 常见路径示例（取消注释并修改为你的路径）:

# Windows 常见路径:
# python_path <- "C:/Python39/python.exe"
# python_path <- "C:/Python310/python.exe"
# python_path <- "C:/Python311/python.exe"
# python_path <- "C:/Users/你的用户名/AppData/Local/Programs/Python/Python39/python.exe"
# python_path <- "C:/Program Files/Python39/python.exe"

# 如果你使用 Anaconda:
# python_path <- "C:/Users/你的用户名/anaconda3/python.exe"
# python_path <- "C:/Users/你的用户名/miniconda3/python.exe"

# ============================================
# 方法1: 自动查找（推荐先试这个）
# ============================================
cat("方法1: 自动查找 Python...\n")
source("best_model/配置Python环境.R")

# 如果自动查找失败，继续下面的方法2

# ============================================
# 方法2: 手动指定路径
# ============================================
if (!py_available()) {
  cat("\n自动查找失败，请手动指定路径\n\n")
  
  # 取消注释下面这行，并修改为你的 Python 路径
  # python_path <- "C:/Python39/python.exe"
  
  # 或者让用户输入
  cat("请输入你的 Python 路径（例如: C:/Python39/python.exe）\n")
  cat("或者按 Enter 跳过，然后手动修改此脚本中的 python_path\n\n")
  
  # 尝试使用
  if (exists("python_path") && file.exists(python_path)) {
    cat(sprintf("正在使用: %s\n", python_path))
    tryCatch({
      use_python(python_path, required = TRUE)
      if (py_available()) {
        cat("✓ 配置成功！\n")
        py_config()
      }
    }, error = function(e) {
      cat("✗ 配置失败:", e$message, "\n")
    })
  } else {
    cat("请修改此脚本，设置正确的 python_path 变量\n")
  }
}

# ============================================
# 方法3: 使用 conda 环境
# ============================================
if (!py_available()) {
  cat("\n尝试使用 conda 环境...\n")
  tryCatch({
    use_condaenv("base", required = FALSE)
    if (py_available()) {
      cat("✓ 使用 conda base 环境成功！\n")
    }
  }, error = function(e) {
    cat("conda 不可用\n")
  })
}

cat("\n========================================\n")
if (py_available()) {
  cat("✓ Python 已配置！\n")
  py_config()
} else {
  cat("✗ Python 仍未配置\n")
  cat("请运行: source('best_model/配置Python环境.R')\n")
}
cat("========================================\n")

