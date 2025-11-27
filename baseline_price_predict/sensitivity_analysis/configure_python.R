### ---------------------------------------------------------
### configure_python.R
### 简单配置Python环境 - 专门为sensitivity_analysis设计
### ---------------------------------------------------------

library(reticulate)

cat("========================================\n")
cat("配置 Python 环境\n")
cat("========================================\n\n")

# =============================================
# 方法1: 检查是否已经配置好
# =============================================
if (py_available()) {
  config <- py_config()
  cat("✓ Python 已经配置好了！\n")
  cat(sprintf("Python 路径: %s\n", config$python))
  cat(sprintf("Python 版本: %s\n", config$version))
  cat("\n可以继续使用模型了！\n")
  cat("========================================\n")
  return(invisible(TRUE))
}

# =============================================
# 方法2: 自动查找Python
# =============================================
cat("正在自动查找 Python...\n\n")

find_python_simple <- function() {
  python_paths <- c()
  
  # 检查系统PATH
  python_cmd <- Sys.which("python")
  python3_cmd <- Sys.which("python3")
  if (python_cmd != "") python_paths <- c(python_paths, python_cmd)
  if (python3_cmd != "") python_paths <- c(python_paths, python3_cmd)
  
  # 检查常见位置（Windows）
  common_locations <- c(
    "C:/Python39/python.exe",
    "C:/Python310/python.exe",
    "C:/Python311/python.exe",
    "C:/Python312/python.exe",
    "C:/Program Files/Python39/python.exe",
    "C:/Program Files/Python310/python.exe",
    "C:/Program Files/Python311/python.exe",
    "C:/Program Files/Python312/python.exe"
  )
  
  # 检查用户目录
  username <- Sys.getenv("USERNAME")
  if (username != "") {
    user_paths <- c(
      sprintf("C:/Users/%s/AppData/Local/Programs/Python/Python39/python.exe", username),
      sprintf("C:/Users/%s/AppData/Local/Programs/Python/Python310/python.exe", username),
      sprintf("C:/Users/%s/AppData/Local/Programs/Python/Python311/python.exe", username),
      sprintf("C:/Users/%s/AppData/Local/Programs/Python/Python312/python.exe", username),
      sprintf("C:/Users/%s/anaconda3/python.exe", username),
      sprintf("C:/Users/%s/miniconda3/python.exe", username)
    )
    common_locations <- c(common_locations, user_paths)
  }
  
  # 验证路径
  valid_paths <- c()
  for (path in c(python_paths, common_locations)) {
    if (file.exists(path)) {
      tryCatch({
        # 测试Python是否可用
        version_output <- system(paste0('"', path, '" --version'), intern = TRUE, ignore.stderr = TRUE)
        if (length(version_output) > 0) {
          valid_paths <- c(valid_paths, path)
          cat(sprintf("  ✓ 找到: %s\n", path))
        }
      }, error = function(e) {})
    }
  }
  
  return(unique(valid_paths))
}

found_pythons <- find_python_simple()

if (length(found_pythons) > 0) {
  cat(sprintf("\n找到 %d 个 Python 安装:\n", length(found_pythons)))
  for (i in seq_along(found_pythons)) {
    cat(sprintf("  [%d] %s\n", i, found_pythons[i]))
  }
  
  # 尝试使用第一个
  cat("\n正在尝试配置第一个 Python...\n")
  selected <- found_pythons[1]
  
  tryCatch({
    use_python(selected, required = TRUE)
    
    if (py_available()) {
      config <- py_config()
      cat("\n✓ 配置成功！\n")
      cat(sprintf("Python 路径: %s\n", config$python))
      cat(sprintf("Python 版本: %s\n", config$version))
      
      # 检查torch
      cat("\n检查 PyTorch...\n")
      if (py_module_available("torch")) {
        cat("✓ PyTorch 已安装\n")
      } else {
        cat("⚠ PyTorch 未安装，正在安装...\n")
        py_install("torch", pip = TRUE)
        if (py_module_available("torch")) {
          cat("✓ PyTorch 安装成功！\n")
        } else {
          cat("✗ PyTorch 安装失败，请手动安装\n")
        }
      }
      
      cat("\n现在可以运行 sensitivity_analysis.R 了！\n")
      cat("========================================\n")
      return(invisible(TRUE))
    }
  }, error = function(e) {
    cat(sprintf("\n✗ 自动配置失败: %s\n", e$message))
  })
}

# =============================================
# 方法3: 手动配置
# =============================================
cat("\n========================================\n")
cat("手动配置方法\n")
cat("========================================\n\n")

cat("如果自动查找失败，请按以下步骤操作：\n\n")

cat("步骤1: 找到你的 Python 安装路径\n")
cat("  - 打开文件管理器（Windows资源管理器）\n")
cat("  - 在地址栏输入: C:\\Users\\你的用户名\\AppData\\Local\\Programs\\Python\n")
cat("  - 或者搜索 'python.exe'\n")
cat("  - 常见位置:\n")
cat("    * C:\\Python39\\python.exe\n")
cat("    * C:\\Python310\\python.exe\n")
cat("    * C:\\Users\\你的用户名\\AppData\\Local\\Programs\\Python\\Python39\\python.exe\n")
cat("    * C:\\Program Files\\Python39\\python.exe\n\n")

cat("步骤2: 在 R 中运行（替换为你的实际路径）:\n")
cat("  library(reticulate)\n")
cat("  use_python('C:/Python39/python.exe')  # 替换为你的路径\n")
cat("  py_available()  # 应该返回 TRUE\n\n")

cat("步骤3: 安装 PyTorch（如果还没有）:\n")
cat("  py_install('torch', pip = TRUE)\n\n")

cat("步骤4: 验证:\n")
cat("  py_module_available('torch')  # 应该返回 TRUE\n\n")

cat("步骤5: 重新运行 sensitivity_analysis.R\n")
cat("  source('sensitivity_analysis.R')\n\n")

cat("========================================\n")
cat("提示: 如果还是不行，可以:\n")
cat("1. 重新安装 Python (https://www.python.org/downloads/)\n")
cat("2. 安装时勾选 'Add Python to PATH'\n")
cat("3. 或者使用 Anaconda (https://www.anaconda.com/)\n")
cat("========================================\n")

