# =============================================
# 简单配置 Python 环境
# 用于 Neural Network 模型训练
# =============================================

library(reticulate)

cat("========================================\n")
cat("配置 Python 环境\n")
cat("========================================\n\n")

# 如果已经配置好，直接返回
if (py_available()) {
  config <- py_config()
  cat("✓ Python 已经配置好了！\n")
  cat(sprintf("Python 路径: %s\n", config$python))
  cat(sprintf("Python 版本: %s\n", config$version))
  
  # 检查 PyTorch
  if (py_module_available("torch")) {
    cat("✓ PyTorch 已安装\n")
    cat("========================================\n")
    return(invisible(TRUE))
  } else {
    cat("PyTorch 未安装，正在安装...\n")
    py_install("torch", pip = TRUE)
    if (py_module_available("torch")) {
      cat("✓ PyTorch 安装成功\n")
      cat("========================================\n")
      return(invisible(TRUE))
    }
  }
}

# =============================================
# 自动查找 Python
# =============================================
cat("正在自动查找 Python...\n\n")

find_python <- function() {
  python_paths <- c()
  
  # 方法1: 检查 PATH
  python_cmd <- Sys.which("python")
  python3_cmd <- Sys.which("python3")
  if (python_cmd != "" && file.exists(python_cmd)) {
    python_paths <- c(python_paths, python_cmd)
  }
  if (python3_cmd != "" && file.exists(python3_cmd)) {
    python_paths <- c(python_paths, python3_cmd)
  }
  
  # 方法2: 检查常见位置 (Windows)
  if (.Platform$OS.type == "windows") {
    common_paths <- c(
      "C:/Python39/python.exe",
      "C:/Python310/python.exe",
      "C:/Python311/python.exe",
      "C:/Python312/python.exe",
      "C:/Program Files/Python39/python.exe",
      "C:/Program Files/Python310/python.exe",
      "C:/Program Files/Python311/python.exe",
      "C:/Program Files/Python312/python.exe"
    )
    
    for (path in common_paths) {
      if (file.exists(path)) {
        python_paths <- c(python_paths, path)
      }
    }
    
    # 检查用户目录下的 Python
    user_home <- Sys.getenv("USERPROFILE")
    if (user_home != "") {
      user_python_paths <- c(
        file.path(user_home, "AppData", "Local", "Programs", "Python", "Python39", "python.exe"),
        file.path(user_home, "AppData", "Local", "Programs", "Python", "Python310", "python.exe"),
        file.path(user_home, "AppData", "Local", "Programs", "Python", "Python311", "python.exe"),
        file.path(user_home, "anaconda3", "python.exe"),
        file.path(user_home, "miniconda3", "python.exe")
      )
      for (path in user_python_paths) {
        if (file.exists(path)) {
          python_paths <- c(python_paths, path)
        }
      }
    }
  } else {
    # Linux/Mac 常见位置
    common_paths <- c(
      "/usr/bin/python3",
      "/usr/local/bin/python3",
      "/opt/homebrew/bin/python3"
    )
    for (path in common_paths) {
      if (file.exists(path)) {
        python_paths <- c(python_paths, path)
      }
    }
  }
  
  return(unique(python_paths))
}

python_paths <- find_python()

if (length(python_paths) == 0) {
  cat("✗ 未找到 Python\n\n")
  cat("请安装 Python 或手动指定路径:\n\n")
  cat("方法1: 安装 Python\n")
  cat("  1. 访问 https://www.python.org/downloads/\n")
  cat("  2. 下载并安装 Python 3.9 或更高版本\n")
  cat("  3. 安装时勾选 'Add Python to PATH'\n")
  cat("  4. 重新运行此脚本\n\n")
  
  cat("方法2: 手动指定路径\n")
  cat("  在 R 中运行:\n")
  cat("    library(reticulate)\n")
  cat("    use_python('C:/Python39/python.exe')  # 替换为你的路径\n")
  cat("    py_available()  # 应该返回 TRUE\n\n")
  
  stop("Python not found. Please install Python or specify the path manually.")
}

# 尝试使用找到的 Python
cat(sprintf("找到 %d 个 Python 安装:\n", length(python_paths)))
for (i in seq_along(python_paths)) {
  cat(sprintf("  %d. %s\n", i, python_paths[i]))
}
cat("\n")

# 尝试第一个找到的 Python
python_path <- python_paths[1]
cat(sprintf("正在使用: %s\n", python_path))

tryCatch({
  use_python(python_path, required = TRUE)
  
  if (py_available()) {
    config <- py_config()
    cat("\n✓ Python 配置成功！\n")
    cat(sprintf("Python 路径: %s\n", config$python))
    cat(sprintf("Python 版本: %s\n", config$version))
  } else {
    stop("配置后 Python 仍不可用")
  }
}, error = function(e) {
  cat("\n✗ 配置失败:", e$message, "\n")
  cat("\n请尝试:\n")
  cat("1. 检查 Python 路径是否正确\n")
  cat("2. 重新安装 Python\n")
  cat("3. 手动指定路径:\n")
  cat("   use_python('你的Python路径', required = TRUE)\n")
  stop("Python configuration failed")
})

# =============================================
# 安装 PyTorch
# =============================================
cat("\n检查 PyTorch...\n")
if (!py_module_available("torch")) {
  cat("PyTorch 未安装，正在安装（这可能需要几分钟）...\n")
  tryCatch({
    py_install("torch", pip = TRUE)
    if (py_module_available("torch")) {
      cat("✓ PyTorch 安装成功\n")
    } else {
      cat("⚠ PyTorch 安装可能失败，请手动安装:\n")
      cat("  在命令行运行: pip install torch\n")
    }
  }, error = function(e) {
    cat("✗ PyTorch 安装失败:", e$message, "\n")
    cat("请手动安装: pip install torch\n")
  })
} else {
  cat("✓ PyTorch 已安装\n")
}

cat("\n========================================\n")
cat("Python 环境配置完成！\n")
cat("========================================\n")

