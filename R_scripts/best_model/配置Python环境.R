# =============================================
# 配置 Python 环境 - 智能查找已安装的 Python
# =============================================

library(reticulate)

cat("========================================\n")
cat("智能查找并配置 Python\n")
cat("========================================\n\n")

# 函数：查找 Python 可执行文件
find_python <- function() {
  python_paths <- c()
  
  # 方法1: 检查 PATH 环境变量
  cat("1. 检查系统 PATH...\n")
  python_cmd <- Sys.which("python")
  python3_cmd <- Sys.which("python3")
  if (python_cmd != "") python_paths <- c(python_paths, python_cmd)
  if (python3_cmd != "") python_paths <- c(python_paths, python3_cmd)
  
  # 方法2: 检查常见安装位置 (Windows)
  cat("2. 检查常见安装位置...\n")
  common_paths <- c(
    "C:/Python*/python.exe",
    "C:/Python*/python3.exe",
    "C:/Users/*/AppData/Local/Programs/Python/*/python.exe",
    "C:/Users/*/AppData/Local/Programs/Python/*/python3.exe",
    "C:/Program Files/Python*/python.exe",
    "C:/Program Files/Python*/python3.exe",
    "C:/Program Files (x86)/Python*/python.exe",
    "C:/Program Files (x86)/Python*/python3.exe",
    "C:/Users/*/anaconda3/python.exe",
    "C:/Users/*/anaconda3/python3.exe",
    "C:/Users/*/miniconda3/python.exe",
    "C:/Users/*/miniconda3/python3.exe"
  )
  
  # 展开通配符路径
  for (pattern in common_paths) {
    # 替换用户目录通配符
    if (grepl("\\*", pattern)) {
      user_dirs <- list.dirs("C:/Users", recursive = FALSE, full.names = TRUE)
      for (user_dir in user_dirs) {
        expanded <- gsub("C:/Users/\\*", basename(user_dir), pattern)
        expanded <- gsub("\\*", "", expanded)
        if (file.exists(expanded)) {
          python_paths <- c(python_paths, expanded)
        }
      }
    } else if (file.exists(pattern)) {
      python_paths <- c(python_paths, pattern)
    }
  }
  
  # 方法3: 使用 PowerShell 查找
  cat("3. 使用 PowerShell 搜索...\n")
  tryCatch({
    ps_cmd <- 'Get-ChildItem -Path "C:\\" -Filter "python.exe" -Recurse -ErrorAction SilentlyContinue | Select-Object -First 5 -ExpandProperty FullName'
    result <- system2("powershell", args = c("-Command", ps_cmd), stdout = TRUE, stderr = TRUE)
    if (length(result) > 0 && !any(grepl("Error", result))) {
      found <- result[result != ""]
      python_paths <- c(python_paths, found)
    }
  }, error = function(e) {})
  
  # 去重并验证
  python_paths <- unique(python_paths)
  valid_paths <- c()
  
  for (path in python_paths) {
    if (file.exists(path)) {
      # 测试 Python 是否可用
      tryCatch({
        version_cmd <- paste0('"', path, '" --version')
        version <- system(version_cmd, intern = TRUE, ignore.stderr = TRUE)
        if (length(version) > 0) {
          valid_paths <- c(valid_paths, path)
          cat(sprintf("  ✓ 找到: %s (%s)\n", path, version[1]))
        }
      }, error = function(e) {})
    }
  }
  
  return(valid_paths)
}

# 主程序
if (py_available()) {
  py_config_info <- py_config()
  cat("✓ Python 已配置！\n")
  cat(sprintf("Python 路径: %s\n", py_config_info$python))
  cat(sprintf("Python 版本: %s\n", py_config_info$version))
  cat("\n可以继续使用模型了！\n")
} else {
  cat("Python 未配置，正在查找...\n\n")
  
  # 查找 Python
  found_pythons <- find_python()
  
  if (length(found_pythons) > 0) {
    cat("\n找到以下 Python 安装:\n")
    for (i in seq_along(found_pythons)) {
      cat(sprintf("  [%d] %s\n", i, found_pythons[i]))
    }
    
    # 尝试使用第一个找到的 Python
    cat("\n正在尝试配置第一个找到的 Python...\n")
    selected_python <- found_pythons[1]
    
    tryCatch({
      use_python(selected_python, required = TRUE)
      
      if (py_available()) {
        py_config_info <- py_config()
        cat("\n✓ Python 配置成功！\n")
        cat(sprintf("Python 路径: %s\n", py_config_info$python))
        cat(sprintf("Python 版本: %s\n", py_config_info$version))
        cat("\n可以继续使用模型了！\n")
      } else {
        stop("配置失败")
      }
    }, error = function(e) {
      cat("\n✗ 自动配置失败\n")
      cat("错误:", e$message, "\n\n")
      
      cat("========================================\n")
      cat("手动配置方法\n")
      cat("========================================\n\n")
      cat("请手动指定 Python 路径:\n\n")
      cat("1. 找到你的 Python 安装路径（通常在以下位置之一）:\n")
      cat("   - C:\\Python3X\\python.exe\n")
      cat("   - C:\\Users\\你的用户名\\AppData\\Local\\Programs\\Python\\Python3X\\python.exe\n")
      cat("   - C:\\Program Files\\Python3X\\python.exe\n\n")
      
      cat("2. 在 R 中运行（替换为你的实际路径）:\n")
      cat("   use_python('你的Python路径')\n\n")
      
      cat("3. 或者，如果你想使用其他找到的 Python:\n")
      for (i in seq_along(found_pythons)) {
        cat(sprintf("   use_python('%s')\n", found_pythons[i]))
      }
    })
  } else {
    cat("\n✗ 未找到 Python 安装\n\n")
    cat("========================================\n")
    cat("请手动指定 Python 路径\n")
    cat("========================================\n\n")
    
    cat("请按以下步骤操作:\n\n")
    cat("1. 找到你的 Python 安装位置:\n")
    cat("   - 打开文件管理器\n")
    cat("   - 搜索 'python.exe'\n")
    cat("   - 或者检查常见位置:\n")
    cat("     * C:\\Python3X\\\n")
    cat("     * C:\\Users\\你的用户名\\AppData\\Local\\Programs\\Python\\\n")
    cat("     * C:\\Program Files\\Python3X\\\n\n")
    
    cat("2. 在 R 中运行（替换为实际路径）:\n")
    cat("   use_python('C:/Python39/python.exe')  # 示例\n\n")
    
    cat("3. 验证配置:\n")
    cat("   py_available()  # 应该返回 TRUE\n\n")
    
    cat("4. 如果还是不行，尝试添加到 PATH:\n")
    cat("   - 右键 '此电脑' > 属性 > 高级系统设置\n")
    cat("   - 环境变量 > 系统变量 > Path > 编辑\n")
    cat("   - 添加 Python 安装目录（例如: C:\\Python39）\n")
    cat("   - 重启 RStudio\n")
  }
}

cat("\n========================================\n")
cat("配置完成\n")
cat("========================================\n")
