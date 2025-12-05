# =============================================
# Initialize Python for Shiny App
# 在Shiny应用启动前运行这个来初始化Python
# =============================================

library(reticulate)

# 初始化Python（如果还没配置）
init_python <- function() {
  if (py_available()) {
    return(TRUE)
  }
  
  # 尝试配置Python
  python_path <- "C:/Users/linda/AppData/Local/Programs/Python/Python310/python.exe"
  
  if (file.exists(python_path)) {
    tryCatch({
      use_python(python_path, required = FALSE)
      py_discover_config()  # 关键：重新发现配置
      
      if (py_available()) {
        return(TRUE)
      }
    }, error = function(e) {
      cat("Python initialization error:", e$message, "\n")
    })
  }
  
  # 如果还是不行，尝试从PATH查找
  python_cmd <- Sys.which("python")
  if (python_cmd != "" && python_cmd != python_path) {
    tryCatch({
      use_python(python_cmd, required = FALSE)
      py_discover_config()
      if (py_available()) {
        return(TRUE)
      }
    }, error = function(e) {})
  }
  
  return(FALSE)
}

# 自动初始化
if (!py_available()) {
  init_python()
}

# 显示状态
if (py_available()) {
  config <- py_config()
  cat("Python initialized successfully!\n")
  cat("Path:", config$python, "\n")
  # version可能是列表，需要转换为字符串
  version_str <- if (is.list(config$version)) {
    paste(config$version, collapse = ".")
  } else {
    as.character(config$version)
  }
  cat("Version:", version_str, "\n")
} else {
  cat("Python not available (will use XGBoost-only mode)\n")
}

