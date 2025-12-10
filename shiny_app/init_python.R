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
  
  # Try to find Python from PATH and common locations
  python_paths <- c(
    Sys.which("python3"),
    Sys.which("python"),
    "/usr/bin/python3",
    "/usr/local/bin/python3"
  )
  
  for (python_path in python_paths) {
    if (python_path != "" && file.exists(python_path)) {
      tryCatch({
        use_python(python_path, required = FALSE)
        py_discover_config()
        
        if (py_available()) {
          return(TRUE)
        }
      }, error = function(e) {
        cat("Python initialization error:", e$message, "\n")
      })
    }
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

