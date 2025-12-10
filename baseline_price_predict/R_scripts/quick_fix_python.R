# =============================================
# 快速修复 Python 配置
# 直接使用已知的 Python 路径
# =============================================

library(reticulate)

cat("========================================\n")
cat("快速修复 Python 配置\n")
cat("========================================\n\n")

# 已知的 Python 路径（从错误信息中获取）
python_paths <- c(
  "C:/Users/linda/AppData/Local/Programs/Python/Python310/python.exe",
  "C:/Users/linda/AppData/Local/Programs/Python/Python312/python.exe"
)

cat("尝试配置 Python...\n\n")

python_configured <- FALSE

for (i in seq_along(python_paths)) {
  python_path <- python_paths[i]
  
  cat(sprintf("尝试 [%d/%d]: %s\n", i, length(python_paths), python_path))
  
  if (!file.exists(python_path)) {
    cat("  ✗ 文件不存在\n\n")
    next
  }
  
  tryCatch({
    # 规范化路径
    python_path_normalized <- normalizePath(python_path, winslash = "/", mustWork = FALSE)
    cat(sprintf("  规范化路径: %s\n", python_path_normalized))
    
    # 尝试配置
    use_python(python_path_normalized, required = FALSE)
    
    if (py_available()) {
      config <- py_config()
      cat("\n✓ Python 配置成功！\n")
      cat(sprintf("Python 路径: %s\n", config$python))
      cat(sprintf("Python 版本: %s\n", as.character(config$version)))
      python_configured <- TRUE
      break
    } else {
      # 尝试强制配置
      cat("  尝试强制配置...\n")
      use_python(python_path_normalized, required = TRUE)
      
      if (py_available()) {
        config <- py_config()
        cat("\n✓ Python 配置成功（强制模式）！\n")
        cat(sprintf("Python 路径: %s\n", config$python))
        cat(sprintf("Python 版本: %s\n", as.character(config$version)))
        python_configured <- TRUE
        break
      }
    }
  }, error = function(e) {
    cat(sprintf("  ✗ 失败: %s\n", e$message))
  })
  
  cat("\n")
}

if (!python_configured) {
  cat("\n✗ 所有路径都失败\n")
  cat("\n请手动运行:\n")
  cat("library(reticulate)\n")
  cat("use_python('C:/Users/linda/AppData/Local/Programs/Python/Python310/python.exe', required = TRUE)\n")
  cat("py_available()  # 应该返回 TRUE\n")
} else {
  cat("\n========================================\n")
  cat("✓ Python 配置完成！\n")
  cat("现在可以运行 evaluate_all_baseline_models.R\n")
  cat("========================================\n")
}

