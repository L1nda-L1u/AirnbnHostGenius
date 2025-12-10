# =============================================
# 测试 Python 和 PyTorch 配置
# =============================================

library(reticulate)

cat("========================================\n")
cat("Testing Python and PyTorch Configuration\n")
cat("========================================\n\n")

# 1. 检查 Python
cat("1. Checking Python...\n")
if (py_available()) {
  config <- py_config()
  cat(sprintf("   ✓ Python is available\n"))
  cat(sprintf("   Python path: %s\n", config$python))
  cat(sprintf("   Python version: %s\n", as.character(config$version)))
} else {
  cat("   ✗ Python is NOT available\n")
  cat("   Attempting to configure...\n")
  
  # 尝试加载配置脚本
  config_script <- file.path(getwd(), "configure_python_simple.R")
  if (!file.exists(config_script)) {
    config_script <- file.path(dirname(getwd()), "configure_python_simple.R")
  }
  
  if (file.exists(config_script)) {
    source(config_script)
  } else {
    cat("   Configuration script not found. Please run:\n")
    cat("   source('R_scripts/configure_python_simple.R')\n")
  }
  
  if (py_available()) {
    config <- py_config()
    cat(sprintf("   ✓ Python configured successfully\n"))
    cat(sprintf("   Python path: %s\n", config$python))
  } else {
    stop("Python configuration failed. Please configure Python first.")
  }
}

cat("\n")

# 2. 检查 PyTorch
cat("2. Checking PyTorch...\n")
if (py_module_available("torch")) {
  cat("   ✓ PyTorch is available\n")
  
  # 获取 PyTorch 版本
  py_run_string("import torch; print(f'   PyTorch version: {torch.__version__}')")
} else {
  cat("   ✗ PyTorch is NOT available\n")
  cat("   Installing PyTorch...\n")
  
  tryCatch({
    py_install("torch", pip = TRUE)
    if (py_module_available("torch")) {
      cat("   ✓ PyTorch installed successfully\n")
      py_run_string("import torch; print(f'   PyTorch version: {torch.__version__}')")
    } else {
      stop("PyTorch installation failed")
    }
  }, error = function(e) {
    cat(sprintf("   ✗ PyTorch installation failed: %s\n", e$message))
    cat("   Please install manually: pip install torch\n")
    stop("PyTorch not available")
  })
}

cat("\n")

# 3. 测试简单的神经网络
cat("3. Testing simple neural network...\n")
tryCatch({
  py_run_string("
import torch
import torch.nn as nn
import numpy as np

# 创建简单的测试数据
X_test = np.random.randn(10, 5).astype(np.float32)
y_test = np.random.randn(10).astype(np.float32)

# 定义简单的模型
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(5, 1)
    
    def forward(self, x):
        return self.fc(x).squeeze()

model = SimpleNN()
X_tensor = torch.FloatTensor(X_test)
y_tensor = torch.FloatTensor(y_test)

# 前向传播
output = model(X_tensor)
loss = nn.MSELoss()(output, y_tensor)

print(f'   ✓ Model created successfully')
print(f'   Input shape: {X_tensor.shape}')
print(f'   Output shape: {output.shape}')
print(f'   Loss: {loss.item():.4f}')
")
  
  cat("   ✓ Neural network test passed!\n")
}, error = function(e) {
  cat(sprintf("   ✗ Neural network test failed: %s\n", e$message))
  if (exists("py_last_error", envir = asNamespace("reticulate"))) {
    tryCatch({
      py_error <- reticulate::py_last_error()
      cat(sprintf("   Python error: %s\n", py_error$message))
    }, error = function(e2) {
      # 忽略
    })
  }
  stop("Neural network test failed")
})

cat("\n========================================\n")
cat("✓ All tests passed! Python and PyTorch are ready.\n")
cat("You can now run evaluate_all_baseline_models.R\n")
cat("========================================\n")

