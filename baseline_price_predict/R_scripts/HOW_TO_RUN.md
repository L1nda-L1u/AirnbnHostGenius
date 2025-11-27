# 如何运行敏感性分析脚本

## 当前问题

如果你看到错误 "cannot open file 'R_scripts/sensitivity_analysis.R': No such file or directory"，说明当前工作目录不正确。

## 解决方案

### 方法1: 切换到正确的目录（推荐）

在RStudio控制台中运行：

```r
# 如果你当前在 baseprice_model 目录中
setwd("..")  # 切换到项目根目录

# 然后运行脚本
source("R_scripts/sensitivity_analysis.R")
```

### 方法2: 使用绝对路径

```r
# 直接使用完整路径
source("C:/Users/linda/Desktop/Data to Product/AirbnbHostGeniusR/baseline_price_predict/R_scripts/sensitivity_analysis.R")
```

### 方法3: 从baseprice_model目录运行（已修复）

脚本现在会自动检测如果你在baseprice_model目录中，可以直接运行：

```r
# 即使你在baseprice_model目录中，也可以运行
source("../R_scripts/sensitivity_analysis.R")
```

## 检查当前目录

运行以下命令查看当前目录：

```r
getwd()
```

## 推荐的目录结构

```
baseline_price_predict/
├── baseprice_model/          # 模型文件在这里
│   ├── best_xgb_log_model.xgb
│   ├── best_price_A2_log_pytorch.pt
│   └── ...
└── R_scripts/                # 脚本在这里
    ├── sensitivity_analysis.R
    └── ...
```

## 最佳实践

1. **在项目根目录运行**（推荐）：
   ```r
   setwd("C:/Users/linda/Desktop/Data to Product/AirbnbHostGeniusR/baseline_price_predict")
   source("R_scripts/sensitivity_analysis.R")
   ```

2. **或者在R_scripts目录运行**：
   ```r
   setwd("C:/Users/linda/Desktop/Data to Product/AirbnbHostGeniusR/baseline_price_predict/R_scripts")
   source("sensitivity_analysis.R")
   ```

3. **脚本会自动查找baseprice_model目录**，无论你在哪个目录运行。

