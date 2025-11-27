# Sensitivity Analysis - 敏感性分析

这个文件夹包含用于测试输入特征变化对baseline定价模型影响的所有脚本。

## 文件说明

- **`sensitivity_analysis.R`** - 主脚本，包含所有核心功能
- **`example_sensitivity_test.R`** - 示例脚本，展示如何使用
- **`README.md`** - 本文件，使用说明

## 功能

- 测试单个特征变化对价格的影响（如：加/减电视amenity）
- 批量测试多个特征的影响
- 支持任何特征类型（amenity、bedrooms、accommodates等）
- 显示价格变化的绝对值和百分比

## 快速开始

### 1. 基本使用

```r
# 加载脚本（从sensitivity_analysis文件夹中运行）
source("sensitivity_analysis.R")

# 获取一个示例房源特征
example <- get_example_features()

# 测试单个特征（例如：TV amenity）
results <- sensitivity_test(example, "amenity_TV", c(0, 1))
```

### 2. 测试Amenity特征

```r
# 查找所有amenity特征
amenity_cols <- get_amenity_features()

# 测试电视的影响
results_tv <- sensitivity_test(example, "amenity_TV", c(0, 1))

# 测试WiFi的影响
results_wifi <- sensitivity_test(example, "amenity_WiFi", c(0, 1))
```

### 3. 批量测试多个特征

```r
# 测试多个amenity
amenities_to_test <- c("amenity_TV", "amenity_WiFi", "amenity_Air.conditioning")
results <- batch_sensitivity_test(example, amenities_to_test)
```

### 4. 测试非amenity特征

```r
# 测试bedrooms的影响
results_bedrooms <- sensitivity_test(example, "bedrooms", c(1, 2, 3, 4))

# 测试accommodates的影响
results_acc <- sensitivity_test(example, "accommodates", c(2, 4, 6, 8))
```

### 5. 运行完整示例

```r
source("example_sensitivity_test.R")
```

## 目录结构

```
baseline_price_predict/
├── baseprice_model/          # 模型文件（自动查找）
│   ├── best_xgb_log_model.xgb
│   ├── best_price_A2_log_pytorch.pt
│   └── ...
└── sensitivity_analysis/     # 敏感性分析脚本（本文件夹）
    ├── sensitivity_analysis.R
    ├── example_sensitivity_test.R
    └── README.md
```

## 如何运行

### 从sensitivity_analysis文件夹运行（推荐）

```r
# 切换到sensitivity_analysis文件夹
setwd("sensitivity_analysis")

# 运行脚本
source("sensitivity_analysis.R")
```

### 从项目根目录运行

```r
# 从项目根目录
source("sensitivity_analysis/sensitivity_analysis.R")
```

### 从任何目录运行

脚本会自动查找`baseprice_model`目录，无论你在哪个目录运行。

## 输出说明

`sensitivity_test()` 函数返回一个数据框，包含：
- `feature_name`: 特征名称
- `feature_value`: 特征值
- `xgb_pred`: XGBoost预测价格
- `nn_pred`: Neural Network预测价格
- `final_price`: 最终stacking预测价格
- `price_change`: 相对于基础价格的变化（绝对值）
- `price_change_pct`: 相对于基础价格的变化（百分比）

## 注意事项

1. 确保 `baseprice_model` 目录存在且包含所有模型文件
2. 需要安装并配置 Python 和 PyTorch（用于Neural Network模型）
3. 特征名称必须与训练数据中的特征名称完全匹配
4. 对于二元特征（如amenity），通常使用 `c(0, 1)` 来测试有无的影响

## 模型说明

这个脚本使用baseprice_model中的最终stacking模型：
- XGBoost模型（用于预测）
- Neural Network模型（PyTorch）
- Ridge Meta Model（融合XGBoost和NN的预测）

最终价格 = intercept + xgb_coef * xgb_pred + nn_coef * nn_pred

## 常见问题

**Q: 如何找到特定的amenity特征？**
A: 使用 `get_amenity_features()` 查看所有amenity特征，然后用 `grep()` 搜索：
```r
amenity_cols <- get_amenity_features()
wifi_features <- grep("WiFi|wifi", amenity_cols, value = TRUE, ignore.case = TRUE)
```

**Q: 如何测试多个特征同时变化？**
A: 创建自定义特征向量：
```r
custom <- example
custom$amenity_TV <- 1
custom$bedrooms <- 3
pred <- predict_price_stacking(custom)
```

**Q: 如何从自己的数据创建特征向量？**
A: 确保特征向量包含所有训练数据中的特征列，缺失的特征会被设为0：
```r
my_features <- data.frame(matrix(0, nrow = 1, ncol = length(feature_cols)))
colnames(my_features) <- feature_cols
# 然后设置你需要的特征值
my_features$bedrooms <- 2
my_features$accommodates <- 4
# ... 等等
```

