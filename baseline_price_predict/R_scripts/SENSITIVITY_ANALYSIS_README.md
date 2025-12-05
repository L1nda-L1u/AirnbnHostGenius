# Sensitivity Analysis - 敏感性分析

这个脚本用于测试输入特征变化对baseline定价模型的影响。

## 功能

- 测试单个特征变化对价格的影响（如：加/减电视amenity）
- 批量测试多个特征的影响
- 支持任何特征类型（amenity、bedrooms、accommodates等）
- 显示价格变化的绝对值和百分比

## 使用方法

### 1. 基本使用

```r
# 加载脚本
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

### 5. 自定义场景测试

```r
# 创建自定义特征向量
custom_features <- example
custom_features$amenity_TV <- 0  # 没有电视
custom_features$bedrooms <- 2

# 预测价格
pred <- predict_price_stacking(custom_features)
cat(sprintf("Predicted price: £%.2f\n", pred$final_price))

# 测试添加电视后的影响
custom_features$amenity_TV <- 1
pred_with_tv <- predict_price_stacking(custom_features)
cat(sprintf("With TV: £%.2f (change: £%.2f)\n", 
            pred_with_tv$final_price,
            pred_with_tv$final_price - pred$final_price))
```

## 运行示例

运行完整的示例脚本：

```r
source("example_sensitivity_test.R")
```

这将展示：
1. 如何测试TV amenity的影响
2. 如何批量测试多个amenity
3. 如何测试非amenity特征（bedrooms, accommodates）
4. 如何创建自定义测试场景

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

