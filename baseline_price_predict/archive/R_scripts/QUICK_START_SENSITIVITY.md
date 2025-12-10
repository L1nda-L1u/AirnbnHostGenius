# 快速开始 - 敏感性分析

## 最简单的使用方式

### 1. 测试电视(TV)amenity的影响

```r
# 加载脚本
source("sensitivity_analysis.R")

# 获取一个示例
example <- get_example_features()

# 查找TV相关的特征
amenity_cols <- get_amenity_features()
tv_feature <- grep("TV", amenity_cols, value = TRUE, ignore.case = TRUE)[1]

# 测试TV的影响
if (!is.na(tv_feature)) {
  results <- sensitivity_test(example, tv_feature, c(0, 1))
  cat(sprintf("\n没有TV: £%.2f\n", results$final_price[1]))
  cat(sprintf("有TV:    £%.2f\n", results$final_price[2]))
  cat(sprintf("价格变化: £%.2f (%.2f%%)\n", 
              results$price_change[2], 
              results$price_change_pct[2]))
}
```

### 2. 测试任何amenity

```r
# 查看所有可用的amenity特征
amenity_cols <- get_amenity_features()
print(amenity_cols)

# 测试任意一个（例如第一个）
if (length(amenity_cols) > 0) {
  test_feature <- amenity_cols[1]
  results <- sensitivity_test(example, test_feature, c(0, 1))
}
```

### 3. 测试其他特征（如bedrooms）

```r
# 测试bedrooms从1到4的影响
results <- sensitivity_test(example, "bedrooms", c(1, 2, 3, 4))
print(results)
```

### 4. 运行完整示例

```r
source("example_sensitivity_test.R")
```

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

