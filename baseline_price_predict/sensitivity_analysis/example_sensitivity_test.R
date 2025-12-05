### ---------------------------------------------------------
### example_sensitivity_test.R
### 示例：如何使用sensitivity_analysis.R测试特征变化对价格的影响
### ---------------------------------------------------------

# 加载敏感性分析脚本
source("sensitivity_analysis.R")

cat("\n")
cat("========================================\n")
cat("Example: Testing Amenity Impact on Price\n")
cat("========================================\n\n")

# =============================================
# 示例1: 测试电视(TV)amenity的影响
# =============================================

cat("Example 1: Testing TV amenity impact\n")
cat("----------------------------------------\n")

# 获取一个示例房源
example <- get_example_features()

# 查找TV相关的amenity特征
amenity_cols <- get_amenity_features()
tv_features <- grep("TV|Television|电视", amenity_cols, value = TRUE, ignore.case = TRUE)

if (length(tv_features) > 0) {
  cat(sprintf("Found TV-related features: %s\n", paste(tv_features, collapse = ", ")))
  
  # 测试第一个TV特征
  tv_feature <- tv_features[1]
  cat(sprintf("\nTesting feature: %s\n", tv_feature))
  
  # 测试：没有TV (0) vs 有TV (1)
  results_tv <- sensitivity_test(example, tv_feature, c(0, 1))
  
  cat("\nSummary:\n")
  cat(sprintf("  Without %s: £%.2f\n", tv_feature, results_tv$final_price[1]))
  cat(sprintf("  With %s:    £%.2f\n", tv_feature, results_tv$final_price[2]))
  cat(sprintf("  Price change: £%.2f (%.2f%%)\n", 
              results_tv$price_change[2], 
              results_tv$price_change_pct[2]))
} else {
  cat("No TV-related amenity features found. Available amenity features:\n")
  cat(paste(head(amenity_cols, 10), collapse = ", "), "...\n")
  
  # 测试第一个amenity特征作为示例
  if (length(amenity_cols) > 0) {
    test_feature <- amenity_cols[1]
    cat(sprintf("\nTesting first amenity feature as example: %s\n", test_feature))
    results <- sensitivity_test(example, test_feature, c(0, 1))
  }
}

# =============================================
# 示例2: 测试多个amenity的影响
# =============================================

cat("\n\n")
cat("Example 2: Testing multiple amenities\n")
cat("----------------------------------------\n")

# 选择几个常见的amenity特征
if (length(amenity_cols) >= 3) {
  test_amenities <- amenity_cols[1:min(3, length(amenity_cols))]
  cat(sprintf("Testing amenities: %s\n", paste(test_amenities, collapse = ", ")))
  
  # 批量测试
  batch_results <- batch_sensitivity_test(example, test_amenities)
  
  # 显示结果摘要
  cat("\nSummary of amenity impacts:\n")
  for (feat in test_amenities) {
    feat_results <- batch_results[batch_results$feature_name == feat, ]
    if (nrow(feat_results) >= 2) {
      base_price <- feat_results$final_price[1]
      with_price <- feat_results$final_price[2]
      change <- with_price - base_price
      change_pct <- (change / base_price) * 100
      
      cat(sprintf("  %s:\n", feat))
      cat(sprintf("    Without: £%.2f, With: £%.2f\n", base_price, with_price))
      cat(sprintf("    Impact: £%.2f (%.2f%%)\n\n", change, change_pct))
    }
  }
}

# =============================================
# 示例3: 测试非amenity特征（如bedrooms, accommodates）
# =============================================

cat("\n\n")
cat("Example 3: Testing non-amenity features\n")
cat("----------------------------------------\n")

# 测试bedrooms的影响
cat("Testing bedrooms impact:\n")
if ("bedrooms" %in% feature_cols) {
  results_bedrooms <- sensitivity_test(example, "bedrooms", c(1, 2, 3, 4))
  
  cat("\nBedrooms impact summary:\n")
  for (i in 1:nrow(results_bedrooms)) {
    cat(sprintf("  %d bedrooms: £%.2f", 
                results_bedrooms$feature_value[i],
                results_bedrooms$final_price[i]))
    if (i > 1) {
      cat(sprintf(" (change: £%.2f, %.2f%%)", 
                  results_bedrooms$price_change[i],
                  results_bedrooms$price_change_pct[i]))
    }
    cat("\n")
  }
}

# 测试accommodates的影响
cat("\nTesting accommodates impact:\n")
if ("accommodates" %in% feature_cols) {
  results_acc <- sensitivity_test(example, "accommodates", c(2, 4, 6, 8))
  
  cat("\nAccommodates impact summary:\n")
  for (i in 1:nrow(results_acc)) {
    cat(sprintf("  %d accommodates: £%.2f", 
                results_acc$feature_value[i],
                results_acc$final_price[i]))
    if (i > 1) {
      cat(sprintf(" (change: £%.2f, %.2f%%)", 
                  results_acc$price_change[i],
                  results_acc$price_change_pct[i]))
    }
    cat("\n")
  }
}

# =============================================
# 示例4: 自定义测试场景
# =============================================

cat("\n\n")
cat("Example 4: Custom scenario testing\n")
cat("----------------------------------------\n")

cat("Scenario: What if we add WiFi and TV to a basic listing?\n")

# 创建一个基础场景（假设没有WiFi和TV）
base_scenario <- example
wifi_feature <- grep("WiFi|Wifi|wifi|无线", amenity_cols, value = TRUE, ignore.case = TRUE)[1]
tv_feature <- grep("TV|Television|电视", amenity_cols, value = TRUE, ignore.case = TRUE)[1]

if (!is.na(wifi_feature) && wifi_feature %in% feature_cols) {
  base_scenario[[wifi_feature]] <- 0
}
if (!is.na(tv_feature) && tv_feature %in% feature_cols) {
  base_scenario[[tv_feature]] <- 0
}

# 预测基础价格
base_pred <- predict_price_stacking(base_scenario)
cat(sprintf("Base scenario (no WiFi, no TV): £%.2f\n", base_pred$final_price))

# 添加WiFi
if (!is.na(wifi_feature) && wifi_feature %in% feature_cols) {
  scenario_wifi <- base_scenario
  scenario_wifi[[wifi_feature]] <- 1
  pred_wifi <- predict_price_stacking(scenario_wifi)
  cat(sprintf("With WiFi: £%.2f (change: £%.2f, %.2f%%)\n", 
              pred_wifi$final_price,
              pred_wifi$final_price - base_pred$final_price,
              (pred_wifi$final_price - base_pred$final_price) / base_pred$final_price * 100))
}

# 添加TV
if (!is.na(tv_feature) && tv_feature %in% feature_cols) {
  scenario_tv <- base_scenario
  scenario_tv[[tv_feature]] <- 1
  pred_tv <- predict_price_stacking(scenario_tv)
  cat(sprintf("With TV: £%.2f (change: £%.2f, %.2f%%)\n", 
              pred_tv$final_price,
              pred_tv$final_price - base_pred$final_price,
              (pred_tv$final_price - base_pred$final_price) / base_pred$final_price * 100))
}

# 同时添加WiFi和TV
if (!is.na(wifi_feature) && !is.na(tv_feature) && 
    wifi_feature %in% feature_cols && tv_feature %in% feature_cols) {
  scenario_both <- base_scenario
  scenario_both[[wifi_feature]] <- 1
  scenario_both[[tv_feature]] <- 1
  pred_both <- predict_price_stacking(scenario_both)
  cat(sprintf("With WiFi + TV: £%.2f (change: £%.2f, %.2f%%)\n", 
              pred_both$final_price,
              pred_both$final_price - base_pred$final_price,
              (pred_both$final_price - base_pred$final_price) / base_pred$final_price * 100))
}

cat("\n")
cat("========================================\n")
cat("Examples completed!\n")
cat("========================================\n")
cat("\nYou can now use these functions to test any feature changes:\n")
cat("  - sensitivity_test() for single feature\n")
cat("  - batch_sensitivity_test() for multiple features\n")
cat("  - predict_price_stacking() for custom predictions\n")

