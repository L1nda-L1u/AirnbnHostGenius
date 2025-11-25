#############################################
# prepare_nn_training_v5_occupancy.R
# 准备预测 occupancy（入住率）的训练数据
# 保留 availability_30/60/90 作为特征
# 目标变量：estimated_occupancy_l365d
#############################################

library(dplyr)
library(tidyr)

df <- clean_df

drop_cols <- c(
  "id", "listing_url", "name", 
  "bathrooms_text", "price", 
  "amenities", "amenities_clean",
  "cluster_label", "cluster_ordered",
  
  # ❌ 删除 revenue（这是另一个目标变量，会造成泄漏）
  "estimated_revenue_l365d"
  
  # ✅ 保留 availability_30, availability_60, availability_90（作为特征）
  # ✅ 保留 estimated_occupancy_l365d（作为目标变量）
)

df <- df %>% select(-any_of(drop_cols))

# price
df$price_num <- as.numeric(df$price_num)

# embedding-ready: 将factor转换为数值ID
df$neighbourhood_id <- as.integer(as.factor(df$neighbourhood_cleansed)) - 1
df$room_type_id <- as.integer(as.factor(df$room_type)) - 1

# ✅ 添加 cluster 价格特征（在转换为ID之前）
if ("location_cluster" %in% names(df)) {
  # 计算每个 cluster 的价格统计（使用中位数，更稳健）
  cluster_price_stats <- df %>%
    group_by(location_cluster) %>%
    summarise(
      cluster_median_price = median(price_num, na.rm = TRUE),
      cluster_mean_price = mean(price_num, na.rm = TRUE),
      cluster_p25_price = quantile(price_num, 0.25, na.rm = TRUE),
      cluster_p75_price = quantile(price_num, 0.75, na.rm = TRUE),
      cluster_count = n(),
      .groups = "drop"
    )
  
  # 合并到原始数据
  df <- df %>%
    left_join(cluster_price_stats, by = "location_cluster")
  
  # 对于缺失的 cluster（如果有），用全局中位数填充
  global_median <- median(df$price_num, na.rm = TRUE)
  df$cluster_median_price <- ifelse(
    is.na(df$cluster_median_price), 
    global_median, 
    df$cluster_median_price
  )
  df$cluster_mean_price <- ifelse(
    is.na(df$cluster_mean_price), 
    global_median, 
    df$cluster_mean_price
  )
  
  cat("✓ 已添加 cluster 价格特征:\n")
  cat("  - cluster_median_price (中位数，推荐使用)\n")
  cat("  - cluster_mean_price (平均值)\n")
  cat("  - cluster_p25_price (25分位数)\n")
  cat("  - cluster_p75_price (75分位数)\n")
  cat("  - cluster_count (cluster 内房源数量)\n")
  
  # 转换为数值ID
  df$location_cluster_id <- as.integer(as.factor(df$location_cluster)) - 1
  df <- df %>% select(-location_cluster)  # 删除factor版本，保留数值版本
  cat("✓ location_cluster 已转换为 location_cluster_id\n")
}

df <- df %>% select(-neighbourhood_cleansed, -room_type)

df <- df %>% mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm=TRUE), .)))

df <- df %>% select(where(~ length(unique(.)) > 1))

# 检查关键列是否存在
cat("\n========================================\n")
cat("Feature Check for Occupancy Prediction\n")
cat("========================================\n\n")

if ("estimated_occupancy_l365d" %in% names(df)) {
  cat("✓ Target variable: estimated_occupancy_l365d\n")
  cat(sprintf("  Non-NA samples: %d (%.1f%%)\n",
              sum(!is.na(df$estimated_occupancy_l365d)),
              sum(!is.na(df$estimated_occupancy_l365d)) / nrow(df) * 100))
} else {
  cat("✗ WARNING: Target variable 'estimated_occupancy_l365d' not found!\n")
}

avail_cols <- c("availability_30", "availability_60", "availability_90")
for (col in avail_cols) {
  if (col %in% names(df)) {
    cat(sprintf("✓ Feature: %s\n", col))
  } else {
    cat(sprintf("✗ WARNING: Feature '%s' not found!\n", col))
  }
}

write.csv(df, "nn_occupancy_training_v5.csv", row.names = FALSE)

cat("\n✔ nn_occupancy_training_v5.csv 已生成（用于预测 occupancy）\n")
cat("\nFeatures included:\n")
cat("  - availability_30, availability_60, availability_90 (作为特征)\n")
cat("  - estimated_occupancy_l365d (作为目标变量)\n")
cat("  - price_num, location, room_type, amenities, etc.\n")
cat("  - cluster price features\n")

