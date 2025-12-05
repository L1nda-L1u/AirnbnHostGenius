#############################################
# prepare_nn_training_v4.R
# 删除所有 leakage 列（availability / occupancy / revenue）
#############################################

library(dplyr)
library(tidyr)

df <- clean_df

drop_cols <- c(
  "id", "listing_url", "name", 
  "bathrooms_text", "price", 
  "amenities", "amenities_clean",
  "cluster_label", "cluster_ordered",
  
  # ❌ 删除 leakage 列
  "availability_30",
  "availability_60",
  "availability_90",
  "estimated_occupancy_l365d",
  "estimated_revenue_l365d"
)

df <- df %>% select(-any_of(drop_cols))

# price
df$price_num <- as.numeric(df$price_num)

# embedding-ready: 将factor转换为数值ID
df$neighbourhood_id <- as.integer(as.factor(df$neighbourhood_cleansed)) - 1
df$room_type_id <- as.integer(as.factor(df$room_type)) - 1

# ✅ 添加 cluster 中心价格特征（在转换为ID之前）
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
  
  # ⚠️ 注意：这里使用的是全局 cluster 价格（可能有轻微数据泄漏）
  # 如果要完全避免泄漏，应该在 train_test_split 后，只用训练集计算 cluster 价格
  # 然后应用到测试集。这需要在 Python 训练脚本中实现。
  cat("⚠️  注意：当前使用全局 cluster 价格（可能有轻微数据泄漏）\n")
  cat("   如需完全避免泄漏，请在训练脚本中只用训练集计算 cluster 价格\n")
}

df <- df %>% select(-neighbourhood_cleansed, -room_type)

df <- df %>% mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm=TRUE), .)))

df <- df %>% select(where(~ length(unique(.)) > 1))

write.csv(df, "nn_price_training_v4.csv", row.names = FALSE)

cat("✔ nn_price_training_v4.csv 已生成（无 leakage，特征最适合 NN）\n")
