#############################################
# prepare_occupancy_training_data.R
# 准备预测 occupancy（入住率）的训练数据
# 目标变量：estimated_occupancy_l365d
#############################################

library(dplyr)
library(tidyr)

# 假设 clean_df 已经存在（从 Dataclean.R 生成）
# 如果没有，需要先运行 Dataclean.R

df <- clean_df

# =============================================
# 分析：预测 occupancy 应该保留哪些列？
# =============================================

cat("========================================\n")
cat("Occupancy Prediction - Feature Analysis\n")
cat("========================================\n\n")

cat("Target variable: estimated_occupancy_l365d (过去365天的估算入住率)\n\n")

cat("Features to KEEP (对预测 occupancy 有用):\n")
cat("  ✓ availability_30, availability_60, availability_90\n")
cat("    - 这些是未来可用天数，与历史入住率相关但不完全相同\n")
cat("    - 可以作为特征使用（不是目标变量）\n\n")

cat("  ✓ price_num, price_per_person (价格相关)\n")
cat("    - 价格影响需求，进而影响入住率\n\n")

cat("  ✓ accommodates, bedrooms, beds, bath_num (房源容量)\n")
cat("    - 容量影响目标客户群体，影响入住率\n\n")

cat("  ✓ room_type (房间类型)\n")
cat("    - Entire home vs Private room 入住率差异大\n\n")

cat("  ✓ location_cluster, neighbourhood_cleansed (地理位置)\n")
cat("    - 地理位置是影响入住率的关键因素\n\n")

cat("  ✓ review_scores_cleanliness, review_scores_location (评分)\n")
cat("    - 评分影响吸引力，进而影响入住率\n\n")

cat("  ✓ amenities (设施)\n")
cat("    - 设施影响吸引力\n\n")

cat("Features to DROP (对预测 occupancy 无用或泄漏):\n")
cat("  ✗ estimated_revenue_l365d\n")
cat("    - 这是另一个目标变量（revenue = price × occupancy），会造成数据泄漏\n\n")

cat("  ✗ id, listing_url, name, bathrooms_text\n")
cat("    - 标识符或文本，对预测无用\n\n")

cat("  ✗ amenities_clean (已转换为 one-hot，保留 one-hot 列)\n\n")

# =============================================
# 删除不需要的列
# =============================================

drop_cols <- c(
  # 标识符和文本
  "id", "listing_url", "name", 
  "bathrooms_text", 
  "amenities", "amenities_clean",  # amenities 已转换为 one-hot
  "cluster_label", "cluster_ordered",
  
  # ❌ 删除 revenue（这是另一个目标变量，会造成泄漏）
  "estimated_revenue_l365d",
  
  # ✅ 保留 availability_30/60/90（这些是特征，不是目标）
  # ✅ 保留 estimated_occupancy_l365d（这是目标变量）
)

df <- df %>% select(-any_of(drop_cols))

# =============================================
# 检查目标变量
# =============================================

cat("\n========================================\n")
cat("Target Variable Check\n")
cat("========================================\n\n")

if ("estimated_occupancy_l365d" %in% names(df)) {
  occupancy_data <- df$estimated_occupancy_l365d
  cat(sprintf("Target variable: estimated_occupancy_l365d\n"))
  cat(sprintf("  Total samples: %d\n", length(occupancy_data)))
  cat(sprintf("  Non-NA samples: %d (%.1f%%)\n", 
              sum(!is.na(occupancy_data)),
              sum(!is.na(occupancy_data)) / length(occupancy_data) * 100))
  cat(sprintf("  NA samples: %d (%.1f%%)\n",
              sum(is.na(occupancy_data)),
              sum(is.na(occupancy_data)) / length(occupancy_data) * 100))
  
  if (sum(!is.na(occupancy_data)) > 0) {
    cat(sprintf("\n  Statistics (non-NA):\n"))
    cat(sprintf("    Mean: %.2f\n", mean(occupancy_data, na.rm = TRUE)))
    cat(sprintf("    Median: %.2f\n", median(occupancy_data, na.rm = TRUE)))
    cat(sprintf("    Min: %.2f\n", min(occupancy_data, na.rm = TRUE)))
    cat(sprintf("    Max: %.2f\n", max(occupancy_data, na.rm = TRUE)))
    cat(sprintf("    SD: %.2f\n", sd(occupancy_data, na.rm = TRUE)))
  }
} else {
  stop("Target variable 'estimated_occupancy_l365d' not found!")
}

# =============================================
# 检查 availability 特征
# =============================================

cat("\n========================================\n")
cat("Availability Features Check\n")
cat("========================================\n\n")

avail_cols <- c("availability_30", "availability_60", "availability_90")
for (col in avail_cols) {
  if (col %in% names(df)) {
    cat(sprintf("✓ %s: present\n", col))
    cat(sprintf("  Non-NA: %d (%.1f%%)\n",
                sum(!is.na(df[[col]])),
                sum(!is.na(df[[col]])) / nrow(df) * 100))
  } else {
    cat(sprintf("✗ %s: missing\n", col))
  }
}

# =============================================
# 数据预处理
# =============================================

cat("\n========================================\n")
cat("Data Preprocessing\n")
cat("========================================\n\n")

# price
df$price_num <- as.numeric(df$price_num)

# embedding-ready: 将factor转换为数值ID
df$neighbourhood_id <- as.integer(as.factor(df$neighbourhood_cleansed)) - 1
df$room_type_id <- as.integer(as.factor(df$room_type)) - 1

# ✅ 添加 cluster 价格特征（在转换为ID之前）
if ("location_cluster" %in% names(df)) {
  # 计算每个 cluster 的价格统计
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
  
  # 对于缺失的 cluster，用全局中位数填充
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
  
  cat("✓ Added cluster price features\n")
  
  # 转换为数值ID
  df$location_cluster_id <- as.integer(as.factor(df$location_cluster)) - 1
  df <- df %>% select(-location_cluster)  # 删除factor版本
  cat("✓ location_cluster converted to location_cluster_id\n")
}

# 删除原始 factor 列（已转换为 ID）
df <- df %>% select(-neighbourhood_cleansed, -room_type)

# 填充缺失值（用中位数）
df <- df %>% mutate(across(where(is.numeric), ~ ifelse(is.na(.), median(., na.rm=TRUE), .)))

# 删除只有一个值的列
df <- df %>% select(where(~ length(unique(.)) > 1))

# =============================================
# 最终检查
# =============================================

cat("\n========================================\n")
cat("Final Dataset Summary\n")
cat("========================================\n\n")

cat(sprintf("Total rows: %d\n", nrow(df)))
cat(sprintf("Total columns: %d\n", ncol(df)))
cat(sprintf("\nColumn names:\n"))
for (i in seq_along(names(df))) {
  cat(sprintf("  %d. %s\n", i, names(df)[i]))
}

# 检查目标变量是否还在
if ("estimated_occupancy_l365d" %in% names(df)) {
  cat("\n✓ Target variable 'estimated_occupancy_l365d' is present\n")
} else {
  stop("ERROR: Target variable 'estimated_occupancy_l365d' was removed!")
}

# 检查 availability 特征是否还在
avail_present <- sum(avail_cols %in% names(df))
cat(sprintf("\n✓ Availability features present: %d/%d\n", avail_present, length(avail_cols)))

# =============================================
# 保存
# =============================================

write.csv(df, "nn_occupancy_training.csv", row.names = FALSE)

cat("\n========================================\n")
cat("✓ nn_occupancy_training.csv generated\n")
cat("========================================\n")
cat("\nNext steps:\n")
cat("  1. Check the dataset: nn_occupancy_training.csv\n")
cat("  2. Train a model to predict 'estimated_occupancy_l365d'\n")
cat("  3. Features include: availability_30/60/90, price, location, etc.\n")
cat("  4. Target: estimated_occupancy_l365d (0-1 or percentage)\n")

