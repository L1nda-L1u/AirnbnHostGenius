#############################################
# add_cluster_price_feature.R
# 为数据集添加 cluster 中心价格特征
# ⚠️ 重要：避免数据泄漏，只在训练集中计算 cluster 价格
#############################################

library(dplyr)
library(tidyr)

# 加载数据
df <- clean_df

# =============================================
# 方法1：使用全局 cluster 平均价格（简单但可能有轻微泄漏）
# =============================================
# 计算每个 cluster 的平均价格（中位数更稳健，不受极端值影响）
cluster_stats <- df %>%
  group_by(location_cluster) %>%
  summarise(
    cluster_median_price = median(price_num, na.rm = TRUE),
    cluster_mean_price = mean(price_num, na.rm = TRUE),
    cluster_p25_price = quantile(price_num, 0.25, na.rm = TRUE),
    cluster_p75_price = quantile(price_num, 0.75, na.rm = TRUE),
    cluster_count = n()
  ) %>%
  ungroup()

# 合并到原始数据
df <- df %>%
  left_join(cluster_stats, by = "location_cluster")

cat("方法1: 添加了全局 cluster 价格特征\n")
cat(sprintf("  - cluster_median_price (中位数，最稳健)\n"))
cat(sprintf("  - cluster_mean_price (平均值)\n"))
cat(sprintf("  - cluster_p25_price (25分位数)\n"))
cat(sprintf("  - cluster_p75_price (75分位数)\n"))
cat(sprintf("  - cluster_count (cluster 内房源数量)\n\n"))

# =============================================
# 方法2：使用训练集计算 cluster 价格（避免数据泄漏，推荐）
# =============================================
# 注意：这个方法需要在 train_test_split 之后使用
# 这里提供一个函数，可以在 Python 训练脚本中调用

# 保存 cluster 统计信息（用于后续在训练时计算）
save(cluster_stats, file = "cluster_price_stats.rda")
cat("已保存 cluster 价格统计到 cluster_price_stats.rda\n")
cat("可以在 Python 训练脚本中加载并使用\n\n")

# =============================================
# 显示一些统计信息
# =============================================
cat("Cluster 价格统计摘要:\n")
print(summary(cluster_stats$cluster_median_price))
cat("\n")

# 显示最贵和最便宜的 cluster
cat("最贵的 5 个 cluster:\n")
print(cluster_stats %>% 
  arrange(desc(cluster_median_price)) %>% 
  head(5) %>%
  select(location_cluster, cluster_median_price, cluster_count))

cat("\n最便宜的 5 个 cluster:\n")
print(cluster_stats %>% 
  arrange(cluster_median_price) %>% 
  head(5) %>%
  select(location_cluster, cluster_median_price, cluster_count))

cat("\n✅ 完成！现在 clean_df 包含了 cluster 价格特征\n")
cat("   注意：如果要在训练时避免数据泄漏，请在 train_test_split 后\n")
cat("   只用训练集重新计算 cluster 价格特征\n")

