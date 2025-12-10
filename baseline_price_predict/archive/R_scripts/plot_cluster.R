############################################
# plot_clusters.R
# 可视化：伦敦 Airbnb Cluster + 按价格排序 + 标注平均价格
############################################

library(ggplot2)
library(dplyr)
library(viridis)

# ---- Step 1: 加载清洗后的数据 ----
source("Dataclean.R")   # clean_df 会在这里可用

# ---- Step 2: 计算每个 cluster 的平均价格 ----
cluster_price <- clean_df %>%
  group_by(location_cluster) %>%
  summarise(mean_price = mean(price_num, na.rm = TRUE)) %>%
  arrange(mean_price)

# ---- Step 3: 重新排序 cluster 因子（便宜 → 贵） ----
clean_df$cluster_ordered <- factor(
  clean_df$location_cluster,
  levels = cluster_price$location_cluster
)

# ---- Step 4: 计算每个 cluster 的经纬度中心（用于 label）
cluster_centers <- clean_df %>%
  group_by(location_cluster) %>%
  summarise(
    center_lon = mean(longitude),
    center_lat = mean(latitude)
  ) %>%
  left_join(cluster_price, by = "location_cluster") %>%
  mutate(label_text = paste0("£", round(mean_price)))
# ---- Step X: 创建带价格的 legend label ----
cluster_price$legend_label <- paste0(
  "Cluster ", cluster_price$location_cluster,
  " (£", round(cluster_price$mean_price), ")"
)

# 为 clean_df 创建带 legend 的因子列
clean_df$cluster_label <- factor(
  clean_df$location_cluster,
  levels = cluster_price$location_cluster,
  labels = cluster_price$legend_label
)

# ---- Step 5: 画图（颜色按价格排序 + 标注平均价格）
print(
  ggplot(clean_df, aes(
    x = longitude,
    y = latitude,
    color = cluster_label   # ← 用新的 legend label
  )) +
    geom_point(alpha = 0.5, size = 1) +
    scale_color_viridis_d(option = "turbo") +
    coord_fixed(ratio = 1.3) +
    labs(
      title = "London Airbnb Clusters Ordered by Average Price",
      x = "Longitude",
      y = "Latitude",
      color = "Cluster (with Avg Price)"
    ) +
    theme_minimal()
)
