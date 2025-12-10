### -----------------------------------------
### Dataclean.R  —— AirbnbHostGenius 清洗流程（稳定版）
### -----------------------------------------

library(readr)
library(dplyr)
library(stringr)
library(purrr)
library(tidyr)
library(jsonlite)

# ---- 1. 读取数据 ----
df <- read_csv("../listings.csv.gz")

# ---- 2. 选择需要的列 ----
df_selected <- df %>%
  select(
    id,
    listing_url,
    name,
    latitude,
    longitude,
    neighbourhood_cleansed,
    room_type,
    accommodates,
    bathrooms,
    bathrooms_text,
    bedrooms,
    beds,
    amenities,
    price,
    availability_30,
    availability_60,
    availability_90,
    estimated_occupancy_l365d,
    estimated_revenue_l365d,
    review_scores_cleanliness,
    review_scores_location
  )

# ---- 3. 清洗 bathrooms_text ----
clean_df <- df_selected %>%
  mutate(
    bath_num = as.numeric(str_extract(bathrooms_text, "\\d+\\.?\\d*")),
    bath_shared = case_when(
      str_detect(bathrooms_text, "shared") ~ 1,
      str_detect(bathrooms_text, "private") ~ 0,
      TRUE ~ 0
    ),
    bath_type_unknown = if_else(is.na(bathrooms_text), 1, 0)
  ) %>%
  mutate(
    bath_num = if_else(is.na(bath_num) & !is.na(bedrooms),
                       bedrooms * 0.7,
                       bath_num)
  ) %>%
  mutate(
    amenities_clean = amenities %>%
      str_replace_all("\\[|\\]", "") %>%
      str_replace_all('"', "") %>%
      str_split(", ")
  )

# ---- 4. Amenities one-hot ----
top_amenities <- clean_df$amenities_clean %>%
  unlist() %>% table() %>%
  sort(decreasing = TRUE) %>%
  head(30) %>% names()

for (a in top_amenities) {
  clean_df[[paste0("amenity_", make.names(a))]] <-
    map_lgl(clean_df$amenities_clean, ~ a %in% .x) %>% as.integer()
}

# ---- 5. 清洗价格并删除 NA ----
clean_df <- clean_df %>%
  mutate(price_num = as.numeric(gsub("[\\$,]", "", price))) %>%
  filter(!is.na(price_num))

# 删除缺 bedrooms 的行（模型需要）
clean_df <- clean_df %>% filter(!is.na(bedrooms))

### ---- 清洗经纬度（强力过滤所有坏值） ----

clean_df <- clean_df %>%
  filter(
    !is.na(latitude), !is.na(longitude),             # 删除 NA
    latitude > -90, latitude < 90,                   # 纬度范围
    longitude > -180, longitude < 180               # 经度范围
  )


# ---- 6. 完善的价格异常值清理（基于accommodates的合理范围）----
cat("清理价格异常值...\n")
n_before <- nrow(clean_df)

# 规则1: 人均价格过滤 - 删除人均价格 > 200镑的数据
clean_df <- clean_df %>%
  mutate(price_per_person = price_num / accommodates) %>%
  filter(price_per_person <= 200)
cat(sprintf("  规则1: 删除人均价格 > 200镑: 剩余 %d 行\n", nrow(clean_df)))

# 规则2: 2人及以下但价格>400（不合理）
clean_df <- clean_df %>% filter(!((accommodates <= 2) & (price_num > 400)))

# 规则3: 4人及以下但价格>600（不合理）
clean_df <- clean_df %>% filter(!((accommodates <= 4) & (price_num > 600)))

# 规则4: 6人及以下但价格>800（不合理）
clean_df <- clean_df %>% filter(!((accommodates <= 6) & (price_num > 800)))

# 规则5: 8人及以下但价格>1000（不合理）
clean_df <- clean_df %>% filter(!((accommodates <= 8) & (price_num > 1000)))

# 规则6: 删除99.5%分位数以上的极端值
upper_quantile <- quantile(clean_df$price_num, 0.995, na.rm = TRUE)
clean_df <- clean_df %>% filter(price_num <= upper_quantile)

# ---- 6.1. 基于 availability 的价格合理性检查 ----
cat("\n基于 availability 检查价格合理性...\n")
n_before_avail <- nrow(clean_df)

# availability_30/60/90 表示未来30/60/90天的可用天数（不是已租天数）
# 如果 availability_90 接近 90，说明未来90天几乎都空着，可能是价格定太高
# 估算：如果 availability_90 >= 85（即90天中只有5天被租），可能价格不合理

# 计算实际租出天数（90 - availability_90）
clean_df <- clean_df %>%
  mutate(
    # 未来90天租出天数（估算）
    booked_days_90 = ifelse(!is.na(availability_90), 90 - availability_90, NA),
    # 未来60天租出天数
    booked_days_60 = ifelse(!is.na(availability_60), 60 - availability_60, NA),
    # 未来30天租出天数
    booked_days_30 = ifelse(!is.na(availability_30), 30 - availability_30, NA),
    # 估算年租出天数（基于90天数据推算）
    estimated_yearly_booked = ifelse(
      !is.na(booked_days_90), 
      booked_days_90 * (365 / 90), 
      NA
    )
  )

# 规则7: 如果估算年租出天数 < 10天，可能是乱定价，删除
# 但要注意：availability 可能缺失，所以只对有效数据应用此规则
clean_df <- clean_df %>%
  filter(
    is.na(estimated_yearly_booked) |  # 如果 availability 数据缺失，保留
    estimated_yearly_booked >= 10     # 如果年租出天数 >= 10天，保留
  )

n_after_avail <- nrow(clean_df)
cat(sprintf("  规则7: 删除年租出天数 < 10天的房源: 删除了 %d 行 (%.2f%%)\n", 
            n_before_avail - n_after_avail,
            (n_before_avail - n_after_avail) / n_before_avail * 100))

# 显示一些统计
if (sum(!is.na(clean_df$estimated_yearly_booked)) > 0) {
  cat("\n  年租出天数统计:\n")
  cat(sprintf("    平均: %.1f 天\n", mean(clean_df$estimated_yearly_booked, na.rm = TRUE)))
  cat(sprintf("    中位数: %.1f 天\n", median(clean_df$estimated_yearly_booked, na.rm = TRUE)))
  cat(sprintf("    最小值: %.1f 天\n", min(clean_df$estimated_yearly_booked, na.rm = TRUE)))
  cat(sprintf("    最大值: %.1f 天\n", max(clean_df$estimated_yearly_booked, na.rm = TRUE)))
  cat(sprintf("    有 availability 数据的房源: %d (%.1f%%)\n",
              sum(!is.na(clean_df$estimated_yearly_booked)),
              sum(!is.na(clean_df$estimated_yearly_booked)) / nrow(clean_df) * 100))
}

# 删除临时计算的列（保留原始 availability 列，因为可能后续有用）
clean_df <- clean_df %>% select(-price_per_person, -booked_days_90, -booked_days_60, 
                                 -booked_days_30, -estimated_yearly_booked)

n_after <- nrow(clean_df)
cat(sprintf("\n总删除: %d 行价格异常值 (%.2f%%)\n", n_before - n_after, 
            (n_before - n_after) / n_before * 100))

# ---- 7. 删除缺经纬度 ----
clean_df <- clean_df %>% filter(!is.na(latitude), !is.na(longitude))

clean_df <- clean_df %>%
  mutate(
    room_type = case_when(
      grepl("Entire", room_type, ignore.case = TRUE) ~ "Entire home/apt",
      grepl("Private", room_type, ignore.case = TRUE) ~ "Private room",
      grepl("Shared", room_type, ignore.case = TRUE) ~ "Shared room",
      
      # 新增： Hotel room 合并为 Private room
      grepl("Hotel", room_type, ignore.case = TRUE) ~ "Private room",
      
      TRUE ~ "Private room"  # 兜底
    )
  )

clean_df$room_type <- as.factor(clean_df$room_type)

# ---- 8. 地段聚类（KMeans 必须在 factor 之前做）----
set.seed(42)
K <- 30
coords <- clean_df %>% select(latitude, longitude) %>% as.matrix()

km_result <- kmeans(coords, centers = K, nstart = 10)
clean_df$location_cluster <- km_result$cluster

# ---- 9. 把因子变量转换为 factor（顺序必须在聚类之后）----
clean_df$room_type <- as.factor(clean_df$room_type)
clean_df$neighbourhood_cleansed <- as.factor(clean_df$neighbourhood_cleansed)
clean_df$location_cluster <- as.factor(clean_df$location_cluster)

# ---- 10. 保存一个包含经纬度的模型用数据集（用于 predict_price） ----
model_df_full <- clean_df %>%
  select(
    price_num,
    room_type,
    accommodates,
    bedrooms,
    bath_num,
    bath_shared,
    neighbourhood_cleansed,
    location_cluster,
    latitude,
    longitude
  )

save(model_df_full, file = "model_df_full.rda")

cat("Dataclean 完成，clean_df & model_df_full 已生成。\n")



