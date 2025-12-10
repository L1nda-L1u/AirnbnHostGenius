### ---------------------------------------------------------
### predict_price.R
### AirbnbHostGenius 最终智能定价引擎（无 NA 稳定版）
### ---------------------------------------------------------

library(dplyr)
library(geosphere)

# 1. 加载数据与模型
source("Dataclean.R")
source("get_comps.R")

load("model_rf_ranger.rda")
load("model_lm.rda")
load("model_meta_stack.rda")

cat("All models loaded.\n")

### ---------------------------------------------------------
### 房型清洗函数（保持房型为训练集可识别的格式）
### ---------------------------------------------------------
clean_room_type <- function(rt) {
  rt <- trimws(rt)
  
  # 映射表
  map <- c(
    "Entire home/apt" = "Entire home/apt",
    "Entire home / apt" = "Entire home/apt",
    "entire home/apt" = "Entire home/apt",
    "Entire Home/Apt" = "Entire home/apt",
    "Private room" = "Private room",
    "private room" = "Private room",
    "Shared room" = "Shared room",
    "shared room" = "Shared room"
  )
  
  if (rt %in% names(map)) {
    return(map[[rt]])
  } else {
    stop(paste("⚠️ 无法识别房型：", rt))
  }
}

### ---------------------------------------------------------
### 主函数：predict_price()
### ---------------------------------------------------------

predict_price <- function(lat, lon, room_type, accommodates, bedrooms, bath_num,
                          radius_km = 1) {
  
  ### 1) 清洗房型
  rt_clean <- clean_room_type(room_type)
  
  ### 2) 获取附近 comps
  comps <- get_comps(
    df = clean_df,
    lat = lat,
    lon = lon,
    room_type = rt_clean,
    accommodates = accommodates,
    bedrooms = bedrooms,
    bath_num = bath_num,
    radius_km = radius_km
  )
  
  local_median <- comps$local_median_price
  local_p25 <- comps$local_p25
  local_p75 <- comps$local_p75
  
  ### 3) 找最近有完整特征的历史房源（使用 model_df_full）
  nearest <- model_df_full %>%
    mutate(dist_km = distHaversine(cbind(longitude, latitude), c(lon, lat))/1000) %>%
    arrange(dist_km) %>% slice(1)
  
  ### 4) 构建用于模型预测的新样本（注意 factor levels 兼容）
  new_data <- data.frame(
    room_type = factor(rt_clean, levels = levels(model_df_full$room_type)),
    accommodates = accommodates,
    bedrooms = bedrooms,
    bath_num = bath_num,
    bath_shared = 0,
    neighbourhood_cleansed = factor(nearest$neighbourhood_cleansed,
                                    levels = levels(model_df_full$neighbourhood_cleansed)),
    location_cluster = factor(nearest$location_cluster,
                              levels = levels(model_df_full$location_cluster))
  )
  
  
  ### 5) RF 预测
  rf_pred <- predict(model_rf, data = new_data)$predictions
  
  ### 6) LM 预测
  lm_pred <- suppressWarnings(predict(model_lm, newdata = new_data))
  
  ### 如果 LM 出现 NA，则 fallback
  if (is.na(lm_pred)) {
    lm_pred <- rf_pred
  }
  
  ### 7) stacking（融合 RF + LM）
  stack_input <- data.frame(
    rf_pred = rf_pred,
    lm_pred = lm_pred
  )
  stack_pred <- predict(meta_model, newdata = stack_input)
  
  ### 8) 最终融合：stacking + comps（市场锚定）
  w_stack <- 0.8
  w_comps <- 0.2
  
  final_price <- as.numeric(w_stack * stack_pred + w_comps * local_median)
  
  ### 9) 返回结果
  return(list(
    input = new_data,
    rf_prediction = rf_pred,
    lm_prediction = lm_pred,
    stack_prediction = stack_pred,
    comps_median = local_median,
    comps_range = c(local_p25, local_p75),
    n_comps = comps$n_comps,
    final_price = final_price
  ))
}

cat("predict_price.R loaded.\n")
