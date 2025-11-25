### -----------------------------------------
### train_meta_model.R
### 训练 stacking meta-model：price ~ rf_pred + lm_pred
### -----------------------------------------

library(dplyr)
library(ranger)

# 1. 加载 clean_df + 两个底层模型
source("Dataclean.R")
load("model_rf_ranger.rda")
load("model_lm.rda")

cat("Data and base models loaded.\n")

# 2. 构建和之前一样的特征表（用于预测 rf / lm）
base_df <- clean_df %>%
  select(
    price_num,
    room_type,
    accommodates,
    bedrooms,
    bath_num,
    bath_shared,
    neighbourhood_cleansed,
    location_cluster
  ) %>%
  na.omit()

# 因子化
base_df$room_type <- as.factor(base_df$room_type)
base_df$neighbourhood_cleansed <- as.factor(base_df$neighbourhood_cleansed)
base_df$location_cluster <- as.factor(base_df$location_cluster)

cat("Base_df ready, rows:", nrow(base_df), "\n")

# 3. 用 RF 预测整个 base_df 的价格（in-sample 或近似）
rf_pred_all <- predict(model_rf, data = base_df)$predictions

# 4. 用 LM 预测同一批数据
lm_pred_all <- predict(model_lm, newdata = base_df)

# 5. 组装 meta 训练集
meta_df <- data.frame(
  price_num = base_df$price_num,
  rf_pred = rf_pred_all,
  lm_pred = lm_pred_all
)

# 6. 训练 stacking meta-model：线性回归
meta_model <- lm(price_num ~ rf_pred + lm_pred, data = meta_df)

summary(meta_model)
coef(meta_model)

# 7. 保存 meta_model
save(meta_model, file = "model_meta_stack.rda")

cat("Meta stacking model saved as model_meta_stack.rda\n")


