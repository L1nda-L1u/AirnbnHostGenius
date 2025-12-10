### -----------------------------------------
### model_rf.R (ranger version)
### Step 3: 随机森林模型（Airbnb 价格预测）
### -----------------------------------------

library(dplyr)
library(caret)
library(ranger)     # ⭐ 使用 ranger！

# 1. 加载清洗后的数据 clean_df
source("Dataclean.R")

cat("Loaded clean_df\n")

# 2. 选择建模特征
model_df <- clean_df %>%
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
  na.omit()   # ⭐ ranger 不接受 NA

# 强制转 factor（ranger 会自动处理）
model_df$room_type <- as.factor(model_df$room_type)
model_df$neighbourhood_cleansed <- as.factor(model_df$neighbourhood_cleansed)
model_df$location_cluster <- as.factor(model_df$location_cluster)

# 3. 划分训练集/测试集
set.seed(123)
train_index <- createDataPartition(model_df$price_num, p = 0.8, list = FALSE)

train_df <- model_df[train_index, ]
test_df  <- model_df[-train_index, ]

cat("Train size:", nrow(train_df), "\n")
cat("Test size:", nrow(test_df), "\n")

# 4. 训练 ranger 随机森林
cat("开始训练 ranger 随机森林...\n")

model_rf <- ranger(
  formula = price_num ~ .,
  data = train_df,
  num.trees = 500,         # ⭐ 300 棵树很快且足够好
  mtry = 3,
  importance = "impurity",
  verbose = TRUE           # ⭐ 显示进度条！
)

cat("ranger 训练完成。\n")

# 5. 在测试集上预测
pred_rf <- predict(model_rf, data = test_df)$predictions

# 6. 模型评估指标：R² + RMSE
rf_r2 <- cor(pred_rf, test_df$price_num)^2
rf_rmse <- sqrt(mean((pred_rf - test_df$price_num)^2))

cat("Random Forest (ranger) R²:", rf_r2, "\n")
cat("Random Forest (ranger) RMSE:", rf_rmse, "\n")

# 7. 查看变量重要性
cat("变量重要性排名（Feature Importance）:\n")
print(importance(model_rf))

# 8. 保存模型
save(model_rf, file = "model_rf_ranger.rda")

cat("Model saved as model_rf_ranger.rda\n")

