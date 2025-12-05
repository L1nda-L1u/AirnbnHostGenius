### -----------------------------------------
### model_xgb.R
### Step: 使用 XGBoost 训练价格预测模型
### -----------------------------------------

library(dplyr)
library(xgboost)
library(Matrix)
library(caret)

# 1. 加载清洗后的数据
source("Dataclean.R")   # clean_df 会在这里被加载
cat("Loaded clean_df.\n")

# 2. 构建模型使用的数据集（与 RF 相同）
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
  na.omit()

cat("Model dataset ready:", nrow(model_df), "rows\n")

# 3. 因子变量
model_df$room_type <- as.factor(model_df$room_type)
model_df$neighbourhood_cleansed <- as.factor(model_df$neighbourhood_cleansed)
model_df$location_cluster <- as.factor(model_df$location_cluster)

# 4. 划分训练集与测试集
set.seed(42)
train_index <- createDataPartition(model_df$price_num, p = 0.8, list = FALSE)

train_df <- model_df[train_index, ]
test_df  <- model_df[-train_index, ]

cat("Train size:", nrow(train_df), "\n")
cat("Test size:", nrow(test_df), "\n")

# 5. 转成 XGBoost 需要的 sparse matrix（自动 one-hot）
train_matrix <- sparse.model.matrix(
  price_num ~ . - 1,
  data = train_df
)

test_matrix <- sparse.model.matrix(
  price_num ~ . - 1,
  data = test_df
)

# 6. 提取 label
train_label <- train_df$price_num
test_label <- test_df$price_num

# 7. 放进 xgb.DMatrix
dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest  <- xgb.DMatrix(data = test_matrix, label = test_label)

# 8. 设置 XGBoost 超参数（非常稳的默认参数）
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eta = 0.05,             # 学习率
  max_depth = 8,          # 树深度
  subsample = 0.8,        # 子采样
  colsample_bytree = 0.8, # 特征采样
  min_child_weight = 3
)

cat("开始训练 XGBoost...\n")

# 9. 训练模型
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  watchlist = list(train = dtrain, test = dtest),
  print_every_n = 20,
  early_stopping_rounds = 50
)

cat("XGBoost training complete.\n")

# 10. 测试集预测
pred_xgb <- predict(xgb_model, newdata = dtest)

# 11. 性能指标
R2  <- cor(pred_xgb, test_label)^2
RMSE <- sqrt(mean((pred_xgb - test_label)^2))
MAE  <- mean(abs(pred_xgb - test_label))

cat("------ XGBoost PERFORMANCE ------\n")
cat("R²:", round(R2, 3), "\n")
cat("RMSE:", round(RMSE, 2), "\n")
cat("MAE:", round(MAE, 2), "\n")

# ±区间评估（更直观）
acc_10 <- mean(abs(pred_xgb - test_label) <= 10)
acc_20 <- mean(abs(pred_xgb - test_label) <= 20)
acc_30 <- mean(abs(pred_xgb - test_label) <= 30)

cat("Accuracy ±£10:", round(acc_10 * 100, 1), "%\n")
cat("Accuracy ±£20:", round(acc_20 * 100, 1), "%\n")
cat("Accuracy ±£30:", round(acc_30 * 100, 1), "%\n")

# 12. 保存模型
save(xgb_model, file = "model_xgb.rda")
cat("XGBoost model saved as model_xgb.rda\n")

# 13. 特征重要性可视化
importance_matrix <- xgb.importance(model = xgb_model)
print(importance_matrix)

xgb.plot.importance(importance_matrix)
cat("Feature importance plotted.\n")
