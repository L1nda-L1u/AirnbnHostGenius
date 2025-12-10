### ---------------------------------------------------------
### test_model_performance.R
### 使用 20% 测试集验证最终智能定价模型（predict_price）
### ---------------------------------------------------------

library(dplyr)
library(ggplot2)

# 1. 载入全部系统组件
source("Dataclean.R")
source("get_comps.R")
source("predict_price.R")

load("model_rf_ranger.rda")
load("model_lm.rda")
load("model_meta_stack.rda")
load("model_df_full.rda")

cat("全部模型和数据载入完毕。\n")

### ---------------------------------------------------------
### Step 1：划分 80/20 的训练 / 测试集（随机）
### ---------------------------------------------------------

set.seed(999)
test_indices <- sample(1:nrow(model_df_full), size = 500)


train_data <- model_df_full[-test_indices, ]
test_data  <- model_df_full[test_indices, ]

cat("Train size:", nrow(train_data), "\n")
cat("Test size:",  nrow(test_data), "\n")

### ---------------------------------------------------------
### Step 2：定义一个安全的 predict_price_for_test()
###  —— 输入 test_df 的真实特征，不用 lat/lon 找 cluster
### ---------------------------------------------------------

predict_price_for_test <- function(row) {
  
  # 从 test_data 一行取出真实特征
  lat <- row["latitude"]
  lon <- row["longitude"]
  room_type <- as.character(row["room_type"])
  accommodates <- row["accommodates"]
  bedrooms <- row["bedrooms"]
  bath_num <- row["bath_num"]
  
  # 用你的最终版本 predict_price()
  res <- predict_price(
    lat = lat,
    lon = lon,
    room_type = room_type,
    accommodates = accommodates,
    bedrooms = bedrooms,
    bath_num = bath_num
  )
  
  return(res$final_price)
}

### ---------------------------------------------------------
### Step 3：对测试集每一条进行预测（可能跑 10–30 秒）
### ---------------------------------------------------------

cat("开始对测试集进行预测...\n")
cat("Debug first 5 test rows:\n")
print(head(test_data[, c("latitude", "longitude")]))

### Step 3：安全的 Rowwise 预测（不会字符化）
test_pred <- vector("numeric", nrow(test_data))

for (i in seq_len(nrow(test_data))) {
  
  row <- test_data[i, ]
  
  res <- predict_price(
    lat = row$latitude,
    lon = row$longitude,
    room_type = as.character(row$room_type),
    accommodates = row$accommodates,
    bedrooms = row$bedrooms,
    bath_num = row$bath_num
  )
  
  test_pred[i] <- res$final_price
}


cat("预测完成！\n")

### ---------------------------------------------------------
### Step 4：构建评估表
### ---------------------------------------------------------

evaluation_df <- data.frame(
  true_price = test_data$price_num,
  pred_price = test_pred
)

### ---------------------------------------------------------
### Step 5：计算 R² 和 RMSE
### ---------------------------------------------------------

R2 <- cor(evaluation_df$true_price, evaluation_df$pred_price)^2
RMSE <- sqrt(mean((evaluation_df$true_price - evaluation_df$pred_price)^2))

cat("测试集 R²:", R2, "\n")
cat("测试集 RMSE:", RMSE, "\n")
### ---------------------------------------------------------
### Step 6：计算更直观的准确率指标
### ---------------------------------------------------------

errors <- evaluation_df$pred_price - evaluation_df$true_price
abs_errors <- abs(errors)

MAE <- mean(abs_errors)     # 平均绝对误差
RMSE <- sqrt(mean(errors^2))
R2 <- cor(evaluation_df$true_price, evaluation_df$pred_price)^2

acc_10 <- mean(abs_errors <= 10)   # ±10 镑内的比例
acc_20 <- mean(abs_errors <= 20)   # ±20 镑内的比例
acc_30 <- mean(abs_errors <= 30)   # ±30 镑内的比例

cat("-------- MODEL ACCURACY --------\n")
cat("MAE:", round(MAE, 2), "\n")
cat("RMSE:", round(RMSE, 2), "\n")
cat("R²:", round(R2, 3), "\n")
cat("Accuracy ±£10:", round(acc_10 * 100, 1), "%\n")
cat("Accuracy ±£20:", round(acc_20 * 100, 1), "%\n")
cat("Accuracy ±£30:", round(acc_30 * 100, 1), "%\n")


### ---------------------------------------------------------
### Step 6：画对比曲线图（真实 vs 预测）
### ---------------------------------------------------------

evaluation_df$index <- 1:nrow(evaluation_df)

print(
  ggplot(evaluation_df, aes(x = index)) +
    geom_line(aes(y = true_price), color = "black", size = 0.8, alpha = 0.8) +
    geom_line(aes(y = pred_price), color = "red", size = 0.8, alpha = 0.7) +
    labs(
      title = "真实价格 vs 模型预测价格（Test Set）",
      x = "Test Sample Index",
      y = "Nightly Price (£)",
      subtitle = "黑色 = 真实价格   红色 = 模型预测"
    ) +
    theme_minimal()
)


### ---------------------------------------------------------
### Step 7：散点图（真实 vs 预测）
### ---------------------------------------------------------

print(
  ggplot(evaluation_df, aes(x = true_price, y = pred_price)) +
    geom_point(alpha = 0.3) +
    geom_abline(slope = 1, intercept = 0, color = "blue") +
    labs(
      title = "Real vs Predicted Nightly Price",
      x = "真实价格 (£)",
      y = "预测价格 (£)"
    ) +
    theme_minimal()
)


cat("模型测试图已生成。\n")
