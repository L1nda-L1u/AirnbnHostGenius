# =============================================
# 如何在R中调用最佳模型
# =============================================

# 方法1：在RStudio中（推荐）
# ----------------------------
# 1. 打开 R_scripts/AirbnbHostGeniusR.Rproj
# 2. 在Console中运行：

source("best_model/load_and_predict.R")

# 3. 加载完成后，使用 predict_price() 函数进行预测
#    注意：需要提供特征向量（feature vector）

# 示例：从数据文件读取一行作为示例
df <- read.csv("best_model/nn_price_training_v4.csv", nrows = 1)
# 提取特征（排除价格列）
features <- as.numeric(df[, setdiff(colnames(df), "price_num")])
# 预测价格
predicted_price <- predict_price(features)
cat("预测价格:", predicted_price, "\n")

# =============================================
# 方法2：如果你有自己的数据
# =============================================

# 首先加载模型
source("best_model/load_and_predict.R")

# 假设你有一个数据框 df，包含所有特征列（与训练数据相同的列）
# 注意：特征列的顺序必须与训练数据一致！

# 提取一行数据的特征
my_features <- as.numeric(df[1, setdiff(colnames(df), "price_num")])

# 预测
my_prediction <- predict_price(my_features)
print(my_prediction)

# =============================================
# 方法3：批量预测多行数据
# =============================================

# 加载模型
source("best_model/load_and_predict.R")

# 读取你的数据
my_data <- read.csv("your_data.csv")  # 替换为你的数据文件

# 批量预测
predictions <- sapply(1:nrow(my_data), function(i) {
  features <- as.numeric(my_data[i, setdiff(colnames(my_data), "price_num")])
  predict_price(features)
})

# 添加到数据框
my_data$predicted_price <- predictions

# 查看结果
head(my_data[, c("price_num", "predicted_price")])  # 如果有真实价格，可以对比

# =============================================
# 注意事项
# =============================================
# 1. 确保你的数据包含所有训练时使用的特征列
# 2. 特征列的顺序必须与训练数据一致
# 3. 如果缺少某些特征，需要先进行数据预处理（参考 Dataclean.R）
# 4. 预测结果是以原始价格单位（不是log转换后的）

