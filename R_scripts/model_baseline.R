### -----------------------------------------
### model_baseline.R
### Step 2: 建立线性回归模型 baseline
### -----------------------------------------

# 1. 加载清洗好的数据
source("Dataclean.R")

# 2. 选择建模特征（X）和目标变量（Y）
# 注意：不能直接丢所有变量，要手动挑选清晰的特征集

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
    # （未来可以加更多特征：amenities one-hot、review_scores等）
  )

# 3. 线性回归模型
model_lm <- lm(price_num ~ ., data = model_df)

# 4. 查看模型结果
summary(model_lm)

# 5. 查看每个变量的系数（解释模型）
coef(summary(model_lm))

# 6. 可视化模型诊断（可选）
par(mfrow=c(2,2))
plot(model_lm)

### 最终模型对象 model_lm 现在已经可用
print("Linear Model baseline 完成！")
### ---- 输出 β 系数（最清晰格式） ----

beta_table <- data.frame(
  Feature = names(coef(model_lm)),
  Beta = coef(model_lm)
)

print(beta_table)

beta_table <- data.frame(
  Feature = names(coef(model_lm)),
  Beta = coef(model_lm)
) %>%
  arrange(desc(abs(Beta)))

print(beta_table)
save(model_lm, file = "model_lm.rda")

