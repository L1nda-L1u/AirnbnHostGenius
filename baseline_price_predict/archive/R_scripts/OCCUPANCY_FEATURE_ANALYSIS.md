# Occupancy Prediction - Feature Analysis

## 目标变量
- **`estimated_occupancy_l365d`**: 过去365天的估算入住率（0-1 或百分比）

## 应该保留的特征

### ✅ 必须保留（对预测 occupancy 有用）

1. **Availability 特征**（之前被删了，现在要加回来）
   - `availability_30`: 未来30天的可用天数
   - `availability_60`: 未来60天的可用天数
   - `availability_90`: 未来90天的可用天数
   - **注意**: 这些是**特征**，不是目标变量。它们反映房源的未来可用性，与历史入住率相关但不完全相同。

2. **价格相关**
   - `price_num`: 价格
   - `cluster_median_price`: 区域中位价格
   - `cluster_mean_price`: 区域平均价格
   - **原因**: 价格影响需求，进而影响入住率

3. **房源容量**
   - `accommodates`: 可容纳人数
   - `bedrooms`: 卧室数
   - `beds`: 床数
   - `bath_num`: 浴室数
   - **原因**: 容量影响目标客户群体，影响入住率

4. **房间类型**
   - `room_type_id`: 房间类型（Entire home/apt, Private room, Shared room）
   - **原因**: Entire home vs Private room 入住率差异大

5. **地理位置**
   - `location_cluster_id`: 位置聚类
   - `neighbourhood_id`: 社区ID
   - `latitude`, `longitude`: 经纬度
   - **原因**: 地理位置是影响入住率的关键因素

6. **评分**
   - `review_scores_cleanliness`: 清洁度评分
   - `review_scores_location`: 位置评分
   - **原因**: 评分影响吸引力，进而影响入住率

7. **设施（Amenities）**
   - `amenity_*`: 各种设施的 one-hot 编码
   - **原因**: 设施影响吸引力

### ❌ 应该删除（对预测 occupancy 无用或造成泄漏）

1. **数据泄漏**
   - `estimated_revenue_l365d`: 这是另一个目标变量（revenue = price × occupancy），会造成数据泄漏

2. **标识符和文本**
   - `id`, `listing_url`, `name`: 标识符，对预测无用
   - `bathrooms_text`: 文本，已转换为数值
   - `amenities`, `amenities_clean`: 已转换为 one-hot

3. **其他**
   - `cluster_label`, `cluster_ordered`: 如果存在，通常无用

## 数据流程

```
原始数据 (listings.csv.gz)
    ↓
Dataclean.R (数据清洗)
    ↓
clean_df (包含所有原始特征)
    ↓
prepare_nn_training_data_v5_occupancy.R
    ↓
nn_occupancy_training_v5.csv (用于训练 occupancy 预测模型)
```

## 使用步骤

1. **运行数据清洗**:
   ```r
   source("Dataclean.R")  # 生成 clean_df
   ```

2. **生成 occupancy 训练数据**:
   ```r
   source("prepare_nn_training_data_v5_occupancy.R")  # 生成 nn_occupancy_training_v5.csv
   ```

3. **训练模型**:
   - 使用 XGBoost、Neural Network 或其他模型
   - 目标变量: `estimated_occupancy_l365d`
   - 特征: 包括 `availability_30/60/90` 等

## 注意事项

1. **Availability vs Occupancy**:
   - `availability_30/60/90` 是**未来**的可用天数（特征）
   - `estimated_occupancy_l365d` 是**过去**的入住率（目标变量）
   - 它们相关但不完全相同，所以 availability 可以作为特征使用

2. **数据缺失**:
   - 检查 `estimated_occupancy_l365d` 的缺失率
   - 如果缺失率很高，可能需要考虑其他目标变量或数据收集方法

3. **特征工程**:
   - 可以考虑创建交互特征（如 price × location）
   - 可以考虑创建比率特征（如 price_per_person）

