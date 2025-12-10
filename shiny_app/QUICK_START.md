# 快速开始指南

## 1. 安装依赖

### R包
```r
install.packages(c("shiny", "shinydashboard", "DT", "leaflet", "plotly", 
                    "dplyr", "geosphere", "xgboost", "reticulate", 
                    "glmnet", "httr", "jsonlite"))
```

### Python和PyTorch
```r
library(reticulate)
# 如果Python未配置，先运行：
source("baseline_price_predict/sensitivity_analysis/configure_python.R")

# 安装PyTorch
py_install("torch", pip = TRUE)
```

## 2. 启动应用

### 方法1：使用启动脚本（推荐）
```r
setwd("shiny_app")
source("run_app.R")
```

### 方法2：直接运行
```r
setwd("shiny_app")
shiny::runApp("app.R")
```

### 方法3：从项目根目录运行
```r
shiny::runApp("shiny_app")
```

## 3. 使用步骤

1. **输入地址**
   - 在"地址或邮编"框中输入地址（例如：London, UK 或 SW1A 1AA）
   - 点击"🔍 查找位置"按钮
   - 系统会自动填充经纬度

2. **填写房源信息**
   - 卧室数
   - 卫生间数
   - 可住人数
   - 床数
   - 房型（Entire home/apt, Private room, Shared room）
   - 评分（清洁度、位置评分）

3. **选择设施**
   - 勾选房源拥有的设施（WiFi、厨房、洗衣机等）

4. **预测价格**
   - 点击"🚀 预测价格"按钮
   - 查看预测结果

## 4. 功能说明

### 地址转经纬度
- 使用OpenStreetMap Nominatim API（免费）
- 支持地址和邮编
- 自动在地图上显示位置

### 价格预测
- 基于Stacking模型（XGBoost + Neural Network）
- 使用49个特征进行预测
- 预测结果以英镑（£）显示

### 地图显示
- 使用Leaflet显示房源位置
- 支持缩放和拖拽

## 5. 故障排除

### 模型加载失败
- 检查`baseline_price_predict/baseprice_model/`目录是否存在
- 检查模型文件是否完整
- 检查Python和PyTorch是否正确安装

### 地址查找失败
- 检查网络连接
- 尝试更详细的地址
- 检查Nominatim API是否可用

### 预测失败
- 确保所有必填字段都已填写
- 确保经纬度已正确获取
- 检查控制台错误信息

## 6. 技术架构

- **前端**: Shiny Dashboard
- **后端**: R + Python (PyTorch)
- **模型**: XGBoost + Neural Network (Stacking)
- **地理编码**: OpenStreetMap Nominatim API
- **地图**: Leaflet

## 7. 自定义

### 更改UI颜色
编辑`app.R`中的CSS部分，修改颜色值：
- 主色调：`#1ABC9C` (蓝绿色)
- 次要色：`#3498DB` (蓝色)
- 背景色：`#ECF0F1` (浅灰)

### 添加更多设施
在`app.R`的`checkboxGroupInput`中添加更多选项。

### 使用Google Geocoding API
在`geocoding.R`中配置API key，使用`geocode_address_google()`函数。

