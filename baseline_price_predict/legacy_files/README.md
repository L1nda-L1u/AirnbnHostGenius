# Airbnb Host Genius

Airbnb 价格预测模型项目 - 使用 XGBoost 和 Neural Network 的 Stacking 集成模型

## 项目结构

```
AirbnbHostGeniusR/
├── R_scripts/              # R 脚本和项目文件
│   ├── AirbnbHostGeniusR.Rproj
│   ├── best_model/        # 最佳模型文件
│   │   ├── load_and_predict.R  # 加载和预测函数
│   │   ├── README.txt     # 使用说明
│   │   └── ...
│   ├── Dataclean.R        # 数据清洗
│   ├── get_comps.R        # 获取可比房源
│   └── ...
├── other_files/           # 其他文件（训练脚本、模型文件等）
│   ├── stack_xgb_nn.py    # Python 训练脚本
│   └── ...
└── README.md              # 本文件
```

## 快速开始

### 1. 配置 Python 环境

在 RStudio 中打开 `R_scripts/AirbnbHostGeniusR.Rproj`，然后运行：

```r
# 配置 Python（如果还没配置）
source("best_model/配置Python环境.R")

# 安装 Python 依赖包
source("best_model/安装Python依赖.R")
```

### 2. 加载模型

```r
source("best_model/load_and_predict.R")
```

### 3. 使用模型预测

```r
# 从数据文件读取示例
df <- read.csv("best_model/nn_price_training_v4.csv", nrows = 1)
features <- as.numeric(df[, setdiff(colnames(df), "price_num")])
predicted_price <- predict_price(features)
print(predicted_price)
```

## 模型说明

### Stacking 模型

最佳模型是 XGBoost + Neural Network 的 Stacking 集成：

- **Base Models**: XGBoost, Neural Network (MLP)
- **Meta Model**: Ridge Regression
- **公式**: `final_price = intercept + xgb_weight * xgb_pred + nn_weight * nn_pred`

### 模型文件

模型文件较大，未包含在 GitHub 中。如需使用，请：

1. 运行 `other_files/stack_xgb_nn.py` 训练模型
2. 或联系项目维护者获取模型文件

## 依赖

### R 包
- `reticulate` - Python 接口
- `dplyr`, `caret` - 数据处理
- `xgboost` - XGBoost 模型
- `torch` - 神经网络（可选，用于 GPU 训练）

### Python 包
- `torch` - PyTorch
- `numpy`, `pandas` - 数据处理
- `xgboost` - XGBoost
- `scikit-learn` - 机器学习工具

## 使用说明

详细使用说明请参考：
- `R_scripts/best_model/README.txt` - 模型使用说明
- `R_scripts/best_model/如何使用模型.R` - 使用示例

## 许可证

[添加许可证信息]

## 作者

[添加作者信息]

