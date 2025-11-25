# 路径检查报告

## 目录结构
```
AirbnbHostGeniusR/
├── best_model/              # 最佳模型（可打包发给同学）
│   ├── nn_price_training_v4.csv
│   ├── xgb_model.json
│   ├── nn.onnx
│   ├── README.txt
│   ├── scaler_xgb.pkl
│   ├── scaler_price.pkl
│   └── meta_ridge_model.pkl
├── R_scripts/               # 所有R文件和R项目
│   ├── *.R
│   ├── *.rda
│   └── AirbnbHostGeniusR.Rproj
└── other_files/             # 其他文件
    ├── *.py
    ├── *.pkl
    └── *.pth
```

## R文件路径检查

### ✅ 正确的路径
- `R_scripts/Dataclean.R` → 读取 `../listings.csv.gz` (上级目录)
- `R_scripts/*.R` → 使用 `source("Dataclean.R")` (同目录)
- `R_scripts/*.R` → 使用 `load("*.rda")` (同目录)

### ⚠️ 需要确认
- `listings.csv.gz` 应该在项目上级目录
- 如果不在，需要修改 `Dataclean.R` 中的路径

## Python文件路径检查

### ✅ 正确的路径
- `other_files/stack_xgb_nn.py` → 读取 `other_files/*.pkl` (同目录)
- `other_files/stack_xgb_nn.py` → 保存到 `../best_model/` (上级目录的best_model文件夹)

## 运行检查

### R脚本
在 `R_scripts/` 目录下运行：
```r
setwd("R_scripts")
source("Dataclean.R")  # 应该能正常工作
```

### Python脚本
在项目根目录运行：
```bash
python other_files/stack_xgb_nn.py  # 会保存到 best_model/
```

## 打包给同学
只需要打包 `best_model/` 文件夹即可！

