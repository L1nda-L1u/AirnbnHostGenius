"""
retrain_all_models.py
重新训练所有模型（使用新的数据集 v4）
"""

import subprocess
import sys
import os

print("="*80)
print("重新训练所有模型（使用新数据集 nn_price_training_v4.csv）")
print("="*80)

# 需要训练的模型列表
models_to_train = [
    {
        'name': 'XGBoost',
        'script': 'train_xgboost_model.py',
        'output_files': ['best_xgb_log_model.pkl', 'scaler_xgb.pkl']
    },
    {
        'name': 'Neural Network (Log-Price)',
        'script': 'train_price_log.py',
        'output_files': ['best_price_A2_log.pth', 'scaler_price.pkl']
    },
    {
        'name': 'Autoencoder + KNN',
        'script': 'autoencoder_knn.py',
        'output_files': ['autoencoder_model.pth', 'ae_scaler.pkl', 
                         'listing_embeddings_train.npy', 'listing_embeddings_predict.npy']
    }
]

print("\n将要训练的模型:")
for i, model in enumerate(models_to_train, 1):
    print(f"  {i}. {model['name']} - {model['script']}")

response = input("\n确认开始训练？(y/n): ")
if response.lower() != 'y':
    print("已取消")
    sys.exit(0)

# 训练每个模型
for i, model in enumerate(models_to_train, 1):
    print("\n" + "="*80)
    print(f"[{i}/{len(models_to_train)}] 训练 {model['name']}...")
    print("="*80)
    
    try:
        # 运行训练脚本
        result = subprocess.run(
            [sys.executable, model['script']],
            capture_output=False,
            text=True
        )
        
        if result.returncode == 0:
            print(f"\n✓ {model['name']} 训练完成")
            
            # 检查输出文件
            all_exist = True
            for file in model['output_files']:
                if os.path.exists(file):
                    print(f"  ✓ {file} 已生成")
                else:
                    print(f"  ✗ {file} 未找到")
                    all_exist = False
            
            if all_exist:
                print(f"  ✅ {model['name']} 所有文件已生成")
            else:
                print(f"  ⚠️  {model['name']} 部分文件缺失")
        else:
            print(f"\n✗ {model['name']} 训练失败 (返回码: {result.returncode})")
            
    except Exception as e:
        print(f"\n✗ {model['name']} 训练出错: {e}")

print("\n" + "="*80)
print("所有模型训练完成！")
print("="*80)
print("\n现在可以运行 compare_all_models_simple.py 来对比结果")
print("（会显示每个模型的散点图和 ±20£ 准确率）")

