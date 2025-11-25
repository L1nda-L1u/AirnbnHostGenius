"""
add_cluster_price_safe.py
安全地添加 cluster 价格特征（避免数据泄漏）
只在训练集中计算 cluster 价格，然后应用到测试集
"""

import pandas as pd
import numpy as np

def add_cluster_price_features(df_train, df_test=None, cluster_col='location_cluster_id', price_col='price_num'):
    """
    为训练集和测试集添加 cluster 价格特征
    
    参数:
        df_train: 训练集 DataFrame（必须包含 cluster_col 和 price_col）
        df_test: 测试集 DataFrame（可选，如果提供会应用训练集的 cluster 价格）
        cluster_col: cluster 列名
        price_col: 价格列名
    
    返回:
        df_train: 添加了 cluster 价格特征的训练集
        df_test: 添加了 cluster 价格特征的测试集（如果提供）
        cluster_price_map: cluster 价格映射字典（用于新数据）
    """
    
    # 在训练集中计算每个 cluster 的价格统计
    cluster_stats = df_train.groupby(cluster_col)[price_col].agg([
        ('cluster_median_price', 'median'),
        ('cluster_mean_price', 'mean'),
        ('cluster_p25_price', lambda x: np.percentile(x, 25)),
        ('cluster_p75_price', lambda x: np.percentile(x, 75)),
        ('cluster_count', 'count')
    ]).reset_index()
    
    # 全局中位数（用于处理训练集中没有的 cluster）
    global_median = df_train[price_col].median()
    
    # 合并到训练集
    df_train = df_train.merge(cluster_stats, on=cluster_col, how='left')
    
    # 填充缺失值（如果有训练集中没有的 cluster）
    df_train['cluster_median_price'] = df_train['cluster_median_price'].fillna(global_median)
    df_train['cluster_mean_price'] = df_train['cluster_mean_price'].fillna(global_median)
    df_train['cluster_p25_price'] = df_train['cluster_p25_price'].fillna(global_median)
    df_train['cluster_p75_price'] = df_train['cluster_p75_price'].fillna(global_median)
    df_train['cluster_count'] = df_train['cluster_count'].fillna(0)
    
    # 创建 cluster 价格映射（用于新数据）
    cluster_price_map = cluster_stats.set_index(cluster_col).to_dict('index')
    
    # 如果有测试集，应用训练集的 cluster 价格
    if df_test is not None:
        df_test = df_test.merge(cluster_stats, on=cluster_col, how='left')
        
        # 对于测试集中训练集没有的 cluster，使用全局中位数
        df_test['cluster_median_price'] = df_test['cluster_median_price'].fillna(global_median)
        df_test['cluster_mean_price'] = df_test['cluster_mean_price'].fillna(global_median)
        df_test['cluster_p25_price'] = df_test['cluster_p25_price'].fillna(global_median)
        df_test['cluster_p75_price'] = df_test['cluster_p75_price'].fillna(global_median)
        df_test['cluster_count'] = df_test['cluster_count'].fillna(0)
        
        return df_train, df_test, cluster_price_map
    else:
        return df_train, None, cluster_price_map


def apply_cluster_price_to_new_data(df_new, cluster_price_map, cluster_col='location_cluster_id', 
                                     global_median=None):
    """
    将训练集计算的 cluster 价格应用到新数据
    
    参数:
        df_new: 新数据 DataFrame
        cluster_price_map: 从 add_cluster_price_features 返回的映射字典
        cluster_col: cluster 列名
        global_median: 全局中位数（用于处理未知 cluster）
    
    返回:
        df_new: 添加了 cluster 价格特征的新数据
    """
    
    # 创建临时 DataFrame 用于合并
    cluster_df = pd.DataFrame([
        {
            cluster_col: cluster_id,
            'cluster_median_price': stats['cluster_median_price'],
            'cluster_mean_price': stats['cluster_mean_price'],
            'cluster_p25_price': stats['cluster_p25_price'],
            'cluster_p75_price': stats['cluster_p75_price'],
            'cluster_count': stats['cluster_count']
        }
        for cluster_id, stats in cluster_price_map.items()
    ])
    
    # 合并
    df_new = df_new.merge(cluster_df, on=cluster_col, how='left')
    
    # 对于未知的 cluster，使用全局中位数
    if global_median is not None:
        df_new['cluster_median_price'] = df_new['cluster_median_price'].fillna(global_median)
        df_new['cluster_mean_price'] = df_new['cluster_mean_price'].fillna(global_median)
        df_new['cluster_p25_price'] = df_new['cluster_p25_price'].fillna(global_median)
        df_new['cluster_p75_price'] = df_new['cluster_p75_price'].fillna(global_median)
        df_new['cluster_count'] = df_new['cluster_count'].fillna(0)
    
    return df_new


# =============================================
# 使用示例
# =============================================
if __name__ == "__main__":
    # 示例：在训练脚本中使用
    """
    import pandas as pd
    from add_cluster_price_safe import add_cluster_price_features
    
    # 加载数据
    df = pd.read_csv("nn_price_training_v4.csv")
    
    # 划分训练集和测试集
    from sklearn.model_selection import train_test_split
    df_train, df_test = train_test_split(df, test_size=0.1, random_state=42)
    
    # 添加 cluster 价格特征（只在训练集中计算）
    df_train, df_test, cluster_price_map = add_cluster_price_features(
        df_train, 
        df_test, 
        cluster_col='location_cluster_id',
        price_col='price_num'
    )
    
    # 现在 df_train 和 df_test 都有了 cluster 价格特征
    # 可以用于训练模型了
    """
    print("Cluster 价格特征添加函数已准备好！")
    print("使用方法请参考文件中的示例代码。")

