import torch
import pickle

# --- 请修改为你自己的文件路径 ---
pt_file_path = '/home/lixianchen/hgcn-master11/hgcn-master/data/disease_lp/processed.pt'
pkl_file_path = '/home/lixianchen/hgcn-master11/hgcn-master/data/disease_lp/processed/disease_lp/lp_usefeats1.pkl'
# ------------------------------------

def inspect_file(path, file_type):
    print(f"\n--- 检查 {file_type} 文件: {path} ---")
    data = None
    try:
        if file_type == 'PyTorch':
            # 允许加载包含非torch对象的pt文件
            data = torch.load(path, weights_only=False)
        elif file_type == 'Pickle':
            with open(path, 'rb') as f:
                data = pickle.load(f)

        print(f"数据对象类型: {type(data)}")

        if isinstance(data, dict):
            print("文件内容 (字典的键):", list(data.keys()))
            # 检查是否存在名为 'features' 或 'x' 的特征键
            features_key = None
            if 'features' in data:
                features_key = 'features'
            elif 'x' in data:
                 features_key = 'x'
            
            if features_key:
                features = data[features_key]
                # 确保特征是torch.Tensor
                if not isinstance(features, torch.Tensor):
                     # 尝试从scipy稀疏矩阵转换
                    if hasattr(features, 'toarray'): 
                        features = torch.tensor(features.toarray())
                    else:
                        features = torch.tensor(features)

                print(f"节点特征形状 ({features_key}.shape): {features.shape}")
                print(f"  - 特征均值: {features.mean():.4f}")
                print(f"  - 特征标准差: {features.std():.4f}  <-- 如果标准差接近1.0，说明已归一化")
            else:
                print("未找到'features'或'x'键。")

    except Exception as e:
        print(f"加载文件失败: {e}")

# 运行检查
inspect_file(pt_file_path, 'PyTorch')
inspect_file(pkl_file_path, 'Pickle')