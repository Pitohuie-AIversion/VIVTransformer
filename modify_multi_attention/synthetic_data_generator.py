"""
合成数据生成器，用于测试模型兼容性
"""
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class SyntheticDataset(Dataset):
    """合成数据集"""
    
    def __init__(self, input_dim, output_dim, num_samples=1000):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_samples = num_samples
        
        # 生成随机数据
        self.inputs = torch.randn(num_samples, input_dim)
        
        # 生成相关的输出（简单的线性变换 + 噪声）
        # 使用一个简单的映射关系
        weight_matrix = torch.randn(input_dim, output_dim) * 0.1
        self.outputs = torch.matmul(self.inputs, weight_matrix) + torch.randn(num_samples, output_dim) * 0.01
        
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

def create_synthetic_dataloader(config):
    """创建合成数据加载器"""
    data_config = config.get('data', {})
    
    # 使用配置文件中的维度，如果没有则使用合理的默认值
    input_dim = data_config.get('input_dim', 16384)  # 默认128*128
    output_dim = data_config.get('output_dim', 16384)  # 默认128*128
    batch_size = data_config.get('batch_size', 4)
    num_samples = data_config.get('num_samples', 1000)
    
    print(f"🔧 合成数据生成器配置: input_dim={input_dim}, output_dim={output_dim}, batch_size={batch_size}")
    
    # 创建数据集
    train_dataset = SyntheticDataset(input_dim, output_dim, num_samples)
    val_dataset = SyntheticDataset(input_dim, output_dim, num_samples // 5)
    test_dataset = SyntheticDataset(input_dim, output_dim, num_samples // 10)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建简单的归一化器（实际上不做任何操作）
    class DummyNormalizer:
        def normalize(self, x):
            return x
        def denormalize(self, x):
            return x
    
    normalizer = DummyNormalizer()
    
    return train_loader, val_loader, test_loader, normalizer

if __name__ == "__main__":
    # 测试合成数据生成器
    config = {
        'data': {
            'input_dim': 400,
            'output_dim': 40000,
            'batch_size': 4,
            'num_samples': 100
        }
    }
    
    train_loader, val_loader, test_loader, normalizer = create_synthetic_dataloader(config)
    
    print("🧪 测试合成数据生成器...")
    for batch_idx, (inputs, outputs) in enumerate(train_loader):
        print(f"批次 {batch_idx}: 输入形状 {inputs.shape}, 输出形状 {outputs.shape}")
        if batch_idx >= 2:  # 只测试前几个批次
            break
    
    print("✅ 合成数据生成器测试完成！")