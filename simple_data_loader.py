#!/usr/bin/env python3
"""
简化的数据加载器模块
用于替代dynamic_resolution_trainer，支持优化测试脚本
"""

import torch
import torch.nn.functional as F
import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class DynamicConfig:
    """动态配置数据类"""
    data_path: str
    input_resolution: Tuple[int, int]
    output_resolution: Tuple[int, int]
    num_samples: int
    batch_size: int

class DynamicResolutionDataset(torch.utils.data.Dataset):
    """简化的动态分辨率数据集"""
    
    def __init__(self, data_path: str, input_resolution: Tuple[int, int], 
                 output_resolution: Tuple[int, int], num_samples: int = 100,
                 normalize_data: bool = True):
        self.data_path = data_path
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.num_samples = min(num_samples, 1000)  # 限制样本数量
        self.normalize_data = normalize_data
        
        # 生成模拟数据
        self._generate_synthetic_data()
        
        logger.info(f"创建数据集: 输入{input_resolution}, 输出{output_resolution}, 样本数{self.num_samples}")
    
    def _generate_synthetic_data(self):
        """生成合成数据用于测试"""
        # 生成输入数据
        h_in, w_in = self.input_resolution
        h_out, w_out = self.output_resolution
        
        # 创建网格
        x = np.linspace(0, 1, w_in)
        y = np.linspace(0, 1, h_in)
        X, Y = np.meshgrid(x, y)
        
        self.input_data = []
        self.output_data = []
        
        for i in range(self.num_samples):
            # 生成输入：简单的波函数
            freq_x = 2 + i * 0.1
            freq_y = 1.5 + i * 0.05
            input_field = np.sin(freq_x * np.pi * X) * np.cos(freq_y * np.pi * Y)
            
            # 添加一些随机噪声
            input_field += 0.1 * np.random.randn(h_in, w_in)
            
            # 生成输出：上采样并应用简单变换
            output_field = F.interpolate(
                torch.tensor(input_field).unsqueeze(0).unsqueeze(0).float(),
                size=(h_out, w_out),
                mode='bilinear',
                align_corners=False
            ).squeeze().numpy()
            
            # 应用简单的物理变换（模拟流场演化）
            output_field = output_field * (1 + 0.1 * np.sin(2 * np.pi * i / self.num_samples))
            
            if self.normalize_data:
                input_field = (input_field - input_field.min()) / (input_field.max() - input_field.min() + 1e-8)
                output_field = (output_field - output_field.min()) / (output_field.max() - output_field.min() + 1e-8)
            
            self.input_data.append(input_field.flatten())
            self.output_data.append(output_field.flatten())
        
        self.input_data = np.array(self.input_data)
        self.output_data = np.array(self.output_data)
        
        # 设置维度信息
        self.latent_input_dim = self.input_data.shape[1]
        self.latent_output_dim = self.output_data.shape[1]
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        input_tensor = torch.tensor(self.input_data[idx], dtype=torch.float32)
        output_tensor = torch.tensor(self.output_data[idx], dtype=torch.float32)
        return input_tensor, output_tensor
    
    def get_data_statistics(self):
        """获取数据统计信息"""
        return {
            'input_shape': self.input_resolution,
            'output_shape': self.output_resolution,
            'input_dim': self.latent_input_dim,
            'output_dim': self.latent_output_dim,
            'num_samples': self.num_samples,
            'input_range': (self.input_data.min(), self.input_data.max()),
            'output_range': (self.output_data.min(), self.output_data.max())
        }

def create_dynamic_config(data_path: str, input_resolution: Tuple[int, int],
                         output_resolution: Tuple[int, int], num_samples: int,
                         batch_size: int) -> DynamicConfig:
    """创建动态配置"""
    return DynamicConfig(
        data_path=data_path,
        input_resolution=input_resolution,
        output_resolution=output_resolution,
        num_samples=num_samples,
        batch_size=batch_size
    )

def get_dynamic_loaders(config: DynamicConfig, train_split: float = 0.7,
                       valid_split: float = 0.15, test_split: float = 0.15):
    """获取数据加载器"""
    # 创建数据集
    dataset = DynamicResolutionDataset(
        data_path=config.data_path,
        input_resolution=config.input_resolution,
        output_resolution=config.output_resolution,
        num_samples=config.num_samples
    )
    
    # 分割数据集
    total_size = len(dataset)
    train_size = int(total_size * train_split)
    valid_size = int(total_size * valid_split)
    test_size = total_size - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.batch_size, shuffle=False
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=config.batch_size, shuffle=False
    )
    
    return train_loader, valid_loader, test_loader, dataset

def log_gpu_memory(stage: str = ""):
    """记录GPU内存使用"""
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated() / 1024**2  # MB
        memory_reserved = torch.cuda.memory_reserved() / 1024**2   # MB
        logger.info(f"[{stage}] GPU内存: 已分配={memory_allocated:.1f}MB, 已保留={memory_reserved:.1f}MB")
    else:
        logger.info(f"[{stage}] 使用CPU模式")

def cleanup_memory():
    """清理内存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("GPU内存已清理")

def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置"""
    required_keys = ['data', 'models']
    for key in required_keys:
        if key not in config:
            logger.error(f"配置中缺少必需的键: {key}")
            return False
    
    data_config = config['data']
    required_data_keys = ['input_resolution', 'output_resolution', 'num_samples', 'batch_size']
    for key in required_data_keys:
        if key not in data_config:
            logger.error(f"数据配置中缺少必需的键: {key}")
            return False
    
    return True

if __name__ == "__main__":
    # 测试数据加载器
    config = create_dynamic_config(
        data_path="dummy_path",
        input_resolution=(32, 32),
        output_resolution=(128, 128),
        num_samples=50,
        batch_size=4
    )
    
    train_loader, valid_loader, test_loader, dataset = get_dynamic_loaders(config)
    
    print(f"训练集: {len(train_loader)} 批次")
    print(f"验证集: {len(valid_loader)} 批次")
    print(f"测试集: {len(test_loader)} 批次")
    
    # 测试一个批次
    for batch_data in train_loader:
        input_data, target_data = batch_data
        print(f"输入形状: {input_data.shape}")
        print(f"目标形状: {target_data.shape}")
        break
    
    print("数据加载器测试完成！")