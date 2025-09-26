#!/usr/bin/env python3
"""
创建测试数据文件，用于验证归一化功能
"""

import h5py
import numpy as np
import os
from pathlib import Path

def create_test_data():
    """创建测试数据文件"""
    
    # 确保数据目录存在
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # 数据参数
    num_samples = 1000
    data_size = 128  # 数据加载器期望的原始数据尺寸
    
    print(f"创建测试数据文件...")
    print(f"样本数量: {num_samples}")
    print(f"数据尺寸: {data_size}x{data_size}")
    
    # 生成模拟数据（数据加载器期望的格式）
    # 创建高分辨率数据，数据加载器会自动裁剪为低分辨率输入
    tensor_data = np.random.uniform(0, 10, (num_samples, data_size, data_size)).astype(np.float32)
    
    # 添加一些结构化模式，使数据更真实
    for i in range(num_samples):
        # 添加空间相关性
        x, y = np.meshgrid(np.linspace(0, 1, data_size), np.linspace(0, 1, data_size))
        pattern = np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
        tensor_data[i] += pattern * 2
        
        # 添加一些噪声
        noise = np.random.normal(0, 0.5, (data_size, data_size))
        tensor_data[i] += noise
    
    # 保存到HDF5文件（使用数据加载器期望的格式）
    data_path = data_dir / "crop_data.h5"
    
    with h5py.File(data_path, 'w') as f:
        # 数据加载器期望'tensor'键
        f.create_dataset('tensor', data=tensor_data, compression='gzip')
        
        # 添加元数据
        f.attrs['num_samples'] = num_samples
        f.attrs['data_shape'] = (data_size, data_size)
        f.attrs['data_range'] = (tensor_data.min(), tensor_data.max())
        f.attrs['description'] = 'Test data for normalization validation'
    
    print(f"✅ 测试数据已保存到: {data_path}")
    print(f"数据范围: [{tensor_data.min():.3f}, {tensor_data.max():.3f}]")
    print(f"数据均值: {tensor_data.mean():.3f}, 标准差: {tensor_data.std():.3f}")
    print(f"文件大小: {data_path.stat().st_size / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    create_test_data()