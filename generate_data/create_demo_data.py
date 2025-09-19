#!/usr/bin/env python3
"""
创建演示数据文件用于网络横向对比测试
"""

import h5py
import numpy as np
from pathlib import Path
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_demo_data(output_path: str, num_samples: int = 500):
    """
    创建演示数据文件
    
    Args:
        output_path: 输出文件路径
        num_samples: 样本数量
    """
    logger.info(f"创建演示数据文件: {output_path}")
    logger.info(f"样本数量: {num_samples}")
    
    # 创建输出目录
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # 生成合成数据
    # 模拟2D Darcy Flow数据格式
    input_data = np.random.randn(num_samples, 128, 128, 1).astype(np.float32)
    output_data = np.random.randn(num_samples, 128, 128, 1).astype(np.float32)
    
    # 添加一些结构化模式使数据更真实
    for i in range(num_samples):
        # 输入数据：添加一些低频模式
        x = np.linspace(0, 2*np.pi, 128)
        y = np.linspace(0, 2*np.pi, 128)
        X, Y = np.meshgrid(x, y)
        
        # 生成多个正弦波的叠加
        pattern = (np.sin(X) * np.cos(Y) + 
                  0.5 * np.sin(2*X) * np.sin(2*Y) + 
                  0.3 * np.cos(3*X) * np.cos(3*Y))
        
        input_data[i, :, :, 0] = pattern + 0.1 * np.random.randn(128, 128)
        
        # 输出数据：输入的非线性变换
        output_data[i, :, :, 0] = np.tanh(pattern * 2) + 0.05 * np.random.randn(128, 128)
    
    # 保存为HDF5格式 - 使用动态训练器期望的格式
    with h5py.File(output_path, 'w') as f:
        # 创建数据组 - 使用'tensor'键（动态训练器期望的格式）
        # 合并输入输出数据，形状为 (samples, channels, height, width)
        # 这里我们使用输出数据作为主要数据
        tensor_data = output_data.transpose(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        f.create_dataset('tensor', data=tensor_data, compression='gzip')
        
        # 也保留原始格式以备其他用途
        f.create_dataset('input', data=input_data, compression='gzip')
        f.create_dataset('output', data=output_data, compression='gzip')
        
        # 添加元数据
        f.attrs['num_samples'] = num_samples
        f.attrs['input_shape'] = input_data.shape
        f.attrs['output_shape'] = output_data.shape
        f.attrs['data_type'] = 'synthetic_darcy_flow'
        f.attrs['description'] = 'Synthetic 2D Darcy Flow data for model comparison testing'
        
        logger.info(f"数据形状:")
        logger.info(f"  输入: {input_data.shape}")
        logger.info(f"  输出: {output_data.shape}")
        
    file_size = Path(output_path).stat().st_size / 1024**2  # MB
    logger.info(f"演示数据文件已创建: {output_path} ({file_size:.2f}MB)")
    
    return output_path

def main():
    """主函数"""
    # 创建演示数据
    demo_data_path = "./preprocessed_data/sample_data.h5"
    create_demo_data(demo_data_path, num_samples=500)
    
    # 验证数据文件
    logger.info("验证创建的数据文件...")
    with h5py.File(demo_data_path, 'r') as f:
        logger.info(f"数据集键: {list(f.keys())}")
        logger.info(f"输入数据形状: {f['input'].shape}")
        logger.info(f"输出数据形状: {f['output'].shape}")
        logger.info(f"元数据: {dict(f.attrs)}")
    
    logger.info("演示数据创建完成!")

if __name__ == "__main__":
    main()