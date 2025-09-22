#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
裁剪数据加载器模块

功能:
1. 基于现有数据集创建裁剪版本的数据加载器
2. 支持输入32x32、输出128x128的数据流
3. 兼容PDEBench数据格式
4. 集成到现有训练框架

作者: AI Assistant
日期: 2025
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class CropDataset(Dataset):
    """
    裁剪数据集类
    
    支持从原始数据创建输入32x32、输出128x128的数据对
    """
    
    def __init__(self, data_path, num_samples=100, crop_size=32, output_size=128):
        """
        初始化裁剪数据集
        
        Args:
            data_path: 原始数据路径
            num_samples: 样本数量
            crop_size: 输入裁剪尺寸
            output_size: 输出尺寸
        """
        self.data_path = data_path
        self.num_samples = num_samples
        self.crop_size = crop_size
        self.output_size = output_size
        
        # 加载和处理数据
        self.inputs, self.outputs = self._load_and_process_data()
        
    def _load_and_process_data(self):
        """
        加载并处理数据
        
        Returns:
            tuple: (inputs, outputs) 张量
        """
        logger.info(f"加载数据: {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            if 'tensor' in f:
                tensor_data = f['tensor']
                # 处理num_samples为None的情况（使用全量数据）
                if self.num_samples is None:
                    actual_samples = tensor_data.shape[0]
                    logger.info(f"使用全量数据，总样本数: {actual_samples}")
                else:
                    actual_samples = min(self.num_samples, tensor_data.shape[0])
                    logger.info(f"使用指定样本数: {actual_samples}")
                data = np.array(tensor_data[:actual_samples], dtype=np.float32)
                
                # 处理数据维度
                if len(data.shape) == 4 and data.shape[1] == 1:
                    data = data.squeeze(1)  # 移除通道维度
                
                logger.info(f"原始数据形状: {data.shape}")
                
                # 创建裁剪输入
                inputs = self._create_spatial_crop(data, self.crop_size)
                
                # 输出保持原始尺寸
                outputs = data
                
                # 转换为张量并调整形状
                inputs = torch.from_numpy(inputs).float()
                outputs = torch.from_numpy(outputs).float()
                
                # 展平空间维度用于MLP，保持空间维度用于CNN
                inputs_flat = inputs.reshape(inputs.shape[0], -1)  # (N, 1024)
                outputs_flat = outputs.reshape(outputs.shape[0], -1)  # (N, 16384)
                
                logger.info(f"输入形状: {inputs_flat.shape}")
                logger.info(f"输出形状: {outputs_flat.shape}")
                
                return inputs_flat, outputs_flat
            else:
                raise ValueError("数据文件中未找到'tensor'键")
    
    def _create_spatial_crop(self, data, crop_size):
        """
        创建空间裁剪
        
        Args:
            data: 原始数据 (N, H, W)
            crop_size: 裁剪尺寸
            
        Returns:
            numpy.ndarray: 裁剪后数据 (N, crop_size, crop_size)
        """
        num_samples, height, width = data.shape
        
        # 中心裁剪
        start_h = (height - crop_size) // 2
        end_h = start_h + crop_size
        start_w = (width - crop_size) // 2
        end_w = start_w + crop_size
        
        cropped_data = data[:, start_h:end_h, start_w:end_w]
        logger.info(f"裁剪区域: [{start_h}:{end_h}, {start_w}:{end_w}]")
        logger.info(f"裁剪后形状: {cropped_data.shape}")
        
        return cropped_data
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

def create_crop_dataloader(config):
    """
    创建裁剪数据加载器
    
    Args:
        config: 配置字典，包含数据路径、批次大小等参数
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 从配置中获取参数
    data_config = config.get('data', {})
    data_path = data_config.get('data_path', '')
    batch_size = data_config.get('batch_size', 32)
    num_samples = data_config.get('num_samples', None)
    
    # 如果num_samples为None，则使用全量数据
    if num_samples is None:
        # 先获取数据文件信息来确定实际样本数
        data_info = get_data_info(data_path)
        num_samples = data_info.get('total_samples', 1000)  # 如果无法获取，使用较大默认值
        logger.info(f"使用全量数据，样本数: {num_samples}")
    
    # 输入输出分辨率
    input_resolution = data_config.get('input_resolution', [32, 32])
    output_resolution = data_config.get('output_resolution', [128, 128])
    
    crop_size = input_resolution[0]  # 假设输入是正方形
    output_size = output_resolution[0]  # 假设输出是正方形
    
    logger.info(f"创建裁剪数据加载器:")
    logger.info(f"  数据路径: {data_path}")
    logger.info(f"  批次大小: {batch_size}")
    logger.info(f"  样本数量: {num_samples}")
    logger.info(f"  输入尺寸: {crop_size}x{crop_size}")
    logger.info(f"  输出尺寸: {output_size}x{output_size}")
    
    # 创建数据集
    dataset = CropDataset(
        data_path=data_path,
        num_samples=num_samples,
        crop_size=crop_size,
        output_size=output_size
    )
    
    # 数据集分割
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0  # Windows兼容性
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    logger.info(f"数据集分割: 训练={len(train_dataset)}, 验证={len(val_dataset)}, 测试={len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def get_data_info(data_path):
    """
    获取数据信息
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        dict: 数据信息字典
    """
    try:
        with h5py.File(data_path, 'r') as f:
            info = {
                'keys': list(f.keys()),
                'file_size': os.path.getsize(data_path) / (1024 * 1024),  # MB
            }
            
            if 'tensor' in f:
                tensor_data = f['tensor']
                info.update({
                    'shape': tensor_data.shape,
                    'dtype': tensor_data.dtype,
                    'total_samples': tensor_data.shape[0]
                })
            
            return info
    except Exception as e:
        logger.error(f"获取数据信息失败: {str(e)}")
        return {}