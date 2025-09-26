#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版裁剪数据加载器模块（包含归一化处理）

功能:
1. 基于现有数据集创建裁剪版本的数据加载器
2. 支持输入32x32、输出128x128的数据流
3. 完整的数据归一化处理
4. 兼容PDEBench数据格式
5. 集成到现有训练框架

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

class EnhancedCropDataset(Dataset):
    """
    增强版裁剪数据集类
    
    支持从原始数据创建输入32x32、输出128x128的数据对，包含完整的归一化处理
    """
    
    def __init__(self, data_path, num_samples=100, crop_size=32, output_size=128, 
                 normalize=True, normalize_method='minmax', normalize_range=(0, 1)):
        """
        初始化增强版裁剪数据集
        
        Args:
            data_path: 原始数据路径
            num_samples: 样本数量
            crop_size: 输入裁剪尺寸
            output_size: 输出尺寸
            normalize: 是否进行归一化
            normalize_method: 归一化方法 ('minmax', 'zscore', 'robust')
            normalize_range: 归一化范围 (仅对minmax有效)
        """
        self.data_path = data_path
        self.num_samples = num_samples
        self.crop_size = crop_size
        self.output_size = output_size
        self.normalize = normalize
        self.normalize_method = normalize_method
        self.normalize_range = normalize_range
        
        # 归一化统计信息
        self.input_stats = {}
        self.output_stats = {}
        
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
            # 尝试多种可能的键名
            possible_keys = ['input', 'data', 'tensor', 'x', 'inputs']
            found_key = None
            
            # 首先列出所有可用的键
            available_keys = list(f.keys())
            logger.info(f"数据文件中的键: {available_keys}")
            
            # 尝试找到合适的键
            for key in possible_keys:
                if key in f:
                    found_key = key
                    break
            
            # 如果没找到预期的键，使用第一个可用的键
            if found_key is None and available_keys:
                found_key = available_keys[0]
                logger.info(f"使用第一个可用键: {found_key}")
            
            if found_key:
                tensor_data = f[found_key]
                logger.info(f"使用数据键: {found_key}，数据形状: {tensor_data.shape}")
                
                if self.num_samples is None:
                    # 使用全量数据
                    actual_samples = tensor_data.shape[0]
                    logger.info(f"使用全量数据，样本数: {actual_samples}")
                else:
                    # 使用指定数量的样本
                    actual_samples = min(self.num_samples, tensor_data.shape[0])
                    logger.info(f"使用指定样本数: {actual_samples}")
                data = np.array(tensor_data[:actual_samples], dtype=np.float32)
                
                # 处理数据维度
                if len(data.shape) == 4 and data.shape[1] == 1:
                    data = data.squeeze(1)  # 移除通道维度
                
                logger.info(f"原始数据形状: {data.shape}")
                logger.info(f"原始数据范围: [{data.min():.6f}, {data.max():.6f}]")
                logger.info(f"原始数据均值: {data.mean():.6f}, 标准差: {data.std():.6f}")
                
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
                
                logger.info(f"裁剪前输入形状: {inputs_flat.shape}")
                logger.info(f"裁剪前输出形状: {outputs_flat.shape}")
                
                # 归一化处理
                if self.normalize:
                    inputs_flat, outputs_flat = self._apply_normalization(inputs_flat, outputs_flat)
                
                logger.info(f"最终输入形状: {inputs_flat.shape}")
                logger.info(f"最终输出形状: {outputs_flat.shape}")
                
                return inputs_flat, outputs_flat
            else:
                raise ValueError(f"数据文件中未找到合适的数据键。可用键: {available_keys}，尝试的键: {possible_keys}")
    
    def _apply_normalization(self, inputs, outputs):
        """
        应用归一化处理
        
        Args:
            inputs: 输入张量 (N, input_dim)
            outputs: 输出张量 (N, output_dim)
            
        Returns:
            tuple: 归一化后的 (inputs, outputs)
        """
        logger.info(f"应用归一化方法: {self.normalize_method}")
        
        if self.normalize_method == 'minmax':
            # Min-Max归一化到指定范围
            inputs_normalized, input_stats = self._minmax_normalize(inputs, self.normalize_range)
            outputs_normalized, output_stats = self._minmax_normalize(outputs, self.normalize_range)
            
        elif self.normalize_method == 'zscore':
            # Z-score标准化
            inputs_normalized, input_stats = self._zscore_normalize(inputs)
            outputs_normalized, output_stats = self._zscore_normalize(outputs)
            
        elif self.normalize_method == 'robust':
            # 鲁棒归一化（使用中位数和四分位距）
            inputs_normalized, input_stats = self._robust_normalize(inputs)
            outputs_normalized, output_stats = self._robust_normalize(outputs)
            
        else:
            raise ValueError(f"不支持的归一化方法: {self.normalize_method}")
        
        # 保存统计信息
        self.input_stats = input_stats
        self.output_stats = output_stats
        
        # 记录归一化后的统计信息
        logger.info(f"输入归一化后范围: [{inputs_normalized.min():.6f}, {inputs_normalized.max():.6f}]")
        logger.info(f"输入归一化后均值: {inputs_normalized.mean():.6f}, 标准差: {inputs_normalized.std():.6f}")
        logger.info(f"输出归一化后范围: [{outputs_normalized.min():.6f}, {outputs_normalized.max():.6f}]")
        logger.info(f"输出归一化后均值: {outputs_normalized.mean():.6f}, 标准差: {outputs_normalized.std():.6f}")
        
        return inputs_normalized, outputs_normalized
    
    def _minmax_normalize(self, data, target_range=(0, 1)):
        """
        Min-Max归一化
        
        Args:
            data: 输入数据张量
            target_range: 目标范围 (min, max)
            
        Returns:
            tuple: (归一化数据, 统计信息)
        """
        data_min = data.min()
        data_max = data.max()
        data_range = data_max - data_min
        
        if data_range == 0:
            logger.warning("数据范围为0，跳过归一化")
            return data, {'min': data_min.item(), 'max': data_max.item(), 'range': 0}
        
        # 归一化到[0, 1]
        normalized = (data - data_min) / data_range
        
        # 缩放到目标范围
        target_min, target_max = target_range
        normalized = normalized * (target_max - target_min) + target_min
        
        stats = {
            'method': 'minmax',
            'original_min': data_min.item(),
            'original_max': data_max.item(),
            'original_range': data_range.item(),
            'target_min': target_min,
            'target_max': target_max
        }
        
        return normalized, stats
    
    def _zscore_normalize(self, data):
        """
        Z-score标准化
        
        Args:
            data: 输入数据张量
            
        Returns:
            tuple: (归一化数据, 统计信息)
        """
        data_mean = data.mean()
        data_std = data.std()
        
        if data_std == 0:
            logger.warning("数据标准差为0，跳过归一化")
            return data, {'mean': data_mean.item(), 'std': 0}
        
        normalized = (data - data_mean) / data_std
        
        stats = {
            'method': 'zscore',
            'mean': data_mean.item(),
            'std': data_std.item()
        }
        
        return normalized, stats
    
    def _robust_normalize(self, data):
        """
        鲁棒归一化（使用中位数和四分位距）
        
        Args:
            data: 输入数据张量
            
        Returns:
            tuple: (归一化数据, 统计信息)
        """
        data_median = data.median()
        q75 = data.quantile(0.75)
        q25 = data.quantile(0.25)
        iqr = q75 - q25
        
        if iqr == 0:
            logger.warning("四分位距为0，跳过归一化")
            return data, {'median': data_median.item(), 'iqr': 0}
        
        normalized = (data - data_median) / iqr
        
        stats = {
            'method': 'robust',
            'median': data_median.item(),
            'q25': q25.item(),
            'q75': q75.item(),
            'iqr': iqr.item()
        }
        
        return normalized, stats
    
    def _create_spatial_crop(self, data, crop_size):
        """
        创建空间裁剪
        
        Args:
            data: 原始数据，可能是 (N, H, W) 或 (N, H, W, C)
            crop_size: 裁剪尺寸
            
        Returns:
            numpy.ndarray: 裁剪后数据 (N, crop_size, crop_size)
        """
        # 处理不同的数据形状
        if len(data.shape) == 4:  # (N, H, W, C)
            num_samples, height, width, channels = data.shape
            # 如果有通道维度，取第一个通道或平均
            if channels == 1:
                data = data.squeeze(-1)  # 移除通道维度
            else:
                data = data.mean(axis=-1)  # 平均所有通道
        elif len(data.shape) == 3:  # (N, H, W)
            num_samples, height, width = data.shape
        else:
            raise ValueError(f"不支持的数据形状: {data.shape}")
        
        # 中心裁剪
        start_h = (height - crop_size) // 2
        end_h = start_h + crop_size
        start_w = (width - crop_size) // 2
        end_w = start_w + crop_size
        
        cropped_data = data[:, start_h:end_h, start_w:end_w]
        logger.info(f"裁剪区域: [{start_h}:{end_h}, {start_w}:{end_w}]")
        logger.info(f"裁剪后形状: {cropped_data.shape}")
        
        return cropped_data
    
    def get_normalization_stats(self):
        """
        获取归一化统计信息
        
        Returns:
            dict: 包含输入和输出归一化统计信息的字典
        """
        return {
            'input_stats': self.input_stats,
            'output_stats': self.output_stats,
            'normalize_method': self.normalize_method,
            'normalize_range': self.normalize_range
        }
    
    def denormalize_outputs(self, normalized_outputs):
        """
        反归一化输出数据
        
        Args:
            normalized_outputs: 归一化的输出张量
            
        Returns:
            torch.Tensor: 反归一化的输出张量
        """
        if not self.normalize or not self.output_stats:
            return normalized_outputs
        
        stats = self.output_stats
        
        if stats['method'] == 'minmax':
            # 反向Min-Max归一化
            target_min, target_max = stats['target_min'], stats['target_max']
            # 从目标范围还原到[0, 1]
            data_01 = (normalized_outputs - target_min) / (target_max - target_min)
            # 从[0, 1]还原到原始范围
            denormalized = data_01 * stats['original_range'] + stats['original_min']
            
        elif stats['method'] == 'zscore':
            # 反向Z-score标准化
            denormalized = normalized_outputs * stats['std'] + stats['mean']
            
        elif stats['method'] == 'robust':
            # 反向鲁棒归一化
            denormalized = normalized_outputs * stats['iqr'] + stats['median']
            
        else:
            denormalized = normalized_outputs
        
        return denormalized
    
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

class DataNormalizer:
    """
    数据归一化器，防止数据泄露
    只从训练集计算归一化统计信息
    """
    
    def __init__(self, train_dataset, method='minmax', target_range=(0, 1)):
        self.method = method
        self.target_range = target_range
        self.input_stats = {}
        self.output_stats = {}
        
        # 从训练集计算统计信息
        self._compute_stats(train_dataset)
    
    def _compute_stats(self, train_dataset):
        """从训练集计算归一化统计信息"""
        logger.info("从训练集计算归一化统计信息...")
        
        # 收集训练集的所有数据
        train_inputs = []
        train_outputs = []
        
        for i in range(len(train_dataset)):
            input_data, output_data = train_dataset[i]
            train_inputs.append(input_data)
            train_outputs.append(output_data)
        
        train_inputs = torch.stack(train_inputs)
        train_outputs = torch.stack(train_outputs)
        
        # 计算输入统计信息
        if self.method == 'minmax':
            self.input_stats = {
                'method': 'minmax',
                'min': train_inputs.min().item(),
                'max': train_inputs.max().item(),
                'target_min': self.target_range[0],
                'target_max': self.target_range[1]
            }
            self.output_stats = {
                'method': 'minmax',
                'min': train_outputs.min().item(),
                'max': train_outputs.max().item(),
                'target_min': self.target_range[0],
                'target_max': self.target_range[1]
            }
        elif self.method == 'zscore':
            self.input_stats = {
                'method': 'zscore',
                'mean': train_inputs.mean().item(),
                'std': train_inputs.std().item()
            }
            self.output_stats = {
                'method': 'zscore',
                'mean': train_outputs.mean().item(),
                'std': train_outputs.std().item()
            }
        
        logger.info(f"输入统计信息: {self.input_stats}")
        logger.info(f"输出统计信息: {self.output_stats}")
    
    def normalize_input(self, data):
        """归一化输入数据"""
        if self.method == 'minmax':
            data_range = self.input_stats['max'] - self.input_stats['min']
            if data_range == 0:
                return data
            normalized = (data - self.input_stats['min']) / data_range
            target_range = self.input_stats['target_max'] - self.input_stats['target_min']
            return normalized * target_range + self.input_stats['target_min']
        elif self.method == 'zscore':
            if self.input_stats['std'] == 0:
                return data
            return (data - self.input_stats['mean']) / self.input_stats['std']
        return data
    
    def normalize_output(self, data):
        """归一化输出数据"""
        if self.method == 'minmax':
            data_range = self.output_stats['max'] - self.output_stats['min']
            if data_range == 0:
                return data
            normalized = (data - self.output_stats['min']) / data_range
            target_range = self.output_stats['target_max'] - self.output_stats['target_min']
            return normalized * target_range + self.output_stats['target_min']
        elif self.method == 'zscore':
            if self.output_stats['std'] == 0:
                return data
            return (data - self.output_stats['mean']) / self.output_stats['std']
        return data
    
    def denormalize_output(self, normalized_data):
        """反归一化输出数据"""
        if self.method == 'minmax':
            target_range = self.output_stats['target_max'] - self.output_stats['target_min']
            if target_range == 0:
                return normalized_data
            # 从目标范围还原到[0, 1]
            data_01 = (normalized_data - self.output_stats['target_min']) / target_range
            # 从[0, 1]还原到原始范围
            data_range = self.output_stats['max'] - self.output_stats['min']
            return data_01 * data_range + self.output_stats['min']
        elif self.method == 'zscore':
            return normalized_data * self.output_stats['std'] + self.output_stats['mean']
        return normalized_data


class NormalizedDataset(Dataset):
    """
    包装数据集以应用归一化
    """
    
    def __init__(self, base_dataset, normalizer):
        self.base_dataset = base_dataset
        self.normalizer = normalizer
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        input_data, output_data = self.base_dataset[idx]
        
        if self.normalizer:
            input_data = self.normalizer.normalize_input(input_data)
            output_data = self.normalizer.normalize_output(output_data)
        
        return input_data, output_data


def create_enhanced_crop_dataloader(config):
    """
    创建增强版裁剪数据加载器，防止数据泄露
    归一化参数只从训练集计算
    
    Args:
        config: 配置字典，包含数据路径、批次大小等参数
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, normalizer)
    """
    # 从配置中获取参数
    data_config = config.get('data', {})
    data_path = data_config.get('data_path', '')
    batch_size = data_config.get('batch_size', 32)
    num_samples = data_config.get('num_samples', None)
    
    # 处理num_samples为None的情况
    if num_samples is None:
        # 尝试从数据文件获取全量样本数
        try:
            import h5py
            with h5py.File(data_path, 'r') as f:
                # 尝试多种可能的键名
                possible_keys = ['input', 'data', 'tensor', 'x', 'inputs']
                found_key = None
                
                # 首先列出所有可用的键
                available_keys = list(f.keys())
                logger.info(f"数据文件中的键: {available_keys}")
                
                # 尝试找到合适的键
                for key in possible_keys:
                    if key in f:
                        found_key = key
                        break
                
                # 如果没找到预期的键，使用第一个可用的键
                if found_key is None and available_keys:
                    found_key = available_keys[0]
                    logger.info(f"使用第一个可用键: {found_key}")
                
                if found_key:
                    num_samples = f[found_key].shape[0]
                    logger.info(f"从数据文件获取全量样本数: {num_samples} (使用键: {found_key})")
                else:
                    num_samples = 1000  # 默认值
                    logger.warning(f"无法从数据文件获取样本数，使用默认值: {num_samples}")
        except Exception as e:
            num_samples = 1000  # 默认值
            logger.warning(f"读取数据文件失败: {e}，使用默认值: {num_samples}")
    else:
        logger.info(f"使用配置指定的样本数: {num_samples}")
    
    # 输入输出分辨率
    input_resolution = data_config.get('input_resolution', [32, 32])
    output_resolution = data_config.get('output_resolution', [128, 128])
    
    crop_size = input_resolution[0]  # 假设输入是正方形
    output_size = output_resolution[0]  # 假设输出是正方形
    
    # 归一化配置
    normalize = data_config.get('normalize', True)
    normalize_method = data_config.get('normalize_method', 'minmax')
    normalize_range = data_config.get('normalize_range', (0, 1))
    
    logger.info(f"创建增强版裁剪数据加载器:")
    logger.info(f"  数据路径: {data_path}")
    logger.info(f"  批次大小: {batch_size}")
    logger.info(f"  样本数量: {num_samples}")
    logger.info(f"  输入尺寸: {crop_size}x{crop_size}")
    logger.info(f"  输出尺寸: {output_size}x{output_size}")
    logger.info(f"  归一化: {normalize}")
    if normalize:
        logger.info(f"  归一化方法: {normalize_method}")
        logger.info(f"  归一化范围: {normalize_range}")
    
    # 创建原始数据集（不进行归一化）
    dataset = EnhancedCropDataset(
        data_path=data_path,
        num_samples=num_samples,
        crop_size=crop_size,
        output_size=output_size,
        normalize=False,  # 先不归一化
        normalize_method=normalize_method,
        normalize_range=normalize_range
    )
    
    # 数据集分割
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    # 创建归一化器（只从训练集计算统计信息）
    normalizer = None
    if normalize:
        normalizer = DataNormalizer(
            train_dataset=train_dataset,
            method=normalize_method,
            target_range=normalize_range
        )
        logger.info("✅ 归一化器已创建，统计信息仅从训练集计算")
    
    # 包装数据集以应用归一化
    if normalizer:
        train_dataset = NormalizedDataset(train_dataset, normalizer)
        val_dataset = NormalizedDataset(val_dataset, normalizer)
        test_dataset = NormalizedDataset(test_dataset, normalizer)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=data_config.get('num_workers', 0)  # Windows兼容性
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=data_config.get('num_workers', 0)
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=data_config.get('num_workers', 0)
    )
    
    logger.info(f"数据集分割: 训练={len(train_dataset)}, 验证={len(val_dataset)}, 测试={len(test_dataset)}")
    logger.info("✅ 数据加载器创建完成，已防止数据泄露")
    
    return train_loader, val_loader, test_loader, normalizer