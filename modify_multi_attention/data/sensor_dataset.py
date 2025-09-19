#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器数据适配器
支持多种传感器数据格式的统一处理

功能:
1. 时间序列传感器数据处理
2. 空间分布传感器数据处理  
3. 多模态传感器数据融合
4. 稀疏传感器到稠密场重建

作者: AI Assistant
日期: 2025
"""

import torch
import numpy as np
import h5py
import pandas as pd
from torch.utils.data import Dataset
from typing import Tuple, Optional, Dict, Any, Union, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class SensorDataset(Dataset):
    """
    传感器数据集类
    支持多种传感器数据格式和处理模式
    """
    
    def __init__(self, 
                 data_path: str,
                 sensor_config: Dict[str, Any],
                 target_resolution: Optional[Tuple[int, int]] = None,
                 normalize_data: bool = True,
                 max_samples: Optional[int] = None):
        """
        初始化传感器数据集
        
        Args:
            data_path: 数据文件路径
            sensor_config: 传感器配置
            target_resolution: 目标分辨率 (H, W)
            normalize_data: 是否归一化数据
            max_samples: 最大样本数量
        """
        self.data_path = Path(data_path)
        self.sensor_config = sensor_config
        self.target_resolution = target_resolution
        self.normalize_data = normalize_data
        self.max_samples = max_samples
        
        # 传感器类型
        self.sensor_type = sensor_config.get('type', 'time_series')
        self.sensor_positions = sensor_config.get('positions', None)
        self.temporal_length = sensor_config.get('temporal_length', 100)
        self.spatial_domain = sensor_config.get('spatial_domain', (1.0, 1.0))
        
        # 加载数据
        self.data, self.targets = self._load_sensor_data()
        
        # 数据统计
        self.data_stats = self._compute_statistics()
        
        logger.info(f"传感器数据集初始化完成:")
        logger.info(f"  数据类型: {self.sensor_type}")
        logger.info(f"  样本数量: {len(self.data)}")
        logger.info(f"  输入维度: {self.data[0].shape if len(self.data) > 0 else 'N/A'}")
        logger.info(f"  目标维度: {self.targets[0].shape if len(self.targets) > 0 else 'N/A'}")
    
    def _load_sensor_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """加载传感器数据"""
        if self.data_path.suffix == '.h5' or self.data_path.suffix == '.hdf5':
            return self._load_hdf5_data()
        elif self.data_path.suffix == '.csv':
            return self._load_csv_data()
        elif self.data_path.suffix == '.pt':
            return self._load_pytorch_data()
        elif self.data_path.suffix == '.npz':
            return self._load_numpy_data()
        else:
            raise ValueError(f"不支持的数据格式: {self.data_path.suffix}")
    
    def _load_hdf5_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """加载HDF5格式传感器数据"""
        data_list = []
        target_list = []
        
        with h5py.File(self.data_path, 'r') as f:
            if self.sensor_type == 'time_series':
                # 时间序列传感器数据
                sensor_data = f['sensor_readings'][:]  # [N, T, S] N样本, T时间, S传感器
                target_data = f['target_field'][:]     # [N, H, W] 目标场
                
                for i in range(len(sensor_data)):
                    if self.max_samples and i >= self.max_samples:
                        break
                    
                    # 传感器读数展平为序列
                    sensor_seq = torch.tensor(sensor_data[i], dtype=torch.float32)
                    sensor_flat = sensor_seq.view(-1)  # [T*S]
                    
                    # 目标场
                    target = torch.tensor(target_data[i], dtype=torch.float32)
                    
                    data_list.append(sensor_flat)
                    target_list.append(target)
            
            elif self.sensor_type == 'spatial_sparse':
                # 空间稀疏传感器数据
                positions = f['sensor_positions'][:]   # [S, 2] 传感器位置
                readings = f['sensor_readings'][:]     # [N, S] 传感器读数
                target_data = f['target_field'][:]     # [N, H, W] 目标场
                
                for i in range(len(readings)):
                    if self.max_samples and i >= self.max_samples:
                        break
                    
                    # 构建稀疏输入：位置+读数
                    pos_tensor = torch.tensor(positions, dtype=torch.float32)
                    read_tensor = torch.tensor(readings[i], dtype=torch.float32)
                    
                    # 拼接位置和读数 [S, 3] (x, y, value)
                    sparse_input = torch.cat([pos_tensor, read_tensor.unsqueeze(-1)], dim=-1)
                    sparse_flat = sparse_input.view(-1)  # [S*3]
                    
                    # 目标场
                    target = torch.tensor(target_data[i], dtype=torch.float32)
                    
                    data_list.append(sparse_flat)
                    target_list.append(target)
        
        return data_list, target_list
    
    def _load_csv_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """加载CSV格式传感器数据"""
        df = pd.read_csv(self.data_path)
        data_list = []
        target_list = []
        
        # 假设CSV格式: timestamp, sensor1, sensor2, ..., target_x, target_y, target_value
        sensor_cols = [col for col in df.columns if col.startswith('sensor')]
        target_cols = [col for col in df.columns if col.startswith('target')]
        
        # 按时间窗口分组
        window_size = self.temporal_length
        for i in range(0, len(df) - window_size + 1, window_size):
            if self.max_samples and len(data_list) >= self.max_samples:
                break
            
            window_data = df.iloc[i:i+window_size]
            
            # 传感器数据
            sensor_data = window_data[sensor_cols].values
            sensor_tensor = torch.tensor(sensor_data, dtype=torch.float32).view(-1)
            
            # 目标数据（使用窗口最后时刻的目标）
            if target_cols:
                target_data = window_data[target_cols].iloc[-1].values
                target_tensor = torch.tensor(target_data, dtype=torch.float32)
            else:
                # 如果没有目标列，使用传感器数据的未来值作为目标
                if i + window_size < len(df):
                    future_data = df.iloc[i+window_size][sensor_cols].values
                    target_tensor = torch.tensor(future_data, dtype=torch.float32)
                else:
                    continue
            
            data_list.append(sensor_tensor)
            target_list.append(target_tensor)
        
        return data_list, target_list
    
    def _load_pytorch_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """加载PyTorch格式传感器数据"""
        data_dict = torch.load(self.data_path)
        
        if isinstance(data_dict, dict):
            sensor_data = data_dict['sensor_data']
            target_data = data_dict['target_data']
        else:
            # 假设是元组 (sensor_data, target_data)
            sensor_data, target_data = data_dict
        
        data_list = []
        target_list = []
        
        for i in range(len(sensor_data)):
            if self.max_samples and i >= self.max_samples:
                break
            
            data_list.append(sensor_data[i].view(-1))
            target_list.append(target_data[i])
        
        return data_list, target_list
    
    def _load_numpy_data(self) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """加载NumPy格式传感器数据"""
        data_dict = np.load(self.data_path)
        
        sensor_data = data_dict['sensor_data']
        target_data = data_dict['target_data']
        
        data_list = []
        target_list = []
        
        for i in range(len(sensor_data)):
            if self.max_samples and i >= self.max_samples:
                break
            
            sensor_tensor = torch.tensor(sensor_data[i], dtype=torch.float32).view(-1)
            target_tensor = torch.tensor(target_data[i], dtype=torch.float32)
            
            data_list.append(sensor_tensor)
            target_list.append(target_tensor)
        
        return data_list, target_list
    
    def _compute_statistics(self) -> Dict[str, float]:
        """计算数据统计信息"""
        if not self.data:
            return {}
        
        # 输入数据统计
        input_data = torch.stack(self.data)
        input_mean = input_data.mean().item()
        input_std = input_data.std().item()
        input_min = input_data.min().item()
        input_max = input_data.max().item()
        
        # 目标数据统计
        target_data = torch.stack([t.view(-1) for t in self.targets])
        target_mean = target_data.mean().item()
        target_std = target_data.std().item()
        target_min = target_data.min().item()
        target_max = target_data.max().item()
        
        return {
            'input_mean': input_mean,
            'input_std': input_std,
            'input_min': input_min,
            'input_max': input_max,
            'target_mean': target_mean,
            'target_std': target_std,
            'target_min': target_min,
            'target_max': target_max,
            'input_dim': self.data[0].numel(),
            'target_dim': self.targets[0].numel()
        }
    
    def _normalize_data(self, data: torch.Tensor, stats_key: str) -> torch.Tensor:
        """归一化数据"""
        if not self.normalize_data:
            return data
        
        mean = self.data_stats[f'{stats_key}_mean']
        std = self.data_stats[f'{stats_key}_std']
        
        return (data - mean) / (std + 1e-8)
    
    def _denormalize_data(self, data: torch.Tensor, stats_key: str) -> torch.Tensor:
        """反归一化数据"""
        if not self.normalize_data:
            return data
        
        mean = self.data_stats[f'{stats_key}_mean']
        std = self.data_stats[f'{stats_key}_std']
        
        return data * (std + 1e-8) + mean
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取数据项"""
        # 输入数据
        input_data = self.data[idx]
        input_normalized = self._normalize_data(input_data, 'input')
        
        # 目标数据
        target_data = self.targets[idx]
        
        # 如果需要调整目标分辨率
        if self.target_resolution and target_data.dim() >= 2:
            target_data = self._resize_target(target_data)
        
        target_normalized = self._normalize_data(target_data.view(-1), 'target')
        
        # 时间步（简单递增）
        time_step = torch.tensor(idx % self.temporal_length, dtype=torch.long)
        
        return input_normalized, target_normalized, time_step
    
    def _resize_target(self, target: torch.Tensor) -> torch.Tensor:
        """调整目标分辨率"""
        if target.dim() == 2:
            # [H, W] -> [1, 1, H, W]
            target = target.unsqueeze(0).unsqueeze(0)
        elif target.dim() == 3:
            # [C, H, W] -> [1, C, H, W]
            target = target.unsqueeze(0)
        
        # 使用双线性插值调整分辨率
        target_resized = torch.nn.functional.interpolate(
            target, 
            size=self.target_resolution, 
            mode='bilinear', 
            align_corners=False
        )
        
        # 移除批次维度
        if target_resized.dim() == 4:
            target_resized = target_resized.squeeze(0)
        
        return target_resized
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        return {
            'dataset_size': len(self),
            'sensor_type': self.sensor_type,
            'input_dimension': self.data_stats.get('input_dim', 0),
            'target_dimension': self.data_stats.get('target_dim', 0),
            'target_resolution': self.target_resolution,
            'statistics': self.data_stats
        }


class MultiModalSensorDataset(SensorDataset):
    """
    多模态传感器数据集
    支持多种传感器类型的融合
    """
    
    def __init__(self, 
                 data_paths: Dict[str, str],
                 sensor_configs: Dict[str, Dict[str, Any]],
                 fusion_mode: str = 'concatenate',
                 **kwargs):
        """
        初始化多模态传感器数据集
        
        Args:
            data_paths: 各模态数据路径 {'modality1': 'path1', ...}
            sensor_configs: 各模态传感器配置
            fusion_mode: 融合模式 ('concatenate', 'attention', 'weighted')
        """
        self.data_paths = data_paths
        self.sensor_configs = sensor_configs
        self.fusion_mode = fusion_mode
        
        # 加载各模态数据
        self.modality_datasets = {}
        for modality, path in data_paths.items():
            config = sensor_configs[modality]
            self.modality_datasets[modality] = SensorDataset(
                path, config, **kwargs
            )
        
        # 确保所有模态数据长度一致
        self.length = min(len(ds) for ds in self.modality_datasets.values())
        
        logger.info(f"多模态传感器数据集初始化完成:")
        logger.info(f"  模态数量: {len(self.modality_datasets)}")
        logger.info(f"  融合模式: {fusion_mode}")
        logger.info(f"  样本数量: {self.length}")
    
    def __len__(self) -> int:
        return self.length
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """获取多模态数据项"""
        modality_inputs = []
        modality_targets = []
        
        # 收集各模态数据
        for modality, dataset in self.modality_datasets.items():
            input_data, target_data, time_step = dataset[idx]
            modality_inputs.append(input_data)
            modality_targets.append(target_data)
        
        # 融合输入数据
        if self.fusion_mode == 'concatenate':
            fused_input = torch.cat(modality_inputs, dim=0)
        elif self.fusion_mode == 'weighted':
            # 简单加权平均（可扩展为学习权重）
            weights = torch.ones(len(modality_inputs)) / len(modality_inputs)
            fused_input = sum(w * inp for w, inp in zip(weights, modality_inputs))
        else:
            # 默认拼接
            fused_input = torch.cat(modality_inputs, dim=0)
        
        # 融合目标数据（取第一个模态的目标）
        fused_target = modality_targets[0]
        
        return fused_input, fused_target, time_step