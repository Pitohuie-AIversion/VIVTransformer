#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试动态分辨率训练器的数据归一化功能
"""

import os
import sys
import numpy as np
import torch
import h5py
import yaml
from pathlib import Path
from typing import Tuple

# 简化版的DynamicResolutionDataset类，仅用于测试归一化功能
class TestDynamicResolutionDataset:
    def __init__(self, 
                 data_path: str,
                 input_resolution: Tuple[int, int],
                 output_resolution: Tuple[int, int],
                 num_samples: int = 100,
                 crop_mode: str = 'center',
                 normalize_data: bool = True):
        self.data_path = Path(data_path)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.num_samples = num_samples
        self.crop_mode = crop_mode
        self.normalize_data = normalize_data
        
        # 归一化相关属性
        self.input_min = None
        self.input_max = None
        self.output_min = None
        self.output_max = None
        
        # 加载原始数据
        self.original_data = self._load_original_data()
        
        # 如果启用归一化，计算归一化参数
        if self.normalize_data:
            self._compute_normalization_params()
    
    def _load_original_data(self) -> np.ndarray:
        """加载原始数据"""
        print(f"加载原始数据: {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            if 'tensor' in f:
                tensor_data = f['tensor']
                print(f"原始数据形状: {tensor_data.shape}")
                
                # 读取指定数量的样本
                actual_samples = min(self.num_samples, tensor_data.shape[0])
                data = np.array(tensor_data[:actual_samples], dtype=np.float32)
                
                # 如果数据是4D，移除通道维度
                if len(data.shape) == 4 and data.shape[1] == 1:
                    data = data.squeeze(1)
                    print(f"移除通道维度后形状: {data.shape}")
                
                return data
            else:
                raise ValueError("数据文件中未找到'tensor'键")
    
    def _crop_data(self, data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """裁剪数据到目标尺寸"""
        _, orig_h, orig_w = data.shape
        target_h, target_w = target_size
        
        if target_h > orig_h or target_w > orig_w:
            raise ValueError(f"目标尺寸 {target_size} 大于原始尺寸 ({orig_h}, {orig_w})")
        
        if self.crop_mode == 'center':
            start_h = (orig_h - target_h) // 2
            start_w = (orig_w - target_w) // 2
        else:
            start_h = 0
            start_w = 0
        
        return data[:, start_h:start_h+target_h, start_w:start_w+target_w]
    
    def _resize_data(self, data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """调整数据尺寸"""
        from scipy.ndimage import zoom
        
        _, orig_h, orig_w = data.shape
        target_h, target_w = target_size
        
        zoom_factors = (1, target_h / orig_h, target_w / orig_w)
        resized_data = zoom(data, zoom_factors, order=1)
        
        return resized_data
    
    def _compute_normalization_params(self):
        """计算归一化参数"""
        print("计算归一化参数...")
        
        input_samples = []
        output_samples = []
        
        # 采样部分数据来计算统计信息
        sample_size = min(10, len(self.original_data))
        for i in range(sample_size):
            original_sample = self.original_data[i:i+1]
            
            # 生成输入数据
            if self.input_resolution == self.original_data.shape[1:]:
                input_data = original_sample[0]
            else:
                input_cropped = self._crop_data(original_sample, self.input_resolution)
                input_data = input_cropped[0]
            
            # 生成输出数据
            if self.output_resolution == self.original_data.shape[1:]:
                output_data = original_sample[0]
            elif self.output_resolution[0] <= self.original_data.shape[1] and self.output_resolution[1] <= self.original_data.shape[2]:
                output_cropped = self._crop_data(original_sample, self.output_resolution)
                output_data = output_cropped[0]
            else:
                try:
                    output_resized = self._resize_data(original_sample, self.output_resolution)
                    output_data = output_resized[0]
                except ImportError:
                    print("警告: scipy不可用，使用简单插值")
                    output_data = original_sample[0]  # 回退到原始数据
            
            input_samples.append(input_data.flatten())
            output_samples.append(output_data.flatten())
        
        # 计算统计信息
        input_array = np.concatenate(input_samples)
        output_array = np.concatenate(output_samples)
        
        self.input_min = np.min(input_array)
        self.input_max = np.max(input_array)
        self.output_min = np.min(output_array)
        self.output_max = np.max(output_array)
        
        print(f"输入数据范围: [{self.input_min:.6f}, {self.input_max:.6f}]")
        print(f"输出数据范围: [{self.output_min:.6f}, {self.output_max:.6f}]")
    
    def _normalize_data(self, data: np.ndarray, data_min: float, data_max: float) -> np.ndarray:
        """归一化数据到[0, 1]范围"""
        if data_max - data_min == 0:
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    def __len__(self):
        return len(self.original_data)
    
    def __getitem__(self, idx):
        """获取单个样本"""
        original_sample = self.original_data[idx:idx+1]
        
        # 生成输入数据
        if self.input_resolution == self.original_data.shape[1:]:
            input_data = original_sample[0]
        else:
            input_cropped = self._crop_data(original_sample, self.input_resolution)
            input_data = input_cropped[0]
        
        # 生成输出数据
        if self.output_resolution == self.original_data.shape[1:]:
            output_data = original_sample[0]
        elif self.output_resolution[0] <= self.original_data.shape[1] and self.output_resolution[1] <= self.original_data.shape[2]:
            output_cropped = self._crop_data(original_sample, self.output_resolution)
            output_data = output_cropped[0]
        else:
            try:
                output_resized = self._resize_data(original_sample, self.output_resolution)
                output_data = output_resized[0]
            except ImportError:
                output_data = original_sample[0]  # 回退到原始数据
        
        # 展平为1D张量
        input_flat = input_data.flatten()
        output_flat = output_data.flatten()
        
        # 应用归一化
        if self.normalize_data:
            input_flat = self._normalize_data(input_flat, self.input_min, self.input_max)
            output_flat = self._normalize_data(output_flat, self.output_min, self.output_max)
        
        return torch.from_numpy(input_flat).float(), torch.from_numpy(output_flat).float()

def test_normalization():
    """
    测试数据归一化功能
    """
    print("=== 测试数据归一化功能 ===")
    
    # 加载配置
    config_path = "dynamic_config.yaml"
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    data_config = config['data']
    
    # 测试数据路径
    test_data_path = data_config['path']
    if not os.path.exists(test_data_path):
        print(f"测试数据文件不存在: {test_data_path}")
        print("请确保数据文件存在或修改配置中的数据路径")
        return
    
    print(f"使用数据文件: {test_data_path}")
    
    # 测试启用归一化的数据集
    print("\n--- 测试启用归一化 ---")
    dataset_normalized = TestDynamicResolutionDataset(
        data_path=test_data_path,
        input_resolution=tuple(data_config['input_resolution']),
        output_resolution=tuple(data_config['output_resolution']),
        num_samples=20,  # 使用较少样本进行测试
        crop_mode=data_config['crop_mode'],
        normalize_data=True
    )
    
    # 测试禁用归一化的数据集
    print("\n--- 测试禁用归一化 ---")
    dataset_raw = TestDynamicResolutionDataset(
        data_path=test_data_path,
        input_resolution=tuple(data_config['input_resolution']),
        output_resolution=tuple(data_config['output_resolution']),
        num_samples=20,
        crop_mode=data_config['crop_mode'],
        normalize_data=False
    )
    
    # 比较数据范围
    print("\n=== 数据范围比较 ===")
    
    # 获取几个样本进行比较
    for i in range(min(3, len(dataset_normalized))):
        # 归一化数据
        input_norm, output_norm = dataset_normalized[i]
        # 原始数据
        input_raw, output_raw = dataset_raw[i]
        
        print(f"\n样本 {i+1}:")
        print(f"  归一化输入范围: [{input_norm.min():.6f}, {input_norm.max():.6f}]")
        print(f"  原始输入范围:   [{input_raw.min():.6f}, {input_raw.max():.6f}]")
        print(f"  归一化输出范围: [{output_norm.min():.6f}, {output_norm.max():.6f}]")
        print(f"  原始输出范围:   [{output_raw.min():.6f}, {output_raw.max():.6f}]")
    
    # 验证归一化参数
    print(f"\n=== 归一化参数 ===")
    print(f"输入数据归一化参数: min={dataset_normalized.input_min:.6f}, max={dataset_normalized.input_max:.6f}")
    print(f"输出数据归一化参数: min={dataset_normalized.output_min:.6f}, max={dataset_normalized.output_max:.6f}")
    
    # 验证归一化是否正确
    input_norm, output_norm = dataset_normalized[0]
    if 0 <= input_norm.min() and input_norm.max() <= 1 and 0 <= output_norm.min() and output_norm.max() <= 1:
        print("\n[OK] 归一化功能正常工作！数据已成功归一化到[0,1]范围")
    else:
        print("\n[ERROR] 归一化功能异常！数据未正确归一化到[0,1]范围")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_normalization()