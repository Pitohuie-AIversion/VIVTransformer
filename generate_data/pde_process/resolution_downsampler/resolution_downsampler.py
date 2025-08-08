#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分辨率降采样器

功能:
1. 对整张图进行分辨率降低处理，而不是裁剪
2. 支持多种降采样方法（双线性插值、最近邻、双三次插值等）
3. 支持自适应降采样比例
4. 保持数据的全局信息，避免裁剪造成的信息丢失
5. 支持批量处理和懒加载模式

作者: AI Assistant
日期: 2025
"""

import os
import sys
import torch
import numpy as np
import h5py
import yaml
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Union
from scipy.ndimage import zoom

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.warning("OpenCV未安装，将使用SciPy作为备选方案")

class ResolutionDownsampler:
    """
    分辨率降采样器类
    专门用于对整张图进行分辨率降低处理
    """

    def __init__(self,
                 downsample_method: str = 'bilinear',
                 preserve_aspect_ratio: bool = True,
                 anti_aliasing: bool = True,
                 backend: str = 'opencv'):
        """
        初始化分辨率降采样器

        Args:
            downsample_method: 降采样方法 ('bilinear', 'nearest', 'bicubic', 'area', 'lanczos')
            preserve_aspect_ratio: 是否保持宽高比
            anti_aliasing: 是否启用抗锯齿
            backend: 底层实现后端 ('opencv' 或 'scipy')
        """
        self.downsample_method = downsample_method
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.anti_aliasing = anti_aliasing
        self.backend = backend
        
        # 验证降采样方法
        valid_methods = ['bilinear', 'nearest', 'bicubic', 'area', 'lanczos']
        if downsample_method not in valid_methods:
            raise ValueError(f"不支持的降采样方法: {downsample_method}. 支持的方法: {valid_methods}")
        
        logger.info(f"分辨率降采样器初始化完成:")
        logger.info(f"  降采样方法: {downsample_method}")
        logger.info(f"  保持宽高比: {preserve_aspect_ratio}")
        logger.info(f"  抗锯齿: {anti_aliasing}")
        logger.info(f"  后端: {backend}")
    
    def _get_opencv_interpolation(self) -> int:
        """获取OpenCV插值方法"""
        if not HAS_OPENCV:
            raise ImportError("OpenCV未安装")
            
        method_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
        return method_map[self.downsample_method]
    
    def _get_scipy_order(self) -> int:
        """获取SciPy插值阶数"""
        order_map = {
            'nearest': 0,
            'bilinear': 1,
            'bicubic': 3,
            'area': 1,  # 使用双线性作为替代
            'lanczos': 3  # 使用双三次作为替代
        }
        return order_map[self.downsample_method]
    
    def downsample_data_opencv(self, 
                              data: np.ndarray, 
                              target_size: Tuple[int, int]) -> np.ndarray:
        """
        使用OpenCV进行降采样（推荐方法，质量更好）
        
        Args:
            data: 输入数据，形状为 (batch, height, width) 或 (height, width)
            target_size: 目标尺寸 (height, width)
            
        Returns:
            降采样后的数据
        """
        if not HAS_OPENCV:
            raise ImportError("OpenCV未安装，无法使用opencv方法")
            
        target_h, target_w = target_size
        
        # 处理不同的输入形状
        if len(data.shape) == 2:
            # 单张图像
            orig_h, orig_w = data.shape
            
            if target_h > orig_h or target_w > orig_w:
                raise ValueError(f"目标尺寸 {target_size} 大于原始尺寸 ({orig_h}, {orig_w})")
            
            # 使用OpenCV进行降采样
            downsampled = cv2.resize(data.astype(np.float32), 
                                   (target_w, target_h), 
                                   interpolation=self._get_opencv_interpolation())
            
            return downsampled
            
        elif len(data.shape) == 3:
            # 批量图像
            batch_size, orig_h, orig_w = data.shape
            
            if target_h > orig_h or target_w > orig_w:
                raise ValueError(f"目标尺寸 {target_size} 大于原始尺寸 ({orig_h}, {orig_w})")
            
            downsampled_data = np.zeros((batch_size, target_h, target_w), dtype=data.dtype)
            
            for i in range(batch_size):
                downsampled_data[i] = cv2.resize(data[i].astype(np.float32), 
                                                (target_w, target_h), 
                                                interpolation=self._get_opencv_interpolation())
            
            return downsampled_data
        else:
            raise ValueError(f"不支持的数据形状: {data.shape}")
    
    def downsample_data_scipy(self, 
                             data: np.ndarray, 
                             target_size: Tuple[int, int]) -> np.ndarray:
        """
        使用SciPy进行降采样（备用方法）
        
        Args:
            data: 输入数据
            target_size: 目标尺寸 (height, width)
            
        Returns:
            降采样后的数据
        """
        target_h, target_w = target_size
        
        if len(data.shape) == 2:
            orig_h, orig_w = data.shape
            zoom_h = target_h / orig_h
            zoom_w = target_w / orig_w
            
            return zoom(data, (zoom_h, zoom_w), order=self._get_scipy_order())
            
        elif len(data.shape) == 3:
            batch_size, orig_h, orig_w = data.shape
            zoom_h = target_h / orig_h
            zoom_w = target_w / orig_w
            
            downsampled_data = np.zeros((batch_size, target_h, target_w), dtype=data.dtype)
            
            for i in range(batch_size):
                downsampled_data[i] = zoom(data[i], (zoom_h, zoom_w), order=self._get_scipy_order())
            
            return downsampled_data
        else:
            raise ValueError(f"不支持的数据形状: {data.shape}")
    
    def downsample_data(self,
                       data: np.ndarray,
                       target_size: Tuple[int, int],
                       method: Optional[str] = None) -> np.ndarray:
        """
        降采样数据的主要接口

        Args:
            data: 输入数据
            target_size: 目标尺寸 (height, width)
            method: 使用的库 ('opencv' 或 'scipy')，默认为初始化时指定的后端
            
        Returns:
            降采样后的数据
        """
        if method is None:
            method = self.backend

        if method == 'opencv':
            if HAS_OPENCV:
                try:
                    return self.downsample_data_opencv(data, target_size)
                except Exception as e:
                    logger.warning(f"OpenCV降采样失败，切换到SciPy: {e}")
                    return self.downsample_data_scipy(data, target_size)
            else:
                logger.warning("OpenCV未安装，使用SciPy方法")
                return self.downsample_data_scipy(data, target_size)
        elif method == 'scipy':
            return self.downsample_data_scipy(data, target_size)
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    def calculate_optimal_size(self, 
                              original_size: Tuple[int, int], 
                              target_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        计算最优的降采样尺寸（保持宽高比）
        
        Args:
            original_size: 原始尺寸 (height, width)
            target_size: 目标尺寸 (height, width)
            
        Returns:
            最优尺寸 (height, width)
        """
        orig_h, orig_w = original_size
        target_h, target_w = target_size
        
        if not self.preserve_aspect_ratio:
            return target_size
        
        # 计算缩放比例
        scale_h = target_h / orig_h
        scale_w = target_w / orig_w
        
        # 使用较小的缩放比例以确保不超过目标尺寸
        scale = min(scale_h, scale_w)
        
        # 计算新的尺寸
        new_h = int(orig_h * scale)
        new_w = int(orig_w * scale)
        
        # 确保不超过目标尺寸
        new_h = min(new_h, target_h)
        new_w = min(new_w, target_w)
        
        logger.info(f"原始尺寸: {original_size} -> 最优尺寸: ({new_h}, {new_w})")
        
        return (new_h, new_w)
    
    def get_downsample_info(self) -> Dict[str, Any]:
        """
        获取降采样器信息
        
        Returns:
            降采样器配置信息
        """
        return {
            'downsample_method': self.downsample_method,
            'preserve_aspect_ratio': self.preserve_aspect_ratio,
            'anti_aliasing': self.anti_aliasing,
            'backend': self.backend,
            'supported_methods': ['bilinear', 'nearest', 'bicubic', 'area', 'lanczos']
        }

class DownsampledResolutionDataset(torch.utils.data.Dataset):
    """
    降采样分辨率数据集类
    使用降采样而不是裁剪来生成不同分辨率的数据
    """
    
    def __init__(self,
                 data_path: str,
                 input_resolution: Tuple[int, int],
                 output_resolution: Tuple[int, int],
                 num_samples: int = 100,
                 downsample_method: str = 'bilinear',
                 preserve_aspect_ratio: bool = True,
                 normalize_data: bool = True,
                 lazy_loading: bool = False,
                 backend: str = 'opencv',
                 anti_aliasing: bool = True):
        """
        初始化降采样分辨率数据集
        
        Args:
            data_path: 原始数据文件路径
            input_resolution: 输入分辨率 (height, width)
            output_resolution: 输出分辨率 (height, width)
            num_samples: 样本数量
            downsample_method: 降采样方法
            preserve_aspect_ratio: 是否保持宽高比
            normalize_data: 是否对数据进行归一化
            lazy_loading: 是否启用懒加载
            backend: 底层实现后端 ('opencv', 'scipy' 或 'auto')
            anti_aliasing: 是否启用抗锯齿
        """
        self.data_path = Path(data_path)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.num_samples = num_samples
        self.normalize_data = normalize_data
        self.lazy_loading = lazy_loading

        # 选择后端
        backend = backend.lower()
        if backend == 'auto':
            backend = 'opencv' if HAS_OPENCV else 'scipy'
        self.backend = backend

        # 初始化降采样器
        self.downsampler = ResolutionDownsampler(
            downsample_method=downsample_method,
            preserve_aspect_ratio=preserve_aspect_ratio,
            anti_aliasing=anti_aliasing,
            backend=self.backend
        )
        
        # 归一化相关属性
        self.input_min = None
        self.input_max = None
        self.output_min = None
        self.output_max = None
        
        # 计算维度
        self.input_dim = input_resolution[0] * input_resolution[1]
        self.output_dim = output_resolution[0] * output_resolution[1]
        
        # 数据形状信息（用于懒加载）
        self.data_shape = None
        
        if lazy_loading:
            self._init_lazy_loading()
            logger.info("🚀 启用懒加载模式，节省内存使用")
        else:
            self.original_data = self._load_original_data()
            
        # 如果启用归一化，计算归一化参数
        if self.normalize_data:
            self._compute_normalization_params()
        
        logger.info(f"降采样数据集初始化完成:")
        logger.info(f"  输入分辨率: {input_resolution} -> {self.input_dim}维")
        logger.info(f"  输出分辨率: {output_resolution} -> {self.output_dim}维")
        logger.info(f"  样本数量: {num_samples}")
        logger.info(f"  降采样方法: {downsample_method}")
        logger.info(f"  数据归一化: {normalize_data}")
        logger.info(f"  懒加载模式: {lazy_loading}")
        logger.info(f"  抗锯齿: {anti_aliasing}")
        logger.info(f"  后端: {self.backend}")
    
    def _init_lazy_loading(self):
        """初始化懒加载模式"""
        try:
            with h5py.File(self.data_path, 'r') as f:
                if 'tensor' not in f:
                    raise ValueError("数据文件中未找到'tensor'键")
                
                self.dataset_name = 'tensor'
                self.data_shape = f[self.dataset_name].shape
                
                logger.info(f"懒加载初始化: {self.dataset_name}")
                logger.info(f"数据形状: {self.data_shape}")
                
        except Exception as e:
            logger.error(f"懒加载初始化失败: {e}")
            raise
    
    def _load_sample_data(self, index: int) -> np.ndarray:
        """懒加载模式下加载单个样本数据"""
        try:
            with h5py.File(self.data_path, 'r') as f:
                data = np.array(f[self.dataset_name][index], dtype=np.float32)
                
                # 如果数据是3D且第一维是通道维度，移除它
                if len(data.shape) == 3 and data.shape[0] == 1:
                    data = data.squeeze(0)
                
                return data
        except Exception as e:
            logger.error(f"加载样本 {index} 失败: {e}")
            raise
    
    def _load_original_data(self) -> np.ndarray:
        """加载原始数据"""
        logger.info(f"加载原始数据: {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            if 'tensor' in f:
                tensor_data = f['tensor']
                logger.info(f"原始数据形状: {tensor_data.shape}")
                
                # 读取指定数量的样本
                actual_samples = min(self.num_samples, tensor_data.shape[0])
                data = np.array(tensor_data[:actual_samples], dtype=np.float32)
                
                # 如果数据是4D，移除通道维度
                if len(data.shape) == 4 and data.shape[1] == 1:
                    data = data.squeeze(1)
                    logger.info(f"移除通道维度后形状: {data.shape}")
                
                return data
            else:
                raise ValueError("数据文件中未找到'tensor'键")
    
    def _compute_normalization_params(self):
        """计算归一化参数"""
        logger.info("计算全局归一化参数...")
        
        if self.lazy_loading:
            # 懒加载模式：通过采样计算归一化参数
            logger.info("懒加载模式：使用采样数据计算归一化参数")
            sample_size = min(100, self.data_shape[0])
            sample_indices = np.linspace(0, self.data_shape[0]-1, sample_size, dtype=int)
            
            sample_data = []
            with h5py.File(self.data_path, 'r') as f:
                for idx in sample_indices:
                    data = np.array(f[self.dataset_name][idx], dtype=np.float32)
                    if len(data.shape) == 3 and data.shape[0] == 1:
                        data = data.squeeze(0)
                    sample_data.append(data)
            
            sample_data = np.array(sample_data)
            global_min = np.min(sample_data)
            global_max = np.max(sample_data)
            logger.info(f"基于 {sample_size} 个样本计算归一化参数")
        else:
            # 传统模式：使用全部加载的数据
            global_min = np.min(self.original_data)
            global_max = np.max(self.original_data)
        
        # 输入和输出都使用相同的全局归一化参数
        self.global_min = global_min
        self.global_max = global_max
        self.input_min = global_min
        self.input_max = global_max
        self.output_min = global_min
        self.output_max = global_max
        
        logger.info(f"全局数据范围: [{global_min:.6f}, {global_max:.6f}]")
    
    def _normalize_data(self, data: np.ndarray, data_min: float, data_max: float) -> np.ndarray:
        """归一化数据到[0, 1]范围"""
        eps = 1e-8
        if abs(data_max - data_min) < eps:
            logger.warning(f"数据范围过小，可能存在常数数据: [{data_min:.6f}, {data_max:.6f}]")
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    def __len__(self):
        if self.lazy_loading:
            return min(self.num_samples, self.data_shape[0])
        else:
            return len(self.original_data)
    
    def __getitem__(self, idx):
        # 获取原始样本
        if self.lazy_loading:
            original_sample = self._load_sample_data(idx)
            original_sample = original_sample[np.newaxis, :]  # 添加batch维度
        else:
            original_sample = self.original_data[idx:idx+1]
        
        # 获取原始形状
        original_shape = self.data_shape[1:] if self.lazy_loading else self.original_data.shape[1:]
        
        # 生成输入数据（降采样到输入分辨率）
        if self.input_resolution == original_shape:
            input_data = original_sample[0]
        else:
            # 使用降采样而不是裁剪
            input_downsampled = self.downsampler.downsample_data(
                original_sample,
                self.input_resolution,
                method=self.backend
            )
            input_data = input_downsampled[0]
        
        # 生成输出数据（降采样到输出分辨率）
        if self.output_resolution == original_shape:
            output_data = original_sample[0]
        else:
            # 使用降采样
            output_downsampled = self.downsampler.downsample_data(
                original_sample,
                self.output_resolution,
                method=self.backend
            )
            output_data = output_downsampled[0]
        
        # 展平为1D
        input_flat = input_data.flatten()
        output_flat = output_data.flatten()
        
        # 应用归一化
        if self.normalize_data:
            input_flat = self._normalize_data(input_flat, self.input_min, self.input_max)
            output_flat = self._normalize_data(output_flat, self.output_min, self.output_max)
        
        return (
            torch.FloatTensor(input_flat),
            torch.FloatTensor(output_flat),
            torch.tensor(idx, dtype=torch.float32)
        )
    
    def get_data_statistics(self):
        """获取数据统计信息"""
        sample_input, sample_output, _ = self[0]
        return {
            'num_samples': len(self),
            'input_shape': self.input_resolution,
            'output_shape': self.output_resolution,
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'input_range': (sample_input.min().item(), sample_input.max().item()),
            'output_range': (sample_output.min().item(), sample_output.max().item()),
            'downsample_method': self.downsampler.downsample_method
        }
    
    def get_downsampler_info(self):
        """获取降采样器信息"""
        return self.downsampler.get_downsample_info()
    
    def get_normalization_info(self):
        """获取归一化信息，保持与原始数据集的兼容性"""
        if self.normalize_data:
            # 使用全局数据范围作为 global_min/max
            global_min = min(float(self.input_min), float(self.output_min))
            global_max = max(float(self.input_max), float(self.output_max))
            return {
                'input_min': float(self.input_min),
                'input_max': float(self.input_max),
                'output_min': float(self.output_min),
                'output_max': float(self.output_max),
                'global_min': global_min,
                'global_max': global_max,
                'original_data_shape': self.data_shape,
                'input_resolution': self.input_resolution,
                'output_resolution': self.output_resolution,
                'normalized': True,
                'normalization_method': 'min_max_global'
            }
        else:
            return {
                'input_min': 0.0,
                'input_max': 1.0,
                'output_min': 0.0,
                'output_max': 1.0,
                'global_min': 0.0,
                'global_max': 1.0,
                'original_data_shape': self.data_shape,
                'input_resolution': self.input_resolution,
                'output_resolution': self.output_resolution,
                'normalized': False,
                'normalization_method': 'none'
            }
    
    def save_normalization_info(self, save_path: str):
        """
        保存归一化信息到文件
        
        Args:
            save_path: 保存路径
        """
        import json
        
        norm_info = self.get_normalization_info()
        
        # 确保保存目录存在
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存为JSON文件
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(norm_info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"归一化信息已保存到: {save_path}")

def demo_resolution_downsampler():
    """
    演示分辨率降采样器的使用
    """
    logger.info("=== 分辨率降采样器演示 ===")
    
    # 创建测试数据
    test_data = np.random.rand(5, 128, 128).astype(np.float32)
    logger.info(f"测试数据形状: {test_data.shape}")
    
    # 创建降采样器
    downsampler = ResolutionDownsampler(
        downsample_method='bilinear',
        preserve_aspect_ratio=True
    )
    
    # 测试不同的目标尺寸
    target_sizes = [(64, 64), (32, 32), (16, 16)]
    
    for target_size in target_sizes:
        logger.info(f"\n降采样到 {target_size}:")
        
        # 降采样
        downsampled = downsampler.downsample_data(test_data, target_size)
        logger.info(f"  原始形状: {test_data.shape}")
        logger.info(f"  降采样后形状: {downsampled.shape}")
        logger.info(f"  数据范围: [{downsampled.min():.4f}, {downsampled.max():.4f}]")
        
        # 计算压缩比
        original_size = test_data.shape[1] * test_data.shape[2]
        new_size = downsampled.shape[1] * downsampled.shape[2]
        compression_ratio = original_size / new_size
        logger.info(f"  压缩比: {compression_ratio:.2f}x")

if __name__ == "__main__":
    # 运行演示
    demo_resolution_downsampler()