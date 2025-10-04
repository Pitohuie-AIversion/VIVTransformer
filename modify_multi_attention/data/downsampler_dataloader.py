#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
降采样数据加载器

功能:
1. 基于现有的crop_dataloader.py，添加降采样功能
2. 支持动态配置裁剪vs降采样模式
3. 集成OpenCV和SciPy降采样方法
4. 兼容现有的数据加载接口

作者: AI Assistant
日期: 2025
"""

import os
import sys
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import h5py
from pathlib import Path
import logging
from typing import Tuple, Optional, Dict, Any, Union

# 设置日志
logger = logging.getLogger(__name__)

# 尝试导入OpenCV
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False
    logger.warning("OpenCV未安装，将使用SciPy作为备选方案")

# 导入SciPy
try:
    from scipy.ndimage import zoom
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.error("SciPy未安装，降采样功能将不可用")

class ResolutionDownsampler:
    """
    分辨率降采样器类
    专门用于对整张图进行分辨率降低处理
    """
    
    def __init__(self, 
                 downsample_method: str = 'bilinear',
                 preserve_aspect_ratio: bool = True,
                 anti_aliasing: bool = True):
        """
        初始化分辨率降采样器
        
        Args:
            downsample_method: 降采样方法 ('bilinear', 'nearest', 'bicubic', 'area', 'lanczos')
            preserve_aspect_ratio: 是否保持宽高比
            anti_aliasing: 是否启用抗锯齿
        """
        self.downsample_method = downsample_method
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.anti_aliasing = anti_aliasing
        
        # 验证降采样方法
        valid_methods = ['bilinear', 'nearest', 'bicubic', 'area', 'lanczos']
        if downsample_method not in valid_methods:
            raise ValueError(f"不支持的降采样方法: {downsample_method}. 支持的方法: {valid_methods}")
        
        logger.info(f"分辨率降采样器初始化完成:")
        logger.info(f"  降采样方法: {downsample_method}")
        logger.info(f"  保持宽高比: {preserve_aspect_ratio}")
        logger.info(f"  抗锯齿: {anti_aliasing}")

    def _get_opencv_interpolation(self):
        """获取OpenCV插值方法"""
        method_map = {
            'nearest': cv2.INTER_NEAREST,
            'bilinear': cv2.INTER_LINEAR,
            'bicubic': cv2.INTER_CUBIC,
            'area': cv2.INTER_AREA,
            'lanczos': cv2.INTER_LANCZOS4
        }
        return method_map.get(self.downsample_method, cv2.INTER_LINEAR)

    def _get_scipy_order(self):
        """获取SciPy插值阶数"""
        order_map = {
            'nearest': 0,
            'bilinear': 1,
            'bicubic': 3,
            'area': 1,  # SciPy没有area方法，使用bilinear
            'lanczos': 3  # SciPy没有lanczos方法，使用bicubic
        }
        return order_map.get(self.downsample_method, 1)

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
            
            # 批量处理
            downsampled_batch = np.zeros((batch_size, target_h, target_w), dtype=data.dtype)
            
            for i in range(batch_size):
                downsampled_batch[i] = cv2.resize(data[i].astype(np.float32), 
                                                (target_w, target_h), 
                                                interpolation=self._get_opencv_interpolation())
            
            return downsampled_batch
        else:
            raise ValueError(f"不支持的数据形状: {data.shape}")

    def downsample_data_scipy(self, 
                             data: np.ndarray, 
                             target_size: Tuple[int, int]) -> np.ndarray:
        """
        使用SciPy进行降采样（备选方法）
        
        Args:
            data: 输入数据
            target_size: 目标尺寸 (height, width)
            
        Returns:
            降采样后的数据
        """
        if not HAS_SCIPY:
            raise ImportError("SciPy未安装，无法使用scipy方法")
            
        target_h, target_w = target_size
        
        # 处理不同的输入形状
        if len(data.shape) == 2:
            orig_h, orig_w = data.shape
            zoom_factors = (target_h / orig_h, target_w / orig_w)
        elif len(data.shape) == 3:
            batch_size, orig_h, orig_w = data.shape
            zoom_factors = (1.0, target_h / orig_h, target_w / orig_w)
        else:
            raise ValueError(f"不支持的数据形状: {data.shape}")
        
        # 使用SciPy进行降采样
        downsampled = zoom(data, zoom_factors, order=self._get_scipy_order())
        
        return downsampled

    def downsample_data(self, 
                       data: np.ndarray, 
                       target_size: Tuple[int, int],
                       method: str = 'opencv') -> np.ndarray:
        """
        降采样数据的主要接口
        
        Args:
            data: 输入数据
            target_size: 目标尺寸 (height, width)
            method: 使用的库 ('opencv' 或 'scipy')
            
        Returns:
            降采样后的数据
        """
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


class DownsampleDataset(Dataset):
    """
    降采样数据集类
    支持裁剪和降采样两种模式
    """
    
    def __init__(self, 
                 data_path: str, 
                 num_samples: int = 100, 
                 input_size: int = 32, 
                 output_size: int = 128,
                 mode: str = 'crop',  # 'crop' 或 'downsample'
                 downsample_method: str = 'bilinear',
                 downsample_backend: str = 'opencv',
                 normalize_data: bool = False,
                 anti_aliasing: bool = True):
        """
        初始化数据集
        
        Args:
            data_path: 数据文件路径
            num_samples: 样本数量
            input_size: 输入尺寸
            output_size: 输出尺寸
            mode: 处理模式 ('crop' 或 'downsample')
            downsample_method: 降采样方法
            downsample_backend: 降采样后端 ('opencv' 或 'scipy')
            normalize_data: 是否归一化数据
            anti_aliasing: 是否启用抗锯齿
        """
        self.data_path = Path(data_path)
        self.num_samples = num_samples
        self.input_size = input_size
        self.output_size = output_size
        self.mode = mode
        self.normalize_data = normalize_data
        self.anti_aliasing = anti_aliasing
        
        # 验证模式
        if mode not in ['crop', 'downsample']:
            raise ValueError(f"不支持的模式: {mode}. 支持的模式: ['crop', 'downsample']")
        
        # 初始化降采样器（如果需要）
        if mode == 'downsample':
            self.downsampler = ResolutionDownsampler(
                downsample_method=downsample_method,
                preserve_aspect_ratio=False,
                anti_aliasing=self.anti_aliasing
            )
            self.downsample_backend = downsample_backend
        
        # 加载和处理数据
        self.inputs, self.outputs = self._load_and_process_data()
        
        logger.info(f"数据集初始化完成:")
        logger.info(f"  模式: {mode}")
        logger.info(f"  输入形状: {self.inputs.shape}")
        logger.info(f"  输出形状: {self.outputs.shape}")
        logger.info(f"  样本数量: {len(self.inputs)}")

    def _load_original_data(self):
        """加载原始数据"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        logger.info(f"正在加载数据: {self.data_path}")
        
        with h5py.File(self.data_path, 'r') as f:
            # 获取数据集信息
            available_keys = list(f.keys())
            logger.info(f"可用数据集: {available_keys}")
            
            # 尝试不同的数据集名称，包括'tensor'
            data_key = None
            for key in ['data', 'u', 'solution', 'field', 'tensor']:
                if key in f:
                    data_key = key
                    break
            
            if data_key is None:
                raise ValueError(f"未找到有效的数据集。可用键: {available_keys}")
            
            # 加载数据
            data = f[data_key][:]
            logger.info(f"使用数据键: {data_key}，原始数据形状: {data.shape}")
            
            # 处理数据形状 - 如果是4维数据，去掉通道维度
            if len(data.shape) == 4 and data.shape[1] == 1:
                data = data.squeeze(1)  # 去掉通道维度 (N, 1, H, W) -> (N, H, W)
                logger.info(f"去掉通道维度后数据形状: {data.shape}")
            
            # 限制样本数量
            if len(data) > self.num_samples:
                data = data[:self.num_samples]
                logger.info(f"限制样本数量到: {self.num_samples}")
            
            return data

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

    def _create_spatial_downsample(self, data, target_size):
        """
        创建空间降采样
        
        Args:
            data: 原始数据 (N, H, W)
            target_size: 目标尺寸
            
        Returns:
            numpy.ndarray: 降采样后数据 (N, target_size, target_size)
        """
        target_shape = (target_size, target_size)
        downsampled_data = self.downsampler.downsample_data(
            data, target_shape, method=self.downsample_backend
        )
        
        logger.info(f"降采样: {data.shape} -> {downsampled_data.shape}")
        logger.info(f"降采样方法: {self.downsampler.downsample_method}")
        logger.info(f"降采样后端: {self.downsample_backend}")
        
        return downsampled_data

    def _load_and_process_data(self):
        """加载并处理数据"""
        # 加载原始数据
        original_data = self._load_original_data()
        
        # 确保数据是3D的 (N, H, W)
        if len(original_data.shape) == 4:
            # 如果是4D，取第一个通道或时间步
            original_data = original_data[:, 0, :, :]
            logger.info(f"转换4D数据到3D: {original_data.shape}")
        elif len(original_data.shape) != 3:
            raise ValueError(f"不支持的数据形状: {original_data.shape}")
        
        # 生成输入数据
        if self.mode == 'crop':
            inputs = self._create_spatial_crop(original_data, self.input_size)
        else:  # downsample
            inputs = self._create_spatial_downsample(original_data, self.input_size)
        
        # 生成输出数据
        if self.output_size == original_data.shape[1]:  # 如果输出尺寸等于原始尺寸
            outputs = original_data
        else:
            if self.mode == 'crop':
                outputs = self._create_spatial_crop(original_data, self.output_size)
            else:  # downsample
                outputs = self._create_spatial_downsample(original_data, self.output_size)
        
        # 数据归一化
        if self.normalize_data:
            inputs = self._normalize_data(inputs)
            outputs = self._normalize_data(outputs)
            logger.info("数据已归一化到 [0, 1] 范围")
        
        # 转换为PyTorch张量
        inputs = torch.FloatTensor(inputs)
        outputs = torch.FloatTensor(outputs)
        
        return inputs, outputs

    def _normalize_data(self, data):
        """归一化数据到[0, 1]范围"""
        data_min = data.min()
        data_max = data.max()
        
        if data_max > data_min:
            normalized = (data - data_min) / (data_max - data_min)
        else:
            normalized = data
        
        return normalized

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        output_data = self.outputs[idx]
        
        # 将3D数据展平为2D格式以兼容现有的训练流程
        # 输入: [height, width] -> [height * width]
        # 输出: [height, width] -> [height * width]
        if len(input_data.shape) == 2:
            input_data = input_data.flatten()
        if len(output_data.shape) == 2:
            output_data = output_data.flatten()
            
        return input_data, output_data


def create_downsample_dataloader(config):
    """
    创建降采样数据加载器
    
    Args:
        config: 配置字典
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 获取数据配置
    data_config = config.get('data', {})
    data_path = data_config.get('data_path')
    
    if not data_path:
        raise ValueError("数据路径未配置")
    
    # 获取数据集参数
    num_samples = data_config.get('num_samples', 100)
    
    # 优先从processing配置中获取分辨率参数
    processing_config = data_config.get('processing', {})
    downsample_config = processing_config.get('downsample', {})
    
    # 尝试多种配置参数名称
    input_size = (
        downsample_config.get('input_resolution') or 
        data_config.get('input_resolution') or 
        data_config.get('input_size', 32)
    )
    output_size = (
        downsample_config.get('output_resolution') or 
        data_config.get('output_resolution') or 
        data_config.get('output_size', 128)
    )
    
    # 处理input_size和output_size可能是列表的情况
    if isinstance(input_size, list):
        input_size = input_size[0]  # 取第一个值，假设是正方形
    if isinstance(output_size, list):
        output_size = output_size[0]  # 取第一个值，假设是正方形
    
    # 获取处理模式配置
    processing_config = data_config.get('processing', {})
    mode = processing_config.get('mode', 'crop')  # 默认为裁剪模式
    
    # 获取降采样配置
    downsample_config = processing_config.get('downsample', {})
    downsample_method = downsample_config.get('method', 'bilinear')
    downsample_backend = downsample_config.get('backend', 'opencv')
    
    # 获取抗锯齿配置
    anti_aliasing = downsample_config.get('antialias', True)  # 默认启用抗锯齿
    
    # 获取归一化配置
    normalize_data = data_config.get('normalize', False)
    
    # 获取数据加载器参数
    batch_size = config.get('training', {}).get('batch_size', 32)
    num_workers = config.get('training', {}).get('num_workers', 0)
    
    logger.info(f"创建数据加载器:")
    logger.info(f"  数据路径: {data_path}")
    logger.info(f"  处理模式: {mode}")
    logger.info(f"  输入尺寸: {input_size}")
    logger.info(f"  输出尺寸: {output_size}")
    logger.info(f"  样本数量: {num_samples}")
    logger.info(f"  批次大小: {batch_size}")
    
    if mode == 'downsample':
        logger.info(f"  降采样方法: {downsample_method}")
        logger.info(f"  降采样后端: {downsample_backend}")
        logger.info(f"  抗锯齿: {anti_aliasing}")
    
    # 创建数据集
    dataset = DownsampleDataset(
        data_path=data_path,
        num_samples=num_samples,
        input_size=input_size,
        output_size=output_size,
        mode=mode,
        downsample_method=downsample_method,
        downsample_backend=downsample_backend,
        normalize_data=normalize_data,
        anti_aliasing=anti_aliasing
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
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    logger.info(f"数据集分割完成:")
    logger.info(f"  训练集: {len(train_dataset)} 样本")
    logger.info(f"  验证集: {len(val_dataset)} 样本")
    logger.info(f"  测试集: {len(test_dataset)} 样本")
    
    # 创建一个简单的归一化器（如果需要归一化）
    normalizer = None
    if normalize_data:
        # 创建一个简单的归一化器类
        class SimpleNormalizer:
            def __init__(self):
                self.name = "SimpleNormalizer"
            
            def __class__(self):
                return SimpleNormalizer
        
        normalizer = SimpleNormalizer()
        logger.info(f"✅ 创建归一化器: {normalizer.__class__.__name__}")
    
    return train_loader, val_loader, test_loader, normalizer


def get_data_info(data_path):
    """
    获取数据文件信息
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        dict: 数据信息
    """
    try:
        with h5py.File(data_path, 'r') as f:
            keys = list(f.keys())
            
            # 获取主要数据集信息
            main_key = None
            for key in ['data', 'u', 'solution', 'field']:
                if key in f:
                    main_key = key
                    break
            
            if main_key:
                data_shape = f[main_key].shape
                data_dtype = f[main_key].dtype
                
                return {
                    'keys': keys,
                    'main_key': main_key,
                    'shape': data_shape,
                    'dtype': str(data_dtype),
                    'file_size': os.path.getsize(data_path)
                }
    except Exception as e:
        logger.error(f"读取数据文件失败: {e}")
        return {}


if __name__ == "__main__":
    # 测试代码
    test_config = {
        'data': {
            'data_path': 'test_data.h5',
            'num_samples': 50,
            'input_size': 32,
            'output_size': 128,
            'processing': {
                'mode': 'downsample',
                'downsample': {
                    'method': 'bilinear',
                    'backend': 'opencv'
                }
            },
            'normalize': True
        },
        'training': {
            'batch_size': 16,
            'num_workers': 0
        }
    }
    
    try:
        train_loader, val_loader, test_loader = create_downsample_dataloader(test_config)
        logger.info("数据加载器创建成功!")
    except Exception as e:
        logger.error(f"数据加载器创建失败: {e}")