#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态分辨率训练器 - 集成SVD模态投影

功能:
1. 根据配置文件动态设置输入输出分辨率
2. 实时生成训练数据，无需预先生成大文件
3. 支持SVD模态投影，将不同分辨率数据投影到统一潜在空间
4. 支持命令行参数和YAML配置文件
5. 灵活的数据集生成和加载
6. 支持多GPU训练 (DataParallel模式)

作者: AI Assistant
日期: 2025
"""

# 设置无头模式环境变量（必须在任何GUI相关导入之前）
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''
os.environ['HEADLESS'] = '1'

# 设置matplotlib后端为Agg（无GUI）
import matplotlib
matplotlib.use('Agg')

import sys
import torch
import numpy as np
import h5py
import yaml
import argparse
import logging
import gc
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# 设置日志（在导入之前）
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加路径
project_root = Path(__file__).parent.parent
modify_multi_attention_path = project_root / 'modify_multi_attention'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(modify_multi_attention_path))
sys.path.insert(0, str(Path(__file__).parent))

# 导入SVD投影模块
try:
    from pde_process.svd_modal_projection import (
        SVDModalProjector, SVDProjectionConfig, create_svd_projector_from_config
    )
    HAS_SVD_PROJECTION = True
    logger.info("✅ 成功导入SVD投影模块")
except ImportError as e:
    HAS_SVD_PROJECTION = False
    logger.warning(f"⚠️ SVD投影模块导入失败: {e}")

# 导入归一化功能
from pde_process.pdebench_data_processor import PDEBenchProcessor, PDEBenchConfig

# 导入新的降分辨率模块
try:
    from pde_process.resolution_downsampler.resolution_downsampler import (
        ResolutionDownsampler, DownsampledResolutionDataset
    )
    HAS_DOWNSAMPLER = True
    logger.info("✅ 成功导入降分辨率模块")
except ImportError as e:
    HAS_DOWNSAMPLER = False
    logger.warning(f"⚠️ 降分辨率模块导入失败: {e}")
    logger.warning("将使用传统的裁剪方法")

# 尝试多种导入方式
try:
    from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel
    from modify_multi_attention.training.trainer import train_model, test_model
    from modify_multi_attention.utils.visualization import plot_losses
    from modify_multi_attention.utils.svd10_loss import TotalLossWithSVD
    from modify_multi_attention.utils.logging_utils import setup_logging
    # 导入增强版SVD损失函数
    from modify_multi_attention.utils.enhanced_svd_loss import (
        EnhancedTotalLossWithSVD, create_enhanced_svd_loss,
        get_global_svd_stats, print_global_svd_stats, reset_global_svd_stats
    )
except ImportError as e:
    logger.warning(f"第一次导入失败: {e}")
    try:
        # 尝试直接导入
        sys.path.append(str(modify_multi_attention_path))
        from mymodels.transformer import TransformerFlowReconstructionModel
        from training.trainer import train_model, test_model
        from utils.visualization import plot_losses
        from utils.svd10_loss import TotalLossWithSVD
        from utils.logging_utils import setup_logging
        # 导入增强版SVD损失函数
        from utils.enhanced_svd_loss import (
            EnhancedTotalLossWithSVD, create_enhanced_svd_loss,
            get_global_svd_stats, print_global_svd_stats, reset_global_svd_stats
        )
    except ImportError as e2:
        logger.error(f"所有导入方式都失败: {e2}")
        logger.error(f"当前Python路径: {sys.path}")
        logger.error(f"项目根目录: {project_root}")
        logger.error(f"modify_multi_attention路径: {modify_multi_attention_path}")
        raise

def log_gpu_memory(stage: str = ""):
    """记录GPU内存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3   # GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        
        logger.info(f"🔍 GPU内存状态 {stage}:")
        logger.info(f"  已分配: {allocated:.2f} GB")
        logger.info(f"  已保留: {reserved:.2f} GB")
        logger.info(f"  峰值分配: {max_allocated:.2f} GB")
        
        # 如果内存使用过高，建议清理
        if allocated > 8.0:  # 8GB阈值
            logger.warning("⚠️  GPU内存使用较高，建议启用懒加载或减少批次大小")
    else:
        logger.info(f"💻 CPU模式 {stage}")

def cleanup_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("🧹 已清理GPU缓存")

def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置文件的合理性"""
    warnings = []
    errors = []
    
    # 检查数据配置
    data_config = config.get('data', {})
    
    # 检查分辨率配置
    input_res = data_config.get('input_resolution', [])
    output_res = data_config.get('output_resolution', [])
    
    if len(input_res) != 2 or len(output_res) != 2:
        errors.append("输入和输出分辨率必须是长度为2的列表")
    
    if input_res and output_res:
        input_dim = input_res[0] * input_res[1]
        output_dim = output_res[0] * output_res[1]
        
        # 检查分辨率是否过高（仅警告，不阻止训练）
        if input_dim > 50000 or output_dim > 50000:
            warnings.append(f"分辨率较高，可能导致内存问题: 输入{input_dim}, 输出{output_dim}")
        
        if output_dim > input_dim * 16:  # 超分辨率倍数过大
            warnings.append(f"输出分辨率({output_res})比输入分辨率({input_res})大很多，可能影响训练效果")
    
    # 检查批次大小
    batch_size = data_config.get('batch_size', 16)
    if batch_size > 64:
        warnings.append(f"批次大小({batch_size})较大，可能导致内存不足")
    
    # 检查高级功能配置
    gradient_config = config.get('gradient', {})
    mixed_precision_config = config.get('mixed_precision', {})
    
    if gradient_config.get('clip_enabled', False) and not gradient_config.get('clip_value'):
        errors.append("启用梯度裁剪时必须设置clip_value")
    
    if mixed_precision_config.get('enabled', False) and not torch.cuda.is_available():
        warnings.append("混合精度训练需要GPU支持，当前为CPU模式")
    
    # 检查降采样配置
    downsampling_config = data_config.get('downsampling', {})
    if downsampling_config.get('enabled', False):
        if not HAS_DOWNSAMPLER:
            errors.append("配置启用了降采样，但降分辨率模块未成功导入")
        else:
            # 验证降采样方法
            method = downsampling_config.get('method', 'bilinear')
            valid_methods = ['bilinear', 'nearest', 'bicubic', 'area', 'lanczos']
            if method not in valid_methods:
                errors.append(f"无效的降采样方法: {method}，支持的方法: {valid_methods}")
            
            # 验证降采样后端
            backend = downsampling_config.get('backend', 'auto')
            valid_backends = ['auto', 'opencv', 'scipy']
            if backend not in valid_backends:
                warnings.append(f"无效的降采样后端: {backend}，支持的后端: {valid_backends}")
    
    # 检查SVD投影配置
    svd_config = data_config.get('svd_projection', {})
    if svd_config.get('enabled', False):
        if not HAS_SVD_PROJECTION:
            errors.append("配置启用了SVD投影，但SVD投影模块未成功导入")
        else:
            # 验证SVD投影参数
            n_modes = svd_config.get('n_modes')
            if n_modes is not None and (not isinstance(n_modes, int) or n_modes <= 0):
                errors.append(f"无效的SVD模态数量: {n_modes}，必须是正整数")
            
            energy_threshold = svd_config.get('energy_threshold')
            if energy_threshold is not None and (not isinstance(energy_threshold, (int, float)) or energy_threshold <= 0 or energy_threshold > 1):
                errors.append(f"无效的能量阈值: {energy_threshold}，必须在(0,1]范围内")
            
            if n_modes is None and energy_threshold is None:
                errors.append("SVD投影必须指定n_modes或energy_threshold中的一个")
    
    # 输出验证结果
    if warnings:
        logger.warning("⚠️  配置验证警告:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    if errors:
        logger.error("❌ 配置验证错误:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    if not warnings and not errors:
        logger.info("✅ 配置验证通过")
    
    # 添加调试信息
    logger.info(f"配置验证详情: warnings={len(warnings)}, errors={len(errors)}")
    
    return True

class DynamicResolutionDataset(torch.utils.data.Dataset):
    """
    动态分辨率数据集类 - 支持SVD模态投影
    根据配置实时生成不同分辨率的输入输出数据，可选地将其投影到统一的潜在空间
    """
    
    def __init__(self, 
                 data_path: str,
                 input_resolution: Tuple[int, int],
                 output_resolution: Tuple[int, int],
                 num_samples: int = 100,
                 crop_mode: str = 'center',
                 normalize_data: bool = True,
                 lazy_loading: bool = False,
                 svd_projector: Optional[SVDModalProjector] = None,
                 use_svd_projection: bool = False):
        """
        初始化动态分辨率数据集
        
        Args:
            data_path: 原始数据文件路径
            input_resolution: 输入分辨率 (height, width)
            output_resolution: 输出分辨率 (height, width)
            num_samples: 样本数量
            crop_mode: 裁剪模式 ('center', 'random', 'corner')
            normalize_data: 是否对数据进行归一化
            lazy_loading: 是否启用懒加载（节省内存）
            svd_projector: SVD投影器实例
            use_svd_projection: 是否使用SVD投影
        """
        self.data_path = Path(data_path)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.num_samples = num_samples
        self.crop_mode = crop_mode
        self.normalize_data = normalize_data
        self.lazy_loading = lazy_loading
        
        # SVD投影相关
        self.svd_projector = svd_projector
        self.use_svd_projection = use_svd_projection and svd_projector is not None and HAS_SVD_PROJECTION
        
        # 归一化相关属性
        self.input_min = None
        self.input_max = None
        self.output_min = None
        self.output_max = None
        
        # 计算维度
        self.input_dim = input_resolution[0] * input_resolution[1]
        self.output_dim = output_resolution[0] * output_resolution[1]
        
        # 如果启用SVD投影，更新潜在空间维度
        if self.use_svd_projection:
            self.latent_input_dim = self.svd_projector.latent_dim
            self.latent_output_dim = self.svd_projector.latent_dim
        else:
            self.latent_input_dim = self.input_dim
            self.latent_output_dim = self.output_dim
        
        # 数据形状信息（用于懒加载）
        self.data_shape = None
        
        if lazy_loading:
            # 懒加载模式：只获取数据形状和归一化参数
            self._init_lazy_loading()
            logger.info("🚀 启用懒加载模式，节省内存使用")
        else:
            # 传统模式：加载所有数据到内存
            self.original_data = self._load_original_data()
            
        # 如果启用归一化，计算归一化参数
        if self.normalize_data:
            self._compute_normalization_params()
        
        logger.info(f"动态数据集初始化完成:")
        logger.info(f"  输入分辨率: {input_resolution} -> {self.input_dim}维")
        logger.info(f"  输出分辨率: {output_resolution} -> {self.output_dim}维")
        if self.use_svd_projection:
            logger.info(f"  SVD投影: 启用 -> 潜在空间维度: {self.latent_input_dim}")
        logger.info(f"  样本数量: {num_samples}")
        logger.info(f"  裁剪模式: {crop_mode}")
        logger.info(f"  数据归一化: {normalize_data}")
        logger.info(f"  懒加载模式: {lazy_loading}")
    
    def _init_lazy_loading(self):
        """
        初始化懒加载模式，获取数据形状信息
        """
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
        """
        懒加载模式下加载单个样本数据
        
        Args:
            index: 样本索引
            
        Returns:
            单个样本数据
        """
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
        """
        加载原始数据
        
        Returns:
            加载的数据数组
        """
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
    
    def _crop_data(self, data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """
        裁剪数据到目标尺寸
        
        Args:
            data: 输入数据 (batch, height, width)
            target_size: 目标尺寸 (height, width)
            
        Returns:
            裁剪后的数据
        """
        _, orig_h, orig_w = data.shape
        target_h, target_w = target_size
        
        if target_h > orig_h or target_w > orig_w:
            raise ValueError(f"目标尺寸 {target_size} 大于原始尺寸 ({orig_h}, {orig_w})")
        
        if self.crop_mode == 'center':
            start_h = (orig_h - target_h) // 2
            start_w = (orig_w - target_w) // 2
        elif self.crop_mode == 'corner':
            start_h = 0
            start_w = 0
        elif self.crop_mode == 'random':
            start_h = np.random.randint(0, orig_h - target_h + 1)
            start_w = np.random.randint(0, orig_w - target_w + 1)
        else:
            raise ValueError(f"不支持的裁剪模式: {self.crop_mode}")
        
        end_h = start_h + target_h
        end_w = start_w + target_w
        
        return data[:, start_h:end_h, start_w:end_w]
    
    def _resize_data(self, data: np.ndarray, target_size: Tuple[int, int], method: str = 'bilinear') -> np.ndarray:
        """
        插值数据到目标尺寸
        
        Args:
            data: 输入数据 (batch, height, width)
            target_size: 目标尺寸 (height, width)
            method: 插值方法
            
        Returns:
            插值后的数据
        """
        from scipy.ndimage import zoom
        
        _, orig_h, orig_w = data.shape
        target_h, target_w = target_size
        
        zoom_factors = (1, target_h / orig_h, target_w / orig_w)
        
        if method == 'bilinear':
            resized_data = zoom(data, zoom_factors, order=1)
        elif method == 'nearest':
            resized_data = zoom(data, zoom_factors, order=0)
        else:
            raise ValueError(f"不支持的插值方法: {method}")
        
        return resized_data
    
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
        
        # 保存归一化信息
        self.normalization_info = {
            'global_min': float(global_min),
            'global_max': float(global_max),
            'normalization_method': 'global_minmax',
            'original_data_shape': self.data_shape if self.lazy_loading else self.original_data.shape,
            'input_resolution': self.input_resolution,
            'output_resolution': self.output_resolution
        }
        
        logger.info(f"全局数据范围: [{global_min:.6f}, {global_max:.6f}]")
        logger.info(f"归一化信息已记录，可用于反归一化")
    
    def _normalize_data(self, data: np.ndarray, data_min: float, data_max: float) -> np.ndarray:
        """归一化数据到[0, 1]范围"""
        eps = 1e-8
        if abs(data_max - data_min) < eps:
            logger.warning(f"数据范围过小，可能存在常数数据: [{data_min:.6f}, {data_max:.6f}]")
            return np.zeros_like(data)
        return (data - data_min) / (data_max - data_min)
    
    def _denormalize_data(self, data: np.ndarray, data_min: float, data_max: float) -> np.ndarray:
        """反归一化数据"""
        return data * (data_max - data_min) + data_min
    
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
        
        # 生成输入数据（裁剪到输入分辨率）
        if self.input_resolution == original_shape:
            input_data = original_sample[0]
        else:
            input_cropped = self._crop_data(original_sample, self.input_resolution)
            input_data = input_cropped[0]
        
        # 生成输出数据
        if self.output_resolution == original_shape:
            output_data = original_sample[0]
        elif self.output_resolution[0] <= original_shape[0] and self.output_resolution[1] <= original_shape[1]:
            # 输出分辨率小于等于原始分辨率，使用裁剪
            output_cropped = self._crop_data(original_sample, self.output_resolution)
            output_data = output_cropped[0]
        else:
            # 输出分辨率大于原始分辨率，使用插值
            output_resized = self._resize_data(original_sample, self.output_resolution)
            output_data = output_resized[0]
        
        # 展平为1D
        input_flat = input_data.flatten()
        output_flat = output_data.flatten()
        
        # 应用归一化
        if self.normalize_data:
            input_flat = self._normalize_data(input_flat, self.input_min, self.input_max)
            output_flat = self._normalize_data(output_flat, self.output_min, self.output_max)
        
        # 应用SVD投影
        if self.use_svd_projection:
            # 确保数据格式为numpy数组
            input_flat = np.ascontiguousarray(input_flat, dtype=np.float32)
            output_flat = np.ascontiguousarray(output_flat, dtype=np.float32)
            
            # 投影到潜在空间
            input_projected = self.svd_projector.transform_input(input_flat.reshape(1, -1))[0]
            output_projected = self.svd_projector.transform_output(output_flat.reshape(1, -1))[0]
            
            return (
                torch.from_numpy(input_projected).contiguous().float(),
                torch.from_numpy(output_projected).contiguous().float(),
                torch.tensor(idx, dtype=torch.long)
            )
        else:
            # 确保数据类型为float32并且内存连续
            input_array = np.ascontiguousarray(input_flat, dtype=np.float32)
            output_array = np.ascontiguousarray(output_flat, dtype=np.float32)
            
            return (
                torch.from_numpy(input_array).contiguous(),
                torch.from_numpy(output_array).contiguous(),
                torch.tensor(idx, dtype=torch.long)
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
            'latent_input_dim': self.latent_input_dim,
            'latent_output_dim': self.latent_output_dim,
            'input_range': (sample_input.min().item(), sample_input.max().item()),
            'output_range': (sample_output.min().item(), sample_output.max().item()),
            'use_svd_projection': self.use_svd_projection
        }
    
    def get_latent_dimensions(self):
        """获取潜在空间维度信息"""
        return {
            'latent_input_dim': self.latent_input_dim,
            'latent_output_dim': self.latent_output_dim,
            'original_input_dim': self.input_dim,
            'original_output_dim': self.output_dim,
            'use_svd_projection': self.use_svd_projection
        }
    
    def get_normalization_info(self):
        """获取归一化信息，用于反归一化"""
        if hasattr(self, 'normalization_info'):
            return self.normalization_info.copy()
        else:
            # 兼容旧版本
            return {
                'global_min': float(self.input_min),
                'global_max': float(self.input_max),
                'normalization_method': 'global_minmax',
                'original_data_shape': self.original_data.shape if hasattr(self, 'original_data') else self.data_shape,
                'input_resolution': self.input_resolution,
                'output_resolution': self.output_resolution
            }
    
    def denormalize_predictions(self, normalized_data):
        """反归一化预测结果，恢复物理信息"""
        if hasattr(self, 'global_min') and hasattr(self, 'global_max'):
            data_min = self.global_min
            data_max = self.global_max
        else:
            data_min = self.output_min
            data_max = self.output_max
        
        # 处理torch.Tensor
        if hasattr(normalized_data, 'cpu'):
            normalized_data = normalized_data.cpu().numpy()
        
        return self._denormalize_data(normalized_data, data_min, data_max)
    
    def save_normalization_info(self, filepath):
        """保存归一化信息到文件"""
        import json
        
        norm_info = self.get_normalization_info()
        
        if filepath.endswith('.json'):
            with open(filepath, 'w') as f:
                json.dump(norm_info, f, indent=2)
        elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'w') as f:
                yaml.dump(norm_info, f, default_flow_style=False)
        else:
            raise ValueError("文件格式不支持，请使用 .json 或 .yaml")
        
        logger.info(f"归一化信息已保存到: {filepath}")
    
    @staticmethod
    def load_normalization_info(filepath):
        """从文件加载归一化信息"""
        import json
        
        if filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                return json.load(f)
        elif filepath.endswith('.yaml') or filepath.endswith('.yml'):
            with open(filepath, 'r') as f:
                return yaml.safe_load(f)
        else:
            raise ValueError("文件格式不支持，请使用 .json 或 .yaml")

def create_dynamic_config(args=None):
    """
    创建动态配置
    
    Args:
        args: 命令行参数
    
    Returns:
        dict: 配置字典
    """
    # 默认配置
    default_config = {
        'data': {
            'path': r"X:\2025\Graduation_project\Pdebench_input_Transformer\VIVTransformer-1\PDEBench\pdebench\data_download\2D_DarcyFlow_beta0.1_Train.hdf5",
            'input_resolution': [32, 32],  # 输入分辨率
            'output_resolution': [128, 128],  # 输出分辨率
            'num_samples': 100,
            'crop_mode': 'center',  # 'center', 'random', 'corner'
            'normalize_data': True,  # 是否对数据进行归一化
            'batch_size': 16,
            'train_ratio': 0.7,
            'valid_ratio': 0.15,
            'test_ratio': 0.15,
            # SVD投影配置
            'svd_projection': {
                'enabled': False,
                'n_modes': 128,  # 保留的模态数量
                'energy_threshold': 0.95,  # 能量保留阈值
                'svd_method': 'svd',  # 'svd', 'randomized_svd'
                'random_state': 42,
                'normalization': 'zscore',  # 'zscore', 'minmax', 'robust', None
                'incremental': False,  # 是否使用增量学习
                'batch_size': 1000,  # 增量学习批次大小
                'memory_efficient': True  # 是否使用内存高效模式
            }
        },
        'training': {
            'epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'patience': 15,
            'min_delta': 1e-6,
            'save_best_model': True,
            'model_save_path': './results/models/dynamic_resolution_model.pth',
            'log_interval': 10
        },
        'model': {
            'num_layers': 4,
            'd_model': 512,
            'num_heads': 8,
            'max_time_steps': 100,
            'attention_type': 'sge'
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': './results/logs/dynamic_resolution_training.log'
        },
        'device': 'auto',
        'seed': 42,
        'visualization': {
            'enabled': True,
            'interval': 10,
            'max_samples': 5
        }
    }
    
    # 如果提供了参数，覆盖默认配置
    if args:
        if hasattr(args, 'input_resolution') and args.input_resolution:
            default_config['data']['input_resolution'] = args.input_resolution
        if hasattr(args, 'output_resolution') and args.output_resolution:
            default_config['data']['output_resolution'] = args.output_resolution
        if hasattr(args, 'num_samples') and args.num_samples:
            default_config['data']['num_samples'] = args.num_samples
        if hasattr(args, 'batch_size') and args.batch_size:
            default_config['data']['batch_size'] = args.batch_size
        if hasattr(args, 'epochs') and args.epochs:
            default_config['training']['epochs'] = args.epochs
        if hasattr(args, 'learning_rate') and args.learning_rate:
            default_config['training']['learning_rate'] = args.learning_rate
        if hasattr(args, 'svd_projection') and args.svd_projection:
            default_config['data']['svd_projection']['enabled'] = True
    
    # 计算模型输入输出维度
    if default_config['data']['svd_projection']['enabled']:
        # 使用SVD投影时，维度由模态数量决定
        latent_dim = default_config['data']['svd_projection']['n_modes']
        default_config['model']['input_dim'] = latent_dim
        default_config['model']['output_dim'] = latent_dim
    else:
        # 使用原始维度
        input_res = default_config['data']['input_resolution']
        output_res = default_config['data']['output_resolution']
        default_config['model']['input_dim'] = input_res[0] * input_res[1]
        default_config['model']['output_dim'] = output_res[0] * output_res[1]
    
    # 设置序列长度（假设单时间步）
    default_config['model']['seq_len'] = 1
    
    return default_config

def get_dynamic_loaders(config):
    """
    创建动态分辨率数据加载器（支持SVD投影）
    
    Args:
        config: 配置字典
        
    Returns:
        train_loader, valid_loader, test_loader, dataset
    """
    from torch.utils.data import DataLoader, random_split
    
    data_config = config['data']
    
    # 创建SVD投影器（如果启用）
    svd_projector = None
    use_svd_projection = False
    
    svd_config = data_config.get('svd_projection', {})
    if svd_config.get('enabled', False) and HAS_SVD_PROJECTION:
        logger.info("=== 创建SVD投影器 ===")
        
        # 创建SVD投影配置
        projection_config = SVDProjectionConfig.from_dict(svd_config)
        
        # 为SVD投影器准备样本数据
        logger.info("准备SVD投影器训练数据...")
        
        # 创建临时数据集用于SVD训练
        temp_dataset = DynamicResolutionDataset(
            data_path=data_config['path'],
            input_resolution=tuple(data_config['input_resolution']),
            output_resolution=tuple(data_config['output_resolution']),
            num_samples=min(data_config['num_samples'], 1000),  # 限制SVD训练样本数量
            crop_mode=data_config.get('crop_mode', 'center'),
            normalize_data=data_config.get('normalize_data', True),
            lazy_loading=data_config.get('lazy_loading', False),
            svd_projector=None,  # 不使用SVD投影
            use_svd_projection=False
        )
        
        # 收集样本数据
        input_samples = []
        output_samples = []
        
        logger.info(f"收集 {len(temp_dataset)} 个样本用于SVD投影器训练...")
        for i in range(len(temp_dataset)):
            input_data, output_data, _ = temp_dataset[i]
            input_samples.append(input_data.numpy())
            output_samples.append(output_data.numpy())
            
            if (i + 1) % 100 == 0:
                logger.info(f"已收集 {i + 1}/{len(temp_dataset)} 个样本")
        
        input_samples = np.array(input_samples)
        output_samples = np.array(output_samples)
        
        logger.info(f"输入样本形状: {input_samples.shape}")
        logger.info(f"输出样本形状: {output_samples.shape}")
        
        # 创建并训练SVD投影器
        svd_projector = create_svd_projector_from_config(projection_config)
        logger.info("训练SVD投影器...")
        
        svd_projector.fit(input_samples, output_samples)
        use_svd_projection = True
        
        logger.info(f"✅ SVD投影器训练完成")
        logger.info(f"输入投影维度: {input_samples.shape[1]} -> {svd_projector.latent_dim}")
        logger.info(f"输出投影维度: {output_samples.shape[1]} -> {svd_projector.latent_dim}")
        
        # 清理临时数据
        del temp_dataset, input_samples, output_samples
        cleanup_memory()
    
    # 创建最终数据集
    dataset = DynamicResolutionDataset(
        data_path=data_config['path'],
        input_resolution=tuple(data_config['input_resolution']),
        output_resolution=tuple(data_config['output_resolution']),
        num_samples=data_config['num_samples'],
        crop_mode=data_config.get('crop_mode', 'center'),
        normalize_data=data_config.get('normalize_data', True),
        lazy_loading=data_config.get('lazy_loading', False),
        svd_projector=svd_projector,
        use_svd_projection=use_svd_projection
    )
    
    # 分割数据集
    train_ratio = data_config.get('train_ratio', 0.7)
    valid_ratio = data_config.get('valid_ratio', 0.15)
    test_ratio = data_config.get('test_ratio', 0.15)
    
    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    valid_size = int(valid_ratio * total_size)
    test_size = total_size - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = random_split(
        dataset, [train_size, valid_size, test_size],
        generator=torch.Generator().manual_seed(config.get('seed', 42))
    )
    
    # 创建数据加载器
    batch_size = data_config.get('batch_size', 16)
    num_workers = config.get('dataloader', {}).get('num_workers', 0)
    pin_memory = config.get('dataloader', {}).get('pin_memory', True)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    logger.info(f"数据集分割完成:")
    logger.info(f"  训练集: {len(train_dataset)} 样本")
    logger.info(f"  验证集: {len(valid_dataset)} 样本")
    logger.info(f"  测试集: {len(test_dataset)} 样本")
    
    return train_loader, valid_loader, test_loader, dataset

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='动态分辨率训练器')
    
    # 添加配置文件参数
    parser.add_argument('--config', type=str, help='配置文件路径')
    
    # 数据相关参数
    parser.add_argument('--input-resolution', type=int, nargs=2, metavar=('H', 'W'),
                        help='输入分辨率，例如: --input-resolution 32 32')
    parser.add_argument('--output-resolution', type=int, nargs=2, metavar=('H', 'W'),
                        help='输出分辨率，例如: --output-resolution 128 128')
    parser.add_argument('--num-samples', type=int, help='样本数量')
    parser.add_argument('--batch-size', type=int, help='批次大小')
    
    # 训练相关参数
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--learning-rate', type=float, help='学习率')
    
    # SVD投影参数
    parser.add_argument('--svd-projection', action='store_true', help='启用SVD投影')
    parser.add_argument('--svd-modes', type=int, help='SVD模态数量')
    
    # 其他参数
    parser.add_argument('--device', type=str, choices=['auto', 'cpu', 'cuda'], default='auto',
                        help='计算设备')
    parser.add_argument('--seed', type=int, help='随机种子')
    
    return parser.parse_args()

def load_config_with_args(config_path, args):
    """
    加载配置文件并合并命令行参数
    
    Args:
        config_path: 配置文件路径
        args: 命令行参数
        
    Returns:
        合并后的配置字典
    """
    # 尝试加载配置文件
    if config_path and Path(config_path).exists():
        logger.info(f"加载配置文件: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        logger.info("使用默认配置")
        config = create_dynamic_config()
    
    # 合并命令行参数
    if args.input_resolution:
        config['data']['input_resolution'] = args.input_resolution
    if args.output_resolution:
        config['data']['output_resolution'] = args.output_resolution
    if args.num_samples:
        config['data']['num_samples'] = args.num_samples
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.device:
        config['device'] = args.device
    if args.seed:
        config['seed'] = args.seed
    if args.svd_projection:
        config['data']['svd_projection']['enabled'] = True
    if args.svd_modes:
        config['data']['svd_projection']['n_modes'] = args.svd_modes
    
    # 重新计算模型维度
    if config['data']['svd_projection']['enabled']:
        latent_dim = config['data']['svd_projection']['n_modes']
        config['model']['input_dim'] = latent_dim
        config['model']['output_dim'] = latent_dim
    else:
        input_res = config['data']['input_resolution']
        output_res = config['data']['output_resolution']
        config['model']['input_dim'] = input_res[0] * input_res[1]
        config['model']['output_dim'] = output_res[0] * output_res[1]
    
    # 确保必要的配置存在
    if 'dataloader' not in config:
        config['dataloader'] = {
            'num_workers': 0,
            'pin_memory': True,
            'prefetch_factor': 2,
            'persistent_workers': False
        }
    
    return config

def setup_enhanced_logging(config):
    """设置增强版日志系统"""
    logging_config = config.get('logging', {})
    log_level = getattr(logging, logging_config.get('level', 'INFO').upper())
    log_format = logging_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_file = logging_config.get('file')
    
    # 创建日志目录
    if log_file:
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
    
    # 配置日志
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"✅ 日志系统已配置，级别: {logging_config.get('level', 'INFO')}")
    if log_file:
        logger.info(f"📝 日志文件: {log_file}")
    
    return logger

def validate_config_detailed(config):
    """详细验证配置"""
    errors = []
    
    # 验证数据配置
    data_config = config.get('data', {})
    if not data_config.get('path'):
        errors.append("缺少数据路径配置")
    elif not Path(data_config['path']).exists():
        errors.append(f"数据文件不存在: {data_config['path']}")
    
    # 验证分辨率
    input_res = data_config.get('input_resolution')
    output_res = data_config.get('output_resolution')
    if not input_res or len(input_res) != 2:
        errors.append("输入分辨率必须是长度为2的列表")
    if not output_res or len(output_res) != 2:
        errors.append("输出分辨率必须是长度为2的列表")
    
    # 验证模型配置
    model_config = config.get('model', {})
    required_model_params = ['input_dim', 'output_dim', 'num_heads', 'num_layers', 'd_model']
    for param in required_model_params:
        if param not in model_config:
            errors.append(f"缺少模型参数: {param}")
    
    # 验证训练配置
    training_config = config.get('training', {})
    if training_config.get('epochs', 0) <= 0:
        errors.append("训练轮数必须大于0")
    if training_config.get('learning_rate', 0) <= 0:
        errors.append("学习率必须大于0")
    
    # 验证SVD配置
    svd_config = data_config.get('svd_projection', {})
    if svd_config.get('enabled', False):
        if not HAS_SVD_PROJECTION:
            errors.append("SVD投影模块未安装，无法启用SVD投影")
        else:
            n_modes = svd_config.get('n_modes')
            energy_threshold = svd_config.get('energy_threshold')
            if not n_modes and not energy_threshold:
                errors.append("SVD投影配置必须指定n_modes或energy_threshold")
    
    return errors

def main():
    """主函数"""
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 加载配置
        config_path = args.config if args.config else 'dynamic_config.yaml'
        
        # 加载和合并配置
        config = load_config_with_args(config_path, args)
        
        # 设置日志
        logger = setup_enhanced_logging(config)
        
        # 验证配置
        config_errors = validate_config_detailed(config)
        if config_errors:
            logger.error("❌ 配置验证失败:")
            for error in config_errors:
                logger.error(f"  - {error}")
            return
        
        logger.info("✅ 配置验证通过")
        
        # 输出配置信息
        logger.info("=== 配置信息 ===")
        logger.info(f"输入分辨率: {config['data']['input_resolution']} -> {config['model']['input_dim']}维")
        logger.info(f"输出分辨率: {config['data']['output_resolution']} -> {config['model']['output_dim']}维")
        logger.info(f"样本数量: {config['data']['num_samples']}")
        logger.info(f"批次大小: {config['data']['batch_size']}")
        logger.info(f"训练轮数: {config['training']['epochs']}")
        logger.info(f"注意力机制: {config['model']['attention_type']}")
        
        # SVD投影信息
        svd_config = config['data'].get('svd_projection', {})
        if svd_config.get('enabled', False):
            logger.info(f"SVD投影: 启用")
            logger.info(f"  模态数量: {svd_config.get('n_modes', 'auto')}")
            logger.info(f"  能量阈值: {svd_config.get('energy_threshold', 'N/A')}")
            logger.info(f"  SVD方法: {svd_config.get('svd_method', 'svd')}")
        else:
            logger.info("SVD投影: 禁用")
        
        # 设置随机种子
        if config.get('seed'):
            torch.manual_seed(config['seed'])
            np.random.seed(config['seed'])
            logger.info(f"🎲 设置随机种子: {config['seed']}")
        
        # 设备配置
        if config['device'] == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(config['device'])
        logger.info(f"🖥️ 使用设备: {device}")
        
        # 内存管理
        log_gpu_memory("初始状态")
        
        # 创建结果目录
        results_dir = Path('./results')
        results_dir.mkdir(exist_ok=True)
        model_save_path = config['training']['model_save_path']
        model_dir = os.path.dirname(model_save_path)
        if model_dir:
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"📁 模型保存目录: {model_dir}")
        
        # 验证配置
        logger.info("=== 验证配置 ===")
        if not validate_config(config):
            logger.error("配置验证失败，请修正配置后重试")
            return
        
        # 创建数据加载器
        logger.info("=== 创建数据加载器 ===")
        train_loader, valid_loader, test_loader, dataset = get_dynamic_loaders(config)
        
        # 获取数据统计信息
        stats = dataset.get_data_statistics()
        logger.info(f"数据统计: {stats}")
        
        # 归一化信息
        if dataset.normalize_data:
            norm_info = dataset.get_normalization_info()
            logger.info("=== 归一化信息 ===")
            logger.info(f"归一化方法: {norm_info['normalization_method']}")
            logger.info(f"全局数据范围: [{norm_info['global_min']:.6f}, {norm_info['global_max']:.6f}]")
            logger.info(f"原始数据形状: {norm_info['original_data_shape']}")
            logger.info(f"输入分辨率: {norm_info['input_resolution']}")
            logger.info(f"输出分辨率: {norm_info['output_resolution']}")
            logger.info("✅ 数据已归一化到[0,1]范围，可用于反归一化恢复物理信息")
        else:
            logger.info("ℹ️  归一化已禁用，使用原始数据范围")
        
        # 创建模型
        logger.info("=== 创建模型 ===")
        model = TransformerFlowReconstructionModel(
            input_dim=config['model']['input_dim'],
            output_dim=config['model']['output_dim'],
            num_heads=config['model']['num_heads'],
            num_layers=config['model']['num_layers'],
            d_model=config['model']['d_model'],
            max_time_steps=config['model']['max_time_steps'],
            attention_type=config['model']['attention_type'],
            seq_len=config['model']['seq_len']
        ).to(device)
        
        logger.info(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 多GPU支持
        use_dataparallel = config.get('use_dataparallel', False)
        if use_dataparallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info(f"🚀 启用多GPU训练: 检测到 {torch.cuda.device_count()} 张GPU")
            model = torch.nn.DataParallel(model)
            logger.info(f"   使用的GPU设备: {list(range(torch.cuda.device_count()))}")
        
        # 创建损失函数
        logger.info("=== 创建损失函数 ===")
        loss_config = config.get('loss', {})
        
        if loss_config.get('use_enhanced_svd', False):
            logger.info("使用增强版SVD损失函数")
            criterion = create_enhanced_svd_loss(config)
        else:
            logger.info("使用标准SVD损失函数")
            criterion = TotalLossWithSVD(
                base_weight=loss_config.get('base_weight', 1.0),
                svd_weights=loss_config.get('svd_weights', [0.1] * 5),
                device=device
            )
        
        # 创建优化器
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 1e-4)
        )
        
        # 学习率调度器
        scheduler_config = config.get('scheduler')
        scheduler = None
        if scheduler_config and scheduler_config.get('enabled', False):
            if scheduler_config['type'].lower() == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_config['step_size'],
                    gamma=scheduler_config['gamma']
                )
            elif scheduler_config['type'].lower() == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=scheduler_config['factor'],
                    patience=scheduler_config['patience']
                )
        
        log_gpu_memory("训练开始前")
        
        # 开始训练
        logger.info("=== 开始训练 ===")
        model, train_losses, valid_losses, test_losses = train_model(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            test_loader=test_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=config['training']['epochs'],
            device=device,
            early_stop_patience=config['training'].get('patience', 15),
            attention_type=config['model']['attention_type'],
            result_dir='./results',
            cfg=config,
            scheduler=scheduler,
            no_pretrained=config.get('no_pretrained', False)
        )
        
        log_gpu_memory("训练完成后")
        
        # 测试模型
        logger.info("=== 开始测试 ===")
        test_loss = test_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            attention_type=config['model']['attention_type'],
            parent_dir='./results',
            cfg=config
        )
        
        logger.info(f"最终测试损失: {test_loss:.6f}")
        
        # 增强版SVD损失函数的性能报告
        svd_config = config.get('loss', {})
        if svd_config.get('use_enhanced_svd', False):
            logger.info("\n=== SVD损失函数性能报告 ===")
            try:
                print_global_svd_stats()
            except Exception as e:
                logger.warning(f"SVD性能报告生成失败: {e}")
        
        # 显示归一化反投影提示
        if dataset.normalize_data:
            logger.info("\n=== 数据预处理和后处理提示 ===")
            logger.info("✅ 训练时使用了归一化数据，测试输出也为归一化结果")
            logger.info("💡 若需获得物理意义的结果，请调用 dataset.denormalize_predictions() 进行反归一化")
            logger.info("📄 归一化信息已保存，可用于生产环境的数据预处理和后处理")
            
            # 保存归一化信息到文件
            norm_save_path = './results/normalization_info.json'
            dataset.save_normalization_info(norm_save_path)
            logger.info(f"📂 归一化信息已保存到: {norm_save_path}")
        
        # SVD投影信息
        svd_config = config['data'].get('svd_projection', {})
        if svd_config.get('enabled', False) and dataset.svd_projector:
            logger.info("\n=== SVD投影结果总结 ===")
            logger.info(f"✅ SVD投影已启用")
            logger.info(f"   输入/输出数据已投影到共同的潜在空间")
            logger.info(f"   SVD方法: {svd_config.get('svd_method', 'svd')}")
            logger.info(f"   保留模态数量: {svd_config.get('n_modes', 'auto')}")
            if svd_config.get('energy_threshold'):
                logger.info(f"   能量阈值: {svd_config.get('energy_threshold')}")
            logger.info("💡 模型输出为潜在空间表示，可通过反投影恢复原始空间")
        
        logger.info("\n=== 训练完成 ===")
        logger.info(f"✅ 训练顺利完成，总耗时: {'-'}")
        logger.info(f"📊 最终训练损失: {train_losses[-1] if train_losses else 'N/A':.6f}")
        logger.info(f"📊 最终验证损失: {valid_losses[-1] if valid_losses else 'N/A':.6f}")
        logger.info(f"📊 最终测试损失: {test_loss:.6f}")
        logger.info(f"💾 模型已保存到: {config['training']['model_save_path']}")
        logger.info(f"📁 结果文件目录: ./results/")
        
        # 清理GPU内存
        cleanup_memory()
        log_gpu_memory("程序结束")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ 用户中断训练")
        cleanup_memory()
        
    except Exception as e:
        logger.error(f"❌ 训练过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        cleanup_memory()
        raise

if __name__ == "__main__":
    main()