#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态分辨率训练器

功能:
1. 根据配置文件动态设置输入输出分辨率
2. 实时生成训练数据，无需预先生成大文件
3. 支持命令行参数和YAML配置文件
4. 灵活的数据集生成和加载
5. 支持多GPU训练 (DataParallel模式)

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

def log_cpu_memory(stage: str = ""):
    """记录CPU内存使用情况"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=0.1)  # 快速采样
        logger.info(f"🔍 CPU状态 {stage}:")
        logger.info(f"  内存使用: {memory.used/1024**3:.2f}GB/{memory.total/1024**3:.2f}GB ({memory.percent:.1f}%)")
        logger.info(f"  CPU使用率: {cpu_percent:.1f}%")
        
        # 如果内存使用过高，给出建议
        if memory.percent > 85:
            logger.warning("⚠️  CPU内存使用较高，建议增加num_workers或启用懒加载")
    except ImportError:
        logger.warning("⚠️  psutil未安装，无法监控CPU内存")

def optimize_cpu_for_data_loading():
    """优化CPU设置以提升数据加载性能"""
    import os
    
    # 设置CPU线程数以充分利用服务器CPU
    cpu_count = os.cpu_count()
    # 为数据加载预留一些CPU核心，避免与训练计算竞争
    optimal_threads = max(1, cpu_count - 2)
    torch.set_num_threads(optimal_threads)
    
    # 设置OpenMP线程数
    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
    os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
    
    logger.info(f"🚀 CPU优化设置: 检测到{cpu_count}个CPU核心，设置{optimal_threads}个线程用于计算")
    
    return optimal_threads, cpu_count

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
    动态分辨率数据集类
    根据配置实时生成不同分辨率的输入输出数据
    """
    
    def __init__(self, 
                 data_path: str,
                 input_resolution: Tuple[int, int],
                 output_resolution: Tuple[int, int],
                 num_samples: int = 100,
                 crop_mode: str = 'center',
                 normalize_data: bool = True,
                 lazy_loading: bool = False):
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
        """
        self.data_path = Path(data_path)
        self.input_resolution = input_resolution
        self.output_resolution = output_resolution
        self.num_samples = num_samples
        self.crop_mode = crop_mode
        self.normalize_data = normalize_data
        self.lazy_loading = lazy_loading
        
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
    
    def _crop_data(self, data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """裁剪数据到目标尺寸"""
        _, orig_h, orig_w = data.shape
        target_h, target_w = target_size
        
        if target_h > orig_h or target_w > orig_w:
            raise ValueError(f"目标尺寸 {target_size} 大于原始尺寸 ({orig_h}, {orig_w})")
        
        if self.crop_mode == 'center':
            # 中心裁剪
            start_h = (orig_h - target_h) // 2
            start_w = (orig_w - target_w) // 2
        elif self.crop_mode == 'corner':
            # 左上角裁剪
            start_h = 0
            start_w = 0
        elif self.crop_mode == 'random':
            # 随机裁剪
            start_h = np.random.randint(0, orig_h - target_h + 1)
            start_w = np.random.randint(0, orig_w - target_w + 1)
        else:
            raise ValueError(f"不支持的裁剪模式: {self.crop_mode}")
        
        end_h = start_h + target_h
        end_w = start_w + target_w
        
        return data[:, start_h:end_h, start_w:end_w]
    
    def _resize_data(self, data: np.ndarray, target_size: Tuple[int, int], method: str = 'bilinear') -> np.ndarray:
        """调整数据尺寸
        
        Args:
            data: 输入数据
            target_size: 目标尺寸
            method: 插值方法 ('nearest', 'bilinear', 'bicubic')
        """
        from scipy.ndimage import zoom
        
        _, orig_h, orig_w = data.shape
        target_h, target_w = target_size
        
        zoom_h = target_h / orig_h
        zoom_w = target_w / orig_w
        
        # 根据插值方法选择order参数
        if method == 'nearest':
            order = 0
        elif method == 'bilinear':
            order = 1
        elif method == 'bicubic':
            order = 3
        else:
            logger.warning(f"未知的插值方法: {method}，使用双线性插值")
            order = 1
        
        resized_data = np.zeros((data.shape[0], target_h, target_w))
        for i in range(data.shape[0]):
            resized_data[i] = zoom(data[i], (zoom_h, zoom_w), order=order)
        
        return resized_data
    
    def _compute_normalization_params(self):
        """计算归一化参数 - 基于全局原始数据统计信息"""
        logger.info("计算全局归一化参数...")
        
        if self.lazy_loading:
            # 懒加载模式：通过采样计算归一化参数
            logger.info("懒加载模式：使用采样数据计算归一化参数")
            sample_size = min(100, self.data_shape[0])  # 采样100个样本或全部数据
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
        
        # 为了兼容性，保留原有的变量名
        self.input_min = global_min
        self.input_max = global_max
        self.output_min = global_min
        self.output_max = global_max
        
        # 记录归一化信息，用于后续反归一化
        if self.lazy_loading:
            original_data_shape = self.data_shape
        else:
            original_data_shape = self.original_data.shape
            
        self.normalization_info = {
            'global_min': float(global_min),
            'global_max': float(global_max),
            'normalization_method': 'global_minmax',
            'original_data_shape': original_data_shape,
            'input_resolution': self.input_resolution,
            'output_resolution': self.output_resolution
        }
        
        logger.info(f"全局数据范围: [{global_min:.6f}, {global_max:.6f}]")
        logger.info(f"归一化信息已记录，可用于反归一化")
    
    def _normalize_data(self, data: np.ndarray, data_min: float, data_max: float) -> np.ndarray:
        """归一化数据到[0, 1]范围"""
        eps = 1e-8  # 添加小的容差，提高数值稳定性
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
        # ===== CPU端数据加载和预处理优化 =====
        # 所有数据加载、裁剪、插值、归一化等重复工作都在CPU上完成
        # 只有最终的tensor创建才涉及内存分配
        
        # 获取原始样本 (CPU操作)
        if self.lazy_loading:
            # 懒加载模式：实时加载单个样本 (I/O在CPU)
            original_sample = self._load_sample_data(idx)
            original_sample = original_sample[np.newaxis, :]  # 添加batch维度
        else:
            # 传统模式：从内存中获取 (CPU内存访问)
            original_sample = self.original_data[idx:idx+1]  # 保持3D形状
        
        # 生成输入数据（裁剪到输入分辨率） - CPU计算
        original_shape = self.data_shape if self.lazy_loading else self.original_data.shape[1:]
        if self.input_resolution == original_shape:
            input_data = original_sample[0]
        else:
            input_cropped = self._crop_data(original_sample, self.input_resolution)
            input_data = input_cropped[0]
        
        # 生成输出数据 - CPU计算
        if self.output_resolution == original_shape:
            output_data = original_sample[0]
        elif self.output_resolution[0] <= original_shape[0] and self.output_resolution[1] <= original_shape[1]:
            # 输出分辨率小于等于原始分辨率，使用裁剪 (CPU)
            output_cropped = self._crop_data(original_sample, self.output_resolution)
            output_data = output_cropped[0]
        else:
            # 输出分辨率大于原始分辨率，使用插值 (CPU)
            output_resized = self._resize_data(original_sample, self.output_resolution)
            output_data = output_resized[0]
        
        # 展平为1D - CPU操作
        input_flat = input_data.flatten()
        output_flat = output_data.flatten()
        
        # 应用归一化 - CPU计算
        if self.normalize_data:
            input_flat = self._normalize_data(input_flat, self.input_min, self.input_max)
            output_flat = self._normalize_data(output_flat, self.output_min, self.output_max)
        
        # 最终tensor创建优化 - 使用torch.from_numpy().contiguous()提升性能
        # 1. torch.from_numpy()避免数据拷贝，直接共享内存
        # 2. .contiguous()确保内存布局连续，优化GPU传输
        # 3. pin_memory=True(在DataLoader中设置)确保数据在页锁定内存中，加速CPU->GPU传输
        
        # 确保数据类型为float32并且内存连续
        input_array = np.ascontiguousarray(input_flat, dtype=np.float32)
        output_array = np.ascontiguousarray(output_flat, dtype=np.float32)
        
        return (
            torch.from_numpy(input_array).contiguous(),   # 零拷贝tensor创建，内存连续
            torch.from_numpy(output_array).contiguous(),  # 零拷贝tensor创建，内存连续
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
            'output_range': (sample_output.min().item(), sample_output.max().item())
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
                'original_data_shape': self.original_data.shape,
                'input_resolution': self.input_resolution,
                'output_resolution': self.output_resolution
            }
    
    def denormalize_predictions(self, normalized_data):
        """反归一化预测结果，恢复物理信息
        
        Args:
            normalized_data: 归一化的数据 (torch.Tensor 或 numpy.ndarray)
            
        Returns:
            反归一化后的物理数据
        """
        if hasattr(self, 'global_min') and hasattr(self, 'global_max'):
            data_min = self.global_min
            data_max = self.global_max
        else:
            # 兼容旧版本
            data_min = self.output_min
            data_max = self.output_max
        
        # 处理torch.Tensor
        if hasattr(normalized_data, 'cpu'):
            normalized_data = normalized_data.cpu().numpy()
        
        return self._denormalize_data(normalized_data, data_min, data_max)
    
    def save_normalization_info(self, filepath):
        """保存归一化信息到文件
        
        Args:
            filepath: 保存路径 (支持 .json 或 .yaml)
        """
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
        """从文件加载归一化信息
        
        Args:
            filepath: 文件路径
            
        Returns:
            归一化信息字典
        """
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
            'test_ratio': 0.15
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
    
    # 如果有命令行参数，更新配置
    if args:
        if args.input_resolution:
            default_config['data']['input_resolution'] = args.input_resolution
        if args.output_resolution:
            default_config['data']['output_resolution'] = args.output_resolution
        if args.batch_size:
            default_config['data']['batch_size'] = args.batch_size
        if args.epochs:
            default_config['training']['epochs'] = args.epochs
        if args.learning_rate:
            default_config['training']['learning_rate'] = args.learning_rate
        if args.num_samples:
            default_config['data']['num_samples'] = args.num_samples
        if args.attention_type:
            default_config['model']['attention_type'] = args.attention_type
        if args.crop_mode:
            default_config['data']['crop_mode'] = args.crop_mode
    
    # 计算输入输出维度
    input_h, input_w = default_config['data']['input_resolution']
    output_h, output_w = default_config['data']['output_resolution']
    
    default_config['model']['input_dim'] = input_h * input_w
    default_config['model']['output_dim'] = output_h * output_w
    # 修复序列长度计算：直接使用输入维度，更符合Transformer的处理逻辑
    default_config['model']['seq_len'] = input_h * input_w
    
    return default_config

def load_config_from_yaml(config_path: str) -> Dict[str, Any]:
    """
    从YAML文件加载配置
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        dict: 配置字典
    """
    logger.info(f"从YAML文件加载配置: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 计算输入输出维度
    if 'data' in config:
        if 'input_resolution' in config['data'] and 'output_resolution' in config['data']:
            input_h, input_w = config['data']['input_resolution']
            output_h, output_w = config['data']['output_resolution']
            
            if 'model' not in config:
                config['model'] = {}
            
            config['model']['input_dim'] = input_h * input_w
            config['model']['output_dim'] = output_h * output_w
            config['model']['seq_len'] = int(np.sqrt(input_h * input_w))
    
    return config

def get_dynamic_loaders(config: Dict[str, Any]):
    """
    获取动态数据加载器
    
    Args:
        config: 配置字典
    
    Returns:
        tuple: (train_loader, valid_loader, test_loader, dataset)
    """
    data_config = config['data']
    
    # 检查是否启用降采样
    downsampling_config = data_config.get('downsampling', {})
    use_downsampling = downsampling_config.get('enabled', False) and HAS_DOWNSAMPLER
    
    if use_downsampling:
        logger.info("🔽 使用降分辨率数据集 (DownsampledResolutionDataset)")
        # 创建降分辨率数据集
        dataset = DownsampledResolutionDataset(
            data_path=data_config['path'],
            input_resolution=tuple(data_config['input_resolution']),
            output_resolution=tuple(data_config['output_resolution']),
            num_samples=data_config['num_samples'],
            downsample_method=downsampling_config.get('method', 'bilinear'),
            preserve_aspect_ratio=downsampling_config.get('preserve_aspect_ratio', True),
            normalize_data=data_config.get('normalize_data', True),
            lazy_loading=data_config.get('lazy_loading', False)
        )
        
        # 记录降采样配置信息
        logger.info(f"降采样方法: {downsampling_config.get('method', 'bilinear')}")
        logger.info(f"保持宽高比: {downsampling_config.get('preserve_aspect_ratio', True)}")
        logger.info(f"抗锯齿: {downsampling_config.get('anti_aliasing', True)}")
        logger.info(f"后端: {downsampling_config.get('backend', 'auto')}")
    else:
        if downsampling_config.get('enabled', False) and not HAS_DOWNSAMPLER:
            logger.warning("⚠️ 配置启用了降采样但模块未导入，回退到传统裁剪方法")
        
        logger.info("✂️ 使用传统裁剪数据集 (DynamicResolutionDataset)")
        # 创建传统动态数据集
        dataset = DynamicResolutionDataset(
            data_path=data_config['path'],
            input_resolution=tuple(data_config['input_resolution']),
            output_resolution=tuple(data_config['output_resolution']),
            num_samples=data_config['num_samples'],
            crop_mode=data_config.get('crop_mode', 'center'),
            normalize_data=data_config.get('normalize_data', True),
            lazy_loading=data_config.get('lazy_loading', False)
        )
        
        # 记录裁剪配置信息
        logger.info(f"裁剪模式: {data_config.get('crop_mode', 'center')}")
    
    # 数据集分割
    total_size = len(dataset)
    train_size = int(total_size * data_config['train_ratio'])
    valid_size = int(total_size * data_config['valid_ratio'])
    test_size = total_size - train_size - valid_size
    
    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )
    
    # 创建数据加载器
    batch_size = data_config['batch_size']
    
    # 从配置中获取数据加载器参数
    dataloader_config = config.get('dataloader', {})
    num_workers = dataloader_config.get('num_workers', 0)
    pin_memory = dataloader_config.get('pin_memory', True)
    drop_last = dataloader_config.get('drop_last', False)
    persistent_workers = dataloader_config.get('persistent_workers', False) and num_workers > 0
    prefetch_factor = dataloader_config.get('prefetch_factor', 2) if num_workers > 0 else None
    
    # 服务器优化：自动调整数据加载器参数
    import os
    cpu_count = os.cpu_count()
    
    # 如果配置的num_workers为0，自动设置为CPU核心数的一半（服务器优化）
    if num_workers == 0 and cpu_count > 4:
        num_workers = min(16, cpu_count // 2)  # 最多16个worker，避免过多进程
        persistent_workers = True
        # 重新设置prefetch_factor，因为num_workers已经改变
        prefetch_factor = dataloader_config.get('prefetch_factor', 2)
        logger.info(f"🚀 服务器优化: 自动设置num_workers={num_workers} (CPU核心数: {cpu_count})")
    
    # 服务器环境下优化prefetch_factor
    if num_workers > 8 and prefetch_factor is not None and prefetch_factor < 4:
        prefetch_factor = 4
        logger.info(f"🚀 服务器优化: 增加prefetch_factor={prefetch_factor}以提升数据流水线效率")
    
    logger.info(f"🔄 数据加载器配置: num_workers={num_workers}, pin_memory={pin_memory}, drop_last={drop_last}, prefetch_factor={prefetch_factor}")
    logger.info(f"🔄 持久化工作进程: {persistent_workers}, CPU核心数: {cpu_count}")
    
    # 构建 DataLoader 参数
    dataloader_kwargs = {
        'batch_size': batch_size,
        'shuffle': True,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': drop_last,
        'persistent_workers': persistent_workers
    }
    
    # 只有在多进程模式下才设置 prefetch_factor
    if num_workers > 0 and prefetch_factor is not None:
        dataloader_kwargs['prefetch_factor'] = prefetch_factor
    
    train_loader = torch.utils.data.DataLoader(train_dataset, **dataloader_kwargs)
    # 构建验证和测试 DataLoader 参数
    valid_test_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'drop_last': False,
        'persistent_workers': persistent_workers
    }
    
    # 只有在多进程模式下才设置 prefetch_factor
    if num_workers > 0 and prefetch_factor is not None:
        valid_test_kwargs['prefetch_factor'] = prefetch_factor
    
    valid_loader = torch.utils.data.DataLoader(valid_dataset, **valid_test_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **valid_test_kwargs)
    
    logger.info(f"数据集分割: 训练集={train_size}, 验证集={valid_size}, 测试集={test_size}")
    
    return train_loader, valid_loader, test_loader, dataset

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='动态分辨率训练器 - 支持灵活配置输入输出分辨率的PDE求解器训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""使用示例:
  # 基本用法
  python dynamic_resolution_trainer.py --input_resolution 32 32 --output_resolution 128 128
  
  # 使用配置文件
  python dynamic_resolution_trainer.py --config my_config.yaml
  
  # 多GPU训练 (DataParallel)
  python dynamic_resolution_trainer.py --input_resolution 32 32 --output_resolution 128 128 --use_dataparallel
  
  # 超分辨率重建
  python dynamic_resolution_trainer.py --input_resolution 16 16 --output_resolution 64 64 --attention_type cbam
  
  # 快速测试
  python dynamic_resolution_trainer.py --input_resolution 32 32 --output_resolution 64 64 --epochs 2 --num_samples 20
        """
    )
    
    # 配置文件
    parser.add_argument('--config', type=str, default=None,
                       help='YAML配置文件路径')
    
    # 数据参数
    data_group = parser.add_argument_group('数据配置')
    data_group.add_argument('--data_path', type=str,
                           help='数据文件路径 (覆盖配置文件中的设置)')
    data_group.add_argument('--input_resolution', type=int, nargs=2, metavar=('H', 'W'),
                           help='输入分辨率 [height width], 例如: --input_resolution 32 32')
    data_group.add_argument('--output_resolution', type=int, nargs=2, metavar=('H', 'W'),
                           help='输出分辨率 [height width], 例如: --output_resolution 128 128')
    data_group.add_argument('--num_samples', type=int,
                           help='样本数量 (默认: 100)')
    data_group.add_argument('--crop_mode', type=str, choices=['center', 'random', 'corner'],
                           help='裁剪模式: center(中心), random(随机), corner(左上角)')
    data_group.add_argument('--batch_size', type=int,
                           help='批次大小 (默认: 16)')
    
    # 训练参数
    train_group = parser.add_argument_group('训练配置')
    train_group.add_argument('--epochs', type=int,
                            help='训练轮数 (默认: 10)')
    train_group.add_argument('--learning_rate', type=float,
                            help='学习率 (默认: 0.001)')
    train_group.add_argument('--weight_decay', type=float,
                            help='权重衰减 (默认: 0.0001)')
    train_group.add_argument('--patience', type=int,
                            help='早停耐心值 (默认: 15)')
    
    # 模型参数
    model_group = parser.add_argument_group('模型配置')
    model_group.add_argument('--num_layers', type=int,
                            help='Transformer层数 (默认: 4)')
    model_group.add_argument('--d_model', type=int,
                            help='模型维度 (默认: 512)')
    model_group.add_argument('--num_heads', type=int,
                            help='注意力头数 (默认: 8)')
    model_group.add_argument('--attention_type', type=str,
                            choices=['sge', 'cbam', 'eca', 'se', 'relative', 'external'],
                            help='注意力机制类型 (默认: sge)')
    
    # 其他参数
    other_group = parser.add_argument_group('其他配置')
    other_group.add_argument('--device', type=str, choices=['auto', 'cuda', 'cpu'],
                            help='设备选择: auto(自动), cuda(GPU), cpu(CPU)')
    other_group.add_argument('--use_dataparallel', action='store_true',
                            help='启用多GPU训练 (DataParallel模式)')
    other_group.add_argument('--no_pretrained', action='store_true',
                            help='不加载预训练模型，从头开始训练')
    other_group.add_argument('--seed', type=int,
                            help='随机种子 (默认: 42)')
    other_group.add_argument('--verbose', action='store_true',
                            help='显示详细日志')
    other_group.add_argument('--no_visualization', action='store_true',
                            help='禁用可视化')
    
    return parser.parse_args()

def load_config_with_args(config_path, args):
    """
    加载配置文件并与命令行参数合并
    """
    # 首先定义默认配置
    default_config = {
        'data': {
            'path': 'X:\\2025\\Graduation_project\\Pdebench_input_Transformer\\VIVTransformer-1\\PDEBench\\pdebench\\data_download\\2D_DarcyFlow_beta0.1_Train.hdf5',
            'input_resolution': [32, 32],
            'output_resolution': [128, 128],
            'num_samples': 100,
            'crop_mode': 'center',
            'batch_size': 16,
            'train_ratio': 0.7,
            'valid_ratio': 0.15,
            'test_ratio': 0.15
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
        'device': 'auto',
        'use_dataparallel': False,
        'no_pretrained': False,
        'seed': 42,
        'visualization': {
            'enabled': True,
            'interval': 10,
            'max_samples': 5
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': './results/logs/dynamic_resolution_training.log'
        },
        'optimizer': {
            'type': 'Adam',
            'betas': [0.9, 0.999],
            'eps': 1e-8,
            'amsgrad': False
        },
        'scheduler': {
            'enabled': True,
            'type': 'cosine',
            'T_max': 100,
            'eta_min': 1e-6,
            'step_size': 30,
            'gamma': 0.1
        },
        'loss': {
            'base_weight': 0.8,
            'svd_weights': [0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
            'topk': 10,
            'loss_type': 'mse_svd',
            'svd_loss_enabled': True,
            'normalize_svd_weights': True
        },
        'dataloader': {
            'num_workers': 0,
            'pin_memory': True,
            'drop_last': False,
            'persistent_workers': False,
            'prefetch_factor': 2
        }
    }
    
    # 从默认配置开始
    config = default_config.copy()
    
    # 如果有配置文件，加载并合并配置文件
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f) or {}
            print(f"✅ 成功加载配置文件: {config_path}")
            
            # 深度合并配置文件到默认配置
            def deep_merge(default_dict, override_dict):
                """深度合并两个字典"""
                result = default_dict.copy()
                for key, value in override_dict.items():
                    if isinstance(value, dict) and key in result and isinstance(result[key], dict):
                        result[key] = deep_merge(result[key], value)
                    else:
                        result[key] = value
                return result
            
            config = deep_merge(config, yaml_config)
            print(f"🔄 配置文件已合并，epochs设置为: {config.get('training', {}).get('epochs', 'N/A')}")
            
        except Exception as e:
            print(f"⚠️ 加载配置文件失败: {e}，使用默认配置")
    else:
        if config_path:
            print(f"⚠️ 配置文件不存在: {config_path}，使用默认配置")
    
    # 命令行参数覆盖配置文件
    if args.data_path:
        config['data']['path'] = args.data_path
    if args.input_resolution:
        config['data']['input_resolution'] = args.input_resolution
    if args.output_resolution:
        config['data']['output_resolution'] = args.output_resolution
    if args.num_samples:
        config['data']['num_samples'] = args.num_samples
    if args.crop_mode:
        config['data']['crop_mode'] = args.crop_mode
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    if args.weight_decay:
        config['training']['weight_decay'] = args.weight_decay
    if args.patience:
        config['training']['patience'] = args.patience
    if args.num_layers:
        config['model']['num_layers'] = args.num_layers
    if args.d_model:
        config['model']['d_model'] = args.d_model
    if args.num_heads:
        config['model']['num_heads'] = args.num_heads
    if args.attention_type:
        config['model']['attention_type'] = args.attention_type
    if args.device:
        config['device'] = args.device
    if args.use_dataparallel:
        config['use_dataparallel'] = True
    if args.no_pretrained:
        config['no_pretrained'] = True
    if args.seed:
        config['seed'] = args.seed
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
    if args.no_visualization:
        config['visualization']['enabled'] = False
    
    # 重新计算输入输出维度
    input_h, input_w = config['data']['input_resolution']
    output_h, output_w = config['data']['output_resolution']
    config['model']['input_dim'] = input_h * input_w
    config['model']['output_dim'] = output_h * output_w
    # 修复序列长度计算：直接使用输入维度
    config['model']['seq_len'] = input_h * input_w
    
    return config

def setup_enhanced_logging(config):
    """
    设置增强的日志系统
    """
    # 创建日志目录
    log_config = config.get('logging', {})
    log_file = log_config.get('file', './logs/dynamic_resolution_training.log')
    log_dir = os.path.dirname(log_file)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    # 设置日志级别
    log_level = getattr(logging, log_config.get('level', 'INFO').upper())
    log_format = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # 清除现有的处理器
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # 配置日志
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, encoding='utf-8')
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"📝 日志系统已初始化，日志文件: {log_file}")
    return logger

def validate_config_detailed(config):
    """
    验证配置的有效性（详细版本，返回错误列表）
    """
    errors = []
    
    # 验证数据路径
    data_path = config['data']['path']
    if not os.path.exists(data_path):
        errors.append(f"数据文件不存在: {data_path}")
    
    # 验证分辨率
    input_res = config['data']['input_resolution']
    output_res = config['data']['output_resolution']
    
    if len(input_res) != 2 or any(x <= 0 for x in input_res):
        errors.append(f"输入分辨率无效: {input_res}")
    
    if len(output_res) != 2 or any(x <= 0 for x in output_res):
        errors.append(f"输出分辨率无效: {output_res}")
    
    # 验证分辨率兼容性
    input_dim = input_res[0] * input_res[1]
    output_dim = output_res[0] * output_res[1]
    if input_dim > 50000 or output_dim > 50000:
        logger.warning(f"分辨率较高，可能导致内存问题: 输入{input_dim}, 输出{output_dim}")
    
    # 验证训练参数
    if config['training']['epochs'] <= 0:
        errors.append(f"训练轮数必须大于0: {config['training']['epochs']}")
    
    if config['training']['learning_rate'] <= 0:
        errors.append(f"学习率必须大于0: {config['training']['learning_rate']}")
    
    if config['data']['batch_size'] <= 0:
        errors.append(f"批次大小必须大于0: {config['data']['batch_size']}")
    
    # 验证模型参数
    if config['model']['num_heads'] <= 0:
        errors.append(f"注意力头数必须大于0: {config['model']['num_heads']}")
    
    if config['model']['d_model'] % config['model']['num_heads'] != 0:
        errors.append(f"模型维度({config['model']['d_model']})必须能被注意力头数({config['model']['num_heads']})整除")
    
    # 验证SVD损失权重（仅在启用SVD损失时验证）
    if 'loss' in config:
        loss_config = config['loss']
        svd_loss_enabled = loss_config.get('svd_loss_enabled', True)
        
        if svd_loss_enabled and 'svd_weights' in loss_config:
            svd_weights = loss_config['svd_weights']
            base_weight = loss_config.get('base_weight', 1.0)
            
            if not isinstance(svd_weights, list) or len(svd_weights) == 0:
                errors.append("SVD权重必须是非空列表")
            
            if any(w < 0 for w in svd_weights):
                errors.append("SVD权重必须为非负数")
            
            if base_weight < 0:
                errors.append("基础权重必须为非负数")
            
            total_weight = base_weight + sum(svd_weights)
            if total_weight == 0:
                errors.append("所有权重之和不能为0")
            
            # 检查权重分布是否合理
            if base_weight / total_weight < 0.1:
                logger.warning(f"基础MSE权重占比过低: {base_weight/total_weight:.3f}")
        elif not svd_loss_enabled:
            logger.info("ℹ️ SVD损失已禁用，跳过SVD权重验证")
    
    # 验证数据格式（如果是HDF5文件）
    if data_path.endswith('.h5') or data_path.endswith('.hdf5'):
        try:
            import h5py
            with h5py.File(data_path, 'r') as f:
                # 检查必要的数据集是否存在
                if 'tensor' not in f.keys():
                    logger.warning(f"HDF5文件缺少'tensor'数据集")
        except ImportError:
            logger.warning("未安装h5py，无法验证HDF5文件格式")
        except Exception as e:
            logger.warning(f"验证HDF5文件时出错: {e}")
    
    return errors

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    try:
        # 如果没有指定配置文件，使用默认配置文件
        config_path = args.config if args.config else 'dynamic_config.yaml'
        
        # 加载和合并配置
        config = load_config_with_args(config_path, args)
        
        # 设置增强的日志系统
        logger = setup_enhanced_logging(config)
        
        # 验证配置
        config_errors = validate_config_detailed(config)
        if config_errors:
            logger.error("❌ 配置验证失败:")
            for error in config_errors:
                logger.error(f"  - {error}")
            return
        
        logger.info("✅ 配置验证通过")
        
        # 显示配置信息
        logger.info("=== 配置信息 ===")
        logger.info(f"输入分辨率: {config['data']['input_resolution']} -> {config['model']['input_dim']}维")
        logger.info(f"输出分辨率: {config['data']['output_resolution']} -> {config['model']['output_dim']}维")
        logger.info(f"样本数量: {config['data']['num_samples']}")
        logger.info(f"批次大小: {config['data']['batch_size']}")
        logger.info(f"训练轮数: {config['training']['epochs']}")
        logger.info(f"注意力机制: {config['model']['attention_type']}")
        
        # 设置随机种子
        if config.get('seed'):
            torch.manual_seed(config['seed'])
            np.random.seed(config['seed'])
            logger.info(f"🎲 设置随机种子: {config['seed']}")
        
        # 服务器优化：优化CPU设置以提升数据加载性能
        optimal_threads, cpu_count = optimize_cpu_for_data_loading()
        
        # 记录初始系统状态
        log_cpu_memory("初始状态")
        log_gpu_memory("初始状态")
        
        # 设置设备
        if config['device'] == 'auto':
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            device = torch.device(config['device'])
        logger.info(f"🖥️ 使用设备: {device}")
        
        # 设置GPU显存限制
        if device.type == 'cuda' and torch.cuda.is_available():
            max_memory_fraction = config.get('max_memory_fraction', 0.8)
            if 0 < max_memory_fraction <= 1.0:
                # 为所有可用的GPU设置显存限制
                gpu_count = torch.cuda.device_count()
                for i in range(gpu_count):
                    torch.cuda.set_per_process_memory_fraction(max_memory_fraction, device=i)
                logger.info(f"💾 设置GPU显存限制: {max_memory_fraction*100:.1f}% (应用于 {gpu_count} 张GPU)")
            else:
                logger.warning(f"⚠️ 无效的max_memory_fraction值: {max_memory_fraction}，应在(0,1]范围内")
        
        # 创建结果目录和模型保存目录
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
        
        # 显示数据统计
        stats = dataset.get_data_statistics()
        logger.info(f"数据统计: {stats}")
        
        # 显示归一化信息
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
        
        # 多GPU支持 (DataParallel)
        use_dataparallel = config.get('use_dataparallel', False)
        if use_dataparallel and torch.cuda.is_available() and torch.cuda.device_count() > 1:
            logger.info(f"🚀 启用多GPU训练: 检测到 {torch.cuda.device_count()} 张GPU")
            
            # 🔧 修复多GPU设备不一致问题
            # 确保模型完全在主设备(cuda:0)上
            if device.type == 'cuda':
                torch.cuda.set_device(0)  # 设置主设备
                model = model.cuda(0)     # 确保模型在主设备上
                logger.info("✅ 已将模型移动到主设备 cuda:0")
            
            # 清理所有GPU缓存，避免设备冲突
            torch.cuda.empty_cache()
            for i in range(torch.cuda.device_count()):
                with torch.cuda.device(i):
                    torch.cuda.empty_cache()
            logger.info("🧹 已清理所有GPU缓存")
            
            model = torch.nn.DataParallel(model)
            logger.info(f"   使用的GPU设备: {list(range(torch.cuda.device_count()))}")
            
            # 显示每个GPU的内存信息
            for i in range(torch.cuda.device_count()):
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"   GPU {i}: {torch.cuda.get_device_name(i)} ({gpu_memory:.1f}GB)")
        elif use_dataparallel:
            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA不可用，无法启用多GPU训练")
            elif torch.cuda.device_count() <= 1:
                logger.warning(f"⚠️ 只检测到 {torch.cuda.device_count()} 张GPU，无法启用多GPU训练")
        else:
            logger.info(f"🖥️ 单GPU训练模式")
        
        # 创建损失函数和优化器
        logger.info("=== 创建增强版损失函数 ===")
        
        # 获取混合精度配置（需要在使用前定义）
        mixed_precision_config = config.get('mixed_precision', {})
        
        # 从配置中获取损失函数参数
        loss_config = config.get('loss', {})
        base_weight = loss_config.get('base_weight', 0.8)
        svd_weights = loss_config.get('svd_weights', [0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01])
        topk = loss_config.get('topk', 10)
        svd_loss_enabled = loss_config.get('svd_loss_enabled', True)
        
        # 增强版SVD损失函数配置
        enhanced_config = loss_config.get('enhanced', {})
        use_enhanced_svd = enhanced_config.get('enabled', True)
        mixed_precision_mode = enhanced_config.get('mixed_precision_mode', mixed_precision_config.get('enabled', False))
        adaptive_weights = enhanced_config.get('adaptive_weights', True)
        fallback_level = enhanced_config.get('fallback_level', 2)
        enable_monitoring = enhanced_config.get('enable_monitoring', True)
        adaptation_interval = enhanced_config.get('adaptation_interval', 10)
        force_svd = enhanced_config.get('force_svd', False)
        
        logger.info(f"损失函数配置:")
        logger.info(f"  - 基础权重 (base_weight): {base_weight}")
        logger.info(f"  - SVD权重 (svd_weights): {svd_weights}")
        logger.info(f"  - SVD模态数 (topk): {topk}")
        logger.info(f"  - 损失类型: {loss_config.get('loss_type', 'enhanced_mse_svd')}")
        logger.info(f"  - SVD损失启用: {svd_loss_enabled}")
        logger.info(f"  - 增强版SVD: {use_enhanced_svd}")
        logger.info(f"  - 混合精度优化: {mixed_precision_mode}")
        logger.info(f"  - 自适应权重: {adaptive_weights}")
        logger.info(f"  - Fallback级别: {fallback_level}")
        logger.info(f"  - 性能监控: {enable_monitoring}")
        logger.info(f"  - 强制SVD: {force_svd}")
        
        # 根据配置决定使用哪种损失函数
        if svd_loss_enabled:
            if use_enhanced_svd:
                logger.info("✅ 启用增强版SVD损失函数 (EnhancedTotalLossWithSVD)")
                
                # 重置全局监控器
                if enable_monitoring:
                    reset_global_svd_stats()
                    logger.info("🔄 已重置SVD性能监控器")
                
                criterion = create_enhanced_svd_loss(
                    base_weight=base_weight,
                    svd_weights=svd_weights,
                    topk=topk,
                    mixed_precision=mixed_precision_mode,
                    adaptive_weights=adaptive_weights,
                    monitoring=enable_monitoring,
                    fallback_level=fallback_level,
                    force_svd=force_svd
                )
                
                # 显示增强版损失函数信息
                logger.info("=== 增强版SVD损失函数配置 ===")
                criterion.print_weight_info()
                
            else:
                logger.info("✅ 启用标准SVD损失函数 (TotalLossWithSVD)")
                criterion = TotalLossWithSVD(
                    base_weight=base_weight,
                    svd_weights=svd_weights,
                    topk=topk
                )
                
                # 显示实际的权重分布信息
                logger.info("=== 标准SVD损失函数权重分布 ===")
                weight_info = criterion.get_weight_info()
                logger.info(f"原始权重总和: {weight_info['weight_sum']:.4f}")
                logger.info(f"归一化后基础权重: {weight_info['normalized_base_weight']:.4f}")
                logger.info(f"归一化后SVD权重: {[f'{w:.4f}' for w in weight_info['normalized_svd_weights']]}")
                logger.info("================================")
        else:
            logger.info("⚠️ 禁用SVD损失，使用标准MSE损失函数")
            criterion = torch.nn.MSELoss()
            logger.info("=== 使用标准MSE损失函数 ===")
            logger.info("SVD计算已完全跳过，节省计算资源")
            logger.info("================================")
        
        # 从配置中获取优化器参数
        optimizer_config = config['optimizer']
        training_config = config['training']
        
        print(f"🔧 优化器配置: 类型={optimizer_config['type']}, 学习率={training_config['learning_rate']}, 权重衰减={training_config['weight_decay']}")
        
        # 根据配置创建优化器
        if optimizer_config['type'].lower() == 'adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=tuple(optimizer_config['betas']),
                eps=float(optimizer_config['eps']),
                amsgrad=bool(optimizer_config['amsgrad'])
            )
            print(f"   Adam参数: betas={optimizer_config['betas']}, eps={optimizer_config['eps']}, amsgrad={optimizer_config['amsgrad']}")
        elif optimizer_config['type'].lower() == 'sgd':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                momentum=optimizer_config.get('momentum', 0.9)
            )
            print(f"   SGD参数: momentum={optimizer_config.get('momentum', 0.9)}")
        elif optimizer_config['type'].lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay'],
                betas=tuple(optimizer_config['betas']),
                eps=float(optimizer_config['eps']),
                amsgrad=bool(optimizer_config['amsgrad'])
            )
            print(f"   AdamW参数: betas={optimizer_config['betas']}, eps={optimizer_config['eps']}, amsgrad={optimizer_config['amsgrad']}")
        else:
            print(f"⚠️  不支持的优化器类型: {optimizer_config['type']}，使用默认Adam")
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=training_config['learning_rate'],
                weight_decay=training_config['weight_decay']
            )
        
        # 创建学习率调度器
        scheduler_config = config['scheduler']
        scheduler = None
        
        if scheduler_config['enabled']:
            print(f"📈 学习率调度器配置: 类型={scheduler_config['type']}, 启用={scheduler_config['enabled']}")
            
            if scheduler_config['type'].lower() == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, 
                    T_max=scheduler_config['T_max']
                )
                print(f"   Cosine参数: T_max={scheduler_config['T_max']}")
            elif scheduler_config['type'].lower() == 'step':
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=scheduler_config['step_size'],
                    gamma=scheduler_config['gamma']
                )
                print(f"   Step参数: step_size={scheduler_config['step_size']}, gamma={scheduler_config['gamma']}")
            elif scheduler_config['type'].lower() == 'exponential':
                scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    optimizer,
                    gamma=scheduler_config['gamma']
                )
                print(f"   Exponential参数: gamma={scheduler_config['gamma']}")
            else:
                print(f"⚠️  不支持的调度器类型: {scheduler_config['type']}")
        else:
            print("📈 学习率调度器: 未启用")
        
        # 显示高级训练功能状态
        gradient_config = config.get('gradient', {})
        
        logger.info("=== 高级训练功能状态 ===")
        logger.info(f"梯度裁剪: {'启用' if gradient_config.get('clip_enabled', False) else '禁用'}")
        if gradient_config.get('clip_enabled', False):
            logger.info(f"  裁剪阈值: {gradient_config.get('clip_value', 1.0)}")
        
        logger.info(f"混合精度训练: {'启用' if mixed_precision_config.get('enabled', False) else '禁用'}")
        if mixed_precision_config.get('enabled', False):
            logger.info(f"  损失缩放: {mixed_precision_config.get('loss_scale', 'dynamic')}")
        
        accumulation_steps = gradient_config.get('accumulation_steps', 1)
        logger.info(f"梯度累积: {'启用' if accumulation_steps > 1 else '禁用'}")
        if accumulation_steps > 1:
            logger.info(f"  累积步数: {accumulation_steps}")
        
        # 内存监控
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
        
        # 显示增强版SVD损失函数的性能报告
        if svd_loss_enabled and use_enhanced_svd and enable_monitoring:
            logger.info("\n=== SVD损失函数性能报告 ===")
            try:
                # 显示性能统计
                criterion.print_performance_stats()
                
                # 显示权重适应历史
                if adaptive_weights:
                    weight_info = criterion.get_weight_info()
                    adaptation_history = weight_info.get('adaptation_history', [])
                    if adaptation_history:
                        logger.info("\n=== 权重自适应历史 ===")
                        logger.info(f"总共进行了 {len(adaptation_history)} 次权重调整")
                        for i, record in enumerate(adaptation_history[-5:]):  # 显示最后5次调整
                            logger.info(f"调整 {i+1}: Epoch {record['epoch']}, "
                                       f"基础权重={record['base_weight']:.4f}, "
                                       f"损失趋势={record['loss_trend']:.4f}")
                    else:
                        logger.info("📊 权重自适应: 训练期间未触发权重调整")
                
                # 保存性能统计到文件
                perf_stats = criterion.get_performance_stats()
                perf_stats_path = results_dir / 'svd_performance_stats.json'
                import json
                with open(perf_stats_path, 'w', encoding='utf-8') as f:
                    json.dump(perf_stats, f, indent=2, ensure_ascii=False)
                logger.info(f"📊 SVD性能统计已保存: {perf_stats_path}")
                
                # 保存权重信息
                weight_info = criterion.get_weight_info()
                weight_info_path = results_dir / 'weight_adaptation_info.json'
                with open(weight_info_path, 'w', encoding='utf-8') as f:
                    json.dump(weight_info, f, indent=2, ensure_ascii=False, default=str)
                logger.info(f"⚖️ 权重适应信息已保存: {weight_info_path}")
                
            except Exception as e:
                logger.warning(f"⚠️ 生成SVD性能报告时出错: {e}")
        
        # 可视化损失
        if config['visualization']['enabled']:
            loss_plot_path = results_dir / 'dynamic_resolution_losses.png'
            plot_losses(train_losses, valid_losses, test_losses, save_path=str(loss_plot_path))
        
        logger.info("🎉 === 训练完成 ===")
        
        # 内存监控和清理 - 服务器优化版本
        log_cpu_memory("训练完成后")
        log_gpu_memory("训练完成后")
        cleanup_memory()
        
        # 服务器资源使用总结
        logger.info("\n=== 服务器资源使用总结 ===")
        logger.info(f"💾 CPU内存: 充分利用多进程数据加载 (num_workers={config['dataloader']['num_workers']})")
        logger.info(f"🚀 GPU计算: 核心模型训练和推理")
        logger.info(f"📊 数据流水线: pin_memory={config['dataloader']['pin_memory']}, prefetch_factor={config['dataloader']['prefetch_factor']}")
        logger.info(f"⚡ 持久化工作进程: persistent_workers={config['dataloader']['persistent_workers']}")
        
        # 保存归一化信息（如果启用了归一化）
        if dataset.normalize_data:
            norm_info_path = results_dir / 'normalization_info.json'
            dataset.save_normalization_info(str(norm_info_path))
            logger.info(f"📊 归一化信息已保存: {norm_info_path}")
            logger.info("💡 使用提示:")
            logger.info("   - 使用 dataset.denormalize_predictions(predictions) 反归一化预测结果")
            logger.info("   - 使用 DynamicResolutionDataset.load_normalization_info(path) 加载归一化信息")
            logger.info("   - 参考 demo_global_normalization.py 了解完整用法")
        
        # 保存最终配置
        config_save_path = results_dir / 'final_config.yaml'
        with open(config_save_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        logger.info(f"💾 最终配置已保存: {config_save_path}")
        
    except KeyboardInterrupt:
        logger.warning("⚠️ 训练被用户中断")
    except FileNotFoundError as e:
        logger.error(f"❌ 文件未找到: {str(e)}")
        logger.error("请检查数据文件路径是否正确")
    except torch.cuda.OutOfMemoryError as e:
        logger.error(f"❌ GPU内存不足: {str(e)}")
        logger.error("建议减小batch_size或使用CPU训练")
        log_gpu_memory("GPU内存不足时")
        log_cpu_memory("GPU内存不足时")
        cleanup_memory()
    except Exception as e:
        logger.error(f"❌ 训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error("详细错误信息请查看上方的堆栈跟踪")

if __name__ == "__main__":
    main()