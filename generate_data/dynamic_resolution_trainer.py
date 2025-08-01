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

import os
import sys
import torch
import numpy as np
import h5py
import yaml
import argparse
import logging
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'modify_multi_attention'))
sys.path.append(str(Path(__file__).parent))

# 导入归一化功能
from pde_process.pdebench_data_processor import PDEBenchProcessor, PDEBenchConfig

from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel
from modify_multi_attention.training.trainer import train_model, test_model
from modify_multi_attention.utils.visualization import plot_losses
from modify_multi_attention.utils.svd10_loss import TotalLossWithSVD
from modify_multi_attention.utils.logging_utils import setup_logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
                 normalize_data: bool = True):
        """
        初始化动态分辨率数据集
        
        Args:
            data_path: 原始数据文件路径
            input_resolution: 输入分辨率 (height, width)
            output_resolution: 输出分辨率 (height, width)
            num_samples: 样本数量
            crop_mode: 裁剪模式 ('center', 'random', 'corner')
            normalize_data: 是否对数据进行归一化
        """
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
        
        # 计算维度
        self.input_dim = input_resolution[0] * input_resolution[1]
        self.output_dim = output_resolution[0] * output_resolution[1]
        
        # 加载原始数据
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
    
    def _resize_data(self, data: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """调整数据尺寸（简单的最近邻插值）"""
        from scipy.ndimage import zoom
        
        _, orig_h, orig_w = data.shape
        target_h, target_w = target_size
        
        zoom_h = target_h / orig_h
        zoom_w = target_w / orig_w
        
        resized_data = np.zeros((data.shape[0], target_h, target_w))
        for i in range(data.shape[0]):
            resized_data[i] = zoom(data[i], (zoom_h, zoom_w), order=1)
        
        return resized_data
    
    def _compute_normalization_params(self):
        """计算归一化参数 - 基于全局原始数据统计信息"""
        logger.info("计算全局归一化参数...")
        
        # 使用全局原始数据计算统计信息，确保输入输出使用相同的归一化标准
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
        self.normalization_info = {
            'global_min': float(global_min),
            'global_max': float(global_max),
            'normalization_method': 'global_minmax',
            'original_data_shape': self.original_data.shape,
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
        return len(self.original_data)
    
    def __getitem__(self, idx):
        # 获取原始样本
        original_sample = self.original_data[idx:idx+1]  # 保持3D形状
        
        # 生成输入数据（裁剪到输入分辨率）
        if self.input_resolution == self.original_data.shape[1:]:
            input_data = original_sample[0]
        else:
            input_cropped = self._crop_data(original_sample, self.input_resolution)
            input_data = input_cropped[0]
        
        # 生成输出数据
        if self.output_resolution == self.original_data.shape[1:]:
            output_data = original_sample[0]
        elif self.output_resolution[0] <= self.original_data.shape[1] and self.output_resolution[1] <= self.original_data.shape[2]:
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
    
    # 创建动态数据集
    dataset = DynamicResolutionDataset(
        data_path=data_config['path'],
        input_resolution=tuple(data_config['input_resolution']),
        output_resolution=tuple(data_config['output_resolution']),
        num_samples=data_config['num_samples'],
        crop_mode=data_config.get('crop_mode', 'center'),
        normalize_data=data_config.get('normalize_data', True)
    )
    
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
    
    logger.info(f"🔄 数据加载器配置: num_workers={num_workers}, pin_memory={pin_memory}, drop_last={drop_last}")
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=persistent_workers
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=persistent_workers
    )
    
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
    # 默认配置
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
        'use_dataparallel': False,  # 启用多GPU训练 (DataParallel)
        'no_pretrained': False,  # 不加载预训练模型，从头开始训练
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
    
    # 如果有配置文件，使用配置文件
    config = default_config.copy()
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    # 深度合并配置
                    for key, value in yaml_config.items():
                        if isinstance(value, dict) and key in config:
                            config[key].update(value)
                        else:
                            config[key] = value
            print(f"✅ 成功加载配置文件: {config_path}")
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

def validate_config(config):
    """
    验证配置的有效性
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
    if input_dim > 10000 or output_dim > 10000:
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
    
    # 验证SVD损失权重
    if 'loss' in config and 'svd_weights' in config['loss']:
        svd_weights = config['loss']['svd_weights']
        base_weight = config['loss'].get('base_weight', 1.0)
        
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
        # 加载和合并配置
        config = load_config_with_args(args.config, args)
        
        # 设置增强的日志系统
        logger = setup_enhanced_logging(config)
        
        # 验证配置
        config_errors = validate_config(config)
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
        logger.info("=== 创建损失函数 ===")
        
        # 从配置中获取损失函数参数
        loss_config = config.get('loss', {})
        base_weight = loss_config.get('base_weight', 0.8)
        svd_weights = loss_config.get('svd_weights', [0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01])
        topk = loss_config.get('topk', 10)
        
        logger.info(f"损失函数配置:")
        logger.info(f"  - 基础权重 (base_weight): {base_weight}")
        logger.info(f"  - SVD权重 (svd_weights): {svd_weights}")
        logger.info(f"  - SVD模态数 (topk): {topk}")
        logger.info(f"  - 损失类型: {loss_config.get('loss_type', 'mse_svd')}")
        logger.info(f"  - SVD损失启用: {loss_config.get('svd_loss_enabled', True)}")
        
        criterion = TotalLossWithSVD(
            base_weight=base_weight,
            svd_weights=svd_weights,
            topk=topk
        )
        
        # 显示实际的权重分布信息
        logger.info("=== 实际损失函数权重分布 ===")
        weight_info = criterion.get_weight_info()
        logger.info(f"原始权重总和: {weight_info['weight_sum']:.4f}")
        logger.info(f"归一化后基础权重: {weight_info['normalized_base_weight']:.4f}")
        logger.info(f"归一化后SVD权重: {[f'{w:.4f}' for w in weight_info['normalized_svd_weights']]}")
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
            early_stop_patience=config['training']['patience'],
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
        
        # 可视化损失
        if config['visualization']['enabled']:
            loss_plot_path = results_dir / 'dynamic_resolution_losses.png'
            plot_losses(train_losses, valid_losses, test_losses, save_path=str(loss_plot_path))
        
        logger.info("🎉 === 训练完成 ===")
        
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
    except Exception as e:
        logger.error(f"❌ 训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()
        logger.error("详细错误信息请查看上方的堆栈跟踪")

if __name__ == "__main__":
    main()