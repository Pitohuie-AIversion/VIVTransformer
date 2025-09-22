#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于真实数据的多模型测试脚本（支持裁剪输入和原始输出）

功能:
1. 使用真实Darcy Flow数据集
2. 输入: 32x32裁剪数据
3. 输出: 128x128原始数据
4. 测试多种模型架构的性能
5. 生成可视化对比图表

参考: generate_data/pde_process/create_input32_output128_dataset.py

作者: AI Assistant
日期: 2025
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import h5py
import argparse
import yaml
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import warnings
import json
from datetime import datetime
import matplotlib
warnings.filterwarnings('ignore')

# 添加models目录到路径
models_dir = Path(__file__).parent / 'models'
sys.path.insert(0, str(models_dir))

# 导入增强模型
try:
    # 尝试多种导入方式
    try:
        from enhanced_transformer import EnhancedTransformer1d, EnhancedTransformer2d
        from enhanced_fno import EnhancedFNO1d, EnhancedFNO2d
        from enhanced_mlp import EnhancedMLP1d, EnhancedMLP2d
        from enhanced_unet import EnhancedUNet1d, EnhancedUNet2d
    except ImportError:
        # 尝试从models目录导入
        from models.enhanced_transformer import EnhancedTransformer1d, EnhancedTransformer2d
        from models.enhanced_fno import EnhancedFNO1d, EnhancedFNO2d
        from models.enhanced_mlp import EnhancedMLP1d, EnhancedMLP2d
        from models.enhanced_unet import EnhancedUNet1d, EnhancedUNet2d
    ENHANCED_MODELS_AVAILABLE = True
    print("✅ 增强模型导入成功")
except ImportError as e:
    print(f"⚠️ 增强模型导入失败: {e}")
    print("📝 将使用内置的简单模型实现")
    ENHANCED_MODELS_AVAILABLE = False

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('Agg')  # 使用非交互式后端

# 设置项目根目录
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 尝试导入增强版数据加载器
try:
    from data.enhanced_crop_dataloader import create_enhanced_crop_dataloader
    ENHANCED_DATALOADER_AVAILABLE = True
    logger.info("✅ 使用增强版数据加载器（支持归一化）")
except ImportError:
    from modify_multi_attention.data.crop_dataloader import create_crop_dataloader
    ENHANCED_DATALOADER_AVAILABLE = False
    logger.warning("⚠️ 增强版数据加载器不可用，使用标准版本（不支持归一化）")

from modify_multi_attention.utils.config import load_config
from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel
from models.enhanced_fno import create_enhanced_fno2d
from models.enhanced_unet import create_enhanced_unet2d
from utils.visualization import create_model_comparison_plots, generate_model_comparison_summary, plot_training_losses, plot_comparison_figure

# ===== 自适应资源管理工具函数 =====

def detect_system_resources():
    """检测系统资源并返回详细信息"""
    resources = {}
    
    # CPU信息
    resources['cpu_count'] = os.cpu_count()
    resources['cpu_usage'] = psutil.cpu_percent(interval=1)
    
    # 内存信息
    memory = psutil.virtual_memory()
    resources['total_memory_gb'] = memory.total / (1024**3)
    resources['available_memory_gb'] = memory.available / (1024**3)
    resources['memory_usage_percent'] = memory.percent
    
    # GPU信息
    if torch.cuda.is_available():
        resources['gpu_count'] = torch.cuda.device_count()
        resources['gpu_devices'] = []
        for i in range(torch.cuda.device_count()):
            gpu_props = torch.cuda.get_device_properties(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            resources['gpu_devices'].append({
                'id': i,
                'name': gpu_props.name,
                'total_memory_gb': gpu_memory,
                'compute_capability': f"{gpu_props.major}.{gpu_props.minor}"
            })
    else:
        resources['gpu_count'] = 0
        resources['gpu_devices'] = []
    
    return resources

def get_gpu_memory_info(device_id=0):
    """获取GPU内存使用信息"""
    if not torch.cuda.is_available():
        return None
    
    try:
        torch.cuda.set_device(device_id)
        total_memory = torch.cuda.get_device_properties(device_id).total_memory
        allocated_memory = torch.cuda.memory_allocated(device_id)
        cached_memory = torch.cuda.memory_reserved(device_id)
        
        return {
            'total_gb': total_memory / (1024**3),
            'allocated_gb': allocated_memory / (1024**3),
            'cached_gb': cached_memory / (1024**3),
            'free_gb': (total_memory - allocated_memory) / (1024**3),
            'usage_percent': (allocated_memory / total_memory) * 100
        }
    except Exception as e:
        logger.warning(f"无法获取GPU内存信息: {e}")
        return None

def adaptive_batch_size(base_batch_size, gpu_memory_gb, model_complexity='medium'):
    """根据GPU内存自适应调整批次大小"""
    if gpu_memory_gb is None:
        return base_batch_size
    
    # 根据模型复杂度设置内存需求系数
    complexity_factors = {
        'simple': 0.5,    # MLP等简单模型
        'medium': 1.0,    # Transformer等中等模型
        'complex': 2.0    # FNO、UNet等复杂模型
    }
    
    factor = complexity_factors.get(model_complexity, 1.0)
    
    # 基于GPU内存调整批次大小
    if gpu_memory_gb >= 40:  # 高端GPU (A100, H100等)
        multiplier = 8
    elif gpu_memory_gb >= 24:  # 中高端GPU (RTX 4090, A6000等)
        multiplier = 4
    elif gpu_memory_gb >= 12:  # 中端GPU (RTX 4070Ti等)
        multiplier = 2
    elif gpu_memory_gb >= 8:   # 入门GPU (RTX 4060Ti等)
        multiplier = 1
    else:  # 低端GPU
        multiplier = 0.5
    
    # 应用复杂度因子
    adjusted_multiplier = multiplier / factor
    new_batch_size = max(1, int(base_batch_size * adjusted_multiplier))
    
    logger.info(f"🎯 自适应批次大小: {base_batch_size} → {new_batch_size} "
                f"(GPU: {gpu_memory_gb:.1f}GB, 复杂度: {model_complexity})")
    
    return new_batch_size

def adaptive_num_workers(base_workers, cpu_count, data_complexity='medium'):
    """根据CPU核心数自适应调整数据加载器工作进程数"""
    # 根据数据处理复杂度设置CPU使用策略
    complexity_factors = {
        'simple': 0.25,   # 简单数据处理
        'medium': 0.5,    # 中等复杂度
        'complex': 0.75   # 复杂数据处理（如降采样）
    }
    
    factor = complexity_factors.get(data_complexity, 0.5)
    
    # 基于CPU核心数计算最优工作进程数
    if cpu_count >= 64:  # 超级服务器
        optimal_workers = int(cpu_count * factor)
    elif cpu_count >= 16:  # 高性能服务器
        optimal_workers = int(cpu_count * factor)
    elif cpu_count >= 8:   # 普通服务器/工作站
        optimal_workers = int(cpu_count * 0.5)
    else:  # 个人电脑
        optimal_workers = max(0, cpu_count - 2)
    
    # 限制在合理范围内
    optimal_workers = max(0, min(32, optimal_workers))
    
    # 如果base_workers为0，使用计算出的值；否则取较大值
    if base_workers == 0:
        final_workers = optimal_workers
    else:
        final_workers = max(base_workers, optimal_workers)
    
    logger.info(f"🚀 自适应工作进程: {base_workers} → {final_workers} "
                f"(CPU: {cpu_count}核, 复杂度: {data_complexity})")
    
    return final_workers

def cleanup_memory():
    """清理内存和GPU缓存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def log_system_status(stage=""):
    """记录系统资源状态"""
    resources = detect_system_resources()
    
    logger.info(f"📊 [{stage}] 系统资源状态:")
    logger.info(f"  💻 CPU: {resources['cpu_count']}核, 使用率: {resources['cpu_usage']:.1f}%")
    logger.info(f"  💾 内存: {resources['available_memory_gb']:.1f}GB可用 / {resources['total_memory_gb']:.1f}GB总计 "
                f"({resources['memory_usage_percent']:.1f}%)")
    
    if resources['gpu_count'] > 0:
        logger.info(f"  🎮 GPU: {resources['gpu_count']}张")
        for gpu in resources['gpu_devices']:
            gpu_mem = get_gpu_memory_info(gpu['id'])
            if gpu_mem:
                logger.info(f"    GPU{gpu['id']}: {gpu['name']} "
                           f"({gpu_mem['free_gb']:.1f}GB可用 / {gpu_mem['total_gb']:.1f}GB总计)")
    else:
        logger.info("  🎮 GPU: 未检测到CUDA设备")
    
    return resources

class SimpleMLP(nn.Module):
    """简单的MLP模型，支持不同输入输出维度"""
    
    def __init__(self, input_dim, output_dim, hidden_dims=None, activation='relu', dropout=0.1, use_batch_norm=True, use_residual=False):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [
                input_dim * 2,
                input_dim * 4,
                output_dim // 2
            ]
        
        self.use_residual = use_residual
        self.layers = nn.ModuleList()
        
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layer_modules = []
            layer_modules.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layer_modules.append(nn.BatchNorm1d(hidden_dim))
            layer_modules.append(nn.ReLU() if activation == 'relu' else nn.GELU())
            layer_modules.append(nn.Dropout(dropout))
            
            self.layers.append(nn.Sequential(*layer_modules))
            prev_dim = hidden_dim
        
        # 输出层
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        logger.info(f"MLP模型架构: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}")
    
    def forward(self, x):
        for layer in self.layers:
            if self.use_residual and x.shape[-1] == layer[0].out_features:
                x = x + layer(x)
            else:
                x = layer(x)
        return self.output_layer(x)

class SimpleTransformer(nn.Module):
    """简化的Transformer模型，支持不同输入输出维度"""
    
    def __init__(self, input_dim, output_dim, d_model=256, num_heads=8, num_layers=3, dropout=0.1, 
                 seq_len=None, pe_type='sinusoidal_1d', use_memory_film=False, max_time_steps=100, 
                 output_head_type='global', time_encoding='embedding'):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.seq_len = seq_len or 1
        
        # 输入投影
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 输出投影
        self.output_projection = nn.Linear(d_model, output_dim)
        
        logger.info(f"Transformer模型: {input_dim} -> {d_model} -> {output_dim}")
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        batch_size = x.size(0)
        
        # 投影到模型维度并添加序列维度
        x = self.input_projection(x)  # (batch_size, d_model)
        x = x.unsqueeze(1)  # (batch_size, 1, d_model)
        
        # Transformer处理
        x = self.transformer(x)  # (batch_size, 1, d_model)
        
        # 移除序列维度并投影到输出维度
        x = x.squeeze(1)  # (batch_size, d_model)
        x = self.output_projection(x)  # (batch_size, output_dim)
        
        return x

class CustomTransformerWrapper(nn.Module):
    """自定义Transformer包装类，适配多注意力机制到简单回归任务"""
    
    def __init__(self, input_dim, output_dim, d_model=256, num_heads=8, num_layers=3, dropout=0.1,
                 attention_type="relative", seq_len=None, pe_type='learnable_1d', 
                 output_head_type='global', time_encoding='embedding', use_memory_film=True, 
                 use_memory_concat=False, max_time_steps=100):
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 推断序列长度
        if seq_len is None:
            # 尝试找到接近正方形的因子分解
            root = int(input_dim ** 0.5)
            if root * root == input_dim:
                seq_len = root * root
            else:
                # 找最接近的因子对
                factors = []
                for i in range(1, int(input_dim ** 0.5) + 1):
                    if input_dim % i == 0:
                        factors.append((i, input_dim // i))
                if factors:
                    h, w = min(factors, key=lambda x: abs(x[0] - x[1]))
                    seq_len = h * w
                else:
                    seq_len = input_dim
        
        self.seq_len = seq_len
        
        # 创建自定义Transformer模型
        self.transformer = TransformerFlowReconstructionModel(
            input_dim=input_dim,
            output_dim=output_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            d_model=d_model,
            max_time_steps=max_time_steps,
            attention_type=attention_type,
            seq_len=seq_len,
            pe_type=pe_type,
            output_head_type=output_head_type,
            time_encoding=time_encoding,
            use_memory_film=use_memory_film,
            use_memory_concat=use_memory_concat
        )
        
        # 固定时间步（用于回归任务）
        self.fixed_time_step = torch.tensor([0], dtype=torch.long)
        
        logger.info(f"CustomTransformer模型: {input_dim} -> {d_model} -> {output_dim}")
        logger.info(f"注意力机制: {attention_type}, 序列长度: {seq_len}")
    
    def forward(self, x):
        # x: (batch_size, input_dim)
        batch_size = x.size(0)
        device = x.device
        
        # 创建固定时间步张量
        time_steps = self.fixed_time_step.expand(batch_size).to(device)
        
        # 调用自定义Transformer
        output = self.transformer(x, time_steps)
        
        return output

class EnhancedFNOWrapper(nn.Module):
    """增强FNO包装类，适配测试框架"""
    
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        
        # 创建增强FNO模型
        self.model = create_enhanced_fno2d(
            num_channels=kwargs.get('num_channels', 1),
            modes1=kwargs.get('modes1', 12),
            modes2=kwargs.get('modes2', 12),
            width=kwargs.get('width', 32),
            initial_step=kwargs.get('initial_step', 10),
            input_resolution=kwargs.get('input_resolution', (32, 32)),
            output_resolution=kwargs.get('output_resolution', (128, 128)),
            use_upsampling=kwargs.get('use_upsampling', True)
        )
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 添加适配层
        self.input_adapter = nn.Linear(input_dim, 1024)  # 32*32
        self.output_adapter = nn.Linear(16384, output_dim)  # 128*128
    
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        batch_size = x.shape[0]
        
        # 通过输入适配层
        x_adapted = self.input_adapter(x)  # [batch_size, 1024]
        
        # 重塑为 [batch_size, channels, height, width]
        x_reshaped = x_adapted.view(batch_size, 1, 32, 32)
        
        # 通过FNO模型
        output = self.model(x_reshaped)  # [batch_size, 1, 128, 128]
        
        # 重塑并通过输出适配层
        output_flat = output.view(batch_size, -1)  # [batch_size, 16384]
        output_final = self.output_adapter(output_flat)  # [batch_size, output_dim]
        
        return output_final

class EnhancedUNetWrapper(nn.Module):
    """增强UNet包装类，适配测试框架"""
    
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__()
        
        # 创建增强UNet模型
        self.model = create_enhanced_unet2d(
            in_channels=kwargs.get('in_channels', 1),
            out_channels=kwargs.get('out_channels', 1),
            init_features=kwargs.get('init_features', 32),
            input_resolution=kwargs.get('input_resolution', (32, 32)),
            output_resolution=kwargs.get('output_resolution', (128, 128)),
            use_upsampling=kwargs.get('use_upsampling', True),
            bilinear=kwargs.get('bilinear', False)
        )
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 添加适配层
        self.input_adapter = nn.Linear(input_dim, 1024)  # 32*32
        self.output_adapter = nn.Linear(16384, output_dim)  # 128*128
    
    def forward(self, x):
        # x shape: [batch_size, input_dim]
        batch_size = x.shape[0]
        
        # 通过输入适配层
        x_adapted = self.input_adapter(x)  # [batch_size, 1024]
        
        # 重塑为 [batch_size, channels, height, width]
        x_reshaped = x_adapted.view(batch_size, 1, 32, 32)
        
        # 通过UNet模型
        output = self.model(x_reshaped)  # [batch_size, 1, 128, 128]
        
        # 重塑并通过输出适配层
        output_flat = output.view(batch_size, -1)  # [batch_size, 16384]
        output_final = self.output_adapter(output_flat)  # [batch_size, output_dim]
        
        return output_final

def get_default_config() -> Dict[str, Any]:
    """获取默认配置"""
    return {
        'experiment': {
            'name': 'default_experiment',
            'mode': 'single_model',
            'output_dir': 'default_results',
            'seed': 42,
            'deterministic': True,
            'log_level': 'INFO',
            'save_intermediate': True
        },
        'data': {
            'data_type': 'synthetic',
            'data_path': 'data/default_dataset.h5',
            'input_dim': 1024,
            'output_dim': 16384,
            'input_resolution': [32, 32],
            'output_resolution': [128, 128],
            'output_channels': 1,
            'crop_mode': 'center',
            'normalize': True,
            'remove_channel_dim': True,
            'batch_size': 32,
            'val_split': 0.2,
            'test_split': 0.1,
            'max_samples': 1000,
            'num_workers': 0,
            'pin_memory': False,
            'shuffle': True
        },
        'training': {
            'epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'warmup_epochs': 5,
            'early_stopping': {
                'patience': 10,
                'min_delta': 0.0001
            },
            'checkpoint': {
                'save_best': True,
                'save_interval': 10,
                'save_last': True
            },
            'device': 'cuda',
            'mixed_precision': False,
            'use_dataparallel': False
        },
        'loss': {
            'mse_weight': 1.0,
            'l1_weight': 0.0,
            'ssim_weight': 0.0,
            'perceptual_weight': 0.0,
            'gradient_weight': 0.0,
            'frequency_weight': 0.0,
            'svd_loss': {
                'enabled': False,
                'base_weight': 0.7,
                'svd_weights': [0.1, 0.1, 0.05, 0.03, 0.02],
                'topk': 5
            }
        },
        'models': {
            'active_model': 'transformer',
            'transformer': {
                'model_type': 'transformer',
                'input_dim': 1024,
                'output_dim': 16384,
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 3,
                'dropout': 0.1,
                'seq_len': 11,
                'pe_type': 'sinusoidal_1d',
                'use_memory_film': False,
                'max_time_steps': 100,
                'output_head_type': 'global',
                'time_encoding': 'embedding'
            }
        },
        'attention': {
            'type': 'self',
            'comparison_types': ['self', 'se', 'cbam', 'eca']
        },
        'evaluation': {
            'metrics': ['mse', 'mae', 'r2', 'psnr', 'ssim'],
            'visualization': {
                'enabled': True,
                'interval': 10,
                'max_samples': 5,
                'save_plots': True,
                'plot_types': ['prediction', 'error', 'loss_curve']
            }
        },
        'comparison': {
            'parameter_fair': {
                'enabled': False,
                'target_params': 5000000,
                'tolerance': 2.0
            },
            'multi_model': {
                'enabled': False,
                'models': ['transformer', 'mlp', 'unet', 'fno'],
                'generate_report': True,
                'generate_plots': True
            },
            'loss_comparison': {
                'enabled': False,
                'loss_configs': [
                    {
                        'name': 'mse_only',
                        'mse_weight': 1.0,
                        'svd_weight': 0.0
                    }
                ]
            }
        },
        'system': {
            'adaptive_resources': True,
            'memory_management': {
                'auto_cleanup': True,
                'cleanup_interval': 100,
                'max_memory_usage': 0.8
            },
            'adaptive_batch_size': {
                'enabled': True,
                'min_batch_size': 4,
                'max_batch_size': 64,
                'memory_threshold': 0.9
            }
        },
        'output': {
            'save_model': True,
            'save_predictions': True,
            'save_metrics': True,
            'generate_report': True,
            'report_format': 'markdown',
            'generate_plots': True,
            'plot_format': 'png',
            'export_results': True,
            'export_format': 'json'
        },
        'debug': {
            'enabled': False,
            'verbose': False,
            'profile_memory': False,
            'profile_time': False,
            'save_debug_info': False
        }
    }

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """验证并修正配置"""
    try:
        # 获取默认配置作为基准
        default_config = get_default_config()
        
        # 确保必要的顶级键存在
        required_keys = ['experiment', 'data', 'training', 'loss', 'models']
        for key in required_keys:
            if key not in config:
                logger.warning(f"配置中缺少必要的键 '{key}'，使用默认值")
                config[key] = default_config[key]
        
        # 验证实验配置
        if 'mode' not in config['experiment']:
            config['experiment']['mode'] = 'single_model'
        
        # 验证数据配置
        data_config = config['data']
        if 'input_dim' not in data_config or 'output_dim' not in data_config:
            logger.warning("数据配置中缺少维度信息，使用默认值")
            data_config.update({
                'input_dim': 1024,
                'output_dim': 16384
            })
        
        # 验证训练配置
        training_config = config['training']
        if 'epochs' not in training_config:
            training_config['epochs'] = 50
        if 'learning_rate' not in training_config:
            training_config['learning_rate'] = 0.001
        if 'batch_size' not in config['data']:
            config['data']['batch_size'] = 32
        
        # 验证模型配置
        if 'active_model' not in config['models']:
            config['models']['active_model'] = 'transformer'
        
        # 确保活跃模型的配置存在
        active_model = config['models']['active_model']
        if active_model not in config['models']:
            logger.warning(f"活跃模型 '{active_model}' 的配置不存在，使用默认配置")
            config['models'][active_model] = default_config['models']['transformer']
        
        # 验证损失配置
        if 'mse_weight' not in config['loss']:
            config['loss']['mse_weight'] = 1.0
        
        logger.info("✅ 配置验证完成")
        return config
        
    except Exception as e:
        logger.error(f"配置验证失败: {e}")
        logger.info("使用默认配置")
        return default_config

def setup_device(config=None):
    """设置设备"""
    if config is None:
        config = {}
    
    # 获取设备配置，可能是字符串或字典
    device_config = config.get('training', {}).get('device', 'auto')
    
    # 如果device_config是字符串，直接处理
    if isinstance(device_config, str):
        if device_config.lower() == 'cpu':
            device = torch.device('cpu')
            logger.info("✅ 使用CPU设备")
        elif device_config.lower() in ['cuda', 'gpu']:
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                logger.info(f"✅ 使用GPU设备: {device}")
            else:
                device = torch.device('cpu')
                logger.info("⚠️ CUDA不可用，回退到CPU设备")
        elif device_config.lower() == 'auto':
            if torch.cuda.is_available():
                device = torch.device('cuda:0')
                logger.info(f"✅ 自动选择GPU设备: {device}")
            else:
                device = torch.device('cpu')
                logger.info("✅ 自动选择CPU设备")
        else:
            # 尝试直接解析设备字符串（如 "cuda:1"）
            try:
                device = torch.device(device_config)
                logger.info(f"✅ 使用指定设备: {device}")
            except:
                device = torch.device('cpu')
                logger.warning(f"⚠️ 无法解析设备配置 '{device_config}'，回退到CPU")
    else:
        # 如果是字典格式，按原来的逻辑处理
        use_cuda = device_config.get('use_cuda', True)
        device_id = device_config.get('device_id', 0)
        
        if use_cuda and torch.cuda.is_available():
            device = torch.device(f'cuda:{device_id}')
            logger.info(f"✅ 使用GPU设备: {device}")
        else:
            device = torch.device('cpu')
            logger.info("✅ 使用CPU设备")
    
    return device

def load_unified_data(config: Dict[str, Any]):
    """加载统一数据"""
    data_config = config.get('data', {})
    data_type = data_config.get('data_type', 'real')
    
    # 如果是合成数据，使用合成数据生成器
    if data_type == 'synthetic':
        from synthetic_data_generator import create_synthetic_dataloader
        return create_synthetic_dataloader(config)
    
    # 使用增强版数据加载器
    if ENHANCED_DATALOADER_AVAILABLE:
        # 增强版数据加载器接受完整的config，并返回4个值（包括normalizer）
        result = create_enhanced_crop_dataloader(config)
        if len(result) == 4:
            train_loader, val_loader, test_loader, normalizer = result
        else:
            train_loader, val_loader, test_loader = result
            normalizer = None
    else:
        # 标准版数据加载器接受config，返回3个值
        train_loader, val_loader, test_loader = create_crop_dataloader(config)
        normalizer = None
    
    return train_loader, val_loader, test_loader, normalizer

def train_unified_model(model, train_loader, val_loader, device, training_config, loss_config):
    """统一模型训练"""
    epochs = training_config.get('epochs', 50)
    learning_rate = training_config.get('learning_rate', 0.001)
    
    # 设置优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # 设置损失函数
    criterion = nn.MSELoss()
    
    train_losses = []
    val_losses = []
    
    logger.info(f"开始训练，共 {epochs} 个epoch")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        epoch_train_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            epoch_train_loss += loss.item()
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        epoch_val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    
    logger.info("训练完成")
    return train_losses, val_losses

def load_unified_config(config_path: str) -> Dict[str, Any]:
    """加载统一配置文件"""
    try:
        # 检查文件是否存在
        if not os.path.exists(config_path):
            logger.error(f"配置文件不存在: {config_path}")
            return get_default_config()
        
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        if not config:
            logger.warning("配置文件为空，使用默认配置")
            return get_default_config()
        
        # 应用预设配置
        if 'presets' in config and 'experiment' in config:
            preset_name = config['experiment'].get('preset')
            if preset_name and preset_name in config['presets']:
                preset_config = config['presets'][preset_name]
                # 深度合并预设配置
                config = merge_configs(config, preset_config)
                logger.info(f"应用预设配置: {preset_name}")
        
        # 验证配置
        config = validate_config(config)
        
        return config
    except yaml.YAMLError as e:
        logger.error(f"YAML解析错误: {e}")
        return get_default_config()
    except Exception as e:
        logger.error(f"加载配置文件失败: {e}")
        return get_default_config()

def merge_configs(base_config: Dict, preset_config: Dict) -> Dict:
    """深度合并配置"""
    result = base_config.copy()
    for key, value in preset_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result

def create_enhanced_model(model_config: Dict[str, Any], input_dim: int, output_dim: int) -> nn.Module:
    """创建增强模型"""
    model_type = model_config.get('model_type', 'mlp').lower()
    
    try:
        if ENHANCED_MODELS_AVAILABLE:
            if model_type == 'transformer':
                return EnhancedTransformer1d(
                    input_channels=1,
                    output_channels=1,
                    d_model=model_config.get('d_model', 128),
                    num_heads=model_config.get('num_heads', 4),
                    num_layers=model_config.get('num_layers', 3),
                    input_resolution=int(np.sqrt(input_dim)),
                    output_resolution=int(np.sqrt(output_dim)),
                    attention_type=model_config.get('attention_type', 'simplified_self_attention'),
                    pe_type=model_config.get('pe_type', 'learnable_1d'),
                    dropout=model_config.get('dropout', 0.1)
                )
            elif model_type == 'mlp':
                # 使用兼容的MLP模型
                from fix_mlp_unet_compatibility import CompatibleEnhancedMLP
                return CompatibleEnhancedMLP(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_dim=model_config.get('hidden_dims', [256])[0] if model_config.get('hidden_dims') else 256,
                    num_layers=len(model_config.get('hidden_dims', [256]))
                )
            elif model_type == 'unet':
                # 使用兼容的UNet模型
                from fix_mlp_unet_compatibility import CompatibleEnhancedUNet
                return CompatibleEnhancedUNet(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    input_spatial_dim=int(np.sqrt(input_dim)),
                    output_spatial_dim=int(np.sqrt(output_dim)),
                    base_channels=model_config.get('base_ch', 32)
                )
            elif model_type == 'fno':
                return EnhancedFNO1d(
                    num_channels=1,
                    modes=model_config.get('modes', 12),
                    width=model_config.get('width', 64),
                    input_resolution=int(np.sqrt(input_dim)),
                    output_resolution=int(np.sqrt(output_dim))
                )
    except Exception as e:
        logging.warning(f"创建增强模型失败: {e}")
    
    # 后备简单模型
    return create_simple_model(model_type, input_dim, output_dim, model_config)

def create_simple_model(model_type: str, input_dim: int, output_dim: int, config: Dict) -> nn.Module:
    """创建简单模型作为后备"""
    if model_type == 'mlp':
        return SimpleMLP(
            input_dim, 
            output_dim, 
            hidden_dims=config.get('hidden_dims', [256, 128]),
            dropout=config.get('dropout', 0.1)
        )
    elif model_type == 'transformer':
        return SimpleTransformer(
            input_dim, 
            output_dim,
            d_model=config.get('d_model', 128),
            num_heads=config.get('num_heads', 4),
            num_layers=config.get('num_layers', 2),
            dropout=config.get('dropout', 0.1)
        )
    else:
        # 默认线性模型
        return nn.Linear(input_dim, output_dim)

def create_model(model_type, input_dim, output_dim, **kwargs):
    """创建模型（保持向后兼容）"""
    if model_type == 'mlp':
        return SimpleMLP(input_dim, output_dim, **kwargs)
    elif model_type == 'transformer':
        return SimpleTransformer(input_dim, output_dim, **kwargs)
    elif model_type == 'custom_transformer':
        return CustomTransformerWrapper(input_dim, output_dim, **kwargs)
    elif model_type == 'fno':
        return EnhancedFNOWrapper(input_dim, output_dim, **kwargs)
    elif model_type == 'unet':
        return EnhancedUNetWrapper(input_dim, output_dim, **kwargs)
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")

def train_model(model, train_loader, val_loader, config, device, normalizer=None):
    """训练模型"""
    model = model.to(device)
    criterion = nn.MSELoss()
    
    # 确保优化器参数类型正确
    learning_rate = float(config['training']['learning_rate'])
    weight_decay = float(config['training']['weight_decay'])
    
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    epochs = config['training']['epochs']
    best_val_loss = float('inf')
    patience = config['training']['patience']
    min_delta = config['training']['min_delta']
    
    # 调试信息
    logger.info(f"训练配置 - epochs: {epochs} (type: {type(epochs)})")
    logger.info(f"训练配置 - patience: {patience} (type: {type(patience)})")
    logger.info(f"训练配置 - min_delta: {min_delta} (type: {type(min_delta)})")
    
    # 确保类型正确
    epochs = int(epochs)
    patience = int(patience)
    min_delta = float(min_delta)
    patience_counter = 0
    
    logger.info(f"转换后 - epochs: {epochs} (type: {type(epochs)})")
    logger.info(f"转换后 - patience: {patience} (type: {type(patience)})")
    logger.info(f"转换后 - min_delta: {min_delta} (type: {type(min_delta)})")
    
    train_losses = []
    val_losses = []
    
    logger.info(f"开始训练，共 {epochs} 轮")
    if normalizer:
        logger.info("✅ 使用归一化训练（数据已在DataLoader中归一化）")
    else:
        logger.info("❌ 未使用归一化训练")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 数据已在DataLoader中归一化，无需重复处理
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = train_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        num_val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 数据已在DataLoader中归一化，无需重复处理
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                num_val_batches += inputs.size(0)
        
        avg_val_loss = val_loss / num_val_batches
        val_losses.append(avg_val_loss)
        
        # 早停检查
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
        
        if epoch % 5 == 0 or epoch == epochs - 1:
            logger.info(f"轮次 {epoch+1}/{epochs}: 训练损失={avg_train_loss:.6f}, 验证损失={avg_val_loss:.6f}")
        
        if patience_counter >= patience:
            logger.info(f"早停于轮次 {epoch+1}")
            break
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'final_epoch': epoch + 1
    }

def evaluate_model(model, test_loader, device, normalizer=None):
    """评估模型"""
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    all_inputs = []  # 添加输入数据收集
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 数据已在DataLoader中归一化，无需重复处理
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            all_inputs.append(inputs.cpu().numpy())  # 收集输入数据
    
    avg_loss = total_loss / total_samples
    
    # 计算R²分数
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    inputs = np.concatenate(all_inputs, axis=0)  # 合并输入数据
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    return {
        'mse': avg_loss,
        'r2_score': r2_score,
        'predictions': predictions,
        'targets': targets,
        'inputs': inputs  # 返回输入数据
    }

def run_unified_training(config_path: str, models: List[str] = None):
    """运行统一训练流程"""
    try:
        # 加载统一配置
        config = load_unified_config(config_path)
        if not config:
            logger.warning("配置加载失败，使用默认配置")
            config = get_default_config()
        
        # 获取实验配置
        experiment_config = config.get('experiment', {})
        mode = experiment_config.get('mode', 'single_model')
        output_dir = experiment_config.get('output_dir', 'unified_training_results')
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"开始统一训练 - 模式: {mode}")
        logger.info(f"输出目录: {output_dir}")
        
        # 根据模式执行不同的训练流程
        if mode == 'single_model':
            return run_single_model_training(config, output_dir)
        elif mode == 'multi_model_comparison':
            return run_multi_model_comparison(config, output_dir, models)
        elif mode == 'parameter_fair_comparison':
            return run_parameter_fair_comparison(config, output_dir)
        elif mode == 'loss_comparison':
            return run_loss_comparison(config, output_dir)
        else:
            logger.error(f"未知的训练模式: {mode}")
            return {}
            
    except Exception as e:
        import traceback
        logger.error(f"统一训练失败: {e}")
        logger.error(f"错误详情: {traceback.format_exc()}")
        return {}

def run_single_model_training(config: Dict, output_dir: str) -> Dict:
    """单模型训练"""
    logger.info("执行单模型训练...")
    
    # 设置设备
    device = setup_device(config)
    
    # 加载数据
    train_loader, val_loader, test_loader, normalizer = load_unified_data(config)
    
    # 获取数据维度
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    output_dim = sample_batch[1].shape[-1]
    
    # 获取激活模型配置
    models_config = config.get('models', {})
    active_model = models_config.get('active_model', 'transformer')
    model_config = models_config.get(active_model, {})
    
    # 创建模型
    model = create_enhanced_model(model_config, input_dim, output_dim)
    model = model.to(device)
    
    # 计算参数量
    param_count = sum(p.numel() for p in model.parameters())
    logger.info(f"模型: {active_model}, 参数量: {param_count:,}")
    
    # 训练模型
    start_time = time.time()
    train_result = train_model(model, train_loader, val_loader, config, device, normalizer)
    train_time = time.time() - start_time
    
    # 提取训练结果
    train_losses = train_result['train_losses']
    val_losses = train_result['val_losses']
    
    # 评估模型
    test_results = evaluate_model(model, test_loader, device)
    test_loss = test_results['mse']
    test_metrics = {
        'r2_score': test_results['r2_score'],
        'mse': test_results['mse']
    }
    
    # 保存结果
    results = {
        active_model: {
            'param_count': param_count,
            'num_parameters': param_count,  # 添加可视化函数期望的字段名
            'train_time': train_time,       # 添加训练时间字段
            'train_losses': train_losses,
            'val_losses': val_losses,
            'test_loss': test_loss,
            'test_metrics': test_metrics,
            'test_r2': test_results['r2_score'],  # 添加可视化函数期望的字段名
            'test_mse': test_results['mse'],      # 添加可视化函数期望的字段名
            'config': model_config,
            # 添加测试数据以支持三联图生成
            'test_inputs': test_results['inputs'],
            'test_predictions': test_results['predictions'],
            'test_targets': test_results['targets']
        }
    }
    
    # 生成报告和可视化
    save_training_results(results, config, output_dir)
    
    return results

def save_training_results(results: Dict, config: Dict, output_dir: str):
    """保存训练结果"""
    try:
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 保存结果为JSON
        results_file = output_path / "training_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            # 转换numpy数组为列表以便JSON序列化
            def convert_to_serializable(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {k: convert_to_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_to_serializable(item) for item in obj]
                elif isinstance(obj, (np.int64, np.int32, np.float64, np.float32)):
                    return obj.item()
                else:
                    return obj
            
            serializable_results = convert_to_serializable(results)
            
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"训练结果已保存到: {results_file}")
        
        # 生成可视化图表
        try:
            logger.info("🎨 开始生成可视化图表...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_files = create_visualization_plots(results, timestamp)
            
            if plot_files:
                logger.info(f"✅ 成功生成 {len(plot_files)} 个可视化图表:")
                for plot_file in plot_files:
                    logger.info(f"  📈 {plot_file}")
                
                # 生成可视化汇总报告
                logger.info("📝 生成可视化汇总报告...")
                summary_path = generate_visualization_summary(results, plot_files, timestamp)
                logger.info(f"📋 可视化汇总报告已保存: {summary_path}")
            else:
                logger.warning("⚠️  没有生成可视化图表（可能没有成功的模型结果）")
                
        except Exception as e:
            logger.error(f"❌ 生成可视化图表时出错: {e}")
            logger.info("📊 JSON结果文件仍然可用")
        
    except Exception as e:
        logger.warning(f"保存训练结果失败: {e}")

def run_multi_model_comparison(config: Dict, output_dir: str, models: List[str] = None) -> Dict:
    """多模型对比训练"""
    logger.info("执行多模型对比训练...")
    
    # 设置设备
    device = setup_device(config)
    
    # 加载数据
    train_loader, val_loader, test_loader, normalizer = load_unified_data(config)
    
    # 获取数据维度
    sample_batch = next(iter(train_loader))
    input_dim = sample_batch[0].shape[-1]
    output_dim = sample_batch[1].shape[-1]
    
    # 确定要对比的模型
    models_config = config.get('models', {})
    comparison_config = config.get('comparison', {}).get('multi_model', {})
    
    if models is None:
        models = comparison_config.get('models', list(models_config.keys()))
        if 'active_model' in models:
            models.remove('active_model')
    
    results = {}
    
    # 训练每个模型
    for model_name in models:
        # 尝试大小写匹配
        model_key = model_name.lower()
        if model_key not in models_config:
            logger.warning(f"跳过未配置的模型: {model_name}")
            continue
            
        logger.info(f"\n{'='*50}")
        logger.info(f"训练模型: {model_name}")
        logger.info(f"{'='*50}")
        
        try:
            model_config = models_config[model_key]
            model = create_enhanced_model(model_config, input_dim, output_dim)
            model = model.to(device)
            
            # 计算参数量
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"参数量: {param_count:,}")
            
            # 训练模型
            training_config = config.get('training', {})
            train_losses, val_losses = train_unified_model(
                model, train_loader, val_loader, device, training_config, config.get('loss', {})
            )
            
            # 评估模型
            eval_results = evaluate_model(model, test_loader, device)
            test_loss = eval_results['mse']
            test_metrics = eval_results
            
            # 保存结果（包含测试数据以支持三联图生成）
            results[model_name] = {
                'param_count': param_count,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'test_loss': test_loss,
                'test_metrics': test_metrics,
                'config': model_config,
                # 添加测试数据以支持三联图生成
                'test_inputs': eval_results['inputs'],
                'test_predictions': eval_results['predictions'],
                'test_targets': eval_results['targets']
            }
            
            logger.info(f"测试损失: {test_loss:.6f}")
            
            # 内存清理
            cleanup_memory()
            
        except Exception as e:
            logger.error(f"模型 {model_name} 训练失败: {e}")
            results[model_name] = {'error': str(e)}
    
    # 生成对比报告
    save_training_results(results, config, output_dir)
    
    return results

def run_model_test(config_path, models=None):
    """运行模型测试（保持向后兼容）"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"加载配置文件: {config_path}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_gpu'] else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 读取自适应资源配置
    adaptive_config = config.get('adaptive_resources', {})
    enable_adaptive = adaptive_config.get('enable_adaptive', True)
    
    if enable_adaptive:
        # 智能资源检测和优化
        system_info = detect_system_resources()
        log_system_status("初始化")
        
        # 自适应批次大小调整
        batch_config = adaptive_config.get('batch_size_adaptation', {})
        if batch_config.get('enable', True):
            original_batch_size = config['data'].get('batch_size', 32)
            strategy = batch_config.get('strategy', 'medium')
            
            if device.type == 'cuda' and torch.cuda.is_available():
                gpu_info = get_gpu_memory_info()
                if gpu_info:
                    optimized_batch_size = adaptive_batch_size(
                        original_batch_size, 
                        gpu_info['total_gb'], 
                        strategy
                    )
                    # 应用批次大小限制
                    min_batch = batch_config.get('min_batch_size', 1)
                    max_batch = batch_config.get('max_batch_size', 64)
                    optimized_batch_size = max(min_batch, min(optimized_batch_size, max_batch))
                    config['data']['batch_size'] = optimized_batch_size
                    logger.info(f"批次大小优化: {original_batch_size} → {optimized_batch_size} (策略: {strategy})")
                    logger.info(f"GPU内存: 总计 {gpu_info['total_gb']:.1f}GB, 已用 {gpu_info['allocated_gb']:.1f}GB, 可用 {gpu_info['free_gb']:.1f}GB")
        
        # 自适应数据加载器工作进程数
        dataloader_config = adaptive_config.get('dataloader_adaptation', {})
        if dataloader_config.get('enable', True):
            original_num_workers = config['data'].get('num_workers', 4)
            strategy = dataloader_config.get('strategy', 'medium')
            optimized_num_workers = adaptive_num_workers(
                original_num_workers, 
                system_info['cpu_count'], 
                strategy
            )
            # 应用工作进程数限制
            min_workers = dataloader_config.get('min_workers', 0)
            max_workers = dataloader_config.get('max_workers', 16)
            optimized_num_workers = max(min_workers, min(optimized_num_workers, max_workers))
            config['data']['num_workers'] = optimized_num_workers
            logger.info(f"工作进程数优化: {original_num_workers} → {optimized_num_workers} (策略: {strategy})")
    else:
        logger.info("自适应资源优化已禁用")
    
    # 创建数据加载器（包含归一化器）
    logger.info("创建数据加载器...")
    
    # 检查归一化配置
    normalize_config = config.get('data', {}).get('normalize', False)
    if normalize_config and ENHANCED_DATALOADER_AVAILABLE:
        logger.info("🎯 启用数据归一化功能")
        train_loader, val_loader, test_loader, normalizer = create_enhanced_crop_dataloader(config)
        
        # 记录归一化信息
        if hasattr(normalizer, 'method'):
            logger.info(f"归一化方法: {normalizer.method}")
            if hasattr(normalizer, 'target_range'):
                logger.info(f"目标范围: {normalizer.target_range}")
    else:
        if normalize_config and not ENHANCED_DATALOADER_AVAILABLE:
            logger.warning("⚠️ 配置要求归一化但增强版数据加载器不可用，使用标准版本")
        train_loader, val_loader, test_loader = create_crop_dataloader(config)
        normalizer = None
    
    # 归一化信息已在数据加载器创建过程中处理
    
    # 获取输入输出维度
    input_dim = config['data']['input_dim']
    output_dim = config['data']['output_dim']
    
    logger.info(f"数据维度 - 输入: {input_dim}, 输出: {output_dim}")
    if normalizer:
        logger.info(f"归一化器状态: 已启用 ({normalizer.__class__.__name__})")
    else:
        logger.info("归一化器状态: 未启用")
    
    # 确定要测试的模型
    if models is None:
        models = list(config['models'].keys())
    
    results = {}
    
    # 测试每个模型
    for model_name in models:
        if model_name not in config['models']:
            logger.warning(f"配置中未找到模型 {model_name}，跳过")
            continue
        
        logger.info(f"\n{'='*50}")
        logger.info(f"测试模型: {model_name}")
        logger.info(f"{'='*50}")
        
        model_config = config['models'][model_name]
        
        try:
            # 创建模型
            logger.info("开始创建模型...")
            model_params = {k: v for k, v in model_config.items() if k not in ['model_type', 'input_dim', 'output_dim']}
            
            # 确保数值参数类型正确
            numeric_params = ['hidden_dims', 'd_model', 'num_heads', 'num_layers', 'dropout', 'seq_len', 'width']
            for param in numeric_params:
                if param in model_params:
                    if isinstance(model_params[param], str):
                        try:
                            # 尝试转换为数值
                            if '.' in model_params[param]:
                                model_params[param] = float(model_params[param])
                            else:
                                model_params[param] = int(model_params[param])
                        except ValueError:
                            logger.warning(f"无法转换参数 {param}: {model_params[param]}")
            
            logger.info(f"模型参数: {model_params}")
            
            logger.info("调用create_model函数...")
            model = create_model(
                model_type=model_config['model_type'],
                input_dim=input_dim,
                output_dim=output_dim,
                **model_params
            )
            logger.info("模型创建成功")
            
            # 计算参数数量
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"模型参数数量: {num_params:,}")
            
            # 训练前内存状态记录
            log_system_status(f"训练前-{model_name}")
            
            # 训练模型
            logger.info("开始训练...")
            logger.info("调用train_model函数...")
            train_start_time = datetime.now()
            train_results = train_model(model, train_loader, val_loader, config, device, normalizer)
            train_time = (datetime.now() - train_start_time).total_seconds()
            
            # 训练后内存清理
            cleanup_memory()
            log_system_status(f"训练后-{model_name}")
            
            # 评估模型
            logger.info("开始评估...")
            eval_start_time = datetime.now()
            eval_results = evaluate_model(model, test_loader, device, normalizer)
            eval_time = (datetime.now() - eval_start_time).total_seconds()
            
            # 评估后内存清理
            cleanup_memory()
            log_system_status(f"评估后-{model_name}")
            
            # 保存结果
            results[model_name] = {
                'model_type': model_config['model_type'],
                'num_parameters': num_params,
                'train_time': train_time,
                'eval_time': eval_time,
                'best_val_loss': train_results['best_val_loss'],
                'final_epoch': train_results['final_epoch'],
                'test_mse': eval_results['mse'],
                'test_r2': eval_results['r2_score'],
                'train_losses': train_results['train_losses'],
                'val_losses': train_results['val_losses']
            }
            
            logger.info(f"✅ {model_name} 测试完成")
            logger.info(f"   测试MSE: {eval_results['mse']:.6f}")
            logger.info(f"   测试R²: {eval_results['r2_score']:.4f}")
            logger.info(f"   训练时间: {train_time:.2f}秒")
            
        except Exception as e:
            import traceback
            logger.error(f"❌ {model_name} 测试失败: {e}")
            logger.error(f"错误详情: {traceback.format_exc()}")
            results[model_name] = {
                'error': str(e),
                'status': 'failed'
            }
    
    # 保存测试报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"crop_model_test_report_{timestamp}.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"裁剪模型测试报告\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"配置文件: {config_path}\n")
        f.write(f"输入维度: {input_dim} (32x32裁剪)\n")
        f.write(f"输出维度: {output_dim} (128x128原始)\n")
        f.write(f"总模型数: {len(models)}\n")
        f.write(f"成功模型数: {len([r for r in results.values() if 'error' not in r])}\n\n")
        
        for model_name, result in results.items():
            f.write(f"模型: {model_name}\n")
            if 'error' in result:
                f.write(f"  状态: 失败\n")
                f.write(f"  错误: {result['error']}\n")
            else:
                f.write(f"  类型: {result['model_type']}\n")
                f.write(f"  参数数量: {result['num_parameters']:,}\n")
                f.write(f"  训练时间: {result['train_time']:.2f}秒\n")
                f.write(f"  评估时间: {result['eval_time']:.2f}秒\n")
                f.write(f"  最佳验证损失: {result['best_val_loss']:.6f}\n")
                f.write(f"  训练轮次: {result['final_epoch']}\n")
                f.write(f"  测试MSE: {result['test_mse']:.6f}\n")
                f.write(f"  测试R²: {result['test_r2']:.4f}\n")
            f.write("\n")
    
    logger.info(f"\n📊 测试报告已保存: {report_path}")
    
    # 生成可视化图表
    try:
        logger.info("🎨 开始生成可视化图表...")
        plot_files = create_visualization_plots(results, timestamp)
        
        if plot_files:
            logger.info(f"✅ 成功生成 {len(plot_files)} 个可视化图表:")
            for plot_file in plot_files:
                logger.info(f"  📈 {plot_file}")
            
            # 生成可视化汇总报告
            logger.info("📝 生成可视化汇总报告...")
            summary_path = generate_visualization_summary(results, plot_files, timestamp)
            logger.info(f"📋 可视化汇总报告已保存: {summary_path}")
        else:
            logger.warning("⚠️  没有生成可视化图表（可能没有成功的模型结果）")
            
    except Exception as e:
        logger.error(f"❌ 生成可视化图表时出错: {e}")
        logger.info("📊 文本报告仍然可用")
    
    return results

def create_visualization_plots(results: Dict[str, Any], timestamp: str) -> List[str]:
    """
    创建可视化图表，调用utils.visualization模块并包装结果到指定文件夹
    
    Args:
        results: 模型测试结果字典
        timestamp: 时间戳字符串
    
    Returns:
        生成的图表文件路径列表
    """
    # 创建可视化结果文件夹
    viz_folder = Path("utils") / "visualization_results" / f"crop_model_test_{timestamp}"
    viz_folder.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 创建可视化结果文件夹: {viz_folder}")
    
    # 过滤成功的结果
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    if not successful_results:
        logger.warning("没有成功的模型结果，跳过可视化")
        return []
    
    try:
        # 调用utils.visualization模块的函数生成图表
        plot_files = create_model_comparison_plots(
            results=successful_results,
            output_dir=str(viz_folder),
            timestamp=timestamp
        )
        
        # 如果有训练损失数据，生成损失曲线图
        for model_name, result in successful_results.items():
            if 'train_losses' in result and 'val_losses' in result:
                # 使用验证损失作为测试损失的替代（因为我们没有单独的测试损失）
                loss_plot_path = plot_training_losses(
                    train_losses=result['train_losses'],
                    valid_losses=result['val_losses'],
                    test_losses=result['val_losses'],  # 使用验证损失作为替代
                    model_name=model_name,
                    output_dir=str(viz_folder),
                    timestamp=timestamp
                )
                if loss_plot_path:
                    plot_files.append(loss_plot_path)
                    logger.info(f"✅ {model_name} 训练损失曲线已保存: {loss_plot_path}")
        
        # 生成三联图（输入-真实-预测对比图）
        triplet_plots = generate_triplet_plots(successful_results, viz_folder, timestamp)
        plot_files.extend(triplet_plots)
        
        logger.info(f"✅ 成功生成 {len(plot_files)} 个可视化图表到文件夹: {viz_folder}")
        return plot_files
        
    except Exception as e:
        logger.error(f"❌ 调用utils.visualization模块时出错: {e}")
        # 回退到原始实现
        return create_fallback_visualization_plots(successful_results, timestamp, viz_folder)

def generate_triplet_plots(results: Dict[str, Any], viz_folder: Path, timestamp: str) -> List[str]:
    """
    生成三联图（输入-真实-预测对比图）
    
    Args:
        results: 模型测试结果字典
        viz_folder: 可视化结果文件夹
        timestamp: 时间戳字符串
    
    Returns:
        生成的三联图文件路径列表
    """
    triplet_plot_files = []
    
    # 为每个模型生成三联图
    for model_name, result in results.items():
        # 检查测试结果数据是否存在（支持两种键名格式）
        if ('inputs' in result and 'targets' in result and 'predictions' in result) or \
           ('test_inputs' in result and 'test_targets' in result and 'test_predictions' in result):
            try:
                # 优先使用标准键名，如果不存在则使用带test_前缀的键名
                inputs = result.get('inputs', result.get('test_inputs'))
                targets = result.get('targets', result.get('test_targets'))
                predictions = result.get('predictions', result.get('test_predictions'))
                
                # 创建模型专用的三联图文件夹
                triplet_dir = viz_folder / f"{model_name}_triplet_plots"
                triplet_dir.mkdir(exist_ok=True)
                
                # 选择几个代表性样本进行可视化（避免生成过多图片）
                num_samples = min(5, len(inputs))  # 最多生成5个样本的三联图
                sample_indices = np.linspace(0, len(inputs)-1, num_samples, dtype=int)
                
                logger.info(f"🎨 为模型 {model_name} 生成 {num_samples} 个三联图样本...")
                
                for i, sample_idx in enumerate(sample_indices):
                    # 获取单个样本数据
                    input_sample = inputs[sample_idx]
                    target_sample = targets[sample_idx] 
                    pred_sample = predictions[sample_idx]
                    
                    # 调试信息：打印原始数据形状
                    logger.info(f"🔍 样本 {sample_idx} 原始数据形状: input={input_sample.shape}, target={target_sample.shape}, pred={pred_sample.shape}")
                    
                    # 处理数据维度：确保是2D图像格式
                    if len(input_sample.shape) == 3:
                        # 如果是3D (C, H, W)，取第一个通道或平均
                        if input_sample.shape[0] == 1:
                            input_2d = input_sample[0]
                            target_2d = target_sample[0] 
                            pred_2d = pred_sample[0]
                        else:
                            # 多通道情况，取平均
                            input_2d = np.mean(input_sample, axis=0)
                            target_2d = np.mean(target_sample, axis=0)
                            pred_2d = np.mean(pred_sample, axis=0)
                    elif len(input_sample.shape) == 2:
                        # 已经是2D
                        input_2d = input_sample
                        target_2d = target_sample
                        pred_2d = pred_sample
                    else:
                        # 1D数据，分别处理每个数组的维度
                        def reshape_1d_to_2d(data, name):
                            data_size = data.shape[0]
                            side_len = int(np.sqrt(data_size))
                            if side_len * side_len == data_size:
                                return data.reshape(side_len, side_len)
                            else:
                                # 无法重塑为正方形，尝试其他合理的矩形形状
                                if data_size == 16384:
                                    return data.reshape(128, 128)
                                elif data_size == 1024:
                                    return data.reshape(32, 32)
                                else:
                                    # 尝试找到合适的因子分解
                                    factors = []
                                    for i in range(1, int(np.sqrt(data_size)) + 1):
                                        if data_size % i == 0:
                                            factors.append((i, data_size // i))
                                    
                                    if factors:
                                        # 选择最接近正方形的形状
                                        h, w = min(factors, key=lambda x: abs(x[0] - x[1]))
                                        return data.reshape(h, w)
                                    else:
                                        logger.warning(f"⚠️ {name} 数据大小 {data_size} 无法重塑为2D图像")
                                        return None
                        
                        input_2d = reshape_1d_to_2d(input_sample, "input")
                        target_2d = reshape_1d_to_2d(target_sample, "target")
                        pred_2d = reshape_1d_to_2d(pred_sample, "pred")
                        
                        # 检查是否有任何数组重塑失败
                        if input_2d is None or target_2d is None or pred_2d is None:
                            logger.warning(f"⚠️ 样本 {sample_idx} 数据重塑失败，跳过")
                            continue
                    
                    # 生成三联图
                    time_step = float(i)  # 使用索引作为时间步
                    epoch = 0  # 测试阶段，epoch设为0
                    
                    # 调用visualization.py中的plot_comparison_figure函数
                    plot_comparison_figure(
                        input_pressure=input_2d,
                        true_pressure=target_2d,
                        predicted_pressure=pred_2d,
                        time_step=time_step,
                        epoch=epoch,
                        attention_type=model_name,  # 使用模型名作为注意力类型
                        idx=sample_idx,
                        parent_dir=str(triplet_dir),
                        mode="test"
                    )
                    
                    # 记录生成的文件路径
                    svg_filename = f"test_epoch_{epoch}_sample_{sample_idx}.svg"
                    svg_path = triplet_dir / model_name / "visualization_results" / svg_filename
                    
                    if svg_path.exists():
                        triplet_plot_files.append(str(svg_path))
                        logger.info(f"  📈 样本 {sample_idx} 三联图已保存: {svg_path}")
                    
                logger.info(f"✅ 模型 {model_name} 三联图生成完成，共 {len([f for f in triplet_plot_files if model_name in f])} 个文件")
                
            except Exception as e:
                logger.error(f"❌ 为模型 {model_name} 生成三联图时出错: {e}")
                continue
    
    logger.info(f"🎨 三联图生成完成，共生成 {len(triplet_plot_files)} 个文件")
    return triplet_plot_files

def create_fallback_visualization_plots(successful_results: Dict[str, Any], timestamp: str, viz_folder: Path) -> List[str]:
    """
    回退的可视化实现（当utils.visualization模块调用失败时使用）
    """
    plot_files = []
    model_names = list(successful_results.keys())
    
    # 设置图表样式
    plt.style.use('default')
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # 1. 性能对比图 (MSE 和 R²)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MSE对比
    mse_values = [successful_results[name]['test_mse'] for name in model_names]
    bars1 = ax1.bar(model_names, mse_values, color=colors[:len(model_names)])
    ax1.set_title('模型测试MSE对比', fontsize=14, fontweight='bold')
    ax1.set_ylabel('MSE损失', fontsize=12)
    ax1.set_xlabel('模型类型', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars1, mse_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.4f}', ha='center', va='bottom', fontsize=10)
    
    # R²对比
    r2_values = [successful_results[name]['test_r2'] for name in model_names]
    bars2 = ax2.bar(model_names, r2_values, color=colors[:len(model_names)])
    ax2.set_title('模型测试R²对比', fontsize=14, fontweight='bold')
    ax2.set_ylabel('R²分数', fontsize=12)
    ax2.set_xlabel('模型类型', fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, value in zip(bars2, r2_values):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    performance_plot = viz_folder / f"model_performance_comparison_{timestamp}.png"
    plt.savefig(performance_plot, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(str(performance_plot))
    logger.info(f"✅ 性能对比图已保存: {performance_plot}")
    
    return plot_files

def generate_visualization_summary(results: Dict[str, Any], plot_files: List[str], timestamp: str) -> str:
    """
    生成可视化结果汇总报告，调用utils.visualization模块的函数
    
    Args:
        results: 模型测试结果字典
        plot_files: 生成的图表文件路径列表
        timestamp: 时间戳字符串
    
    Returns:
        汇总报告文件路径
    """
    # 创建可视化结果文件夹
    viz_folder = Path("utils") / "visualization_results" / f"crop_model_test_{timestamp}"
    viz_folder.mkdir(parents=True, exist_ok=True)
    
    summary_path = viz_folder / f"visualization_summary_{timestamp}.md"
    
    try:
        # 调用utils.visualization模块的函数生成汇总报告
        actual_summary_path = generate_model_comparison_summary(
            results=results,
            plot_files=plot_files,
            output_dir=str(viz_folder),
            timestamp=timestamp
        )
        logger.info(f"✅ 可视化汇总报告已保存到: {actual_summary_path}")
        return actual_summary_path
        
    except Exception as e:
        logger.error(f"❌ 调用utils.visualization模块生成汇总报告时出错: {e}")
        # 回退到原始实现
        return create_fallback_summary(results, plot_files, timestamp, summary_path)

def create_fallback_summary(results: Dict[str, Any], plot_files: List[str], timestamp: str, summary_path: Path) -> str:
    """
    回退的汇总报告实现（当utils.visualization模块调用失败时使用）
    """
    # 过滤成功的结果
    successful_results = {k: v for k, v in results.items() if 'error' not in v}
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f"# 裁剪模型测试可视化汇总报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**测试模型数量**: {len(results)}\n")
        f.write(f"**成功模型数量**: {len(successful_results)}\n")
        f.write(f"**生成图表数量**: {len(plot_files)}\n\n")
        
        # 生成的图表列表
        f.write("## 📊 生成的可视化图表\n\n")
        for i, plot_file in enumerate(plot_files, 1):
            f.write(f"{i}. `{plot_file}`\n")
        f.write("\n")
        
        # 模型性能汇总
        if successful_results:
            f.write("## 🏆 模型性能汇总\n\n")
            f.write("| 模型 | 测试MSE | 测试R² | 参数量 | 训练时间(s) |\n")
            f.write("|------|---------|--------|--------|-------------|\n")
            
            for name, result in successful_results.items():
                f.write(f"| {name} | {result['test_mse']:.6f} | {result['test_r2']:.4f} | "
                       f"{result['num_parameters']:,} | {result['train_time']:.2f} |\n")
            f.write("\n")
            
            # 性能排名
            f.write("### 🥇 按性能排名 (MSE越小越好)\n\n")
            sorted_by_mse = sorted(successful_results.items(), key=lambda x: x[1]['test_mse'])
            for i, (name, result) in enumerate(sorted_by_mse, 1):
                f.write(f"{i}. **{name}**: MSE={result['test_mse']:.6f}\n")
            f.write("\n")
            
            f.write("### 🥇 按性能排名 (R²越大越好)\n\n")
            sorted_by_r2 = sorted(successful_results.items(), key=lambda x: x[1]['test_r2'], reverse=True)
            for i, (name, result) in enumerate(sorted_by_r2, 1):
                f.write(f"{i}. **{name}**: R²={result['test_r2']:.4f}\n")
            f.write("\n")
        
        # 失败的模型
        failed_results = {k: v for k, v in results.items() if 'error' in v}
        if failed_results:
            f.write("## ❌ 失败的模型\n\n")
            for name, result in failed_results.items():
                f.write(f"- **{name}**: {result['error']}\n")
            f.write("\n")
    
    return summary_path

def main():
    parser = argparse.ArgumentParser(description='运行统一训练脚本')
    parser.add_argument('--config', default='configs/unified_training_config.yaml',
                        help='统一配置文件路径')
    parser.add_argument('--models', default='all',
                        help='要训练的模型，用逗号分隔，或使用"all"训练所有模型')
    parser.add_argument('--mode', type=str, default=None, 
                       choices=['unified', 'legacy', 'single_model', 'multi_model_comparison', 
                               'parameter_fair_comparison', 'loss_comparison'], 
                       help='训练模式：如果不指定，将使用配置文件中的experiment.mode')
    parser.add_argument('--preset', type=str, default=None,
                       help='使用预设配置：quick_test, full_training, parameter_comparison, loss_study')
    
    args = parser.parse_args()
    
    # 加载配置文件
    try:
        config = load_unified_config(args.config)
        if args.preset:
            # 应用预设配置
            if args.preset in config.get('presets', {}):
                preset_config = config['presets'][args.preset]
                config = merge_configs(config, preset_config)
                logger.info(f"✅ 应用预设配置: {args.preset}")
            else:
                logger.warning(f"⚠️ 预设配置 '{args.preset}' 不存在，使用默认配置")
    except Exception as e:
        logger.error(f"❌ 配置文件加载失败: {e}")
        return
    
    # 确定训练模式
    if args.mode:
        # 命令行指定的模式优先
        experiment_mode = args.mode
        if experiment_mode in ['unified', 'legacy']:
            # 兼容旧的模式名称
            experiment_mode = 'single_model' if experiment_mode == 'unified' else 'legacy'
    else:
        # 使用配置文件中的模式
        experiment_mode = config.get('experiment', {}).get('mode', 'single_model')
    
    # 确定要训练的模型
    if args.models.lower() == 'all':
        models = None  # 让函数自动从配置文件读取
    else:
        models = [model.strip() for model in args.models.split(',')]
    
    logger.info("🚀 开始统一训练...")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"实验模式: {experiment_mode}")
    logger.info(f"训练模型: {models if models else '从配置文件自动读取'}")
    
    # 根据实验模式执行相应的训练
    if experiment_mode in ['single_model', 'unified']:
        run_unified_training(args.config, models)
    elif experiment_mode == 'multi_model_comparison':
        # 启用多模型对比
        config['comparison']['multi_model']['enabled'] = True
        run_unified_training(args.config, models)
    elif experiment_mode == 'parameter_fair_comparison':
        # 启用参数公平对比
        config['comparison']['parameter_fair']['enabled'] = True
        config['comparison']['multi_model']['enabled'] = True
        run_unified_training(args.config, models)
    elif experiment_mode == 'loss_comparison':
        # 启用损失函数对比
        config['comparison']['loss_comparison']['enabled'] = True
        run_unified_training(args.config, models)
    else:
        # 使用传统模式（向后兼容）
        if args.models.lower() == 'all':
            try:
                # 加载配置文件获取所有可用模型
                with open(args.config, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                available_models = list(config.get('models', {}).keys())
                models = available_models
                logger.info(f"自动检测到可用模型: {available_models}")
            except Exception as e:
                logger.warning(f"无法读取配置文件中的模型列表: {e}")
                logger.info("使用默认模型列表")
                models = ['mlp', 'transformer', 'custom_transformer', 'fno', 'unet']
        else:
            models = [m.strip() for m in args.models.split(',')]
        
        logger.info("开始传统模式测试...")
        logger.info(f"配置文件: {args.config}")
        logger.info(f"测试模型: {models}")
        
        results = run_model_test(args.config, models)
    
    logger.info("🎉 训练完成！")

if __name__ == "__main__":
    main()


def set_optimizer_and_learning_rate_scheduler(config: Dict[str, Any]) -> Tuple[torch.optim.Optimizer, Optional[torch.optim.lr_scheduler._LRScheduler]]:
    """设置优化器和学习率调度器"""
    training_config = config.get('training', {})
    
    # 创建优化器
    optimizer_name = training_config.get('optimizer', 'adam').lower()
    lr = training_config.get('learning_rate', 0.001)
    weight_decay = training_config.get('weight_decay', 1e-5)
    
    if optimizer_name == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = training_config.get('momentum', 0.9)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        logger.warning(f"未知优化器: {optimizer_name}，使用Adam")
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 创建学习率调度器
    scheduler = None
    scheduler_config = training_config.get('scheduler')
    if scheduler_config:
        scheduler_type = scheduler_config.get('type', 'step').lower()
        
        if scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 30)
            gamma = scheduler_config.get('gamma', 0.1)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_type == 'cosine':
            T_max = scheduler_config.get('T_max', training_config.get('epochs', 50))
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
        elif scheduler_type == 'plateau':
            patience = scheduler_config.get('patience', 10)
            factor = scheduler_config.get('factor', 0.5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience, factor=factor)
    
    return optimizer, scheduler