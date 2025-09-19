#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基于真实数据的多模型测试脚本（支持裁剪输入和原始输出）

功能:
1. 使用真实Darcy Flow数据集
2. 输入: 32x32裁剪数据
3. 输出: 128x128原始数据
4. 测试多种模型架构的性能

参考: generate_data/pde_process/create_input32_output128_dataset.py

作者: AI Assistant
日期: 2025
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import json
import psutil
import gc
import time

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# 导入自定义模块
from modify_multi_attention.data.crop_dataloader import create_crop_dataloader
from modify_multi_attention.utils.config import load_config
from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel
from models.enhanced_fno import create_enhanced_fno2d
from models.enhanced_unet import create_enhanced_unet2d

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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

def create_model(model_type, input_dim, output_dim, **kwargs):
    """创建模型"""
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

def train_model(model, train_loader, val_loader, config, device):
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
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        num_batches = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            
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
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                num_val_batches += 1
        
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

def evaluate_model(model, test_loader, device):
    """评估模型"""
    model.eval()
    criterion = nn.MSELoss()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)
            
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    
    # 计算R²分数
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    
    return {
        'mse': avg_loss,
        'r2_score': r2_score,
        'predictions': predictions,
        'targets': targets
    }

def run_model_test(config_path, models=None):
    """运行模型测试"""
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info(f"加载配置文件: {config_path}")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() and config['device']['use_cuda'] else 'cpu')
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
    
    # 创建数据加载器
    logger.info("创建数据加载器...")
    train_loader, val_loader, test_loader = create_crop_dataloader(config)
    
    # 获取输入输出维度
    input_dim = config['data']['input_dim']
    output_dim = config['data']['output_dim']
    
    logger.info(f"数据维度 - 输入: {input_dim}, 输出: {output_dim}")
    
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
            train_results = train_model(model, train_loader, val_loader, config, device)
            train_time = (datetime.now() - train_start_time).total_seconds()
            
            # 训练后内存清理
            cleanup_memory()
            log_system_status(f"训练后-{model_name}")
            
            # 评估模型
            logger.info("开始评估...")
            eval_start_time = datetime.now()
            eval_results = evaluate_model(model, test_loader, device)
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
                'test_r2': eval_results['r2_score']
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
    
    return results

def main():
    parser = argparse.ArgumentParser(description='运行裁剪模型测试')
    parser.add_argument('--config', default='modify_multi_attention/configs/real_data_crop_config.yaml',
                        help='配置文件路径')
    parser.add_argument('--models', default='all',
                        help='要测试的模型，用逗号分隔，或使用"all"运行所有模型')
    
    args = parser.parse_args()
    
    # 如果指定为"all"，则从配置文件中读取所有可用模型
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
    
    logger.info("开始裁剪模型测试...")
    logger.info(f"配置文件: {args.config}")
    logger.info(f"测试模型: {models}")
    
    results = run_model_test(args.config, models)
    
    logger.info("🎉 测试完成！")

if __name__ == "__main__":
    main()