#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络横向对比测试脚本

功能:
1. 集成动态分辨率训练器和多种模型架构
2. 实现统一的性能评估指标
3. 生成详细的对比报告和可视化结果
4. 支持多种模型类型的横向对比

作者: AI Assistant
日期: 2025
"""

# 设置无头模式环境变量
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''
os.environ['HEADLESS'] = '1'

import matplotlib
matplotlib.use('Agg')

import sys
import torch
import torch.nn.functional as F
import numpy as np
import yaml
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加路径
project_root = Path(__file__).parent.parent
modify_multi_attention_path = project_root / 'modify_multi_attention'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(modify_multi_attention_path))
sys.path.insert(0, str(Path(__file__).parent))

# 导入动态分辨率训练器
from dynamic_resolution_trainer import (
    DynamicResolutionDataset, get_dynamic_loaders, create_dynamic_config,
    log_gpu_memory, cleanup_memory
)

# 导入模型
try:
    from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel
    from modify_multi_attention.training.trainer import train_model, test_model
    from modify_multi_attention.utils.enhanced_svd_loss import create_enhanced_svd_loss
except ImportError:
    try:
        sys.path.append(str(modify_multi_attention_path))
        from mymodels.transformer import TransformerFlowReconstructionModel
        from training.trainer import train_model, test_model
        from utils.enhanced_svd_loss import create_enhanced_svd_loss
    except ImportError as e:
        logger.error(f"无法导入Transformer模型: {e}")
        TransformerFlowReconstructionModel = None

# 导入增强模型
try:
    from modify_multi_attention.models.enhanced_mlp import EnhancedMLP, EnhancedMLP1d
    from modify_multi_attention.models.enhanced_fno import EnhancedFNO1d, EnhancedFNO2d
    from modify_multi_attention.models.enhanced_unet import EnhancedUNet1d, EnhancedUNet2d
    from modify_multi_attention.models.enhanced_pinn import EnhancedPINN
except ImportError as e:
    logger.warning(f"部分增强模型导入失败: {e}")
    EnhancedMLP = EnhancedMLP1d = None
    EnhancedFNO1d = EnhancedFNO2d = None
    EnhancedUNet1d = EnhancedUNet2d = None
    EnhancedPINN = None

@dataclass
class ModelConfig:
    """模型配置数据类"""
    name: str
    model_type: str
    enabled: bool
    params: Dict[str, Any]
    target_params: Optional[int] = None

@dataclass
class TestResult:
    """测试结果数据类"""
    model_name: str
    model_type: str
    train_time: float
    test_time: float
    memory_usage: float
    param_count: int
    metrics: Dict[str, float]
    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    success: bool = True
    error_msg: str = ""

class PerformanceMetrics:
    """性能评估指标计算器"""
    
    @staticmethod
    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """均方误差"""
        return float(np.mean((y_true - y_pred) ** 2))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """平均绝对误差"""
        return float(np.mean(np.abs(y_true - y_pred)))
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """均方根误差"""
        return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    @staticmethod
    def psnr(y_true: np.ndarray, y_pred: np.ndarray, max_val: float = 1.0) -> float:
        """峰值信噪比"""
        mse = np.mean((y_true - y_pred) ** 2)
        if mse == 0:
            return float('inf')
        return float(20 * np.log10(max_val / np.sqrt(mse)))
    
    @staticmethod
    def ssim_simple(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """简化的结构相似性指数"""
        # 简化版SSIM计算
        mu1 = np.mean(y_true)
        mu2 = np.mean(y_pred)
        sigma1 = np.var(y_true)
        sigma2 = np.var(y_pred)
        sigma12 = np.mean((y_true - mu1) * (y_pred - mu2))
        
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1 + sigma2 + c2))
        return float(ssim)
    
    @staticmethod
    def correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """皮尔逊相关系数"""
        try:
            corr, _ = pearsonr(y_true.flatten(), y_pred.flatten())
            return float(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    @classmethod
    def compute_all_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算所有指标"""
        return {
            'mse': cls.mse(y_true, y_pred),
            'mae': cls.mae(y_true, y_pred),
            'rmse': cls.rmse(y_true, y_pred),
            'psnr': cls.psnr(y_true, y_pred),
            'ssim': cls.ssim_simple(y_true, y_pred),
            'correlation': cls.correlation(y_true, y_pred)
        }

class ModelFactory:
    """模型工厂类"""
    
    @staticmethod
    def create_model(config: ModelConfig, input_dim: int, output_dim: int, 
                    input_hw: Tuple[int, int], device: torch.device) -> torch.nn.Module:
        """根据配置创建模型"""
        model_type = config.model_type.lower()
        params = config.params
        
        try:
            if model_type == 'transformer' and TransformerFlowReconstructionModel is not None:
                model = TransformerFlowReconstructionModel(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    num_heads=params.get('num_heads', 8),
                    num_layers=params.get('num_layers', 6),
                    d_model=params.get('d_model', 512),
                    max_time_steps=params.get('max_time_steps', 1),
                    attention_type=params.get('attention_type', 'sge'),
                    seq_len=params.get('seq_len', 32),
                    input_hw=input_hw,
                    pe_type=params.get('pe_type', 'learnable_1d'),
                    output_head_type=params.get('output_head_type', 'global'),
                    out_channels_per_token=params.get('out_channels_per_token')
                )
                
            elif model_type == 'enhanced_mlp' and EnhancedMLP1d is not None:
                # 修复：MLP模型的input_channels应该是实际的通道数，而不是input_dim
                model = EnhancedMLP1d(
                    input_channels=1,  # 修复：使用实际的输入通道数
                    output_channels=1,  # 修复：使用实际的输出通道数
                    input_resolution=input_hw[0],
                    output_resolution=input_hw[0],
                    hidden_dim=params.get('hidden_dim', 128),
                    num_layers=params.get('num_layers', 6),
                    fourier_mapping_size=params.get('fourier_features', 32)  # 修复：使用更小的fourier_mapping_size
                )
                
            elif model_type == 'enhanced_fno1d' and EnhancedFNO1d is not None:
                model = EnhancedFNO1d(
                    num_channels=1,
                    modes=params.get('modes', 16),
                    width=params.get('width', 64),
                    input_resolution=input_hw[0],
                    output_resolution=input_hw[0]
                )
                
            elif model_type == 'enhanced_unet1d' and EnhancedUNet1d is not None:
                model = EnhancedUNet1d(
                    in_channels=1,  # 修复：使用实际的输入通道数
                    out_channels=1,
                    init_features=params.get('init_features', 32),
                    input_resolution=input_hw[0],
                    output_resolution=input_hw[0]
                )
                
            elif model_type == 'fno1d':
                # 使用基础FNO1d模型（如果可用）
                try:
                    from models.fno import FNO1d
                    model = FNO1d(
                        num_channels=params.get('num_channels', 1),
                        modes=params.get('modes', 16),
                        width=params.get('width', 64)
                    )
                except ImportError:
                    # 如果没有基础FNO1d，使用增强版本
                    if EnhancedFNO1d is not None:
                        model = EnhancedFNO1d(
                            num_channels=1,
                            modes=params.get('modes', 16),
                            width=params.get('width', 64),
                            input_resolution=input_hw[0],
                            output_resolution=input_hw[0]
                        )
                    else:
                        raise ValueError(f"FNO1d模型不可用: {model_type}")
                        
            elif model_type == 'unet1d':
                # 使用基础UNet1d模型（如果可用）
                try:
                    from models.unet import UNet1d
                    model = UNet1d(
                        in_channels=8,  # 修复：使用实际的输入通道数
                        out_channels=params.get('out_channels', 1),
                        init_features=params.get('init_features', 32)
                    )
                except ImportError:
                    # 如果没有基础UNet1d，使用增强版本
                    if EnhancedUNet1d is not None:
                        model = EnhancedUNet1d(
                            in_channels=8,  # 修复：使用实际的输入通道数
                            out_channels=1,
                            init_features=params.get('init_features', 32),
                            input_resolution=input_hw[0],
                            output_resolution=input_hw[0]
                        )
                    else:
                        raise ValueError(f"UNet1d模型不可用: {model_type}")
                
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
                
            return model.to(device)
            
        except Exception as e:
            logger.error(f"创建模型 {config.name} 失败: {e}")
            raise

class NetworkComparison:
    """网络横向对比测试主类"""
    
    def __init__(self, config_path: str, results_dir: str = "./comparison_results"):
        self.config_path = config_path
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载配置
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        # 设备设置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 性能指标计算器
        self.metrics = PerformanceMetrics()
        
        # 测试结果存储
        self.results: List[TestResult] = []
        
    def load_models_config(self) -> List[ModelConfig]:
        """加载模型配置"""
        models_config = []
        models_dict = self.config.get('models', {})
        
        for model_name, model_params in models_dict.items():
            if model_params.get('enabled', False):
                config = ModelConfig(
                    name=model_name,
                    model_type=model_params['type'],
                    enabled=True,
                    params=model_params,
                    target_params=model_params.get('target_params')
                )
                models_config.append(config)
                
        logger.info(f"加载了 {len(models_config)} 个启用的模型配置")
        return models_config
    
    def prepare_data(self) -> Tuple[torch.utils.data.DataLoader, ...]:
        """准备数据加载器"""
        logger.info("准备数据加载器...")
        
        # 创建动态配置
        dynamic_config = create_dynamic_config()
        
        # 更新配置
        dynamic_config.update(self.config)
        
        # 获取数据加载器
        train_loader, valid_loader, test_loader, dataset = get_dynamic_loaders(dynamic_config)
        
        logger.info(f"数据集大小: 训练={len(train_loader.dataset)}, "
                   f"验证={len(valid_loader.dataset)}, 测试={len(test_loader.dataset)}")
        
        return train_loader, valid_loader, test_loader, dataset
    
    def train_and_evaluate_model(self, model_config: ModelConfig, 
                                train_loader, valid_loader, test_loader, 
                                dataset) -> TestResult:
        """训练和评估单个模型"""
        logger.info(f"\n{'='*60}")
        logger.info(f"开始测试模型: {model_config.name} ({model_config.model_type})")
        logger.info(f"{'='*60}")
        
        try:
            # 获取数据维度信息
            sample_batch = next(iter(train_loader))
            # DynamicResolutionDataset返回三个值：(input, output, index)
            if len(sample_batch) == 3:
                input_data, target_data, _ = sample_batch
            else:
                input_data, target_data = sample_batch
            
            input_dim = input_data.shape[-1] if len(input_data.shape) > 2 else input_data.shape[1]
            output_dim = target_data.shape[-1] if len(target_data.shape) > 2 else target_data.shape[1]
            
            # 获取输入分辨率
            input_hw = tuple(self.config['data']['input_resolution'])
            
            # 创建模型
            start_time = time.time()
            model = ModelFactory.create_model(model_config, input_dim, output_dim, input_hw, self.device)
            
            # 计算参数数量
            param_count = sum(p.numel() for p in model.parameters())
            logger.info(f"模型参数数量: {param_count:,}")
            
            # 记录内存使用
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                memory_before = torch.cuda.memory_allocated()
            else:
                memory_before = 0
            
            # 训练模型（简化版，只训练几个epoch）
            train_start = time.time()
            
            if model_config.model_type == 'transformer':
                # 使用现有的训练器
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                criterion = create_enhanced_svd_loss()
                
                model.train()
                train_losses = []
                
                # 简化训练（只训练3个epoch）
                for epoch in range(3):
                    epoch_loss = 0
                    for batch_idx, batch_data in enumerate(train_loader):
                         if batch_idx >= 10:  # 只训练前10个batch
                             break
                         
                         # 处理数据解包
                         if len(batch_data) == 3:
                             data, target, _ = batch_data
                         else:
                             data, target = batch_data
                             
                         data, target = data.to(self.device), target.to(self.device)
                         
                         # 调整数据形状以适应不同模型
                         if len(data.shape) == 4:  # [B, C, H, W]
                             if model_config.model_type in ['enhanced_mlp']:
                                 # MLP期望[batch, height, width, channels]格式
                                 batch_size, channels, height, width = data.shape
                                 data = data.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
                                 target = target.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
                             elif model_config.model_type in ['enhanced_fno1d', 'fno1d']:
                                 # FNO1d期望[batch, resolution, channels]格式
                                 batch_size, channels, height, width = data.shape
                                 # 将2D数据转换为1D：[B, C, H, W] -> [B, H*W, C]
                                 data = data.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
                                 target = target.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, target.shape[1])
                             elif model_config.model_type in ['enhanced_unet1d', 'unet1d']:
                               # UNet1d需要1通道输入 [batch, 1, length]
                               batch_size, channels, height, width = data.shape
                               # 将多通道数据平均为1通道
                               data = data.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                               data = data.view(batch_size, 1, height * width)  # [B, 1, H*W]
                               # target也转换为1通道
                               target = target.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                               target = target.view(batch_size, 1, height * width)  # [B, 1, H*W]
                               
                               # 确保数据长度适合UNet1d的池化操作（需要能被16整除）
                               current_length = height * width
                               # 将长度调整为最接近的能被16整除的数
                               target_length = ((current_length + 15) // 16) * 16
                               if current_length != target_length:
                                   # 使用插值调整长度
                                   data = F.interpolate(data, size=target_length, mode='linear', align_corners=False)
                                   target = F.interpolate(target, size=target_length, mode='linear', align_corners=False)
                         
                         optimizer.zero_grad()
                         # 为Transformer模型添加时间步参数
                         if model_config.model_type == 'transformer':
                             # 创建时间步张量
                             time_steps = torch.zeros(data.shape[0], dtype=torch.long, device=self.device)
                             output = model(data, time_steps)
                         else:
                             output = model(data)
                         loss = criterion(output, target)
                         loss.backward()
                         optimizer.step()
                         
                         epoch_loss += loss.item()
                    
                    avg_loss = epoch_loss / min(10, len(train_loader))
                    train_losses.append(avg_loss)
                    logger.info(f"Epoch {epoch+1}/3, Loss: {avg_loss:.6f}")
            else:
                # 其他模型的简化训练
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
                criterion = torch.nn.MSELoss()
                
                model.train()
                for epoch in range(3):
                    epoch_loss = 0
                    for batch_idx, batch_data in enumerate(train_loader):
                        if batch_idx >= 10:
                            break
                        
                        # 处理数据解包
                        if len(batch_data) == 3:
                            data, target, _ = batch_data
                        else:
                            data, target = batch_data
                            
                        data, target = data.to(self.device), target.to(self.device)
                        
                        # 调整数据形状以适应不同模型
                        if len(data.shape) == 4:  # [B, C, H, W]
                            if model_config.model_type in ['enhanced_mlp']:
                                # MLP期望[batch, height, width, channels]格式
                                batch_size, channels, height, width = data.shape
                                data = data.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
                                target = target.permute(0, 2, 3, 1).contiguous()  # [B, H, W, C]
                            elif model_config.model_type in ['enhanced_fno1d', 'fno1d']:
                                # FNO1d期望[batch, resolution, channels]格式
                                batch_size, channels, height, width = data.shape
                                # 将2D数据转换为1D：[B, C, H, W] -> [B, H*W, C]
                                data = data.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
                                target = target.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, target.shape[1])
                            elif model_config.model_type in ['enhanced_unet1d', 'unet1d']:
                              # UNet1d需要1通道输入 [batch, 1, length]
                              batch_size, channels, height, width = data.shape
                              # 将多通道数据平均为1通道
                              data = data.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                              data = data.view(batch_size, 1, height * width)  # [B, 1, H*W]
                              # target也转换为1通道
                              target = target.mean(dim=1, keepdim=True)  # [B, 1, H, W]
                              target = target.view(batch_size, 1, height * width)  # [B, 1, H*W]
                              
                              # 确保数据长度适合UNet1d的池化操作（需要能被16整除）
                              current_length = height * width
                              # 将长度调整为最接近的能被16整除的数
                              target_length = ((current_length + 15) // 16) * 16
                              if current_length != target_length:
                                  # 使用插值调整长度
                                  data = F.interpolate(data, size=target_length, mode='linear', align_corners=False)
                                  target = F.interpolate(target, size=target_length, mode='linear', align_corners=False)
                        
                        optimizer.zero_grad()
                        # 为不同模型添加特定参数
                        if model_config.model_type == 'transformer':
                            # 创建时间步张量
                            time_steps = torch.zeros(data.shape[0], dtype=torch.long, device=self.device)
                            output = model(data, time_steps)
                        elif model_config.model_type in ['fno1d', 'enhanced_fno1d']:
                            # FNO模型需要grid参数
                            if len(data.shape) == 3:  # [batch, resolution, channels]
                                grid = torch.linspace(0, 1, data.shape[1], device=self.device).unsqueeze(0).unsqueeze(-1)
                                grid = grid.expand(data.shape[0], -1, -1)
                                output = model(data, grid)
                            else:
                                output = model(data)
                        else:
                            output = model(data)
                        
                        # 确保输出和目标形状匹配
                        if output.shape != target.shape:
                            if len(output.shape) == 2 and len(target.shape) == 4:
                                target = target.view(target.shape[0], -1)
                            elif len(output.shape) == 4 and len(target.shape) == 2:
                                output = output.view(output.shape[0], -1)
                        
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()
                        
                        epoch_loss += loss.item()
                    
                    avg_loss = epoch_loss / min(10, len(train_loader))
                    logger.info(f"Epoch {epoch+1}/3, Loss: {avg_loss:.6f}")
            
            train_time = time.time() - train_start
            
            # 测试模型
            test_start = time.time()
            model.eval()
            
            all_predictions = []
            all_targets = []
            
            with torch.no_grad():
                for batch_idx, batch_data in enumerate(test_loader):
                     if batch_idx >= 5:  # 只测试前5个batch
                         break
                     
                     # 处理数据解包
                     if len(batch_data) == 3:
                         data, target, _ = batch_data
                     else:
                         data, target = batch_data
                         
                     data, target = data.to(self.device), target.to(self.device)
                     
                     # 调整数据形状
                     if len(data.shape) == 4:
                         if model_config.model_type in ['enhanced_mlp']:
                             # MLP期望[batch, resolution, channels]格式
                             batch_size, channels, height, width = data.shape
                             data = data.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
                             target = target.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, target.shape[1])
                         elif model_config.model_type in ['enhanced_fno1d', 'fno1d']:
                             # FNO1d期望[batch, resolution, channels]格式
                             batch_size, channels, height, width = data.shape
                             # 将2D数据转换为1D：[B, C, H, W] -> [B, H*W, C]
                             data = data.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, channels)
                             target = target.permute(0, 2, 3, 1).contiguous().view(batch_size, height * width, target.shape[1])
                         elif model_config.model_type in ['enhanced_unet1d', 'unet1d']:
                             # UNet1d需要保持通道维度 [batch, channels, length]
                             batch_size, channels, height, width = data.shape
                             data = data.view(batch_size, channels, height * width)
                             # 保持target的通道维度以匹配模型输出
                             target = target.view(batch_size, target.shape[1], height * width)
                     
                     # 为不同模型添加特定参数
                     if model_config.model_type == 'transformer':
                         # 创建时间步张量
                         time_steps = torch.zeros(data.shape[0], dtype=torch.long, device=self.device)
                         output = model(data, time_steps)
                     elif model_config.model_type in ['fno1d', 'enhanced_fno1d']:
                         # FNO模型需要grid参数
                         if len(data.shape) == 3:  # [batch, resolution, channels]
                             grid = torch.linspace(0, 1, data.shape[1], device=self.device).unsqueeze(0).unsqueeze(-1)
                             grid = grid.expand(data.shape[0], -1, -1)
                             output = model(data, grid)
                         else:
                             output = model(data)
                     else:
                         output = model(data)
                     
                     # 确保输出和目标形状匹配
                     if output.shape != target.shape:
                         if len(output.shape) == 2 and len(target.shape) == 4:
                             target = target.view(target.shape[0], -1)
                         elif len(output.shape) == 4 and len(target.shape) == 2:
                             output = output.view(output.shape[0], -1)
                     
                     all_predictions.append(output.cpu().numpy())
                     all_targets.append(target.cpu().numpy())
            
            test_time = time.time() - test_start
            
            # 合并预测结果
            predictions = np.concatenate(all_predictions, axis=0)
            targets = np.concatenate(all_targets, axis=0)
            
            # 计算性能指标
            metrics = self.metrics.compute_all_metrics(targets, predictions)
            
            # 记录内存使用
            if torch.cuda.is_available():
                memory_after = torch.cuda.memory_allocated()
                memory_usage = (memory_after - memory_before) / 1024**2  # MB
            else:
                memory_usage = 0
            
            # 创建测试结果
            result = TestResult(
                model_name=model_config.name,
                model_type=model_config.model_type,
                train_time=train_time,
                test_time=test_time,
                memory_usage=memory_usage,
                param_count=param_count,
                metrics=metrics,
                predictions=predictions[:100],  # 只保存前100个样本
                targets=targets[:100],
                success=True
            )
            
            logger.info(f"模型 {model_config.name} 测试完成")
            logger.info(f"训练时间: {train_time:.2f}s, 测试时间: {test_time:.2f}s")
            logger.info(f"内存使用: {memory_usage:.2f}MB, 参数数量: {param_count:,}")
            logger.info(f"MSE: {metrics['mse']:.6f}, PSNR: {metrics['psnr']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"模型 {model_config.name} 测试失败: {e}")
            return TestResult(
                model_name=model_config.name,
                model_type=model_config.model_type,
                train_time=0,
                test_time=0,
                memory_usage=0,
                param_count=0,
                metrics={},
                success=False,
                error_msg=str(e)
            )
        
        finally:
            # 清理内存
            cleanup_memory()
    
    def run_comparison(self):
        """运行完整的横向对比测试"""
        logger.info("开始网络横向对比测试")
        
        # 加载模型配置
        models_config = self.load_models_config()
        
        if not models_config:
            logger.error("没有找到启用的模型配置")
            return
        
        # 准备数据
        train_loader, valid_loader, test_loader, dataset = self.prepare_data()
        
        # 测试每个模型
        for model_config in models_config:
            try:
                result = self.train_and_evaluate_model(
                    model_config, train_loader, valid_loader, test_loader, dataset
                )
                self.results.append(result)
                
            except Exception as e:
                logger.error(f"模型 {model_config.name} 测试过程中发生错误: {e}")
                continue
        
        # 生成报告
        self.generate_report()
        
        logger.info(f"横向对比测试完成，结果保存在: {self.results_dir}")
    
    def generate_report(self):
        """生成详细的对比报告"""
        logger.info("生成对比报告...")
        
        # 过滤成功的结果
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        # 生成文本报告
        self._generate_text_report(successful_results, failed_results)
        
        # 生成可视化报告
        if successful_results:
            self._generate_visualization_report(successful_results)
        
        # 保存详细结果
        self._save_detailed_results()
    
    def _generate_text_report(self, successful_results: List[TestResult], 
                             failed_results: List[TestResult]):
        """生成文本报告"""
        report_path = self.results_dir / "comparison_report.txt"
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("网络横向对比测试报告\n")
            f.write("=" * 50 + "\n\n")
            
            # 测试概览
            f.write(f"测试时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"总模型数: {len(self.results)}\n")
            f.write(f"成功测试: {len(successful_results)}\n")
            f.write(f"失败测试: {len(failed_results)}\n\n")
            
            if successful_results:
                # 性能排名
                f.write("性能排名 (按MSE排序):\n")
                f.write("-" * 30 + "\n")
                
                sorted_results = sorted(successful_results, key=lambda x: x.metrics.get('mse', float('inf')))
                
                for i, result in enumerate(sorted_results, 1):
                    f.write(f"{i}. {result.model_name} ({result.model_type})\n")
                    f.write(f"   MSE: {result.metrics.get('mse', 'N/A'):.6f}\n")
                    f.write(f"   PSNR: {result.metrics.get('psnr', 'N/A'):.2f}\n")
                    f.write(f"   参数数量: {result.param_count:,}\n")
                    f.write(f"   训练时间: {result.train_time:.2f}s\n\n")
                
                # 详细指标对比
                f.write("\n详细指标对比:\n")
                f.write("-" * 50 + "\n")
                
                # 表头
                f.write(f"{'模型名称':<20} {'MSE':<12} {'PSNR':<8} {'SSIM':<8} {'相关性':<8} {'参数数':<10}\n")
                f.write("-" * 80 + "\n")
                
                for result in sorted_results:
                    f.write(f"{result.model_name:<20} ")
                    f.write(f"{result.metrics.get('mse', 0):<12.6f} ")
                    f.write(f"{result.metrics.get('psnr', 0):<8.2f} ")
                    f.write(f"{result.metrics.get('ssim', 0):<8.4f} ")
                    f.write(f"{result.metrics.get('correlation', 0):<8.4f} ")
                    f.write(f"{result.param_count:<10,}\n")
            
            # 失败的模型
            if failed_results:
                f.write("\n\n失败的模型:\n")
                f.write("-" * 30 + "\n")
                for result in failed_results:
                    f.write(f"- {result.model_name} ({result.model_type}): {result.error_msg}\n")
        
        logger.info(f"文本报告已保存: {report_path}")
    
    def _generate_visualization_report(self, results: List[TestResult]):
        """生成可视化报告"""
        # 设置绘图样式
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 创建图表
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('网络横向对比测试结果', fontsize=16, fontweight='bold')
        
        # 提取数据
        model_names = [r.model_name for r in results]
        mse_values = [r.metrics.get('mse', 0) for r in results]
        psnr_values = [r.metrics.get('psnr', 0) for r in results]
        param_counts = [r.param_count for r in results]
        train_times = [r.train_time for r in results]
        memory_usage = [r.memory_usage for r in results]
        correlations = [r.metrics.get('correlation', 0) for r in results]
        
        # 1. MSE对比
        axes[0, 0].bar(range(len(model_names)), mse_values)
        axes[0, 0].set_title('均方误差 (MSE) 对比')
        axes[0, 0].set_ylabel('MSE')
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. PSNR对比
        axes[0, 1].bar(range(len(model_names)), psnr_values)
        axes[0, 1].set_title('峰值信噪比 (PSNR) 对比')
        axes[0, 1].set_ylabel('PSNR (dB)')
        axes[0, 1].set_xticks(range(len(model_names)))
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 参数数量对比
        axes[0, 2].bar(range(len(model_names)), param_counts)
        axes[0, 2].set_title('模型参数数量对比')
        axes[0, 2].set_ylabel('参数数量')
        axes[0, 2].set_xticks(range(len(model_names)))
        axes[0, 2].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. 训练时间对比
        axes[1, 0].bar(range(len(model_names)), train_times)
        axes[1, 0].set_title('训练时间对比')
        axes[1, 0].set_ylabel('时间 (秒)')
        axes[1, 0].set_xticks(range(len(model_names)))
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. 内存使用对比
        axes[1, 1].bar(range(len(model_names)), memory_usage)
        axes[1, 1].set_title('内存使用对比')
        axes[1, 1].set_ylabel('内存 (MB)')
        axes[1, 1].set_xticks(range(len(model_names)))
        axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. 相关性对比
        axes[1, 2].bar(range(len(model_names)), correlations)
        axes[1, 2].set_title('预测相关性对比')
        axes[1, 2].set_ylabel('皮尔逊相关系数')
        axes[1, 2].set_xticks(range(len(model_names)))
        axes[1, 2].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        chart_path = self.results_dir / "comparison_charts.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"可视化报告已保存: {chart_path}")
        
        # 生成性能效率散点图
        self._generate_efficiency_plot(results)
    
    def _generate_efficiency_plot(self, results: List[TestResult]):
        """生成性能效率散点图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 提取数据
        param_counts = [r.param_count for r in results]
        mse_values = [r.metrics.get('mse', 0) for r in results]
        train_times = [r.train_time for r in results]
        model_names = [r.model_name for r in results]
        
        # 1. 参数数量 vs 性能
        ax1.scatter(param_counts, mse_values, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            ax1.annotate(name, (param_counts[i], mse_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax1.set_xlabel('参数数量')
        ax1.set_ylabel('MSE')
        ax1.set_title('参数数量 vs 性能')
        ax1.grid(True, alpha=0.3)
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        
        # 2. 训练时间 vs 性能
        ax2.scatter(train_times, mse_values, s=100, alpha=0.7)
        for i, name in enumerate(model_names):
            ax2.annotate(name, (train_times[i], mse_values[i]), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax2.set_xlabel('训练时间 (秒)')
        ax2.set_ylabel('MSE')
        ax2.set_title('训练时间 vs 性能')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        plt.tight_layout()
        
        # 保存图表
        efficiency_path = self.results_dir / "efficiency_analysis.png"
        plt.savefig(efficiency_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"效率分析图已保存: {efficiency_path}")
    
    def _save_detailed_results(self):
        """保存详细结果到JSON文件"""
        results_data = []
        
        for result in self.results:
            result_dict = {
                'model_name': result.model_name,
                'model_type': result.model_type,
                'success': result.success,
                'train_time': result.train_time,
                'test_time': result.test_time,
                'memory_usage': result.memory_usage,
                'param_count': result.param_count,
                'metrics': result.metrics,
                'error_msg': result.error_msg
            }
            results_data.append(result_dict)
        
        # 保存到JSON文件
        json_path = self.results_dir / "detailed_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"详细结果已保存: {json_path}")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='网络横向对比测试')
    parser.add_argument('--config', type=str, default='enhanced_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--results_dir', type=str, default='./comparison_results',
                       help='结果保存目录')
    
    args = parser.parse_args()
    
    # 检查配置文件是否存在
    if not Path(args.config).exists():
        logger.error(f"配置文件不存在: {args.config}")
        return
    
    # 创建对比测试实例
    comparison = NetworkComparison(args.config, args.results_dir)
    
    # 运行对比测试
    comparison.run_comparison()
    
    logger.info("网络横向对比测试完成！")

if __name__ == "__main__":
    main()