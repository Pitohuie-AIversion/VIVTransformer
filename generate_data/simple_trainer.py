#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的训练器 - 用于分离式训练的第二阶段

功能:
1. 加载预处理好的数据
2. 使用与dynamic_resolution_trainer.py完全相同的训练代码
3. 支持断点续训
4. 优化的内存使用

作者: AI Assistant
日期: 2025
"""

# 设置环境变量
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''
os.environ['HEADLESS'] = '1'

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

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加项目路径
project_root = Path(__file__).parent.parent
modify_multi_attention_path = project_root / 'modify_multi_attention'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(modify_multi_attention_path))
sys.path.insert(0, str(Path(__file__).parent))

# 导入模块 - 与dynamic_resolution_trainer.py完全相同的导入逻辑
try:
    from modify_multi_attention.mymodels.transformer import TransformerFlowReconstructionModel
    from modify_multi_attention.training.trainer import train_model, test_model
    from modify_multi_attention.utils.visualization import plot_losses
    from modify_multi_attention.utils.svd10_loss import TotalLossWithSVD
    from modify_multi_attention.utils.logging_utils import setup_logging
    from modify_multi_attention.utils.enhanced_svd_loss import (
        EnhancedTotalLossWithSVD, create_enhanced_svd_loss,
        get_global_svd_stats, print_global_svd_stats, reset_global_svd_stats
    )
except ImportError as e:
    logger.warning(f"第一次导入失败: {e}")
    try:
        # 备用导入路径
        sys.path.append(str(modify_multi_attention_path))
        from mymodels.transformer import TransformerFlowReconstructionModel
        from training.trainer import train_model, test_model
        from utils.visualization import plot_losses
        from utils.svd10_loss import TotalLossWithSVD
        from utils.logging_utils import setup_logging
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

# 复制dynamic_resolution_trainer.py中的工具函数
def log_gpu_memory(stage: str = ""):
    """记录GPU内存使用情况"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            logger.info(f"🖥️ GPU {i} 内存 {stage}: 已分配={allocated:.2f}GB, 已保留={reserved:.2f}GB")
    else:
        logger.info(f"🖥️ GPU内存 {stage}: CUDA不可用")

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
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = process.memory_percent()
        
        system_memory = psutil.virtual_memory()
        system_total_gb = system_memory.total / 1024**3
        system_used_gb = system_memory.used / 1024**3
        system_percent = system_memory.percent
        
        logger.info(f"💾 CPU内存 {stage}: 进程={memory_mb:.1f}MB ({memory_percent:.1f}%), 系统={system_used_gb:.1f}/{system_total_gb:.1f}GB ({system_percent:.1f}%)")
    except ImportError:
        logger.info(f"💾 CPU内存 {stage}: psutil未安装，无法监控")

def optimize_cpu_for_data_loading():
    """优化CPU设置以提升数据加载性能"""
    import os
    cpu_count = os.cpu_count()
    
    # 设置OpenMP线程数
    optimal_threads = min(16, cpu_count)
    os.environ['OMP_NUM_THREADS'] = str(optimal_threads)
    os.environ['MKL_NUM_THREADS'] = str(optimal_threads)
    os.environ['NUMEXPR_NUM_THREADS'] = str(optimal_threads)
    
    logger.info(f"🚀 CPU优化: 设置线程数={optimal_threads} (总CPU核心数: {cpu_count})")
    logger.info(f"   OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    logger.info(f"   MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS')}")
    logger.info(f"   NUMEXPR_NUM_THREADS={os.environ.get('NUMEXPR_NUM_THREADS')}")
    
    return optimal_threads, cpu_count

class PreprocessedDataset(torch.utils.data.Dataset):
    """
    预处理数据集类 - 从HDF5文件加载预处理好的数据
    """
    
    def __init__(self, data_path: str):
        """
        初始化预处理数据集
        
        Args:
            data_path: 预处理数据文件路径
        """
        self.data_path = data_path
        
        # 加载数据
        with h5py.File(data_path, 'r') as f:
            self.inputs = torch.from_numpy(f['inputs'][:]).float()
            self.outputs = torch.from_numpy(f['outputs'][:]).float()
            
        logger.info(f"加载预处理数据: {data_path}")
        logger.info(f"输入形状: {self.inputs.shape}")
        logger.info(f"输出形状: {self.outputs.shape}")
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.outputs[idx]

def create_data_loaders(data_source: str, config: Dict[str, Any]):
    """
    创建数据加载器
    
    Args:
        data_source: 预处理数据目录路径或单个文件路径
        config: 配置字典
        
    Returns:
        tuple: (train_loader, valid_loader, test_loader)
    """
    from pathlib import Path
    
    data_path = Path(data_source)
    
    if data_path.is_dir():
        # 目录模式：分别加载训练、验证、测试数据
        train_path = data_path / 'train_data.h5'
        valid_path = data_path / 'valid_data.h5'
        test_path = data_path / 'test_data.h5'
        
        # 检查文件是否存在
        for file_path, name in [(train_path, '训练'), (valid_path, '验证'), (test_path, '测试')]:
            if not file_path.exists():
                raise FileNotFoundError(f"{name}数据文件不存在: {file_path}")
        
        # 创建数据集
        train_dataset = PreprocessedDataset(str(train_path))
        valid_dataset = PreprocessedDataset(str(valid_path))
        test_dataset = PreprocessedDataset(str(test_path))
        
        logger.info(f"从目录加载预处理数据: {data_source}")
        logger.info(f"训练集大小: {len(train_dataset)}")
        logger.info(f"验证集大小: {len(valid_dataset)}")
        logger.info(f"测试集大小: {len(test_dataset)}")
        
    else:
        # 文件模式：从单个文件分割数据集（兼容模式）
        dataset = PreprocessedDataset(str(data_path))
        
        # 数据集分割
        total_size = len(dataset)
        train_ratio = config['data'].get('train_ratio', 0.7)
        valid_ratio = config['data'].get('valid_ratio', 0.15)
        test_ratio = config['data'].get('test_ratio', 0.15)
        
        train_size = int(total_size * train_ratio)
        valid_size = int(total_size * valid_ratio)
        test_size = total_size - train_size - valid_size
        
        train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
            dataset, [train_size, valid_size, test_size]
        )
        
        logger.info(f"从文件加载并分割数据: {data_source}")
        logger.info(f"数据集分割: 训练集={train_size}, 验证集={valid_size}, 测试集={test_size}")
    
    # 从配置中获取数据加载器参数
    dataloader_config = config.get('dataloader', {})
    batch_size = config['data']['batch_size']
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
    
    return train_loader, valid_loader, test_loader

def load_default_config():
    """
    加载默认配置 - 与dynamic_resolution_trainer.py兼容
    """
    return {
        'data': {
            'batch_size': 16,
            'train_ratio': 0.7,
            'valid_ratio': 0.15,
            'test_ratio': 0.15
        },
        'model': {
            'input_dim': 1024,  # 32*32
            'output_dim': 16384,  # 128*128
            'num_heads': 8,
            'num_layers': 4,
            'd_model': 512,
            'max_time_steps': 100,
            'attention_type': 'sge',
            'seq_len': 1
        },
        'training': {
            'epochs': 10,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'patience': 15,
            'model_save_path': './results/models/simple_trainer_model.pth'
        },
        'optimizer': {
            'type': 'adam',
            'betas': [0.9, 0.999],
            'eps': 1e-8,
            'amsgrad': False
        },
        'scheduler': {
            'enabled': True,
            'type': 'cosine',
            'T_max': 10
        },
        'loss': {
            'base_weight': 0.8,
            'svd_weights': [0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
            'topk': 10,
            'svd_loss_enabled': True,
            'enhanced': {
                'enabled': True,
                'mixed_precision_mode': False,
                'adaptive_weights': True,
                'fallback_level': 2,
                'enable_monitoring': True,
                'adaptation_interval': 10,
                'force_svd': False
            }
        },
        'dataloader': {
            'num_workers': 0,
            'pin_memory': True,
            'drop_last': False,
            'persistent_workers': False,
            'prefetch_factor': 2
        },
        'mixed_precision': {
            'enabled': False,
            'loss_scale': 'dynamic'
        },
        'gradient': {
            'clip_enabled': False,
            'clip_value': 1.0,
            'accumulation_steps': 1
        },
        'visualization': {
            'enabled': True
        },
        'device': 'auto',
        'use_dataparallel': False,
        'max_memory_fraction': 0.8,
        'seed': 42
    }

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(
        description='简化训练器 - 使用预处理数据进行训练',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data_dir', type=str, required=True,
                       help='预处理数据目录路径')
    parser.add_argument('--data_path', type=str, default=None,
                       help='预处理数据文件路径（兼容性参数）')
    parser.add_argument('--config', type=str, default=None,
                       help='配置文件路径')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='学习率')
    parser.add_argument('--device', type=str, default=None,
                       help='设备 (cpu/cuda/auto)')
    parser.add_argument('--use_dataparallel', action='store_true',
                       help='启用多GPU训练')
    parser.add_argument('--resume', type=str,
                       help='恢复训练的检查点路径')
    
    return parser.parse_args()

def main():
    """
    主函数 - 使用与dynamic_resolution_trainer.py完全相同的训练逻辑
    """
    # 解析命令行参数
    args = parse_arguments()
    
    try:
        # 处理数据路径参数兼容性
        if args.data_path is not None:
            # 兼容旧的--data_path参数
            data_source = args.data_path
            logger.info(f"使用兼容模式: 数据文件路径 {data_source}")
        else:
            # 使用新的--data_dir参数
            data_source = args.data_dir
            logger.info(f"使用目录模式: 预处理数据目录 {data_source}")
        
        # 加载配置
        if args.config and os.path.exists(args.config):
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✅ 加载配置文件: {args.config}")
        else:
            config = load_default_config()
            logger.info("✅ 使用默认配置")
        
        # 命令行参数覆盖配置
        if args.epochs is not None:
            config['training']['epochs'] = args.epochs
        if args.batch_size is not None:
            config['data']['batch_size'] = args.batch_size
        if args.learning_rate is not None:
            config['training']['learning_rate'] = args.learning_rate
        if args.device is not None:
            config['device'] = args.device
        if args.use_dataparallel:
            config['use_dataparallel'] = True
        
        # 显示配置信息
        logger.info("=== 配置信息 ===")
        logger.info(f"输入维度: {config['model']['input_dim']}")
        logger.info(f"输出维度: {config['model']['output_dim']}")
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
        
        # 创建数据加载器
        logger.info("=== 创建数据加载器 ===")
        train_loader, valid_loader, test_loader = create_data_loaders(data_source, config)
        
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
        
        # 创建损失函数和优化器 - 与dynamic_resolution_trainer.py完全相同
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
        
        # 处理恢复训练
        start_epoch = 0
        if args.resume:
            if os.path.exists(args.resume):
                logger.info(f"🔄 恢复训练从检查点: {args.resume}")
                try:
                    checkpoint = torch.load(args.resume, map_location=device)
                    
                    # 加载模型状态
                    if hasattr(model, 'module'):  # DataParallel模型
                        model.module.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        model.load_state_dict(checkpoint['model_state_dict'])
                    
                    # 加载优化器状态
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    
                    # 加载调度器状态
                    if scheduler and 'scheduler_state_dict' in checkpoint:
                        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    
                    # 获取起始epoch
                    start_epoch = checkpoint.get('epoch', 0) + 1
                    
                    logger.info(f"✅ 成功恢复训练，从第 {start_epoch} 轮开始")
                    
                except Exception as e:
                    logger.error(f"❌ 恢复训练失败: {e}")
                    logger.info("将从头开始训练")
                    start_epoch = 0
            else:
                logger.warning(f"⚠️ 检查点文件不存在: {args.resume}")
                logger.info("将从头开始训练")
        
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
            loss_plot_path = results_dir / 'simple_trainer_losses.png'
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