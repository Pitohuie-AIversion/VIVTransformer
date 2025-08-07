#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化训练器 - 专门用于加载预处理数据进行训练

功能:
1. 直接加载预处理好的HDF5数据文件
2. 避免训练过程中的数据处理，减少CPU占用
3. 专注于模型训练和优化
4. 支持断点续训和模型保存

作者: AI Assistant
日期: 2025
"""

# 设置无头模式环境变量
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''
os.environ['HEADLESS'] = '1'

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import yaml
import argparse
import logging
import json
import time
import gc
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
from tqdm import tqdm

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

# 导入模型和工具
try:
    from models.viv_transformer import VIVTransformer
    from utils.loss_functions import CombinedLoss
    from utils.metrics import calculate_metrics
    from utils.visualization import save_training_plots
    HAS_MODULES = True
    logger.info("✅ 成功导入模型和工具模块")
except ImportError as e:
    HAS_MODULES = False
    logger.warning(f"⚠️ 模块导入失败: {e}")
    logger.warning("将使用基础模块")

class PreprocessedDataset(Dataset):
    """
    预处理数据集类
    直接加载预处理好的HDF5数据
    """
    
    def __init__(self, data_path: str, device: torch.device = None):
        """
        初始化数据集
        
        Args:
            data_path: 预处理数据文件路径
            device: 设备
        """
        self.data_path = data_path
        self.device = device or torch.device('cpu')
        
        # 加载数据
        logger.info(f"加载预处理数据: {data_path}")
        with h5py.File(data_path, 'r') as f:
            self.input_data = torch.from_numpy(np.array(f['input'])).float()
            self.output_data = torch.from_numpy(np.array(f['output'])).float()
        
        logger.info(f"数据加载完成:")
        logger.info(f"  输入数据形状: {self.input_data.shape}")
        logger.info(f"  输出数据形状: {self.output_data.shape}")
        logger.info(f"  样本数量: {len(self.input_data)}")
    
    def __len__(self) -> int:
        return len(self.input_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            (输入数据, 输出数据)
        """
        input_sample = self.input_data[idx]
        output_sample = self.output_data[idx]
        
        # 添加通道维度
        if len(input_sample.shape) == 2:
            input_sample = input_sample.unsqueeze(0)
        if len(output_sample.shape) == 2:
            output_sample = output_sample.unsqueeze(0)
        
        return input_sample, output_sample

class SimpleTrainer:
    """
    简化训练器类
    专门用于训练预处理好的数据
    """
    
    def __init__(self, config: Dict[str, Any], data_dir: str):
        """
        初始化训练器
        
        Args:
            config: 配置字典
            data_dir: 预处理数据目录
        """
        self.config = config
        self.data_dir = Path(data_dir)
        
        # 训练配置
        self.training_config = config['training']
        self.model_config = config['model']
        
        # 设备配置
        self.setup_device()
        
        # 加载数据
        self.setup_data()
        
        # 创建模型
        self.setup_model()
        
        # 设置优化器和损失函数
        self.setup_optimizer()
        self.setup_loss_function()
        
        # 训练状态
        self.current_epoch = 0
        self.best_loss = float('inf')
        self.train_losses = []
        self.valid_losses = []
        
        logger.info("✅ 简化训练器初始化完成")
    
    def setup_device(self):
        """
        设置计算设备
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            
            # GPU内存限制
            gpu_memory_limit = self.config.get('gpu_memory_limit_gb')
            if gpu_memory_limit:
                torch.cuda.set_per_process_memory_fraction(gpu_memory_limit / torch.cuda.get_device_properties(0).total_memory * 1024**3)
                logger.info(f"设置GPU内存限制: {gpu_memory_limit}GB")
            
            logger.info(f"使用GPU: {torch.cuda.get_device_name()}")
            logger.info(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
        else:
            self.device = torch.device('cpu')
            logger.info("使用CPU训练")
    
    def setup_data(self):
        """
        设置数据加载器
        """
        # 数据路径
        train_path = self.data_dir / 'train_data.h5'
        valid_path = self.data_dir / 'valid_data.h5'
        test_path = self.data_dir / 'test_data.h5'
        
        # 检查文件存在性
        for path in [train_path, valid_path, test_path]:
            if not path.exists():
                raise FileNotFoundError(f"预处理数据文件不存在: {path}")
        
        # 创建数据集
        self.train_dataset = PreprocessedDataset(str(train_path), self.device)
        self.valid_dataset = PreprocessedDataset(str(valid_path), self.device)
        self.test_dataset = PreprocessedDataset(str(test_path), self.device)
        
        # 数据加载器配置
        dataloader_config = self.config.get('dataloader', {})
        batch_size = dataloader_config.get('batch_size', 8)
        num_workers = min(dataloader_config.get('num_workers', 2), 2)  # 限制worker数量
        pin_memory = dataloader_config.get('pin_memory', False)  # 关闭pin_memory减少内存使用
        
        # 创建数据加载器
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )
        
        logger.info(f"数据加载器设置完成:")
        logger.info(f"  批次大小: {batch_size}")
        logger.info(f"  工作进程: {num_workers}")
        logger.info(f"  训练批次: {len(self.train_loader)}")
        logger.info(f"  验证批次: {len(self.valid_loader)}")
        logger.info(f"  测试批次: {len(self.test_loader)}")
    
    def setup_model(self):
        """
        设置模型
        """
        # 获取数据形状信息
        sample_input, sample_output = self.train_dataset[0]
        input_shape = sample_input.shape[1:]  # 去除batch维度
        output_shape = sample_output.shape[1:]  # 去除batch维度
        
        logger.info(f"模型输入形状: {input_shape}")
        logger.info(f"模型输出形状: {output_shape}")
        
        if HAS_MODULES:
            # 使用VIVTransformer模型
            self.model = VIVTransformer(
                input_resolution=input_shape,
                output_resolution=output_shape,
                **self.model_config
            )
        else:
            # 使用简单的CNN模型作为备选
            self.model = self.create_simple_cnn(input_shape, output_shape)
        
        self.model = self.model.to(self.device)
        
        # 多GPU支持
        if torch.cuda.device_count() > 1:
            logger.info(f"使用 {torch.cuda.device_count()} 个GPU进行训练")
            self.model = nn.DataParallel(self.model)
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"模型参数统计:")
        logger.info(f"  总参数: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
    
    def create_simple_cnn(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> nn.Module:
        """
        创建简单的CNN模型作为备选
        
        Args:
            input_shape: 输入形状
            output_shape: 输出形状
            
        Returns:
            CNN模型
        """
        class SimpleCNN(nn.Module):
            def __init__(self, input_channels, output_channels, hidden_dim=64):
                super().__init__()
                self.encoder = nn.Sequential(
                    nn.Conv2d(input_channels, hidden_dim, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, hidden_dim * 2, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim * 2, hidden_dim, 3, padding=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, output_channels, 3, padding=1)
                )
            
            def forward(self, x):
                return self.encoder(x)
        
        input_channels = input_shape[0] if len(input_shape) == 3 else 1
        output_channels = output_shape[0] if len(output_shape) == 3 else 1
        
        logger.info(f"使用简单CNN模型: {input_channels} -> {output_channels} 通道")
        return SimpleCNN(input_channels, output_channels)
    
    def setup_optimizer(self):
        """
        设置优化器
        """
        optimizer_config = self.training_config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam').lower()
        learning_rate = optimizer_config.get('learning_rate', 1e-4)
        weight_decay = optimizer_config.get('weight_decay', 1e-5)
        
        if optimizer_type == 'adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            logger.warning(f"未知的优化器类型: {optimizer_type}，使用Adam")
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # 学习率调度器
        scheduler_config = self.training_config.get('scheduler', {})
        if scheduler_config.get('enabled', False):
            scheduler_type = scheduler_config.get('type', 'step').lower()
            if scheduler_type == 'step':
                self.scheduler = optim.lr_scheduler.StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 10),
                    gamma=scheduler_config.get('gamma', 0.1)
                )
            elif scheduler_type == 'cosine':
                self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer,
                    T_max=self.training_config.get('epochs', 100)
                )
            else:
                self.scheduler = None
        else:
            self.scheduler = None
        
        logger.info(f"优化器设置: {optimizer_type.upper()}, 学习率: {learning_rate}")
        if self.scheduler:
            logger.info(f"学习率调度器: {scheduler_config.get('type', 'step')}")
    
    def setup_loss_function(self):
        """
        设置损失函数
        """
        loss_config = self.training_config.get('loss', {})
        loss_type = loss_config.get('type', 'mse').lower()
        
        if HAS_MODULES and loss_type == 'combined':
            self.criterion = CombinedLoss(**loss_config)
        else:
            if loss_type == 'mse':
                self.criterion = nn.MSELoss()
            elif loss_type == 'l1':
                self.criterion = nn.L1Loss()
            elif loss_type == 'huber':
                self.criterion = nn.SmoothL1Loss()
            else:
                logger.warning(f"未知的损失函数类型: {loss_type}，使用MSE")
                self.criterion = nn.MSELoss()
        
        logger.info(f"损失函数: {loss_type.upper()}")
    
    def train_epoch(self) -> float:
        """
        训练一个epoch
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            # 数据移到设备
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # 计算损失
            loss = self.criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            
            # 梯度裁剪
            grad_clip = self.training_config.get('gradient_clip', 1.0)
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip)
            
            # 优化器步进
            self.optimizer.step()
            
            # 累计损失
            total_loss += loss.item()
            
            # 更新进度条
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Avg': f'{total_loss / (batch_idx + 1):.6f}'
            })
            
            # 内存清理
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self) -> float:
        """
        验证模型
        
        Returns:
            平均验证损失
        """
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.valid_loader)
        
        with torch.no_grad():
            for inputs, targets in tqdm(self.valid_loader, desc="验证"):
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, filepath: str, is_best: bool = False):
        """
        保存检查点
        
        Args:
            filepath: 保存路径
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_loss': self.best_loss,
            'train_losses': self.train_losses,
            'valid_losses': self.valid_losses,
            'config': self.config
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = str(Path(filepath).parent / 'best_model.pth')
            torch.save(checkpoint, best_path)
            logger.info(f"💾 最佳模型已保存: {best_path}")
    
    def train(self):
        """
        执行完整的训练过程
        """
        epochs = self.training_config.get('epochs', 100)
        save_interval = self.training_config.get('save_interval', 10)
        
        logger.info(f"🚀 开始训练，总轮数: {epochs}")
        
        start_time = time.time()
        
        try:
            for epoch in range(epochs):
                self.current_epoch = epoch
                
                # 训练
                train_loss = self.train_epoch()
                self.train_losses.append(train_loss)
                
                # 验证
                valid_loss = self.validate()
                self.valid_losses.append(valid_loss)
                
                # 学习率调度
                if self.scheduler:
                    self.scheduler.step()
                
                # 记录日志
                logger.info(f"Epoch {epoch + 1}/{epochs}:")
                logger.info(f"  训练损失: {train_loss:.6f}")
                logger.info(f"  验证损失: {valid_loss:.6f}")
                if self.scheduler:
                    logger.info(f"  学习率: {self.scheduler.get_last_lr()[0]:.2e}")
                
                # 保存最佳模型
                is_best = valid_loss < self.best_loss
                if is_best:
                    self.best_loss = valid_loss
                    logger.info(f"🎉 发现更好的模型！验证损失: {valid_loss:.6f}")
                
                # 定期保存检查点
                if (epoch + 1) % save_interval == 0 or is_best:
                    checkpoint_path = f"checkpoint_epoch_{epoch + 1}.pth"
                    self.save_checkpoint(checkpoint_path, is_best)
                    logger.info(f"💾 检查点已保存: {checkpoint_path}")
                
                # 内存清理
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        except KeyboardInterrupt:
            logger.warning("⚠️ 训练被用户中断")
            # 保存当前状态
            self.save_checkpoint(f"interrupted_epoch_{self.current_epoch}.pth")
        
        end_time = time.time()
        training_time = end_time - start_time
        
        logger.info(f"🎉 训练完成！")
        logger.info(f"⏱️ 总训练时间: {training_time / 3600:.2f}小时")
        logger.info(f"🏆 最佳验证损失: {self.best_loss:.6f}")
        
        # 保存训练曲线
        if HAS_MODULES:
            try:
                save_training_plots(self.train_losses, self.valid_losses, 'training_curves.png')
                logger.info("📊 训练曲线已保存: training_curves.png")
            except Exception as e:
                logger.warning(f"保存训练曲线失败: {e}")

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='简化训练器')
    parser.add_argument('--config', type=str, default='dynamic_config_server_optimized_final.yaml',
                       help='配置文件路径')
    parser.add_argument('--data_dir', type=str, default='./preprocessed_data',
                       help='预处理数据目录')
    parser.add_argument('--resume', type=str,
                       help='恢复训练的检查点路径')
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    加载配置文件
    
    Args:
        config_path: 配置文件路径
        
    Returns:
        配置字典
    """
    logger.info(f"加载配置文件: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    logger.info("✅ 配置文件加载成功")
    return config

def main():
    """
    主函数
    """
    # 解析命令行参数
    args = parse_arguments()
    
    try:
        # 加载配置
        config = load_config(args.config)
        
        # 检查预处理数据目录
        data_dir = Path(args.data_dir)
        if not data_dir.exists():
            logger.error(f"❌ 预处理数据目录不存在: {data_dir}")
            logger.error("请先运行 data_preprocessor.py 生成预处理数据")
            return
        
        # 创建训练器
        trainer = SimpleTrainer(config, args.data_dir)
        
        # 恢复训练（如果指定）
        if args.resume:
            logger.info(f"恢复训练: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=trainer.device)
            trainer.model.load_state_dict(checkpoint['model_state_dict'])
            trainer.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            trainer.current_epoch = checkpoint['epoch']
            trainer.best_loss = checkpoint['best_loss']
            trainer.train_losses = checkpoint.get('train_losses', [])
            trainer.valid_losses = checkpoint.get('valid_losses', [])
            if trainer.scheduler and 'scheduler_state_dict' in checkpoint:
                trainer.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info(f"✅ 从第 {trainer.current_epoch} 轮恢复训练")
        
        # 开始训练
        trainer.train()
        
    except KeyboardInterrupt:
        logger.warning("⚠️ 程序被用户中断")
    except FileNotFoundError as e:
        logger.error(f"❌ 文件未找到: {str(e)}")
    except Exception as e:
        logger.error(f"❌ 训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()