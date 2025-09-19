#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
传感器数据测试脚本
演示如何使用传感器数据适配器进行稀疏观测到稠密重建

功能:
1. 加载传感器数据
2. 训练Custom Transformer模型
3. 评估重建性能
4. 可视化结果

作者: AI Assistant
日期: 2025
"""

import os
import sys
import torch
import numpy as np
import yaml
import logging
import argparse
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple
import time

# 设置项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

# 导入模块
from data.sensor_dataset import SensorDataset, MultiModalSensorDataset
from mymodels.transformer import TransformerFlowReconstructionModel
from utils.config import load_config
from utils.loss import get_loss_function
from torch.utils.data import DataLoader, random_split

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SensorTransformerWrapper:
    """传感器数据专用Transformer包装器"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device(config['training']['device'] if torch.cuda.is_available() else 'cpu')
        
        # 数据配置
        self.data_config = config['data']
        self.sensor_config = self.data_config['sensor_config']
        self.target_resolution = tuple(self.data_config['target_resolution'])
        
        # 模型配置
        self.model_config = config['model']
        
        # 训练配置
        self.training_config = config['training']
        
        logger.info(f"传感器Transformer初始化:")
        logger.info(f"  传感器类型: {self.sensor_config['type']}")
        logger.info(f"  目标分辨率: {self.target_resolution}")
        logger.info(f"  设备: {self.device}")
    
    def create_dataset(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """创建数据集和数据加载器"""
        logger.info("创建传感器数据集...")
        
        # 检查是否为多模态
        if self.config.get('multi_modal', {}).get('enabled', False):
            dataset = MultiModalSensorDataset(
                data_paths=self.config['multi_modal']['data_paths'],
                sensor_configs=self.config['multi_modal']['sensor_configs'],
                fusion_mode=self.config['multi_modal']['fusion_mode'],
                target_resolution=self.target_resolution,
                normalize_data=self.data_config['normalize_data'],
                max_samples=self.data_config.get('max_samples')
            )
        else:
            dataset = SensorDataset(
                data_path=self.data_config['data_path'],
                sensor_config=self.sensor_config,
                target_resolution=self.target_resolution,
                normalize_data=self.data_config['normalize_data'],
                max_samples=self.data_config.get('max_samples')
            )
        
        # 数据集分割
        total_size = len(dataset)
        train_size = int(self.data_config['train_ratio'] * total_size)
        valid_size = int(self.data_config['valid_ratio'] * total_size)
        test_size = total_size - train_size - valid_size
        
        train_dataset, valid_dataset, test_dataset = random_split(
            dataset, [train_size, valid_size, test_size]
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=self.data_config['shuffle'],
            num_workers=self.data_config['num_workers']
        )
        
        valid_loader = DataLoader(
            valid_dataset,
            batch_size=self.data_config['batch_size'],
            shuffle=False,
            num_workers=self.data_config['num_workers']
        )
        
        test_batch_size = self.config['evaluation'].get('test_batch_size', self.data_config['batch_size'])
        test_loader = DataLoader(
            test_dataset,
            batch_size=test_batch_size,
            shuffle=False,
            num_workers=self.data_config['num_workers']
        )
        
        # 更新模型配置中的维度信息
        sample_input, sample_target, _ = dataset[0]
        self.model_config['input_dim'] = sample_input.numel()
        self.model_config['output_dim'] = sample_target.numel()
        
        logger.info(f"数据集创建完成:")
        logger.info(f"  训练集: {len(train_dataset)} 样本")
        logger.info(f"  验证集: {len(valid_dataset)} 样本")
        logger.info(f"  测试集: {len(test_dataset)} 样本")
        logger.info(f"  输入维度: {self.model_config['input_dim']}")
        logger.info(f"  输出维度: {self.model_config['output_dim']}")
        
        return train_loader, valid_loader, test_loader
    
    def create_model(self) -> TransformerFlowReconstructionModel:
        """创建Transformer模型"""
        logger.info("创建Transformer模型...")
        
        # 推断序列长度
        seq_len = int(np.sqrt(self.model_config['input_dim']))
        if seq_len * seq_len != self.model_config['input_dim']:
            seq_len = self.model_config['input_dim']  # 1D序列
        
        model = TransformerFlowReconstructionModel(
            input_dim=self.model_config['input_dim'],
            output_dim=self.model_config['output_dim'],
            d_model=self.model_config['d_model'],
            num_heads=self.model_config['num_heads'],
            num_layers=self.model_config.get('num_layers', 6),
            seq_len=seq_len,
            attention_type=self.model_config['attention_type'],
            pe_type=self.model_config.get('position_encoding', 'learnable_1d'),
            output_head_type=self.model_config['output_head_type']
        ).to(self.device)
        
        # 计算模型参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"模型创建完成:")
        logger.info(f"  总参数: {total_params:,}")
        logger.info(f"  可训练参数: {trainable_params:,}")
        logger.info(f"  序列长度: {seq_len}")
        
        return model
    
    def train_model(self, model: TransformerFlowReconstructionModel, 
                   train_loader: DataLoader, valid_loader: DataLoader) -> Dict[str, list]:
        """训练模型"""
        logger.info("开始训练模型...")
        
        # 优化器
        learning_rate = float(self.training_config['learning_rate'])
        weight_decay = float(self.training_config.get('weight_decay', 1e-5))
        
        if self.training_config['optimizer'] == 'adamw':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=learning_rate
            )
        
        # 学习率调度器
        epochs = int(self.training_config['epochs'])
        if self.training_config.get('scheduler') == 'cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs
            )
        else:
            scheduler = None
        
        # 损失函数
        criterion = get_loss_function(self.training_config['loss_function'])
        
        # 训练历史
        history = {
            'train_loss': [],
            'valid_loss': [],
            'learning_rate': []
        }
        
        best_valid_loss = float('inf')
        patience_counter = 0
        early_stopping_patience = int(self.training_config.get('early_stopping_patience', 20))
        
        for epoch in range(epochs):
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            for batch_idx, (inputs, targets, time_steps) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                time_steps = time_steps.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model(inputs, time_steps)
                loss = criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
            
            avg_train_loss = train_loss / train_batches
            
            # 验证阶段
            model.eval()
            valid_loss = 0.0
            valid_batches = 0
            
            with torch.no_grad():
                for inputs, targets, time_steps in valid_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    time_steps = time_steps.to(self.device)
                    
                    outputs = model(inputs, time_steps)
                    loss = criterion(outputs, targets)
                    
                    valid_loss += loss.item()
                    valid_batches += 1
            
            avg_valid_loss = valid_loss / valid_batches
            
            # 更新学习率
            if scheduler:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = self.training_config['learning_rate']
            
            # 记录历史
            history['train_loss'].append(avg_train_loss)
            history['valid_loss'].append(avg_valid_loss)
            history['learning_rate'].append(current_lr)
            
            # 早停检查
            if avg_valid_loss < best_valid_loss:
                best_valid_loss = avg_valid_loss
                patience_counter = 0
                
                # 保存最佳模型
                if self.training_config['save_checkpoint']:
                    torch.save(model.state_dict(), 'best_sensor_model.pth')
            else:
                patience_counter += 1
            
            # 打印进度
            if epoch % 10 == 0 or epoch == self.training_config['epochs'] - 1:
                logger.info(f"Epoch {epoch+1}/{self.training_config['epochs']}: "
                          f"Train Loss: {avg_train_loss:.6f}, "
                          f"Valid Loss: {avg_valid_loss:.6f}, "
                          f"LR: {current_lr:.2e}")
            
            # 早停
            if patience_counter >= early_stopping_patience:
                logger.info(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
        
        logger.info(f"训练完成，最佳验证损失: {best_valid_loss:.6f}")
        return history
    
    def evaluate_model(self, model: TransformerFlowReconstructionModel, 
                      test_loader: DataLoader) -> Dict[str, float]:
        """评估模型"""
        logger.info("评估模型性能...")
        
        model.eval()
        
        # 评估指标
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0
        
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets, time_steps in test_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                time_steps = time_steps.to(self.device)
                
                outputs = model(inputs, time_steps)
                
                # 计算指标
                mse = torch.nn.functional.mse_loss(outputs, targets, reduction='sum')
                mae = torch.nn.functional.l1_loss(outputs, targets, reduction='sum')
                
                total_mse += mse.item()
                total_mae += mae.item()
                total_samples += targets.numel()
                
                # 收集预测结果
                all_predictions.append(outputs.cpu())
                all_targets.append(targets.cpu())
        
        # 计算平均指标
        avg_mse = total_mse / total_samples
        avg_mae = total_mae / total_samples
        
        # 计算R²和相关系数
        all_predictions = torch.cat(all_predictions, dim=0).view(-1)
        all_targets = torch.cat(all_targets, dim=0).view(-1)
        
        # R²
        ss_res = torch.sum((all_targets - all_predictions) ** 2)
        ss_tot = torch.sum((all_targets - torch.mean(all_targets)) ** 2)
        r2 = 1 - ss_res / ss_tot
        
        # 相关系数
        correlation = torch.corrcoef(torch.stack([all_predictions, all_targets]))[0, 1]
        
        metrics = {
            'mse': avg_mse,
            'mae': avg_mae,
            'rmse': np.sqrt(avg_mse),
            'r2': r2.item(),
            'correlation': correlation.item()
        }
        
        logger.info("评估结果:")
        for metric, value in metrics.items():
            logger.info(f"  {metric.upper()}: {value:.6f}")
        
        return metrics
    
    def visualize_results(self, model: TransformerFlowReconstructionModel, 
                         test_loader: DataLoader, num_samples: int = 4):
        """可视化重建结果"""
        logger.info("生成可视化结果...")
        
        model.eval()
        
        # 获取测试样本
        test_iter = iter(test_loader)
        inputs, targets, time_steps = next(test_iter)
        
        inputs = inputs[:num_samples].to(self.device)
        targets = targets[:num_samples].to(self.device)
        time_steps = time_steps[:num_samples].to(self.device)
        
        with torch.no_grad():
            outputs = model(inputs, time_steps)
        
        # 转换为numpy
        inputs_np = inputs.cpu().numpy()
        targets_np = targets.cpu().numpy()
        outputs_np = outputs.cpu().numpy()
        
        # 重塑为2D图像（如果可能）
        H, W = self.target_resolution
        
        fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i in range(num_samples):
            # 输入（传感器数据可视化）
            if len(inputs_np[i]) == H * W:
                input_img = inputs_np[i].reshape(H, W)
            else:
                # 如果输入不是图像格式，显示为1D信号
                input_img = np.zeros((H, W))
                input_img[0, :len(inputs_np[i])] = inputs_np[i][:W]
            
            # 目标和预测
            target_img = targets_np[i].reshape(H, W)
            output_img = outputs_np[i].reshape(H, W)
            
            # 绘制
            im1 = axes[i, 0].imshow(input_img, cmap='viridis')
            axes[i, 0].set_title(f'Sample {i+1}: Input (Sensors)')
            axes[i, 0].axis('off')
            plt.colorbar(im1, ax=axes[i, 0])
            
            im2 = axes[i, 1].imshow(target_img, cmap='viridis')
            axes[i, 1].set_title(f'Sample {i+1}: Target')
            axes[i, 1].axis('off')
            plt.colorbar(im2, ax=axes[i, 1])
            
            im3 = axes[i, 2].imshow(output_img, cmap='viridis')
            axes[i, 2].set_title(f'Sample {i+1}: Prediction')
            axes[i, 2].axis('off')
            plt.colorbar(im3, ax=axes[i, 2])
        
        plt.tight_layout()
        plt.savefig('sensor_reconstruction_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info("可视化结果已保存为 sensor_reconstruction_results.png")


def create_synthetic_sensor_data(output_path: str, config: Dict[str, Any]):
    """创建合成传感器数据用于测试"""
    logger.info("创建合成传感器数据...")
    
    sensor_config = config['data']['sensor_config']
    target_resolution = config['data']['target_resolution']
    num_samples = config['data'].get('max_samples', 1000)
    
    H, W = target_resolution
    
    if sensor_config['type'] == 'time_series':
        # 时间序列传感器数据
        num_sensors = sensor_config['num_sensors']
        temporal_length = sensor_config['temporal_length']
        
        # 生成合成数据
        sensor_data = np.random.randn(num_samples, temporal_length, num_sensors)
        target_data = np.random.randn(num_samples, H, W)
        
        # 添加一些相关性
        for i in range(num_samples):
            # 传感器数据的平均值影响目标场
            sensor_mean = sensor_data[i].mean(axis=0)
            for j, mean_val in enumerate(sensor_mean):
                x, y = j % H, (j * 7) % W
                target_data[i, x, y] += mean_val * 0.5
    
    elif sensor_config['type'] == 'spatial_sparse':
        # 空间稀疏传感器数据
        num_sensors = sensor_config['num_sensors']
        
        # 随机传感器位置
        positions = np.random.rand(num_sensors, 2)
        
        # 传感器读数
        sensor_readings = np.random.randn(num_samples, num_sensors)
        target_data = np.random.randn(num_samples, H, W)
        
        # 添加空间相关性
        for i in range(num_samples):
            for j, (x_pos, y_pos) in enumerate(positions):
                x_idx = int(x_pos * H)
                y_idx = int(y_pos * W)
                target_data[i, x_idx, y_idx] += sensor_readings[i, j] * 0.8
    
    # 保存数据
    import h5py
    with h5py.File(output_path, 'w') as f:
        if sensor_config['type'] == 'time_series':
            f.create_dataset('sensor_readings', data=sensor_data)
        elif sensor_config['type'] == 'spatial_sparse':
            f.create_dataset('sensor_positions', data=positions)
            f.create_dataset('sensor_readings', data=sensor_readings)
        
        f.create_dataset('target_field', data=target_data)
    
    logger.info(f"合成传感器数据已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='传感器数据测试脚本')
    parser.add_argument('--config', type=str, default='configs/sensor_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--create_data', action='store_true',
                       help='创建合成测试数据')
    parser.add_argument('--data_path', type=str, default='synthetic_sensor_data.h5',
                       help='数据文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新数据路径
    config['data']['data_path'] = args.data_path
    
    # 创建合成数据（如果需要）
    if args.create_data:
        create_synthetic_sensor_data(args.data_path, config)
    
    # 创建传感器Transformer包装器
    sensor_transformer = SensorTransformerWrapper(config)
    
    # 创建数据集
    train_loader, valid_loader, test_loader = sensor_transformer.create_dataset()
    
    # 创建模型
    model = sensor_transformer.create_model()
    
    # 训练模型
    history = sensor_transformer.train_model(model, train_loader, valid_loader)
    
    # 评估模型
    metrics = sensor_transformer.evaluate_model(model, test_loader)
    
    # 可视化结果
    sensor_transformer.visualize_results(model, test_loader)
    
    logger.info("传感器数据测试完成！")


if __name__ == "__main__":
    main()