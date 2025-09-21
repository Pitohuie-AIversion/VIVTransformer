#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
归一化集成示例

展示如何在现有的 run_crop_model_test.py 中集成归一化功能

核心修改点:
1. 数据加载器中添加归一化处理
2. 训练过程中记录归一化统计信息
3. 模型评估时考虑归一化影响
4. 可视化归一化效果

作者: AI Assistant
日期: 2025
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)

class NormalizationHandler:
    """归一化处理器"""
    
    def __init__(self, method='minmax', target_range=(0, 1)):
        """
        初始化归一化处理器
        
        Args:
            method: 归一化方法 ('minmax', 'zscore', 'robust', 'none')
            target_range: 目标范围 (仅对minmax有效)
        """
        self.method = method
        self.target_range = target_range
        self.input_stats = {}
        self.output_stats = {}
        self.fitted = False
    
    def fit(self, inputs: torch.Tensor, outputs: torch.Tensor):
        """
        拟合归一化参数
        
        Args:
            inputs: 输入数据 (N, input_dim)
            outputs: 输出数据 (N, output_dim)
        """
        logger.info(f"拟合归一化参数，方法: {self.method}")
        
        if self.method == 'none':
            self.fitted = True
            return
        
        # 计算输入统计信息
        self.input_stats = self._compute_stats(inputs)
        self.output_stats = self._compute_stats(outputs)
        
        logger.info(f"输入数据统计: 均值={self.input_stats['mean']:.6f}, "
                   f"标准差={self.input_stats['std']:.6f}, "
                   f"范围=[{self.input_stats['min']:.6f}, {self.input_stats['max']:.6f}]")
        
        logger.info(f"输出数据统计: 均值={self.output_stats['mean']:.6f}, "
                   f"标准差={self.output_stats['std']:.6f}, "
                   f"范围=[{self.output_stats['min']:.6f}, {self.output_stats['max']:.6f}]")
        
        self.fitted = True
    
    def _compute_stats(self, data: torch.Tensor) -> Dict[str, float]:
        """计算数据统计信息"""
        stats = {
            'mean': data.mean().item(),
            'std': data.std().item(),
            'min': data.min().item(),
            'max': data.max().item(),
            'median': data.median().item()
        }
        
        if self.method == 'robust':
            stats['q25'] = data.quantile(0.25).item()
            stats['q75'] = data.quantile(0.75).item()
            stats['iqr'] = stats['q75'] - stats['q25']
        
        return stats
    
    def transform_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """归一化输入数据"""
        if not self.fitted or self.method == 'none':
            return inputs
        
        return self._apply_normalization(inputs, self.input_stats)
    
    def transform_outputs(self, outputs: torch.Tensor) -> torch.Tensor:
        """归一化输出数据"""
        if not self.fitted or self.method == 'none':
            return outputs
        
        return self._apply_normalization(outputs, self.output_stats)
    
    def inverse_transform_outputs(self, normalized_outputs: torch.Tensor) -> torch.Tensor:
        """反归一化输出数据"""
        if not self.fitted or self.method == 'none':
            return normalized_outputs
        
        return self._apply_inverse_normalization(normalized_outputs, self.output_stats)
    
    def _apply_normalization(self, data: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
        """应用归一化"""
        if self.method == 'minmax':
            data_range = stats['max'] - stats['min']
            if data_range == 0:
                return data
            
            # 归一化到[0, 1]
            normalized = (data - stats['min']) / data_range
            
            # 缩放到目标范围
            target_min, target_max = self.target_range
            normalized = normalized * (target_max - target_min) + target_min
            
            return normalized
        
        elif self.method == 'zscore':
            if stats['std'] == 0:
                return data
            return (data - stats['mean']) / stats['std']
        
        elif self.method == 'robust':
            if stats['iqr'] == 0:
                return data
            return (data - stats['median']) / stats['iqr']
        
        else:
            return data
    
    def _apply_inverse_normalization(self, data: torch.Tensor, stats: Dict[str, float]) -> torch.Tensor:
        """应用反归一化"""
        if self.method == 'minmax':
            target_min, target_max = self.target_range
            # 从目标范围还原到[0, 1]
            data_01 = (data - target_min) / (target_max - target_min)
            # 从[0, 1]还原到原始范围
            return data_01 * (stats['max'] - stats['min']) + stats['min']
        
        elif self.method == 'zscore':
            return data * stats['std'] + stats['mean']
        
        elif self.method == 'robust':
            return data * stats['iqr'] + stats['median']
        
        else:
            return data
    
    def get_stats(self) -> Dict[str, Any]:
        """获取归一化统计信息"""
        return {
            'method': self.method,
            'target_range': self.target_range,
            'input_stats': self.input_stats,
            'output_stats': self.output_stats,
            'fitted': self.fitted
        }

def create_normalized_dataloader(data_path: str, config: Dict[str, Any]) -> Tuple[Any, Any, Any, NormalizationHandler]:
    """
    创建带归一化的数据加载器
    
    这是对现有 create_crop_dataloader 函数的增强版本
    
    Args:
        data_path: 数据文件路径
        config: 配置字典
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, normalizer)
    """
    # 导入现有的数据加载器
    from data.crop_dataloader import create_crop_dataloader
    
    # 创建原始数据加载器
    train_loader, val_loader, test_loader = create_crop_dataloader(config)
    
    # 获取归一化配置
    normalize_config = config.get('data', {}).get('normalization', {})
    method = normalize_config.get('method', 'minmax')
    target_range = normalize_config.get('target_range', (0, 1))
    
    logger.info(f"创建归一化处理器: 方法={method}, 范围={target_range}")
    
    # 创建归一化处理器
    normalizer = NormalizationHandler(method=method, target_range=target_range)
    
    # 如果启用归一化，拟合参数
    if method != 'none':
        # 收集训练数据样本用于拟合归一化参数
        inputs_list = []
        outputs_list = []
        
        for i, (inputs, outputs) in enumerate(train_loader):
            inputs_list.append(inputs)
            outputs_list.append(outputs)
            if i >= 10:  # 限制样本数量以节省内存
                break
        
        all_inputs = torch.cat(inputs_list, dim=0)
        all_outputs = torch.cat(outputs_list, dim=0)
        
        # 拟合归一化参数
        normalizer.fit(all_inputs, all_outputs)
        
        # 创建归一化的数据加载器
        train_loader = NormalizedDataLoader(train_loader, normalizer)
        val_loader = NormalizedDataLoader(val_loader, normalizer)
        test_loader = NormalizedDataLoader(test_loader, normalizer)
    
    return train_loader, val_loader, test_loader, normalizer

class NormalizedDataLoader:
    """归一化数据加载器包装器"""
    
    def __init__(self, dataloader, normalizer: NormalizationHandler):
        """
        初始化归一化数据加载器
        
        Args:
            dataloader: 原始数据加载器
            normalizer: 归一化处理器
        """
        self.dataloader = dataloader
        self.normalizer = normalizer
    
    def __iter__(self):
        """迭代器"""
        for inputs, outputs in self.dataloader:
            # 归一化输入和输出
            normalized_inputs = self.normalizer.transform_inputs(inputs)
            normalized_outputs = self.normalizer.transform_outputs(outputs)
            yield normalized_inputs, normalized_outputs
    
    def __len__(self):
        """数据长度"""
        return len(self.dataloader)
    
    @property
    def dataset(self):
        """数据集属性"""
        return self.dataloader.dataset
    
    @property
    def batch_size(self):
        """批次大小属性"""
        return self.dataloader.batch_size

def train_model_with_normalization(model, train_loader, val_loader, config, device, normalizer: NormalizationHandler):
    """
    带归一化的模型训练函数
    
    这是对现有 train_model 函数的增强版本
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        config: 配置字典
        device: 设备
        normalizer: 归一化处理器
        
    Returns:
        dict: 训练结果，包含归一化信息
    """
    # 获取训练配置
    training_config = config.get('training', {})
    epochs = training_config.get('epochs', 50)
    learning_rate = training_config.get('learning_rate', 0.001)
    
    # 创建优化器和损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    # 训练历史
    train_losses = []
    val_losses = []
    
    logger.info(f"开始训练，归一化方法: {normalizer.method}")
    
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs}, 训练损失: {train_loss:.6f}, 验证损失: {val_loss:.6f}")
    
    # 返回训练结果
    result = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'normalization_stats': normalizer.get_stats()
    }
    
    logger.info(f"训练完成，最终训练损失: {train_losses[-1]:.6f}, 最终验证损失: {val_losses[-1]:.6f}")
    
    return result

def evaluate_model_with_normalization(model, test_loader, device, normalizer: NormalizationHandler):
    """
    带归一化的模型评估函数
    
    这是对现有 evaluate_model 函数的增强版本
    
    Args:
        model: 模型
        test_loader: 测试数据加载器
        device: 设备
        normalizer: 归一化处理器
        
    Returns:
        dict: 评估结果，包含原始尺度和归一化尺度的指标
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_predictions_original = []
    all_targets_original = []
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 模型预测（归一化尺度）
            predictions = model(inputs)
            
            # 反归一化到原始尺度
            predictions_original = normalizer.inverse_transform_outputs(predictions)
            targets_original = normalizer.inverse_transform_outputs(targets)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
            all_predictions_original.append(predictions_original.cpu())
            all_targets_original.append(targets_original.cpu())
    
    # 合并所有预测和目标
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_predictions_original = torch.cat(all_predictions_original, dim=0)
    all_targets_original = torch.cat(all_targets_original, dim=0)
    
    # 计算归一化尺度的指标
    mse_normalized = torch.mean((all_predictions - all_targets) ** 2).item()
    mae_normalized = torch.mean(torch.abs(all_predictions - all_targets)).item()
    
    # 计算R²（归一化尺度）
    ss_res_normalized = torch.sum((all_targets - all_predictions) ** 2).item()
    ss_tot_normalized = torch.sum((all_targets - torch.mean(all_targets)) ** 2).item()
    r2_normalized = 1 - (ss_res_normalized / ss_tot_normalized) if ss_tot_normalized != 0 else 0
    
    # 计算原始尺度的指标
    mse_original = torch.mean((all_predictions_original - all_targets_original) ** 2).item()
    mae_original = torch.mean(torch.abs(all_predictions_original - all_targets_original)).item()
    
    # 计算R²（原始尺度）
    ss_res_original = torch.sum((all_targets_original - all_predictions_original) ** 2).item()
    ss_tot_original = torch.sum((all_targets_original - torch.mean(all_targets_original)) ** 2).item()
    r2_original = 1 - (ss_res_original / ss_tot_original) if ss_tot_original != 0 else 0
    
    result = {
        # 归一化尺度指标
        'mse_normalized': mse_normalized,
        'mae_normalized': mae_normalized,
        'r2_normalized': r2_normalized,
        
        # 原始尺度指标
        'mse_original': mse_original,
        'mae_original': mae_original,
        'r2_original': r2_original,
        
        # 主要指标（通常使用原始尺度）
        'mse': mse_original,
        'mae': mae_original,
        'r2': r2_original,
        
        # 归一化信息
        'normalization_method': normalizer.method,
        'normalization_stats': normalizer.get_stats()
    }
    
    logger.info(f"评估完成:")
    logger.info(f"  原始尺度 - MSE: {mse_original:.6f}, MAE: {mae_original:.6f}, R²: {r2_original:.6f}")
    logger.info(f"  归一化尺度 - MSE: {mse_normalized:.6f}, MAE: {mae_normalized:.6f}, R²: {r2_normalized:.6f}")
    
    return result

# 示例配置文件格式
EXAMPLE_CONFIG = {
    "data": {
        "data_path": "data/2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_Train.hdf5",
        "batch_size": 32,
        "num_samples": 100,
        "num_workers": 0,
        
        # 归一化配置
        "normalization": {
            "method": "minmax",      # 'minmax', 'zscore', 'robust', 'none'
            "target_range": [0, 1]   # 仅对minmax有效
        }
    },
    
    "training": {
        "epochs": 50,
        "learning_rate": 0.001
    },
    
    "models": {
        "mlp": {
            "model_type": "mlp",
            "hidden_dims": [512, 256, 128],
            "activation": "relu",
            "dropout": 0.1
        }
    }
}

def main_example():
    """主函数示例"""
    # 这个函数展示如何在现有的 run_crop_model_test.py 中集成归一化功能
    
    config = EXAMPLE_CONFIG
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. 创建带归一化的数据加载器
    data_path = config['data']['data_path']
    train_loader, val_loader, test_loader, normalizer = create_normalized_dataloader(data_path, config)
    
    # 2. 获取数据维度
    sample_input, sample_output = next(iter(train_loader))
    input_dim = sample_input.shape[1]
    output_dim = sample_output.shape[1]
    
    # 3. 创建模型（这里使用简单的MLP作为示例）
    from run_crop_model_test import SimpleMLP
    model = SimpleMLP(input_dim, output_dim, hidden_dims=[512, 256, 128])
    model = model.to(device)
    
    # 4. 训练模型
    train_result = train_model_with_normalization(model, train_loader, val_loader, config, device, normalizer)
    
    # 5. 评估模型
    eval_result = evaluate_model_with_normalization(model, test_loader, device, normalizer)
    
    # 6. 输出结果
    print(f"训练完成!")
    print(f"归一化方法: {normalizer.method}")
    print(f"最终训练损失: {train_result['final_train_loss']:.6f}")
    print(f"最终验证损失: {train_result['final_val_loss']:.6f}")
    print(f"测试MSE (原始尺度): {eval_result['mse_original']:.6f}")
    print(f"测试R² (原始尺度): {eval_result['r2_original']:.6f}")

if __name__ == "__main__":
    main_example()