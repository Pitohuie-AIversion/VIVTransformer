#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试输入32x32、输出128x128的模型效果

功能:
1. 加载训练好的模型
2. 测试模型预测效果
3. 可视化输入32x32和输出128x128的对比
4. 计算重建误差

作者: AI Assistant
日期: 2025
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'modify_multi_attention'))
sys.path.append(str(Path(__file__).parent))

from mymodels.transformer import TransformerFlowReconstructionModel
from train_with_processed_data import ProcessedPDEBenchDataset

# 设置中文字体
# 统一使用全局 sitecustomize.py 的中文字体和负号设置
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model(model_path, input_dim=1024, output_dim=16384, device='cpu'):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        input_dim: 输入维度
        output_dim: 输出维度
        device: 设备
    
    Returns:
        model: 加载的模型
    """
    logger.info(f"加载模型: {model_path}")
    
    # 创建模型
    model = TransformerFlowReconstructionModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_layers=4,
        d_model=512,
        num_heads=8,
        max_time_steps=100,
        seq_len=32,
        attention_type='sge'
    )
    
    # 加载权重
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    
    logger.info(f"模型加载完成，参数数量: {sum(p.numel() for p in model.parameters())}")
    return model

def load_test_data(data_path):
    """
    加载测试数据
    
    Args:
        data_path: 数据文件路径
    
    Returns:
        dataset: 数据集对象
    """
    logger.info(f"加载测试数据: {data_path}")
    dataset = ProcessedPDEBenchDataset(data_path)
    logger.info(f"数据集大小: {len(dataset)}")
    return dataset

def test_model_prediction(model, dataset, num_samples=5, device='cpu'):
    """
    测试模型预测效果
    
    Args:
        model: 训练好的模型
        dataset: 测试数据集
        num_samples: 测试样本数量
        device: 设备
    
    Returns:
        results: 预测结果字典
    """
    logger.info(f"测试模型预测，样本数量: {num_samples}")
    
    results = {
        'inputs': [],
        'targets': [],
        'predictions': [],
        'errors': []
    }
    
    with torch.no_grad():
        for i in range(min(num_samples, len(dataset))):
            input_data, target_data, time_step = dataset[i]
            
            # 添加批次维度
            input_batch = input_data.unsqueeze(0).to(device)  # [1, 1024]
            time_batch = time_step.unsqueeze(0).to(device)    # [1]
            target_batch = target_data.unsqueeze(0).to(device) # [1, 16384]
            
            # 模型预测
            prediction = model(input_batch, time_batch)  # [1, 16384]
            
            # 计算误差
            error = torch.mean((prediction - target_batch) ** 2).item()
            
            # 保存结果
            results['inputs'].append(input_data.cpu().numpy())
            results['targets'].append(target_data.cpu().numpy())
            results['predictions'].append(prediction.squeeze(0).cpu().numpy())
            results['errors'].append(error)
            
            logger.info(f"样本 {i+1}: MSE = {error:.6f}")
    
    avg_error = np.mean(results['errors'])
    logger.info(f"平均MSE: {avg_error:.6f}")
    
    return results

def visualize_results(results, save_path='test_results_visualization.png'):
    """
    可视化预测结果
    
    Args:
        results: 预测结果字典
        save_path: 保存路径
    """
    logger.info("创建可视化图表")
    
    num_samples = len(results['inputs'])
    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5*num_samples))
    
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # 重塑数据
        input_2d = results['inputs'][i].reshape(32, 32)
        target_2d = results['targets'][i].reshape(128, 128)
        pred_2d = results['predictions'][i].reshape(128, 128)
        
        # 输入 (32x32)
        im1 = axes[i, 0].imshow(input_2d, cmap='viridis', aspect='equal')
        axes[i, 0].set_title(f'输入 32x32 (样本 {i+1})')
        axes[i, 0].set_xlabel('X')
        axes[i, 0].set_ylabel('Y')
        plt.colorbar(im1, ax=axes[i, 0])
        
        # 目标输出 (128x128)
        im2 = axes[i, 1].imshow(target_2d, cmap='viridis', aspect='equal')
        axes[i, 1].set_title(f'目标输出 128x128 (样本 {i+1})')
        axes[i, 1].set_xlabel('X')
        axes[i, 1].set_ylabel('Y')
        plt.colorbar(im2, ax=axes[i, 1])
        
        # 预测输出 (128x128)
        im3 = axes[i, 2].imshow(pred_2d, cmap='viridis', aspect='equal')
        axes[i, 2].set_title(f'预测输出 128x128 (样本 {i+1})\nMSE: {results["errors"][i]:.6f}')
        axes[i, 2].set_xlabel('X')
        axes[i, 2].set_ylabel('Y')
        plt.colorbar(im3, ax=axes[i, 2])
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    logger.info(f"可视化结果保存到: {save_path}")
    plt.show()

def analyze_reconstruction_quality(results):
    """
    分析重建质量
    
    Args:
        results: 预测结果字典
    """
    logger.info("=== 重建质量分析 ===")
    
    errors = np.array(results['errors'])
    
    logger.info(f"样本数量: {len(errors)}")
    logger.info(f"平均MSE: {errors.mean():.6f}")
    logger.info(f"MSE标准差: {errors.std():.6f}")
    logger.info(f"最小MSE: {errors.min():.6f}")
    logger.info(f"最大MSE: {errors.max():.6f}")
    
    # 计算相对误差
    for i, (target, pred) in enumerate(zip(results['targets'], results['predictions'])):
        target_norm = np.linalg.norm(target)
        pred_norm = np.linalg.norm(pred)
        relative_error = np.linalg.norm(target - pred) / target_norm
        logger.info(f"样本 {i+1}: 相对误差 = {relative_error:.6f}, 目标范数 = {target_norm:.6f}, 预测范数 = {pred_norm:.6f}")

def main():
    """
    主函数
    """
    # 配置
    model_path = "training_results_sge/best_model_sge.pt"
    data_path = "input32_output128_dataset.npz"
    num_test_samples = 3
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    try:
        # 1. 加载模型
        logger.info("=== 步骤1: 加载模型 ===")
        model = load_model(model_path, device=device)
        
        # 2. 加载测试数据
        logger.info("=== 步骤2: 加载测试数据 ===")
        dataset = load_test_data(data_path)
        
        # 3. 测试模型预测
        logger.info("=== 步骤3: 测试模型预测 ===")
        results = test_model_prediction(model, dataset, num_test_samples, device)
        
        # 4. 分析重建质量
        logger.info("=== 步骤4: 分析重建质量 ===")
        analyze_reconstruction_quality(results)
        
        # 5. 可视化结果
        logger.info("=== 步骤5: 可视化结果 ===")
        visualize_results(results)
        
        logger.info("=== 测试完成 ===")
        
    except Exception as e:
        logger.error(f"测试过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()