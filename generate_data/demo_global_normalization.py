#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
全局归一化演示脚本

展示如何使用改进后的全局归一化功能：
1. 基于全局原始数据计算归一化参数
2. 确保输入输出使用相同的归一化标准
3. 保存和加载归一化信息
4. 反归一化恢复物理信息

作者: AI Assistant
日期: 2025-01-27
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))
from dynamic_resolution_trainer import DynamicResolutionDataset

def demo_global_normalization():
    """演示全局归一化功能"""
    print("=" * 60)
    print("全局归一化功能演示")
    print("=" * 60)
    
    # 1. 创建示例数据
    print("\n1. 创建示例数据...")
    
    # 创建一个简单的测试数据文件
    test_data_path = "test_global_norm_data.h5"
    
    # 生成测试数据：200x200的压力场数据
    np.random.seed(42)
    test_data = np.random.randn(50, 200, 200) * 10 + 100  # 均值100，标准差10
    
    import h5py
    with h5py.File(test_data_path, 'w') as f:
        f.create_dataset('pressure', data=test_data)
    
    print(f"测试数据形状: {test_data.shape}")
    print(f"原始数据范围: [{test_data.min():.3f}, {test_data.max():.3f}]")
    
    # 2. 创建数据集（输入20x20，输出128x128）
    print("\n2. 创建动态分辨率数据集...")
    
    dataset = DynamicResolutionDataset(
        data_path=test_data_path,
        input_resolution=(20, 20),
        output_resolution=(128, 128),
        num_samples=10,
        normalize_data=True
    )
    
    # 3. 查看归一化信息
    print("\n3. 归一化信息:")
    norm_info = dataset.get_normalization_info()
    for key, value in norm_info.items():
        print(f"  {key}: {value}")
    
    # 4. 获取样本数据
    print("\n4. 样本数据分析:")
    input_data, output_data, idx = dataset[0]
    
    print(f"输入数据形状: {input_data.shape}")
    print(f"输出数据形状: {output_data.shape}")
    print(f"输入数据范围: [{input_data.min():.6f}, {input_data.max():.6f}]")
    print(f"输出数据范围: [{output_data.min():.6f}, {output_data.max():.6f}]")
    
    # 5. 演示反归一化
    print("\n5. 反归一化演示:")
    
    # 反归一化输入数据
    denorm_input = dataset.denormalize_predictions(input_data)
    denorm_output = dataset.denormalize_predictions(output_data)
    
    print(f"反归一化后输入范围: [{denorm_input.min():.3f}, {denorm_input.max():.3f}]")
    print(f"反归一化后输出范围: [{denorm_output.min():.3f}, {denorm_output.max():.3f}]")
    
    # 6. 保存归一化信息
    print("\n6. 保存归一化信息...")
    
    norm_info_path = "normalization_info.json"
    dataset.save_normalization_info(norm_info_path)
    
    # 7. 加载归一化信息
    print("\n7. 加载归一化信息...")
    
    loaded_norm_info = DynamicResolutionDataset.load_normalization_info(norm_info_path)
    print("加载的归一化信息:")
    for key, value in loaded_norm_info.items():
        print(f"  {key}: {value}")
    
    # 8. 验证一致性
    print("\n8. 验证归一化一致性:")
    
    # 检查输入输出是否使用相同的归一化标准
    print(f"输入归一化参数: min={dataset.input_min:.6f}, max={dataset.input_max:.6f}")
    print(f"输出归一化参数: min={dataset.output_min:.6f}, max={dataset.output_max:.6f}")
    print(f"全局归一化参数: min={dataset.global_min:.6f}, max={dataset.global_max:.6f}")
    
    is_consistent = (dataset.input_min == dataset.output_min and 
                    dataset.input_max == dataset.output_max)
    print(f"归一化一致性: {'✓ 一致' if is_consistent else '✗ 不一致'}")
    
    # 9. 清理临时文件
    print("\n9. 清理临时文件...")
    if os.path.exists(test_data_path):
        os.remove(test_data_path)
    if os.path.exists(norm_info_path):
        os.remove(norm_info_path)
    
    print("\n演示完成！")

def demo_denormalization_workflow():
    """演示完整的归一化-训练-反归一化工作流程"""
    print("\n" + "=" * 60)
    print("完整工作流程演示")
    print("=" * 60)
    
    # 模拟训练后的预测结果
    print("\n1. 模拟训练预测结果...")
    
    # 创建示例数据
    test_data_path = "test_workflow_data.h5"
    np.random.seed(123)
    test_data = np.random.randn(20, 100, 100) * 5 + 50
    
    import h5py
    with h5py.File(test_data_path, 'w') as f:
        f.create_dataset('pressure', data=test_data)
    
    # 创建数据集
    dataset = DynamicResolutionDataset(
        data_path=test_data_path,
        input_resolution=(32, 32),
        output_resolution=(64, 64),
        num_samples=5,
        normalize_data=True
    )
    
    # 保存归一化信息（实际应用中在训练前保存）
    norm_info_path = "training_normalization.yaml"
    dataset.save_normalization_info(norm_info_path)
    
    # 2. 模拟模型预测（归一化的结果）
    print("\n2. 模拟模型预测...")
    
    # 获取一个真实的归一化输出作为"预测结果"
    _, true_output, _ = dataset[0]
    
    # 添加一些噪声模拟预测误差
    predicted_output = true_output + torch.randn_like(true_output) * 0.01
    
    print(f"预测结果形状: {predicted_output.shape}")
    print(f"预测结果范围: [{predicted_output.min():.6f}, {predicted_output.max():.6f}]")
    
    # 3. 反归一化恢复物理信息
    print("\n3. 反归一化恢复物理信息...")
    
    # 方法1: 使用数据集对象
    physical_prediction = dataset.denormalize_predictions(predicted_output)
    
    # 方法2: 使用保存的归一化信息（推荐用于部署）
    loaded_norm_info = DynamicResolutionDataset.load_normalization_info(norm_info_path)
    
    # 手动反归一化
    global_min = loaded_norm_info['global_min']
    global_max = loaded_norm_info['global_max']
    
    predicted_np = predicted_output.numpy()
    physical_prediction_manual = predicted_np * (global_max - global_min) + global_min
    
    print(f"物理预测结果范围: [{physical_prediction.min():.3f}, {physical_prediction.max():.3f}]")
    print(f"手动反归一化结果范围: [{physical_prediction_manual.min():.3f}, {physical_prediction_manual.max():.3f}]")
    
    # 验证两种方法的一致性
    diff = np.abs(physical_prediction - physical_prediction_manual).max()
    print(f"两种反归一化方法的最大差异: {diff:.10f}")
    
    # 4. 与原始物理数据比较
    print("\n4. 与原始数据比较...")
    print(f"原始数据范围: [{test_data.min():.3f}, {test_data.max():.3f}]")
    print(f"预测数据范围: [{physical_prediction.min():.3f}, {physical_prediction.max():.3f}]")
    
    # 清理
    if os.path.exists(test_data_path):
        os.remove(test_data_path)
    if os.path.exists(norm_info_path):
        os.remove(norm_info_path)
    
    print("\n工作流程演示完成！")

if __name__ == "__main__":
    try:
        demo_global_normalization()
        demo_denormalization_workflow()
        
        print("\n" + "=" * 60)
        print("总结:")
        print("✓ 实现了基于全局数据的归一化")
        print("✓ 确保输入输出使用相同的归一化标准")
        print("✓ 提供了完整的反归一化功能")
        print("✓ 支持归一化信息的保存和加载")
        print("✓ 适用于输入是输出子集的场景")
        print("=" * 60)
        
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()