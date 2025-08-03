#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示训练日志中归一化信息的显示

本脚本展示了修改后的训练代码如何在日志中显示归一化相关信息，
包括数据统计、归一化参数、以及训练完成后的使用提示。

作者: AI Assistant
日期: 2025
"""

import os
import sys
import numpy as np
import h5py
import yaml
import logging
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

from dynamic_resolution_trainer import DynamicResolutionDataset

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_demo_data():
    """创建演示数据"""
    demo_data_path = "demo_training_log_data.h5"
    
    # 创建示例数据
    np.random.seed(42)
    demo_data = np.random.randn(50, 64, 64) * 10 + 25  # 模拟物理数据
    
    with h5py.File(demo_data_path, 'w') as f:
        f.create_dataset('tensor', data=demo_data)
    
    logger.info(f"✅ 演示数据已创建: {demo_data_path}")
    logger.info(f"   数据形状: {demo_data.shape}")
    logger.info(f"   数据范围: [{demo_data.min():.3f}, {demo_data.max():.3f}]")
    
    return demo_data_path

def demo_training_log_normalization():
    """演示训练日志中的归一化信息显示"""
    print("=" * 60)
    print("训练日志中归一化信息显示演示")
    print("=" * 60)
    
    # 1. 创建演示数据
    print("\n1. 创建演示数据...")
    demo_data_path = create_demo_data()
    
    # 2. 演示启用归一化的情况
    print("\n2. 启用归一化的训练日志:")
    print("-" * 40)
    
    logger.info("=== 创建数据加载器 ===")
    
    # 创建启用归一化的数据集
    dataset_normalized = DynamicResolutionDataset(
        data_path=demo_data_path,
        input_resolution=(32, 32),
        output_resolution=(64, 64),
        num_samples=20,
        crop_mode='center',
        normalize_data=True  # 启用归一化
    )
    
    # 显示数据统计（模拟训练脚本中的日志）
    stats = dataset_normalized.get_data_statistics()
    logger.info(f"数据统计: {stats}")
    
    # 显示归一化信息（新增的日志内容）
    if dataset_normalized.normalize_data:
        norm_info = dataset_normalized.get_normalization_info()
        logger.info("=== 归一化信息 ===")
        logger.info(f"归一化方法: {norm_info['normalization_method']}")
        logger.info(f"全局数据范围: [{norm_info['global_min']:.6f}, {norm_info['global_max']:.6f}]")
        logger.info(f"原始数据形状: {norm_info['original_data_shape']}")
        logger.info(f"输入分辨率: {norm_info['input_resolution']}")
        logger.info(f"输出分辨率: {norm_info['output_resolution']}")
        logger.info("✅ 数据已归一化到[0,1]范围，可用于反归一化恢复物理信息")
    
    # 3. 演示禁用归一化的情况
    print("\n3. 禁用归一化的训练日志:")
    print("-" * 40)
    
    # 创建禁用归一化的数据集
    dataset_no_norm = DynamicResolutionDataset(
        data_path=demo_data_path,
        input_resolution=(32, 32),
        output_resolution=(64, 64),
        num_samples=20,
        crop_mode='center',
        normalize_data=False  # 禁用归一化
    )
    
    # 显示数据统计
    stats_no_norm = dataset_no_norm.get_data_statistics()
    logger.info(f"数据统计: {stats_no_norm}")
    
    # 显示归一化状态
    if dataset_no_norm.normalize_data:
        # 这部分不会执行
        pass
    else:
        logger.info("ℹ️  归一化已禁用，使用原始数据范围")
    
    # 4. 演示训练完成后的归一化提示
    print("\n4. 训练完成后的归一化提示:")
    print("-" * 40)
    
    logger.info("🎉 === 训练完成 ===")
    
    # 模拟保存归一化信息
    if dataset_normalized.normalize_data:
        norm_info_path = "demo_normalization_info.json"
        dataset_normalized.save_normalization_info(norm_info_path)
        logger.info(f"📊 归一化信息已保存: {norm_info_path}")
        logger.info("💡 使用提示:")
        logger.info("   - 使用 dataset.denormalize_predictions(predictions) 反归一化预测结果")
        logger.info("   - 使用 DynamicResolutionDataset.load_normalization_info(path) 加载归一化信息")
        logger.info("   - 参考 demo_global_normalization.py 了解完整用法")
    
    # 5. 数据范围对比
    print("\n5. 归一化前后数据范围对比:")
    print("-" * 40)
    
    # 获取样本数据
    input_norm, output_norm, _ = dataset_normalized[0]
    input_raw, output_raw, _ = dataset_no_norm[0]
    
    print(f"归一化数据:")
    print(f"  输入范围: [{input_norm.min():.6f}, {input_norm.max():.6f}]")
    print(f"  输出范围: [{output_norm.min():.6f}, {output_norm.max():.6f}]")
    
    print(f"原始数据:")
    print(f"  输入范围: [{input_raw.min():.3f}, {input_raw.max():.3f}]")
    print(f"  输出范围: [{output_raw.min():.3f}, {output_raw.max():.3f}]")
    
    # 6. 清理临时文件
    print("\n6. 清理临时文件...")
    cleanup_files = [demo_data_path, "demo_normalization_info.json"]
    for file_path in cleanup_files:
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"   已删除: {file_path}")

def demo_log_comparison():
    """对比修改前后的日志输出"""
    print("\n" + "=" * 60)
    print("修改前后日志对比")
    print("=" * 60)
    
    print("\n📋 修改前的日志输出:")
    print("-" * 30)
    print("INFO - 数据统计: {'num_samples': 20, 'input_shape': (32, 32), ...}")
    print("INFO - === 创建模型 ===")
    print("INFO - 模型参数数量: 1234567")
    
    print("\n📋 修改后的日志输出:")
    print("-" * 30)
    print("INFO - 数据统计: {'num_samples': 20, 'input_shape': (32, 32), ...}")
    print("INFO - === 归一化信息 ===")
    print("INFO - 归一化方法: global_minmax")
    print("INFO - 全局数据范围: [15.234567, 34.567890]")
    print("INFO - 原始数据形状: (50, 64, 64)")
    print("INFO - 输入分辨率: (32, 32)")
    print("INFO - 输出分辨率: (64, 64)")
    print("INFO - ✅ 数据已归一化到[0,1]范围，可用于反归一化恢复物理信息")
    print("INFO - === 创建模型 ===")
    print("INFO - 模型参数数量: 1234567")
    
    print("\n📋 训练完成时的新增日志:")
    print("-" * 30)
    print("INFO - 🎉 === 训练完成 ===")
    print("INFO - 📊 归一化信息已保存: ./results/normalization_info.json")
    print("INFO - 💡 使用提示:")
    print("INFO -    - 使用 dataset.denormalize_predictions(predictions) 反归一化预测结果")
    print("INFO -    - 使用 DynamicResolutionDataset.load_normalization_info(path) 加载归一化信息")
    print("INFO -    - 参考 demo_global_normalization.py 了解完整用法")
    print("INFO - 💾 最终配置已保存: ./results/final_config.yaml")

if __name__ == "__main__":
    try:
        demo_training_log_normalization()
        demo_log_comparison()
        
        print("\n" + "=" * 60)
        print("总结:")
        print("✅ 训练日志已更新，现在会显示:")
        print("   1. 详细的归一化信息（方法、数据范围、形状等）")
        print("   2. 归一化状态提示（启用/禁用）")
        print("   3. 训练完成后的使用指南")
        print("   4. 自动保存归一化信息到文件")
        print("\n💡 这些改进让用户能够:")
        print("   - 清楚了解数据的归一化状态")
        print("   - 获得反归一化的使用指导")
        print("   - 方便地恢复物理信息")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()