#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDEBench数据处理器使用示例

本脚本展示如何使用PDEBenchProcessor来截取和处理PDEBench数据集

作者: AI Assistant
日期: 2025
版本: 1.0
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdebench_data_processor import PDEBenchProcessor, PDEBenchConfig
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def example_basic_usage():
    """
    基本使用示例：处理Darcy Flow数据
    """
    logger.info("=== 基本使用示例 ===")
    
    # 创建配置
    config = PDEBenchConfig()
    config.MAX_SAMPLES = 100  # 限制样本数量
    config.START_TIME_IDX = 0  # 从第0个时间步开始
    config.END_TIME_IDX = 50   # 到第50个时间步结束
    config.TIME_STEP_INTERVAL = 2  # 每隔2个时间步取一个
    config.OUTPUT_FILE = 'darcy_flow_processed.pt'
    
    # 创建处理器
    processor = PDEBenchProcessor(config)
    
    # 处理Darcy Flow数据
    success = processor.process_pde_dataset('darcy', sequence_length=5)
    
    if success:
        # 保存处理后的数据
        processor.save_processed_data()
        logger.info("[OK] Darcy Flow数据处理完成")
        return True
    else:
        logger.error("[FAIL] Darcy Flow数据处理失败")
        return False


def example_custom_spatial_processing():
    """
    自定义空间处理示例：空间裁剪和下采样
    """
    logger.info("=== 自定义空间处理示例 ===")
    
    # 创建配置
    config = PDEBenchConfig()
    config.MAX_SAMPLES = 50
    config.SPATIAL_CROP = ((10, 50), (10, 50))  # 裁剪空间区域
    config.SPATIAL_DOWNSAMPLE = 2  # 2倍下采样
    config.FLATTEN_SPATIAL = False  # 保持空间维度
    config.OUTPUT_FILE = 'darcy_flow_spatial_processed.pt'
    
    # 创建处理器
    processor = PDEBenchProcessor(config)
    
    # 处理数据
    success = processor.process_pde_dataset('darcy', sequence_length=3)
    
    if success:
        processor.save_processed_data()
        logger.info("[OK] 空间处理示例完成")
        return True
    else:
        logger.error("[FAIL] 空间处理示例失败")
        return False


def example_multiple_pde_types():
    """
    多种PDE类型处理示例
    """
    logger.info("=== 多种PDE类型处理示例 ===")
    
    # 创建配置
    config = PDEBenchConfig()
    config.MAX_SAMPLES = 30
    config.END_TIME_IDX = 20
    config.OUTPUT_FILE = 'multiple_pde_processed.pt'
    
    # 创建处理器
    processor = PDEBenchProcessor(config)
    
    # 处理多种PDE类型
    pde_types = ['darcy']  # 可以添加其他类型，如 'burgers', 'advection'
    
    success_count = 0
    for pde_type in pde_types:
        logger.info(f"正在处理 {pde_type} 数据...")
        if processor.process_pde_dataset(pde_type, sequence_length=2):
            success_count += 1
            logger.info(f"[OK] {pde_type} 处理成功")
        else:
            logger.error(f"[FAIL] {pde_type} 处理失败")
    
    if success_count > 0:
        processor.save_processed_data()
        logger.info(f"[OK] 成功处理 {success_count}/{len(pde_types)} 种PDE类型")
        return True
    else:
        logger.error("[FAIL] 所有PDE类型处理失败")
        return False


def example_load_and_analyze():
    """
    加载和分析处理后的数据
    """
    logger.info("=== 数据加载和分析示例 ===")
    
    # 查找已处理的数据文件
    data_files = [
        'darcy_flow_processed.pt',
        'darcy_flow_spatial_processed.pt',
        'multiple_pde_processed.pt'
    ]
    
    for data_file in data_files:
        if os.path.exists(data_file):
            logger.info(f"分析文件: {data_file}")
            
            try:
                # 加载数据
                data = torch.load(data_file)
                
                logger.info(f"  包含的PDE类型: {data['pde_types']}")
                
                for pde_type in data['pde_types']:
                    pde_data = data['processed_data'][pde_type]
                    logger.info(f"  {pde_type}:")
                    logger.info(f"    输入形状: {pde_data['inputs'].shape}")
                    logger.info(f"    输出形状: {pde_data['outputs'].shape}")
                    logger.info(f"    样本数量: {len(pde_data['sample_info'])}")
                    
                    # 计算数据统计
                    inputs = pde_data['inputs']
                    outputs = pde_data['outputs']
                    
                    logger.info(f"    输入数据范围: [{inputs.min():.6f}, {inputs.max():.6f}]")
                    logger.info(f"    输出数据范围: [{outputs.min():.6f}, {outputs.max():.6f}]")
                    logger.info(f"    输入数据均值: {inputs.mean():.6f}")
                    logger.info(f"    输出数据均值: {outputs.mean():.6f}")
                
                logger.info("-" * 50)
                
            except Exception as e:
                logger.error(f"加载文件 {data_file} 时出错: {str(e)}")
        else:
            logger.info(f"文件不存在: {data_file}")


def example_quick_test():
    """
    快速测试：验证数据处理流程
    """
    logger.info("=== 快速测试 ===")
    
    # 创建最小配置
    config = PDEBenchConfig()
    config.MAX_SAMPLES = 5  # 只处理5个样本
    config.END_TIME_IDX = 10  # 只取前10个时间步
    config.OUTPUT_FILE = 'quick_test.pt'
    
    # 创建处理器
    processor = PDEBenchProcessor(config)
    
    # 快速测试
    success = processor.process_pde_dataset('darcy', sequence_length=1)
    
    if success:
        processor.save_processed_data()
        
        # 验证保存的数据
        if os.path.exists('quick_test.pt'):
            data = torch.load('quick_test.pt')
            logger.info("[OK] 快速测试成功")
            logger.info(f"  处理的PDE类型: {data['pde_types']}")
            
            # 清理测试文件
            os.remove('quick_test.pt')
            logger.info("  测试文件已清理")
            
            return True
        else:
            logger.error("[FAIL] 数据文件未生成")
            return False
    else:
        logger.error("[FAIL] 快速测试失败")
        return False


def main():
    """
    主函数：运行所有示例
    """
    logger.info("PDEBench数据处理器使用示例")
    logger.info("=" * 60)
    
    # 检查PDEBench数据路径
    config = PDEBenchConfig()
    if not os.path.exists(config.PDEBENCH_ROOT):
        logger.error(f"PDEBench数据路径不存在: {config.PDEBENCH_ROOT}")
        logger.error("请确保PDEBench数据已下载并放置在正确位置")
        return
    
    # 运行示例
    examples = [
        ("快速测试", example_quick_test),
        ("基本使用", example_basic_usage),
        ("自定义空间处理", example_custom_spatial_processing),
        ("多种PDE类型处理", example_multiple_pde_types),
        ("数据加载和分析", example_load_and_analyze)
    ]
    
    success_count = 0
    for name, func in examples:
        try:
            logger.info(f"\n运行示例: {name}")
            if func():
                success_count += 1
        except Exception as e:
            logger.error(f"示例 {name} 运行时出错: {str(e)}")
    
    logger.info(f"\n=== 总结 ===")
    logger.info(f"成功运行 {success_count}/{len(examples)} 个示例")
    
    if success_count > 0:
        logger.info("\n生成的文件:")
        for file in os.listdir('.'):
            if file.endswith('_processed.pt'):
                logger.info(f"  - {file}")


if __name__ == "__main__":
    main()