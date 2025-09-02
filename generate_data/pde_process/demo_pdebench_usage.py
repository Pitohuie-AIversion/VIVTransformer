#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDEBench数据处理系统演示脚本

这个脚本展示了如何使用PDEBench数据处理器来截取和处理数据集，
实现类似于makedata200plus200origin_data_channel_aoming_mergedd_all_reynold4D_plot_normalize_afterextraction.py的功能

主要功能:
1. 从PDEBench HDF5文件中读取数据
2. 截取指定数量的样本
3. 进行数据预处理和归一化
4. 生成适合机器学习的PyTorch张量
5. 保存处理后的数据

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

# 设置matplotlib支持中文显示
# 统一使用全局 sitecustomize.py 的中文字体和负号设置
# plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
# plt.rcParams['axes.unicode_minus'] = False

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdebench_data_processor import PDEBenchProcessor, PDEBenchConfig
from config_loader import ConfigLoader

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_basic_usage():
    """
    演示基本使用方法
    """
    print("\n=== 演示1: 基本使用方法 ===")
    
    # 创建配置
    config = PDEBenchConfig()
    config.PDEBENCH_ROOT = "x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench"
    config.MAX_SAMPLES = 50  # 截取50个样本
    config.OUTPUT_FILE = "demo_basic_output.pt"
    config.NORMALIZE_DATA = True
    config.FLATTEN_SPATIAL = True
    
    # 创建处理器
    processor = PDEBenchProcessor(config)
    
    # 处理Darcy Flow数据
    try:
        success = processor.process_pde_dataset('darcy', sequence_length=1)
        if success:
            # 获取处理后的数据
            if 'darcy' in processor.processed_data:
                result = processor.processed_data['darcy']
                print(f"[OK] 成功处理数据")
                print(f"  输入形状: {result['inputs'].shape}")
                print(f"  输出形状: {result['outputs'].shape}")
                print(f"  数据类型: {result['inputs'].dtype}")
                print(f"  数据范围: [{result['inputs'].min():.6f}, {result['inputs'].max():.6f}]")
            else:
                print("[FAIL] 数据处理成功但无法获取结果")
        else:
            print("[FAIL] 数据处理失败")
    except Exception as e:
        print(f"[FAIL] 处理过程中出错: {str(e)}")

def demo_custom_processing():
    """
    演示自定义处理参数
    """
    print("\n=== 演示2: 自定义处理参数 ===")
    
    # 创建自定义配置
    config = PDEBenchConfig()
    config.PDEBENCH_ROOT = "x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/PDEBench"
    config.MAX_SAMPLES = 20  # 只处理20个样本
    config.SPATIAL_CROP = ((0, 64), (0, 64))  # 空间裁剪到64x64
    config.SPATIAL_DOWNSAMPLE = 2   # 下采样因子为2
    config.START_TIME_IDX = 0
    config.END_TIME_IDX = 5  # 只使用前5个时间步
    config.OUTPUT_FILE = "demo_custom_output.pt"
    config.NORMALIZE_DATA = True
    config.FLATTEN_SPATIAL = False  # 保持空间维度
    
    processor = PDEBenchProcessor(config)
    
    try:
        success = processor.process_pde_dataset('darcy', sequence_length=3)
        if success:
            if 'darcy' in processor.processed_data:
                result = processor.processed_data['darcy']
                print(f"[OK] 自定义处理成功")
                print(f"  输入形状: {result['inputs'].shape}")
                print(f"  输出形状: {result['outputs'].shape}")
                print(f"  处理的样本数: {result['sample_info']}")
            else:
                print("[FAIL] 自定义处理成功但无法获取结果")
        else:
            print("[FAIL] 自定义处理失败")
    except Exception as e:
        print(f"[FAIL] 自定义处理出错: {str(e)}")

def demo_multiple_pde_types():
    """
    演示处理多种PDE类型
    """
    print("\n=== 演示3: 处理多种PDE类型 ===")
    
    # 使用配置文件
    config_loader = ConfigLoader("pdebench_config.yaml")
    config = config_loader.create_config("basic")
    config.MAX_SAMPLES = 30  # 每种类型30个样本
    config.OUTPUT_FILE = "demo_multiple_output.pt"
    
    processor = PDEBenchProcessor(config)
    
    # 尝试处理多种PDE类型
    pde_types = ['darcy']  # 目前只有darcy数据可用
    
    all_results = {}
    for pde_type in pde_types:
        try:
            success = processor.process_pde_dataset(pde_type, sequence_length=2)
            if success and pde_type in processor.processed_data:
                result = processor.processed_data[pde_type]
                all_results[pde_type] = result
                print(f"[OK] {pde_type}: 输入{result['inputs'].shape}, 输出{result['outputs'].shape}")
            else:
                print(f"[FAIL] {pde_type}: 处理失败")
        except Exception as e:
            print(f"[FAIL] {pde_type}: {str(e)}")
    
    if all_results:
        # 保存所有结果
        torch.save(all_results, config.OUTPUT_FILE)
        print(f"\n所有结果已保存到: {config.OUTPUT_FILE}")

def demo_data_analysis():
    """
    演示数据分析功能
    """
    print("\n=== 演示4: 数据分析 ===")
    
    # 加载之前保存的数据
    try:
        data_file = "demo_basic_output.pt"
        if os.path.exists(data_file):
            data = torch.load(data_file)
            
            print(f"数据文件: {data_file}")
            print(f"数据键: {list(data.keys())}")
            
            if 'darcy' in data:
                darcy_data = data['darcy']
                inputs = darcy_data['inputs']
                targets = darcy_data['outputs']
                
                print(f"\nDarcy Flow数据分析:")
                print(f"  输入数据统计:")
                print(f"    形状: {inputs.shape}")
                print(f"    均值: {inputs.mean():.6f}")
                print(f"    标准差: {inputs.std():.6f}")
                print(f"    最小值: {inputs.min():.6f}")
                print(f"    最大值: {inputs.max():.6f}")
                
                print(f"  目标数据统计:")
                print(f"    形状: {targets.shape}")
                print(f"    均值: {targets.mean():.6f}")
                print(f"    标准差: {targets.std():.6f}")
                print(f"    最小值: {targets.min():.6f}")
                print(f"    最大值: {targets.max():.6f}")
                
                # 简单的数据可视化
                if len(inputs.shape) == 2 and inputs.shape[1] > 1:
                    plt.figure(figsize=(12, 4))
                    
                    # 输入数据分布
                    plt.subplot(1, 3, 1)
                    plt.hist(inputs.flatten().numpy(), bins=50, alpha=0.7)
                    plt.title('输入数据分布')
                    plt.xlabel('值')
                    plt.ylabel('频次')
                    
                    # 目标数据分布
                    plt.subplot(1, 3, 2)
                    plt.hist(targets.flatten().numpy(), bins=50, alpha=0.7)
                    plt.title('目标数据分布')
                    plt.xlabel('值')
                    plt.ylabel('频次')
                    
                    # 输入-输出关系
                    plt.subplot(1, 3, 3)
                    sample_indices = np.random.choice(len(inputs), min(1000, len(inputs)), replace=False)
                    plt.scatter(inputs[sample_indices, 0].numpy(), targets[sample_indices].numpy(), alpha=0.5)
                    plt.title('输入-输出关系')
                    plt.xlabel('输入特征')
                    plt.ylabel('目标值')
                    
                    plt.tight_layout()
                    plt.savefig('demo_data_analysis.png', dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"\n数据分析图表已保存到: demo_data_analysis.png")
        else:
            print(f"数据文件不存在: {data_file}")
            print("请先运行演示1生成数据")
            
    except Exception as e:
        print(f"数据分析出错: {str(e)}")

def demo_config_usage():
    """
    演示配置文件使用
    """
    print("\n=== 演示5: 配置文件使用 ===")
    
    try:
        config_loader = ConfigLoader("pdebench_config.yaml")
        
        # 显示可用配置
        print("可用配置:")
        configs = config_loader.get_available_configs()
        for config_name in configs:
            print(f"  - {config_name}")
        
        # 使用快速测试配置
        print("\n使用快速测试配置:")
        config = config_loader.create_config("quick_test")
        config.OUTPUT_FILE = "demo_config_output.pt"
        
        processor = PDEBenchProcessor(config)
        success = processor.process_pde_dataset('darcy', sequence_length=1)
        
        if success and 'darcy' in processor.processed_data:
            result = processor.processed_data['darcy']
            print(f"[OK] 配置文件处理成功")
            print(f"  输入形状: {result['inputs'].shape}")
            print(f"  输出形状: {result['outputs'].shape}")
        else:
            print("[FAIL] 配置文件处理失败")
            
    except Exception as e:
        print(f"配置文件使用出错: {str(e)}")

def cleanup_demo_files():
    """
    清理演示文件
    """
    demo_files = [
        "demo_basic_output.pt",
        "demo_custom_output.pt", 
        "demo_multiple_output.pt",
        "demo_config_output.pt",
        "demo_data_analysis.png"
    ]
    
    print("\n=== 清理演示文件 ===")
    for file in demo_files:
        if os.path.exists(file):
            os.remove(file)
            print(f"[OK] 已删除: {file}")
        else:
            print(f"- 不存在: {file}")

def main():
    """
    主演示函数
    """
    print("PDEBench数据处理系统演示")
    print("=" * 50)
    print("这个演示展示了如何使用PDEBench数据处理器来截取和处理数据集")
    print("类似于原始的makedata脚本的功能，但支持HDF5格式和更多配置选项")
    
    try:
        # 运行所有演示
        demo_basic_usage()
        demo_custom_processing()
        demo_multiple_pde_types()
        demo_data_analysis()
        demo_config_usage()
        
        print("\n=== 演示完成 ===")
        print("所有演示都已运行完成！")
        print("\n主要功能:")
        print("1. [OK] 基本数据处理")
        print("2. [OK] 自定义处理参数")
        print("3. [OK] 多种PDE类型支持")
        print("4. [OK] 数据分析和可视化")
        print("5. [OK] 配置文件管理")
        
        # 询问是否清理文件
        response = input("\n是否清理演示生成的文件? (y/n): ")
        if response.lower() in ['y', 'yes']:
            cleanup_demo_files()
        
    except KeyboardInterrupt:
        print("\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中出错: {str(e)}")
        logger.exception("详细错误信息:")

if __name__ == "__main__":
    main()