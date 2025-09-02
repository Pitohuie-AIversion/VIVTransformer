#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态分辨率训练器归一化功能使用示例
"""

import os
import sys
import yaml
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'modify_multi_attention'))
sys.path.append(str(Path(__file__).parent))

from dynamic_resolution_trainer import get_dynamic_loaders, load_config_from_yaml

def demo_normalization_usage():
    """
    演示如何使用归一化功能
    """
    print("=== 动态分辨率训练器归一化功能演示 ===")
    
    # 1. 加载配置
    config_path = "dynamic_config.yaml"
    print(f"\n1. 加载配置文件: {config_path}")
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    config = load_config_from_yaml(config_path)
    
    # 2. 显示归一化配置
    normalize_enabled = config['data'].get('normalize_data', True)
    print(f"\n2. 归一化配置:")
    print(f"   - 启用归一化: {normalize_enabled}")
    print(f"   - 输入分辨率: {config['data']['input_resolution']}")
    print(f"   - 输出分辨率: {config['data']['output_resolution']}")
    print(f"   - 样本数量: {config['data']['num_samples']}")
    
    # 3. 创建数据加载器
    print(f"\n3. 创建数据加载器...")
    try:
        train_loader, val_loader, test_loader = get_dynamic_loaders(config)
        print(f"   [OK] 数据加载器创建成功")
        print(f"   - 训练集批次数: {len(train_loader)}")
        print(f"   - 验证集批次数: {len(val_loader)}")
        print(f"   - 测试集批次数: {len(test_loader)}")
        
        # 4. 检查数据范围
        print(f"\n4. 检查数据范围...")
        
        # 获取一个批次的数据
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            print(f"   批次 {batch_idx + 1}:")
            print(f"   - 输入数据形状: {inputs.shape}")
            print(f"   - 输出数据形状: {targets.shape}")
            print(f"   - 输入数据范围: [{inputs.min():.6f}, {inputs.max():.6f}]")
            print(f"   - 输出数据范围: [{targets.min():.6f}, {targets.max():.6f}]")
            
            if normalize_enabled:
                if 0 <= inputs.min() and inputs.max() <= 1 and 0 <= targets.min() and targets.max() <= 1:
                    print(f"   [OK] 数据已正确归一化到[0,1]范围")
                else:
                    print(f"   [ERROR] 数据归一化异常")
            else:
                print(f"   [INFO] 归一化已禁用，显示原始数据范围")
            
            # 只检查第一个批次
            break
            
    except Exception as e:
        print(f"   [ERROR] 创建数据加载器失败: {e}")
        return
    
    # 5. 配置建议
    print(f"\n5. 配置建议:")
    print(f"   - 对于神经网络训练，建议启用归一化 (normalize_data: true)")
    print(f"   - 归一化可以提高训练稳定性和收敛速度")
    print(f"   - 如需使用原始数据范围，可设置 normalize_data: false")
    
    print(f"\n=== 演示完成 ===")

def show_config_options():
    """
    显示配置选项说明
    """
    print("\n=== 归一化配置选项说明 ===")
    print("""
在 dynamic_config.yaml 中的 data 部分添加:

data:
  # ... 其他配置 ...
  normalize_data: true    # 启用归一化 (推荐)
  # normalize_data: false # 禁用归一化

归一化功能:
- 启用时: 将数据归一化到 [0, 1] 范围
- 禁用时: 使用原始数据范围
- 自动计算归一化参数 (min/max)
- 支持输入和输出数据分别归一化
    """)

if __name__ == "__main__":
    demo_normalization_usage()
    show_config_options()