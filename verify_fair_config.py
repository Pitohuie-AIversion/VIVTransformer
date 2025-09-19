#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证公平对比配置的参数量

作者: AI Assistant
日期: 2025
"""

import torch
import yaml
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from modify_multi_attention.run_crop_model_test import (
    SimpleMLP, SimpleTransformer, CustomTransformerWrapper, 
    EnhancedFNOWrapper, EnhancedUNetWrapper
)

def count_parameters(model):
    """计算模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def verify_config(config_path):
    """验证配置文件的参数量"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    input_dim = config['data']['input_dim']
    output_dim = config['data']['output_dim']
    
    print(f"验证配置文件: {config_path}")
    print(f"输入维度: {input_dim}, 输出维度: {output_dim}")
    print("="*70)
    print(f"{'模型名称':<20} {'模型类型':<15} {'参数量':<15} {'相对比例':<10}")
    print("="*70)
    
    results = {}
    param_counts = []
    
    for model_name, model_config in config['models'].items():
        try:
            model_type = model_config['model_type']
            model_params = {k: v for k, v in model_config.items() 
                          if k not in ['model_type', 'input_dim', 'output_dim']}
            
            # 创建模型
            if model_type == 'mlp':
                model = SimpleMLP(input_dim, output_dim, **model_params)
            elif model_type == 'transformer':
                model = SimpleTransformer(input_dim, output_dim, **model_params)
            elif model_type == 'custom_transformer':
                model = CustomTransformerWrapper(input_dim, output_dim, **model_params)
            elif model_type == 'fno':
                model = EnhancedFNOWrapper(input_dim, output_dim, **model_params)
            elif model_type == 'unet':
                model = EnhancedUNetWrapper(input_dim, output_dim, **model_params)
            else:
                continue
            
            param_count = count_parameters(model)
            param_counts.append(param_count)
            results[model_name] = {
                'type': model_type,
                'parameters': param_count,
                'config': model_params
            }
            
            print(f"{model_name:<20} {model_type:<15} {param_count:>12,} {'':>10}")
            
        except Exception as e:
            print(f"{model_name:<20} {'错误':<15} {str(e):<15}")
    
    # 计算统计信息
    if param_counts:
        min_params = min(param_counts)
        max_params = max(param_counts)
        avg_params = sum(param_counts) / len(param_counts)
        ratio = max_params / min_params
        
        print("="*70)
        print(f"统计信息:")
        print(f"  最小参数量: {min_params:,}")
        print(f"  最大参数量: {max_params:,}")
        print(f"  平均参数量: {avg_params:,.0f}")
        print(f"  最大/最小比例: {ratio:.2f}x")
        
        # 判断是否公平
        if ratio <= 2.0:
            print(f"  ✅ 参数量差异在合理范围内 (≤2x)")
        elif ratio <= 5.0:
            print(f"  ⚠️  参数量差异较大 (2-5x)")
        else:
            print(f"  ❌ 参数量差异过大 (>5x)")
    
    return results

if __name__ == "__main__":
    print("验证公平对比配置")
    print("="*70)
    
    # 验证原始配置
    print("\n1. 原始配置:")
    original_results = verify_config("configs/real_data_crop_config.yaml")
    
    # 验证公平对比配置
    print("\n2. 公平对比配置:")
    fair_results = verify_config("configs/fair_comparison_config.yaml")
    
    print("\n验证完成！")