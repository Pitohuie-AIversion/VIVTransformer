#!/usr/bin/env python3
"""
验证小中大参数量配置的参数计数脚本
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, List

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 导入模型创建函数
from run_crop_model_test import create_enhanced_model, SimpleMLP

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_parameters(param_count):
    """格式化参数数量显示"""
    if param_count >= 1e6:
        return f"{param_count/1e6:.2f}M"
    elif param_count >= 1e3:
        return f"{param_count/1e3:.2f}K"
    else:
        return str(param_count)

def create_model_from_config(model_type: str, model_config: Dict[str, Any], input_dim: int = 1024, output_dim: int = 1024):
    """根据配置创建模型"""
    try:
        if model_type == 'mlp':
            # 直接创建SimpleMLP
            model = SimpleMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=model_config.get('hidden_dims', [256]),
                activation=model_config.get('activation', 'relu'),
                dropout=model_config.get('dropout', 0.1),
                use_batch_norm=model_config.get('use_batch_norm', True),
                use_residual=model_config.get('use_residual', False)
            )
        else:
            # 使用通用的create_enhanced_model函数
            model = create_enhanced_model(model_config, input_dim, output_dim)
        
        return model
    except Exception as e:
        print(f"创建{model_type}模型失败: {e}")
        return None

def analyze_config_file(config_path: str):
    """分析单个配置文件的参数量"""
    print(f"\n分析配置文件: {config_path}")
    print("=" * 60)
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"读取配置文件失败: {e}")
        return {}
    
    models_config = config.get('models', {})
    results = {}
    
    for model_name, model_config in models_config.items():
        print(f"\n{model_name.upper()} 模型:")
        print("-" * 30)
        
        # 打印配置信息
        if model_name == 'transformer':
            print(f"  d_model: {model_config.get('d_model', 'N/A')}")
            print(f"  num_heads: {model_config.get('num_heads', 'N/A')}")
            print(f"  num_layers: {model_config.get('num_layers', 'N/A')}")
        elif model_name == 'unet':
            print(f"  channels: {model_config.get('channels', 'N/A')}")
        elif model_name == 'mlp':
            print(f"  hidden_dims: {model_config.get('hidden_dims', 'N/A')}")
        elif model_name == 'fno':
            print(f"  modes: {model_config.get('modes', 'N/A')}")
            print(f"  width: {model_config.get('width', 'N/A')}")
            print(f"  num_layers: {model_config.get('num_layers', 'N/A')}")
        
        # 创建模型并计算参数
        model = create_model_from_config(model_name, model_config)
        if model is not None:
            param_count = count_parameters(model)
            results[model_name] = param_count
            print(f"  参数量: {format_parameters(param_count)} ({param_count:,})")
        else:
            results[model_name] = 0
            print(f"  参数量: 模型创建失败")
    
    return results

def main():
    """主函数"""
    print("🔍 验证参数量配置")
    print("=" * 80)
    
    # 配置文件路径
    config_files = {
        'small': 'unified_training_config_small.yaml',
        'medium': 'unified_training_config_medium.yaml', 
        'large': 'unified_training_config_large.yaml'
    }
    
    all_results = {}
    
    # 分析每个配置文件
    for scale, config_file in config_files.items():
        if os.path.exists(config_file):
            results = analyze_config_file(config_file)
            all_results[scale] = results
        else:
            print(f"\n⚠️ 配置文件不存在: {config_file}")
    
    # 生成对比表格
    if all_results:
        print("\n" + "=" * 80)
        print("📊 参数量对比表格")
        print("=" * 80)
        
        # 表头
        print(f"{'模型':<12} {'小型':<15} {'中型':<15} {'大型':<15}")
        print("-" * 60)
        
        # 模型列表
        model_names = ['transformer', 'unet', 'mlp', 'fno']
        
        for model in model_names:
            row = f"{model.upper():<12}"
            for scale in ['small', 'medium', 'large']:
                if scale in all_results and model in all_results[scale]:
                    param_count = all_results[scale][model]
                    row += f"{format_parameters(param_count):<15}"
                else:
                    row += f"{'N/A':<15}"
            print(row)
        
        # 统计信息
        print("\n" + "=" * 80)
        print("📈 统计信息")
        print("=" * 80)
        
        for scale in ['small', 'medium', 'large']:
            if scale in all_results:
                params = list(all_results[scale].values())
                params = [p for p in params if p > 0]  # 过滤掉失败的模型
                
                if params:
                    total_params = sum(params)
                    max_params = max(params)
                    min_params = min(params)
                    ratio = max_params / min_params if min_params > 0 else float('inf')
                    
                    print(f"\n{scale.upper()} 配置:")
                    print(f"  总参数量: {format_parameters(total_params)}")
                    print(f"  最大参数量: {format_parameters(max_params)}")
                    print(f"  最小参数量: {format_parameters(min_params)}")
                    print(f"  参数比例: {ratio:.2f}:1")

if __name__ == "__main__":
    main()