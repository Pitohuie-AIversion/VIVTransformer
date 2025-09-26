#!/usr/bin/env python3
"""
验证三个不同参数量级配置的参数统计
"""

import yaml
import torch
import torch.nn as nn
from pathlib import Path
import sys
import os

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 导入模型创建函数
from run_crop_model_test import SimpleMLP, create_enhanced_model

def count_parameters(model):
    """计算模型参数数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_params(num_params):
    """格式化参数数量显示"""
    if num_params >= 1e6:
        return f"{num_params/1e6:.2f}M"
    elif num_params >= 1e3:
        return f"{num_params/1e3:.2f}K"
    else:
        return str(num_params)

def create_model_from_config(model_name, model_config, input_dim, output_dim):
    """根据配置创建模型"""
    try:
        if model_name == 'mlp':
            model = SimpleMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=model_config.get('hidden_dims', [256, 128]),
                activation=model_config.get('activation', 'relu'),
                dropout=model_config.get('dropout', 0.1),
                use_batch_norm=model_config.get('use_batch_norm', True),
                use_residual=model_config.get('use_residual', False)
            )
        else:
            # 对于其他模型类型，使用create_enhanced_model
            model = create_enhanced_model(
                model_config=model_config,
                input_dim=input_dim,
                output_dim=output_dim
            )
        
        return model
    except Exception as e:
        print(f"Error creating {model_name} model: {e}")
        return None

def analyze_config(config_path, scale_name):
    """分析单个配置文件"""
    print(f"\n{'='*60}")
    print(f"分析 {scale_name} 参数量配置: {config_path}")
    print(f"{'='*60}")
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    models_config = config['models']
    input_dim = config['data']['input_dim']
    output_dim = config['data']['output_dim']
    
    results = {}
    
    # 分析每个模型
    for model_name, model_config in models_config.items():
        print(f"\n{model_name.upper()} 配置:")
        
        # 打印关键配置参数
        if model_name == 'transformer':
            print(f"  d_model: {model_config.get('d_model')}")
            print(f"  num_heads: {model_config.get('num_heads')}")
            print(f"  num_layers: {model_config.get('num_layers')}")
            print(f"  use_memory_film: {model_config.get('use_memory_film', False)}")
            print(f"  use_memory_concat: {model_config.get('use_memory_concat', False)}")
        elif model_name == 'unet':
            print(f"  channels: {model_config.get('channels')}")
            print(f"  input_spatial_dim: {model_config.get('input_spatial_dim')}")
        elif model_name == 'mlp':
            print(f"  hidden_dims: {model_config.get('hidden_dims')}")
            print(f"  use_residual: {model_config.get('use_residual', False)}")
        elif model_name == 'fno':
            print(f"  modes: {model_config.get('modes')}")
            print(f"  width: {model_config.get('width')}")
            print(f"  num_layers: {model_config.get('num_layers')}")
        
        # 创建模型并计算参数
        model = create_model_from_config(model_name, model_config, input_dim, output_dim)
        if model is not None:
            param_count = count_parameters(model)
            results[model_name] = param_count
            print(f"  参数量: {format_params(param_count)} ({param_count:,})")
        else:
            results[model_name] = 0
            print(f"  参数量: 模型创建失败")
    
    return results

def main():
    """主函数"""
    config_files = [
        ('unified_training_config_small.yaml', '小参数量'),
        ('unified_training_config_medium.yaml', '中参数量'),
        ('unified_training_config_large.yaml', '大参数量')
    ]
    
    all_results = {}
    
    # 分析每个配置
    for config_file, scale_name in config_files:
        config_path = project_root / config_file
        if config_path.exists():
            results = analyze_config(config_path, scale_name)
            all_results[scale_name] = results
        else:
            print(f"配置文件不存在: {config_path}")
    
    # 生成对比表格
    print(f"\n{'='*80}")
    print("参数量对比总结")
    print(f"{'='*80}")
    
    model_names = ['transformer', 'unet', 'mlp', 'fno']
    
    # 表头
    print(f"{'模型':<12} {'小参数量':<15} {'中参数量':<15} {'大参数量':<15}")
    print("-" * 60)
    
    # 每个模型的对比
    for model_name in model_names:
        row = f"{model_name:<12}"
        for scale_name in ['小参数量', '中参数量', '大参数量']:
            if scale_name in all_results and model_name in all_results[scale_name]:
                param_count = all_results[scale_name][model_name]
                row += f" {format_params(param_count):<14}"
            else:
                row += f" {'N/A':<14}"
        print(row)
    
    # 计算每个量级的总参数量和比例
    print(f"\n{'='*60}")
    print("量级统计")
    print(f"{'='*60}")
    
    for scale_name in ['小参数量', '中参数量', '大参数量']:
        if scale_name in all_results:
            total_params = sum(all_results[scale_name].values())
            max_params = max(all_results[scale_name].values()) if all_results[scale_name].values() else 0
            min_params = min(all_results[scale_name].values()) if all_results[scale_name].values() else 0
            ratio = max_params / min_params if min_params > 0 else 0
            
            print(f"\n{scale_name}:")
            print(f"  总参数量: {format_params(total_params)}")
            print(f"  最大模型: {format_params(max_params)}")
            print(f"  最小模型: {format_params(min_params)}")
            print(f"  参数比例: {ratio:.2f}:1")

if __name__ == "__main__":
    main()