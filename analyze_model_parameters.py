#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型参数量分析脚本

功能:
1. 分析当前配置中各模型的参数量
2. 提供参数量统一的配置建议
3. 生成公平对比的配置文件

作者: AI Assistant
日期: 2025
"""

import torch
import torch.nn as nn
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

def analyze_current_config():
    """分析当前配置的参数量"""
    config_path = "configs/real_data_crop_config.yaml"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    input_dim = config['data']['input_dim']
    output_dim = config['data']['output_dim']
    
    print(f"输入维度: {input_dim}, 输出维度: {output_dim}")
    print("="*60)
    
    results = {}
    
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
            results[model_name] = {
                'type': model_type,
                'parameters': param_count,
                'config': model_params
            }
            
            print(f"{model_name:20} | {model_type:15} | {param_count:>12,} 参数")
            
        except Exception as e:
            print(f"{model_name:20} | 错误: {e}")
    
    return results

def generate_unified_configs():
    """生成参数量统一的配置"""
    input_dim = 1024
    output_dim = 16384
    
    # 目标参数量级别
    target_params = [
        ("small", 100_000),    # 10万参数
        ("medium", 500_000),   # 50万参数
        ("large", 1_000_000),  # 100万参数
    ]
    
    configs = {}
    
    for size_name, target_param in target_params:
        print(f"\n生成 {size_name} 配置 (目标: {target_param:,} 参数)")
        print("="*50)
        
        config = {
            'mlp': generate_mlp_config(input_dim, output_dim, target_param),
            'transformer': generate_transformer_config(input_dim, output_dim, target_param),
            'custom_transformer': generate_custom_transformer_config(input_dim, output_dim, target_param),
            'fno': generate_fno_config(input_dim, output_dim, target_param),
            'unet': generate_unet_config(input_dim, output_dim, target_param)
        }
        
        # 验证参数量
        for model_name, model_config in config.items():
            try:
                model_type = model_config['model_type']
                model_params = {k: v for k, v in model_config.items() 
                              if k not in ['model_type', 'input_dim', 'output_dim']}
                
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
                
                param_count = count_parameters(model)
                ratio = param_count / target_param
                
                print(f"{model_name:20} | {param_count:>10,} | {ratio:>6.2f}x")
                
            except Exception as e:
                print(f"{model_name:20} | 错误: {e}")
        
        configs[size_name] = config
    
    return configs

def generate_mlp_config(input_dim, output_dim, target_params):
    """生成MLP配置"""
    # 估算隐藏层维度
    if target_params <= 100_000:
        hidden_dims = [512, 1024, 512]
    elif target_params <= 500_000:
        hidden_dims = [1024, 2048, 4096, 2048]
    else:
        hidden_dims = [2048, 4096, 8192, 4096]
    
    return {
        'model_type': 'mlp',
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dims': hidden_dims,
        'activation': 'gelu',
        'dropout': 0.1,
        'use_batch_norm': True,
        'use_residual': True
    }

def generate_transformer_config(input_dim, output_dim, target_params):
    """生成Transformer配置"""
    if target_params <= 100_000:
        d_model, num_heads, num_layers = 128, 4, 2
    elif target_params <= 500_000:
        d_model, num_heads, num_layers = 256, 8, 3
    else:
        d_model, num_heads, num_layers = 512, 8, 4
    
    return {
        'model_type': 'transformer',
        'input_dim': input_dim,
        'output_dim': output_dim,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'dropout': 0.1,
        'seq_len': 11,
        'pe_type': 'sinusoidal_1d',
        'use_memory_film': False,
        'max_time_steps': 100,
        'output_head_type': 'global',
        'time_encoding': 'embedding'
    }

def generate_custom_transformer_config(input_dim, output_dim, target_params):
    """生成CustomTransformer配置"""
    if target_params <= 100_000:
        d_model, num_heads, num_layers = 128, 4, 2
    elif target_params <= 500_000:
        d_model, num_heads, num_layers = 256, 8, 3
    else:
        d_model, num_heads, num_layers = 512, 8, 4
    
    return {
        'model_type': 'custom_transformer',
        'input_dim': input_dim,
        'output_dim': output_dim,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'dropout': 0.1,
        'attention_type': 'relative',
        'seq_len': None,
        'pe_type': 'learnable_1d',
        'output_head_type': 'global',
        'time_encoding': 'embedding',
        'use_memory_film': True,
        'use_memory_concat': False,
        'max_time_steps': 100
    }

def generate_fno_config(input_dim, output_dim, target_params):
    """生成FNO配置"""
    if target_params <= 100_000:
        width, num_layers = 64, 3
    elif target_params <= 500_000:
        width, num_layers = 128, 4
    else:
        width, num_layers = 256, 4
    
    return {
        'model_type': 'fno',
        'input_dim': input_dim,
        'output_dim': output_dim,
        'width': width,
        'num_layers': num_layers,
        'activation': 'gelu',
        'input_resolution': [32, 32],
        'output_resolution': [128, 128]
    }

def generate_unet_config(input_dim, output_dim, target_params):
    """生成UNet配置"""
    if target_params <= 100_000:
        features = [32, 64, 128, 64]
    elif target_params <= 500_000:
        features = [64, 128, 256, 128]
    else:
        features = [128, 256, 512, 256]
    
    return {
        'model_type': 'unet',
        'input_dim': input_dim,
        'output_dim': output_dim,
        'in_channels': 1,
        'out_channels': 1,
        'features': features,
        'activation': 'relu',
        'use_attention': False,
        'input_resolution': [32, 32],
        'output_resolution': [128, 128]
    }

def save_unified_config(size_name, models_config):
    """保存统一参数量的配置文件"""
    base_config_path = "configs/real_data_crop_config.yaml"
    
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # 更新模型配置
    base_config['models'] = models_config
    
    # 保存新配置
    output_path = f"configs/unified_{size_name}_config.yaml"
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(base_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"保存配置文件: {output_path}")

if __name__ == "__main__":
    print("模型参数量分析")
    print("="*60)
    
    # 分析当前配置
    print("\n1. 当前配置参数量分析:")
    current_results = analyze_current_config()
    
    # 生成统一配置
    print("\n2. 生成参数量统一的配置:")
    unified_configs = generate_unified_configs()
    
    # 保存配置文件
    print("\n3. 保存配置文件:")
    for size_name, models_config in unified_configs.items():
        save_unified_config(size_name, models_config)
    
    print("\n分析完成！")
    print("\n建议:")
    print("1. 使用 unified_medium_config.yaml 进行公平对比")
    print("2. 所有模型参数量控制在 50万 左右")
    print("3. 可以分别测试 small/medium/large 三个级别")