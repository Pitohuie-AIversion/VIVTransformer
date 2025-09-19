#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公平对比配置生成脚本

目标: 将所有模型参数量控制在相似范围内 (约50万参数)
确保横向对比的公平性

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

def find_optimal_mlp_config(input_dim, output_dim, target_params=500_000):
    """寻找最优MLP配置"""
    best_config = None
    best_diff = float('inf')
    
    # 尝试不同的隐藏层配置
    configs = [
        [512, 1024, 512],
        [768, 1536, 768],
        [1024, 2048, 1024],
        [1024, 1536, 1024],
        [1536, 2048, 1536],
        [1024, 2048, 2048, 1024],
        [512, 1024, 2048, 1024],
        [768, 1536, 2048, 1024],
    ]
    
    for hidden_dims in configs:
        try:
            model = SimpleMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=hidden_dims,
                activation='gelu',
                dropout=0.1,
                use_batch_norm=True,
                use_residual=True
            )
            
            param_count = count_parameters(model)
            diff = abs(param_count - target_params)
            
            if diff < best_diff:
                best_diff = diff
                best_config = {
                    'hidden_dims': hidden_dims,
                    'param_count': param_count
                }
                
        except Exception as e:
            continue
    
    return best_config

def find_optimal_transformer_config(input_dim, output_dim, target_params=500_000):
    """寻找最优Transformer配置"""
    best_config = None
    best_diff = float('inf')
    
    # 尝试不同的配置组合
    configs = [
        {'d_model': 128, 'num_heads': 4, 'num_layers': 2},
        {'d_model': 128, 'num_heads': 8, 'num_layers': 2},
        {'d_model': 192, 'num_heads': 6, 'num_layers': 2},
        {'d_model': 256, 'num_heads': 8, 'num_layers': 2},
        {'d_model': 128, 'num_heads': 4, 'num_layers': 3},
        {'d_model': 192, 'num_heads': 6, 'num_layers': 3},
        {'d_model': 256, 'num_heads': 8, 'num_layers': 3},
        {'d_model': 384, 'num_heads': 6, 'num_layers': 2},
    ]
    
    for config in configs:
        try:
            model = SimpleTransformer(
                input_dim=input_dim,
                output_dim=output_dim,
                d_model=config['d_model'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                dropout=0.1,
                seq_len=11
            )
            
            param_count = count_parameters(model)
            diff = abs(param_count - target_params)
            
            if diff < best_diff:
                best_diff = diff
                best_config = {**config, 'param_count': param_count}
                
        except Exception as e:
            continue
    
    return best_config

def find_optimal_custom_transformer_config(input_dim, output_dim, target_params=500_000):
    """寻找最优CustomTransformer配置"""
    best_config = None
    best_diff = float('inf')
    
    # 尝试不同的配置组合 (使用较小的参数)
    configs = [
        {'d_model': 64, 'num_heads': 4, 'num_layers': 2},
        {'d_model': 96, 'num_heads': 6, 'num_layers': 2},
        {'d_model': 128, 'num_heads': 8, 'num_layers': 2},
        {'d_model': 64, 'num_heads': 4, 'num_layers': 3},
        {'d_model': 96, 'num_heads': 6, 'num_layers': 3},
        {'d_model': 128, 'num_heads': 4, 'num_layers': 2},
        {'d_model': 160, 'num_heads': 8, 'num_layers': 2},
    ]
    
    for config in configs:
        try:
            model = CustomTransformerWrapper(
                input_dim=input_dim,
                output_dim=output_dim,
                d_model=config['d_model'],
                num_heads=config['num_heads'],
                num_layers=config['num_layers'],
                dropout=0.1,
                attention_type='relative',
                seq_len=None,
                pe_type='learnable_1d',
                output_head_type='global',
                time_encoding='embedding',
                use_memory_film=True,
                use_memory_concat=False,
                max_time_steps=100
            )
            
            param_count = count_parameters(model)
            diff = abs(param_count - target_params)
            
            if diff < best_diff:
                best_diff = diff
                best_config = {**config, 'param_count': param_count}
                
        except Exception as e:
            continue
    
    return best_config

def find_optimal_fno_config(input_dim, output_dim, target_params=500_000):
    """寻找最优FNO配置"""
    best_config = None
    best_diff = float('inf')
    
    # 尝试不同的配置组合
    configs = [
        {'width': 32, 'num_layers': 3},
        {'width': 48, 'num_layers': 3},
        {'width': 64, 'num_layers': 3},
        {'width': 32, 'num_layers': 4},
        {'width': 48, 'num_layers': 4},
        {'width': 64, 'num_layers': 4},
        {'width': 80, 'num_layers': 3},
        {'width': 96, 'num_layers': 3},
    ]
    
    for config in configs:
        try:
            model = EnhancedFNOWrapper(
                input_dim=input_dim,
                output_dim=output_dim,
                width=config['width'],
                num_layers=config['num_layers'],
                activation='gelu',
                input_resolution=[32, 32],
                output_resolution=[128, 128]
            )
            
            param_count = count_parameters(model)
            diff = abs(param_count - target_params)
            
            if diff < best_diff:
                best_diff = diff
                best_config = {**config, 'param_count': param_count}
                
        except Exception as e:
            continue
    
    return best_config

def find_optimal_unet_config(input_dim, output_dim, target_params=500_000):
    """寻找最优UNet配置"""
    best_config = None
    best_diff = float('inf')
    
    # 尝试不同的配置组合
    configs = [
        {'features': [16, 32, 64, 32]},
        {'features': [24, 48, 96, 48]},
        {'features': [32, 64, 128, 64]},
        {'features': [16, 32, 64, 128, 64]},
        {'features': [24, 48, 96, 128, 64]},
        {'features': [32, 64, 96, 64]},
        {'features': [40, 80, 120, 80]},
    ]
    
    for config in configs:
        try:
            model = EnhancedUNetWrapper(
                input_dim=input_dim,
                output_dim=output_dim,
                in_channels=1,
                out_channels=1,
                features=config['features'],
                activation='relu',
                use_attention=False,
                input_resolution=[32, 32],
                output_resolution=[128, 128]
            )
            
            param_count = count_parameters(model)
            diff = abs(param_count - target_params)
            
            if diff < best_diff:
                best_diff = diff
                best_config = {**config, 'param_count': param_count}
                
        except Exception as e:
            continue
    
    return best_config

def create_fair_comparison_config():
    """创建公平对比配置"""
    input_dim = 1024
    output_dim = 16384
    target_params = 500_000  # 50万参数
    
    print(f"目标参数量: {target_params:,}")
    print("="*60)
    
    # 寻找各模型的最优配置
    print("正在寻找最优配置...")
    
    mlp_config = find_optimal_mlp_config(input_dim, output_dim, target_params)
    transformer_config = find_optimal_transformer_config(input_dim, output_dim, target_params)
    custom_transformer_config = find_optimal_custom_transformer_config(input_dim, output_dim, target_params)
    fno_config = find_optimal_fno_config(input_dim, output_dim, target_params)
    unet_config = find_optimal_unet_config(input_dim, output_dim, target_params)
    
    # 显示结果
    print("\n最优配置结果:")
    print("="*60)
    
    configs = {
        'mlp': mlp_config,
        'transformer': transformer_config,
        'custom_transformer': custom_transformer_config,
        'fno': fno_config,
        'unet': unet_config
    }
    
    for model_name, config in configs.items():
        if config:
            param_count = config['param_count']
            ratio = param_count / target_params
            print(f"{model_name:20} | {param_count:>10,} | {ratio:>6.2f}x")
        else:
            print(f"{model_name:20} | 配置失败")
    
    # 生成YAML配置
    yaml_config = generate_yaml_config(configs, input_dim, output_dim)
    
    return yaml_config, configs

def generate_yaml_config(configs, input_dim, output_dim):
    """生成YAML配置"""
    yaml_config = {}
    
    # MLP配置
    if configs['mlp']:
        yaml_config['mlp'] = {
            'model_type': 'mlp',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'hidden_dims': configs['mlp']['hidden_dims'],
            'activation': 'gelu',
            'dropout': 0.1,
            'use_batch_norm': True,
            'use_residual': True
        }
    
    # Transformer配置
    if configs['transformer']:
        yaml_config['transformer'] = {
            'model_type': 'transformer',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'd_model': configs['transformer']['d_model'],
            'num_heads': configs['transformer']['num_heads'],
            'num_layers': configs['transformer']['num_layers'],
            'dropout': 0.1,
            'seq_len': 11,
            'pe_type': 'sinusoidal_1d',
            'use_memory_film': False,
            'max_time_steps': 100,
            'output_head_type': 'global',
            'time_encoding': 'embedding'
        }
    
    # CustomTransformer配置
    if configs['custom_transformer']:
        yaml_config['custom_transformer'] = {
            'model_type': 'custom_transformer',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'd_model': configs['custom_transformer']['d_model'],
            'num_heads': configs['custom_transformer']['num_heads'],
            'num_layers': configs['custom_transformer']['num_layers'],
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
    
    # FNO配置
    if configs['fno']:
        yaml_config['fno'] = {
            'model_type': 'fno',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'width': configs['fno']['width'],
            'num_layers': configs['fno']['num_layers'],
            'activation': 'gelu',
            'input_resolution': [32, 32],
            'output_resolution': [128, 128]
        }
    
    # UNet配置
    if configs['unet']:
        yaml_config['unet'] = {
            'model_type': 'unet',
            'input_dim': input_dim,
            'output_dim': output_dim,
            'in_channels': 1,
            'out_channels': 1,
            'features': configs['unet']['features'],
            'activation': 'relu',
            'use_attention': False,
            'input_resolution': [32, 32],
            'output_resolution': [128, 128]
        }
    
    return yaml_config

def save_fair_config(yaml_config):
    """保存公平对比配置"""
    # 加载基础配置
    base_config_path = "configs/real_data_crop_config.yaml"
    
    with open(base_config_path, 'r', encoding='utf-8') as f:
        base_config = yaml.safe_load(f)
    
    # 更新模型配置
    base_config['models'] = yaml_config
    
    # 保存新配置
    output_path = "configs/fair_comparison_config.yaml"
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(base_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n保存公平对比配置: {output_path}")
    
    return output_path

if __name__ == "__main__":
    print("创建公平对比配置")
    print("="*60)
    
    # 创建配置
    yaml_config, configs = create_fair_comparison_config()
    
    # 保存配置
    config_path = save_fair_config(yaml_config)
    
    print("\n配置创建完成！")
    print(f"使用配置文件: {config_path}")
    print("\n建议:")
    print("1. 所有模型参数量控制在50万左右")
    print("2. 可以进行公平的横向对比")
    print("3. 运行测试: python run_crop_model_test.py --config configs/fair_comparison_config.yaml")