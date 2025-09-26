#!/usr/bin/env python3
"""
模型参数量分析脚本 - 统一参量级横向对比
分析当前配置文件中各模型的参数量，设计统一参量级的配置方案
"""

import yaml
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import os

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def calculate_transformer_params(config):
    """计算Transformer模型参数量"""
    d_model = config['d_model']
    num_heads = config['num_heads']
    num_layers = config['num_layers']
    input_dim = config['input_dim']
    output_dim = config['output_dim']
    seq_len = config.get('seq_len', 11)
    if seq_len is None:
        seq_len = 11  # 默认序列长度
    
    # 输入投影层
    input_proj = input_dim * d_model
    
    # 位置编码（如果是learnable）
    pos_encoding = 0
    if config.get('pe_type') == 'learnable_1d':
        pos_encoding = seq_len * d_model
    
    # 每层Transformer参数
    # Multi-head attention: Q, K, V projections + output projection
    attention_params = 4 * d_model * d_model
    # Feed-forward network: 通常是4*d_model的隐藏层
    ffn_params = d_model * (4 * d_model) + (4 * d_model) * d_model
    # Layer normalization: 2 layers per transformer block
    ln_params = 2 * d_model * 2
    
    transformer_layer_params = attention_params + ffn_params + ln_params
    total_transformer_params = transformer_layer_params * num_layers
    
    # 输出投影层
    output_proj = d_model * output_dim
    
    total_params = input_proj + pos_encoding + total_transformer_params + output_proj
    return total_params

def calculate_mlp_params(config):
    """计算MLP模型参数量"""
    input_dim = config['input_dim']
    output_dim = config['output_dim']
    hidden_dims = config['hidden_dims']
    use_batch_norm = config.get('use_batch_norm', False)
    
    total_params = 0
    prev_dim = input_dim
    
    for hidden_dim in hidden_dims:
        # 线性层参数
        total_params += prev_dim * hidden_dim + hidden_dim  # weights + bias
        
        # 批归一化参数
        if use_batch_norm:
            total_params += hidden_dim * 2  # gamma + beta
        
        prev_dim = hidden_dim
    
    # 输出层
    total_params += prev_dim * output_dim + output_dim
    
    return total_params

def calculate_fno_params(config):
    """计算FNO模型参数量（简化估算）"""
    width = config['width']
    num_layers = config['num_layers']
    input_dim = config['input_dim']
    output_dim = config['output_dim']
    
    # 输入投影
    input_proj = input_dim * width
    
    # FNO层参数（简化估算）
    # 每层包含傅里叶变换权重和线性变换
    fno_layer_params = width * width * 2  # 复数权重
    total_fno_params = fno_layer_params * num_layers
    
    # 输出投影
    output_proj = width * output_dim
    
    total_params = input_proj + total_fno_params + output_proj
    return total_params

def calculate_unet_params(config):
    """计算UNet模型参数量"""
    features = config['features']
    in_channels = config['in_channels']
    out_channels = config['out_channels']
    
    total_params = 0
    
    # 编码器
    prev_channels = in_channels
    for feature in features:
        # 卷积层参数 (假设3x3卷积)
        total_params += prev_channels * feature * 9 + feature  # weights + bias
        prev_channels = feature
    
    # 解码器（对称结构）
    for i in range(len(features) - 2, -1, -1):
        feature = features[i]
        # 上采样 + 卷积
        total_params += prev_channels * feature * 9 + feature
        prev_channels = feature
    
    # 输出层
    total_params += prev_channels * out_channels * 1 + out_channels
    
    return total_params

def analyze_current_config(config_path):
    """分析当前配置文件中各模型的参数量"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    models_config = config['models']
    results = {}
    
    for model_name, model_config in models_config.items():
        model_type = model_config['model_type']
        
        if model_type in ['transformer', 'custom_transformer']:
            params = calculate_transformer_params(model_config)
        elif model_type == 'mlp':
            params = calculate_mlp_params(model_config)
        elif model_type == 'fno':
            params = calculate_fno_params(model_config)
        elif model_type == 'unet':
            params = calculate_unet_params(model_config)
        else:
            params = 0
        
        results[model_name] = {
            'type': model_type,
            'params': params,
            'params_M': params / 1e6,  # 百万参数
            'config': model_config
        }
    
    return results

def design_unified_config(target_params_M=1.0):
    """设计统一参量级的模型配置"""
    target_params = int(target_params_M * 1e6)
    
    unified_configs = {}
    
    # Transformer配置（目标1M参数）
    # 通过调整d_model和num_layers来控制参数量
    transformer_config = {
        'model_type': 'transformer',
        'input_dim': 1024,
        'output_dim': 16384,
        'seq_len': 11,
        'd_model': 128,  # 减小模型维度
        'num_heads': 8,
        'num_layers': 2,  # 减少层数
        'dropout': 0.1,
        'pe_type': 'sinusoidal_1d',
        'use_memory_film': False,
        'max_time_steps': 100,
        'output_head_type': 'global',
        'time_encoding': 'embedding'
    }
    
    # 自定义Transformer配置
    custom_transformer_config = transformer_config.copy()
    custom_transformer_config.update({
        'model_type': 'custom_transformer',
        'attention_type': 'relative',
        'pe_type': 'learnable_1d',
        'use_memory_film': True,
        'use_memory_concat': False
    })
    
    # MLP配置（目标1M参数）
    mlp_config = {
        'model_type': 'mlp',
        'input_dim': 1024,
        'output_dim': 16384,
        'hidden_dims': [512, 1024, 512],  # 减少隐藏层维度
        'activation': 'gelu',
        'dropout': 0.1,
        'use_batch_norm': True,
        'use_residual': True
    }
    
    # FNO配置（目标1M参数）
    fno_config = {
        'model_type': 'fno',
        'input_dim': 1024,
        'output_dim': 16384,
        'width': 64,  # 减小宽度
        'num_layers': 3,  # 减少层数
        'activation': 'gelu',
        'input_resolution': [32, 32],
        'output_resolution': [128, 128]
    }
    
    # UNet配置（目标1M参数）
    unet_config = {
        'model_type': 'unet',
        'input_dim': 1024,
        'output_dim': 16384,
        'in_channels': 1,
        'out_channels': 1,
        'features': [32, 64, 128, 64, 32],  # 减少特征维度
        'activation': 'relu',
        'use_attention': False,
        'input_resolution': [32, 32],
        'output_resolution': [128, 128]
    }
    
    unified_configs = {
        'transformer': transformer_config,
        'custom_transformer': custom_transformer_config,
        'mlp': mlp_config,
        'fno': fno_config,
        'unet': unet_config
    }
    
    return unified_configs

def verify_unified_params(unified_configs):
    """验证统一配置的参数量"""
    results = {}
    
    for model_name, config in unified_configs.items():
        model_type = config['model_type']
        
        if model_type in ['transformer', 'custom_transformer']:
            params = calculate_transformer_params(config)
        elif model_type == 'mlp':
            params = calculate_mlp_params(config)
        elif model_type == 'fno':
            params = calculate_fno_params(config)
        elif model_type == 'unet':
            params = calculate_unet_params(config)
        else:
            params = 0
        
        results[model_name] = {
            'params': params,
            'params_M': params / 1e6
        }
    
    return results

def main():
    """主函数"""
    config_path = "configs/crop_model_with_normalization.yaml"
    
    print("=" * 60)
    print("模型参数量分析 - 统一参量级横向对比")
    print("=" * 60)
    
    # 分析当前配置
    print("\n1. 当前配置文件中各模型参数量分析:")
    print("-" * 40)
    current_results = analyze_current_config(config_path)
    
    for model_name, info in current_results.items():
        print(f"{model_name:20} ({info['type']:15}): {info['params_M']:8.2f}M 参数")
    
    # 计算参数量差异
    params_list = [info['params_M'] for info in current_results.values()]
    max_params = max(params_list)
    min_params = min(params_list)
    ratio = max_params / min_params if min_params > 0 else float('inf')
    
    print(f"\n参数量范围: {min_params:.2f}M - {max_params:.2f}M")
    print(f"最大差异倍数: {ratio:.1f}x")
    
    # 设计统一配置
    print("\n2. 统一参量级配置设计 (目标: ~1M参数):")
    print("-" * 40)
    unified_configs = design_unified_config(target_params_M=1.0)
    unified_results = verify_unified_params(unified_configs)
    
    for model_name, info in unified_results.items():
        print(f"{model_name:20}: {info['params_M']:8.2f}M 参数")
    
    # 验证统一性
    unified_params_list = [info['params_M'] for info in unified_results.values()]
    unified_max = max(unified_params_list)
    unified_min = min(unified_params_list)
    unified_ratio = unified_max / unified_min if unified_min > 0 else float('inf')
    
    print(f"\n统一后参数量范围: {unified_min:.2f}M - {unified_max:.2f}M")
    print(f"最大差异倍数: {unified_ratio:.1f}x")
    
    # 保存统一配置
    output_config = {
        'models': unified_configs,
        'analysis': {
            'target_params_M': 1.0,
            'actual_range': f"{unified_min:.2f}M - {unified_max:.2f}M",
            'max_ratio': unified_ratio
        }
    }
    
    output_path = "configs/unified_params_config.yaml"
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(output_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n3. 统一配置已保存到: {output_path}")
    print("\n建议:")
    print("- 所有模型参数量控制在1M左右，确保公平对比")
    print("- 可根据需要调整target_params_M参数")
    print("- 建议使用统一的训练配置和数据预处理")

if __name__ == "__main__":
    main()