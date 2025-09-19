#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
详细分析所有模型的参数量构成和优化策略

作者: AI Assistant
日期: 2025
"""

import torch
import yaml
import sys
from pathlib import Path
import math

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

def analyze_model_structure(model, model_name):
    """分析模型结构和参数分布"""
    print(f"\n=== {model_name} 模型结构分析 ===")
    total_params = 0
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            param_count = param.numel()
            total_params += param_count
            print(f"  {name}: {param.shape} -> {param_count:,} 参数")
    
    print(f"  总参数量: {total_params:,}")
    return total_params

def find_optimal_mlp_config(input_dim, output_dim, target_params):
    """寻找MLP的最优配置"""
    print(f"\n寻找MLP最优配置 (目标: {target_params:,} 参数)")
    
    best_config = None
    best_diff = float('inf')
    
    # 尝试不同的隐藏层配置
    for num_layers in [1, 2, 3]:
        for hidden_dim in range(64, 2048, 64):
            if num_layers == 1:
                hidden_dims = [hidden_dim]
            elif num_layers == 2:
                hidden_dims = [hidden_dim, hidden_dim // 2]
            else:
                hidden_dims = [hidden_dim, hidden_dim, hidden_dim // 2]
            
            # 计算参数量
            params = input_dim * hidden_dims[0] + hidden_dims[0]  # 第一层
            for i in range(len(hidden_dims) - 1):
                params += hidden_dims[i] * hidden_dims[i+1] + hidden_dims[i+1]
            params += hidden_dims[-1] * output_dim + output_dim  # 输出层
            
            diff = abs(params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_config = {
                    'hidden_dims': hidden_dims,
                    'params': params,
                    'diff': diff
                }
    
    print(f"  最优配置: {best_config['hidden_dims']}")
    print(f"  参数量: {best_config['params']:,} (差异: {best_config['diff']:,})")
    return best_config

def find_optimal_transformer_config(input_dim, output_dim, target_params):
    """寻找Transformer的最优配置"""
    print(f"\n寻找Transformer最优配置 (目标: {target_params:,} 参数)")
    
    best_config = None
    best_diff = float('inf')
    
    for d_model in [64, 96, 128, 192, 256, 320, 384]:
        for num_heads in [2, 4, 6, 8]:
            if d_model % num_heads != 0:
                continue
            for num_layers in [1, 2, 3, 4]:
                # 估算参数量
                # 输入投影
                params = input_dim * d_model + d_model
                
                # Transformer层
                for _ in range(num_layers):
                    # 自注意力
                    params += 3 * d_model * d_model + 3 * d_model  # Q, K, V
                    params += d_model * d_model + d_model  # 输出投影
                    params += 2 * d_model  # LayerNorm
                    
                    # FFN
                    ffn_dim = 4 * d_model
                    params += d_model * ffn_dim + ffn_dim
                    params += ffn_dim * d_model + d_model
                    params += 2 * d_model  # LayerNorm
                
                # 输出投影
                params += d_model * output_dim + output_dim
                
                diff = abs(params - target_params)
                if diff < best_diff:
                    best_diff = diff
                    best_config = {
                        'd_model': d_model,
                        'num_heads': num_heads,
                        'num_layers': num_layers,
                        'params': params,
                        'diff': diff
                    }
    
    print(f"  最优配置: d_model={best_config['d_model']}, heads={best_config['num_heads']}, layers={best_config['num_layers']}")
    print(f"  参数量: {best_config['params']:,} (差异: {best_config['diff']:,})")
    return best_config

def find_optimal_fno_config(input_dim, output_dim, target_params):
    """寻找FNO的最优配置"""
    print(f"\n寻找FNO最优配置 (目标: {target_params:,} 参数)")
    
    best_config = None
    best_diff = float('inf')
    
    # FNO参数量主要由modes和width决定
    for width in [16, 24, 32, 48, 64, 96]:
        for modes in [8, 12, 16, 20, 24]:
            for num_layers in [2, 3, 4]:
                # 估算参数量 (简化计算)
                params = input_dim * width + width  # 输入投影
                
                # FNO层
                for _ in range(num_layers):
                    # 频域卷积
                    params += width * width * modes * modes * 2  # 复数权重
                    params += width * width + width  # 空域卷积
                    params += width  # bias
                
                params += width * output_dim + output_dim  # 输出投影
                
                diff = abs(params - target_params)
                if diff < best_diff:
                    best_diff = diff
                    best_config = {
                        'width': width,
                        'modes': modes,
                        'num_layers': num_layers,
                        'params': params,
                        'diff': diff
                    }
    
    print(f"  最优配置: width={best_config['width']}, modes={best_config['modes']}, layers={best_config['num_layers']}")
    print(f"  参数量: {best_config['params']:,} (差异: {best_config['diff']:,})")
    return best_config

def find_optimal_unet_config(input_dim, output_dim, target_params):
    """寻找UNet的最优配置"""
    print(f"\n寻找UNet最优配置 (目标: {target_params:,} 参数)")
    
    best_config = None
    best_diff = float('inf')
    
    for base_channels in [16, 24, 32, 48, 64]:
        for num_levels in [2, 3, 4]:
            # 估算参数量 (简化计算)
            params = 0
            current_channels = base_channels
            
            # 编码器
            for level in range(num_levels):
                if level == 0:
                    params += 1 * current_channels * 9 + current_channels  # 第一层卷积
                else:
                    prev_channels = current_channels // 2
                    params += prev_channels * current_channels * 9 + current_channels
                
                params += current_channels * current_channels * 9 + current_channels  # 第二层卷积
                current_channels *= 2
            
            # 解码器
            for level in range(num_levels):
                prev_channels = current_channels
                current_channels //= 2
                params += prev_channels * current_channels * 4 + current_channels  # 上采样
                params += (current_channels * 2) * current_channels * 9 + current_channels  # 卷积
                params += current_channels * current_channels * 9 + current_channels
            
            # 输出层
            params += current_channels * 1 * 1 + 1
            
            diff = abs(params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_config = {
                    'base_channels': base_channels,
                    'num_levels': num_levels,
                    'params': params,
                    'diff': diff
                }
    
    print(f"  最优配置: base_channels={best_config['base_channels']}, levels={best_config['num_levels']}")
    print(f"  参数量: {best_config['params']:,} (差异: {best_config['diff']:,})")
    return best_config

def generate_ultra_fair_config(target_params=5000000):
    """生成超公平配置"""
    input_dim = 1024
    output_dim = 16384
    
    print(f"生成超公平配置 (目标参数量: {target_params:,})")
    print("="*70)
    
    # 寻找各模型最优配置
    mlp_config = find_optimal_mlp_config(input_dim, output_dim, target_params)
    transformer_config = find_optimal_transformer_config(input_dim, output_dim, target_params)
    fno_config = find_optimal_fno_config(input_dim, output_dim, target_params)
    unet_config = find_optimal_unet_config(input_dim, output_dim, target_params)
    
    # 生成配置文件
    config = {
        'data': {
            'data_path': "x:/2025/Graduation_project/Pdebench_input_Transformer/VIVTransformer-1/generate_data/preprocessed_data/darcy_flow_input32_output128_dataset.h5",
            'input_dim': input_dim,
            'output_dim': output_dim,
            'crop_mode': "center",
            'output_resolution': [128, 128],
            'output_channels': 1,
            'batch_size': 32,
            'val_split': 0.2,
            'test_split': 0.1,
            'max_samples': 1000,
            'preprocessing': {
                'normalize': True,
                'remove_channel_dim': True
            }
        },
        'training': {
            'epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'patience': 10,
            'min_delta': 0.0001
        },
        'loss': {
            'mse_only': True,
            'mse_weight': 1.0,
            'l1_weight': 0.0,
            'ssim_weight': 0.0,
            'perceptual_weight': 0.0,
            'gradient_weight': 0.0,
            'frequency_weight': 0.0
        },
        'attention': {
            'type': "self"
        },
        'models': {
            'mlp': {
                'model_type': 'mlp',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'hidden_dims': mlp_config['hidden_dims'],
                'activation': 'gelu',
                'dropout': 0.1,
                'use_batch_norm': True,
                'use_residual': False
            },
            'transformer': {
                'model_type': 'transformer',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'd_model': transformer_config['d_model'],
                'num_heads': transformer_config['num_heads'],
                'num_layers': transformer_config['num_layers'],
                'dropout': 0.1,
                'seq_len': 11,
                'pe_type': 'sinusoidal_1d',
                'use_memory_film': False,
                'max_time_steps': 100,
                'output_head_type': 'global',
                'time_encoding': 'embedding'
            },
            'custom_transformer': {
                'model_type': 'custom_transformer',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'd_model': 24,  # 极小配置
                'num_heads': 2,
                'num_layers': 1,
                'dropout': 0.1,
                'attention_type': "relative",
                'seq_len': None,
                'pe_type': "learnable_1d",
                'output_head_type': "global",
                'time_encoding': "embedding",
                'use_memory_film': False,
                'use_memory_concat': False,
                'max_time_steps': 100
            },
            'fno': {
                'model_type': 'fno',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'width': fno_config['width'],
                'modes': fno_config['modes'],
                'num_layers': fno_config['num_layers'],
                'activation': 'gelu',
                'resolution': [32, 32]
            },
            'unet': {
                'model_type': 'unet',
                'input_dim': input_dim,
                'output_dim': output_dim,
                'base_channels': unet_config['base_channels'],
                'num_levels': unet_config['num_levels'],
                'activation': 'gelu',
                'resolution': [32, 32]
            }
        },
        'device': {
            'use_cuda': True,
            'device_id': 0,
            'mixed_precision': False
        },
        'visualization': {
            'enabled': True,
            'interval': 10,
            'max_samples': 5
        },
        'output': {
            'results_dir': "./ultra_fair_results",
            'save_models': True,
            'save_predictions': True,
            'generate_plots': True
        }
    }
    
    return config

if __name__ == "__main__":
    print("详细分析所有模型参数量构成")
    print("="*70)
    
    # 生成超公平配置
    config = generate_ultra_fair_config(target_params=5000000)
    
    # 保存配置
    config_path = "modify_multi_attention/configs/ultra_fair_all_models_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n超公平配置已保存到: {config_path}")
    print("\n建议:")
    print("1. 所有模型参数量控制在500万左右")
    print("2. FNO和UNet使用极简配置")
    print("3. CustomTransformer大幅简化")
    print("4. 确保公平对比的科学性")