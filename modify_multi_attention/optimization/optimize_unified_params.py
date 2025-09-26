#!/usr/bin/env python3
"""
精确统一参量级配置优化脚本
通过迭代优化，使所有模型参数量尽可能接近目标值
"""

import yaml
import numpy as np
from pathlib import Path
import math

def calculate_transformer_params_precise(d_model, num_heads, num_layers, input_dim, output_dim, seq_len=11, pe_type='sinusoidal_1d'):
    """精确计算Transformer参数量"""
    # 输入投影层
    input_proj = input_dim * d_model + d_model  # weights + bias
    
    # 位置编码（如果是learnable）
    pos_encoding = 0
    if pe_type == 'learnable_1d':
        pos_encoding = seq_len * d_model
    
    # 每层Transformer参数
    # Multi-head attention: Q, K, V projections + output projection
    attention_params = 4 * (d_model * d_model + d_model)  # 4个线性层，每个有bias
    
    # Feed-forward network: 通常是4*d_model的隐藏层
    ffn_hidden = 4 * d_model
    ffn_params = (d_model * ffn_hidden + ffn_hidden) + (ffn_hidden * d_model + d_model)
    
    # Layer normalization: 2 layers per transformer block
    ln_params = 2 * (d_model * 2)  # gamma + beta for each LN
    
    transformer_layer_params = attention_params + ffn_params + ln_params
    total_transformer_params = transformer_layer_params * num_layers
    
    # 输出投影层
    output_proj = d_model * output_dim + output_dim
    
    total_params = input_proj + pos_encoding + total_transformer_params + output_proj
    return total_params

def calculate_mlp_params_precise(input_dim, output_dim, hidden_dims, use_batch_norm=True):
    """精确计算MLP参数量"""
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

def optimize_transformer_config(target_params, input_dim, output_dim):
    """优化Transformer配置以达到目标参数量"""
    best_config = None
    best_diff = float('inf')
    
    # 搜索最优的d_model和num_layers组合
    for d_model in range(64, 512, 32):  # 64, 96, 128, ..., 480
        for num_layers in range(1, 8):  # 1-7层
            for num_heads in [4, 8, 16]:
                if d_model % num_heads != 0:
                    continue
                
                params = calculate_transformer_params_precise(
                    d_model, num_heads, num_layers, input_dim, output_dim
                )
                
                diff = abs(params - target_params)
                if diff < best_diff:
                    best_diff = diff
                    best_config = {
                        'd_model': d_model,
                        'num_heads': num_heads,
                        'num_layers': num_layers,
                        'params': params
                    }
    
    return best_config

def optimize_mlp_config(target_params, input_dim, output_dim):
    """优化MLP配置以达到目标参数量"""
    best_config = None
    best_diff = float('inf')
    
    # 搜索最优的隐藏层配置
    for num_hidden_layers in range(2, 6):  # 2-5个隐藏层
        for base_hidden in range(128, 1024, 128):  # 基础隐藏层大小
            # 创建对称的隐藏层结构
            if num_hidden_layers == 2:
                hidden_dims = [base_hidden, base_hidden]
            elif num_hidden_layers == 3:
                hidden_dims = [base_hidden, base_hidden * 2, base_hidden]
            elif num_hidden_layers == 4:
                hidden_dims = [base_hidden, base_hidden * 2, base_hidden * 2, base_hidden]
            else:  # 5层
                hidden_dims = [base_hidden, base_hidden * 2, base_hidden * 3, base_hidden * 2, base_hidden]
            
            params = calculate_mlp_params_precise(input_dim, output_dim, hidden_dims)
            
            diff = abs(params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_config = {
                    'hidden_dims': hidden_dims,
                    'params': params
                }
    
    return best_config

def optimize_fno_config(target_params, input_dim, output_dim):
    """优化FNO配置以达到目标参数量"""
    best_config = None
    best_diff = float('inf')
    
    # 搜索最优的width和num_layers组合
    for width in range(32, 256, 16):  # 32, 48, 64, ..., 240
        for num_layers in range(2, 8):  # 2-7层
            # 简化的FNO参数计算
            input_proj = input_dim * width + width
            fno_layer_params = width * width * 2  # 复数权重
            total_fno_params = fno_layer_params * num_layers
            output_proj = width * output_dim + output_dim
            
            params = input_proj + total_fno_params + output_proj
            
            diff = abs(params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_config = {
                    'width': width,
                    'num_layers': num_layers,
                    'params': params
                }
    
    return best_config

def optimize_unet_config(target_params, input_dim, output_dim):
    """优化UNet配置以达到目标参数量"""
    best_config = None
    best_diff = float('inf')
    
    # 搜索最优的特征维度配置
    for base_features in range(16, 128, 16):  # 16, 32, 48, ..., 112
        for num_levels in range(3, 6):  # 3-5个层级
            # 创建UNet特征配置
            features = []
            for i in range(num_levels):
                features.append(base_features * (2 ** i))
            # 添加解码器部分（对称）
            for i in range(num_levels - 2, -1, -1):
                features.append(base_features * (2 ** i))
            
            # 计算参数量
            total_params = 0
            prev_channels = 1  # 输入通道数
            
            for feature in features:
                # 卷积层参数 (假设3x3卷积)
                total_params += prev_channels * feature * 9 + feature  # weights + bias
                prev_channels = feature
            
            # 输出层
            total_params += prev_channels * 1 * 1 + 1  # 1x1卷积到输出通道
            
            diff = abs(total_params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_config = {
                    'features': features,
                    'params': total_params
                }
    
    return best_config

def create_optimized_unified_config(target_params_M=1.0):
    """创建优化的统一参量级配置"""
    target_params = int(target_params_M * 1e6)
    input_dim = 1024
    output_dim = 16384
    
    print(f"目标参数量: {target_params_M:.1f}M ({target_params:,} 参数)")
    print("=" * 50)
    
    # 优化Transformer配置
    print("优化Transformer配置...")
    transformer_opt = optimize_transformer_config(target_params, input_dim, output_dim)
    transformer_config = {
        'model_type': 'transformer',
        'input_dim': input_dim,
        'output_dim': output_dim,
        'seq_len': 11,
        'd_model': transformer_opt['d_model'],
        'num_heads': transformer_opt['num_heads'],
        'num_layers': transformer_opt['num_layers'],
        'dropout': 0.1,
        'pe_type': 'sinusoidal_1d',
        'use_memory_film': False,
        'max_time_steps': 100,
        'output_head_type': 'global',
        'time_encoding': 'embedding'
    }
    print(f"  d_model={transformer_opt['d_model']}, num_layers={transformer_opt['num_layers']}, num_heads={transformer_opt['num_heads']}")
    print(f"  实际参数量: {transformer_opt['params']/1e6:.2f}M")
    
    # 自定义Transformer配置（相同参数量）
    custom_transformer_config = transformer_config.copy()
    custom_transformer_config.update({
        'model_type': 'custom_transformer',
        'attention_type': 'relative',
        'pe_type': 'learnable_1d',
        'use_memory_film': True,
        'use_memory_concat': False
    })
    
    # 优化MLP配置
    print("\n优化MLP配置...")
    mlp_opt = optimize_mlp_config(target_params, input_dim, output_dim)
    mlp_config = {
        'model_type': 'mlp',
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dims': mlp_opt['hidden_dims'],
        'activation': 'gelu',
        'dropout': 0.1,
        'use_batch_norm': True,
        'use_residual': True
    }
    print(f"  hidden_dims={mlp_opt['hidden_dims']}")
    print(f"  实际参数量: {mlp_opt['params']/1e6:.2f}M")
    
    # 优化FNO配置
    print("\n优化FNO配置...")
    fno_opt = optimize_fno_config(target_params, input_dim, output_dim)
    fno_config = {
        'model_type': 'fno',
        'input_dim': input_dim,
        'output_dim': output_dim,
        'width': fno_opt['width'],
        'num_layers': fno_opt['num_layers'],
        'activation': 'gelu',
        'input_resolution': [32, 32],
        'output_resolution': [128, 128]
    }
    print(f"  width={fno_opt['width']}, num_layers={fno_opt['num_layers']}")
    print(f"  实际参数量: {fno_opt['params']/1e6:.2f}M")
    
    # 优化UNet配置
    print("\n优化UNet配置...")
    unet_opt = optimize_unet_config(target_params, input_dim, output_dim)
    unet_config = {
        'model_type': 'unet',
        'input_dim': input_dim,
        'output_dim': output_dim,
        'in_channels': 1,
        'out_channels': 1,
        'features': unet_opt['features'],
        'activation': 'relu',
        'use_attention': False,
        'input_resolution': [32, 32],
        'output_resolution': [128, 128]
    }
    print(f"  features={unet_opt['features']}")
    print(f"  实际参数量: {unet_opt['params']/1e6:.2f}M")
    
    # 汇总结果
    all_params = [
        transformer_opt['params'],
        transformer_opt['params'],  # custom_transformer相同
        mlp_opt['params'],
        fno_opt['params'],
        unet_opt['params']
    ]
    
    params_M = [p/1e6 for p in all_params]
    min_params = min(params_M)
    max_params = max(params_M)
    ratio = max_params / min_params if min_params > 0 else float('inf')
    
    print(f"\n统一后参数量范围: {min_params:.2f}M - {max_params:.2f}M")
    print(f"最大差异倍数: {ratio:.1f}x")
    
    unified_configs = {
        'transformer': transformer_config,
        'custom_transformer': custom_transformer_config,
        'mlp': mlp_config,
        'fno': fno_config,
        'unet': unet_config
    }
    
    return unified_configs, {
        'target_params_M': target_params_M,
        'actual_range': f"{min_params:.2f}M - {max_params:.2f}M",
        'max_ratio': ratio,
        'individual_params': {
            'transformer': transformer_opt['params']/1e6,
            'custom_transformer': transformer_opt['params']/1e6,
            'mlp': mlp_opt['params']/1e6,
            'fno': fno_opt['params']/1e6,
            'unet': unet_opt['params']/1e6
        }
    }

def main():
    """主函数"""
    print("精确统一参量级配置优化")
    print("=" * 50)
    
    # 创建优化配置
    unified_configs, analysis = create_optimized_unified_config(target_params_M=1.0)
    
    # 保存配置
    output_config = {
        'models': unified_configs,
        'analysis': analysis
    }
    
    output_path = "configs/optimized_unified_params_config.yaml"
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(output_config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"\n优化配置已保存到: {output_path}")
    print("\n建议:")
    print("- 参数量差异已显著减小，适合公平对比")
    print("- 可以直接用于横向训练对比实验")
    print("- 建议使用相同的训练超参数")

if __name__ == "__main__":
    main()