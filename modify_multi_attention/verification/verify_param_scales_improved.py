#!/usr/bin/env python3
"""
验证小中大参数量配置的参数计数脚本（改进版）
使用简单模型确保配置正确应用
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
from run_crop_model_test import create_simple_model, SimpleMLP, SimpleTransformer

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

class SimpleUNet(nn.Module):
    """简单的UNet实现，支持可配置的通道数"""
    def __init__(self, input_dim, output_dim, channels=[32, 64, 128], **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 计算空间维度
        spatial_dim = int(np.sqrt(input_dim))
        
        # 编码器
        self.encoder = nn.ModuleList()
        in_ch = 1
        for ch in channels:
            self.encoder.append(nn.Sequential(
                nn.Conv1d(in_ch, ch, 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(ch, ch, 3, padding=1),
                nn.ReLU()
            ))
            in_ch = ch
        
        # 解码器
        self.decoder = nn.ModuleList()
        for i in range(len(channels)-1, 0, -1):
            self.decoder.append(nn.Sequential(
                nn.Conv1d(channels[i] + channels[i-1], channels[i-1], 3, padding=1),
                nn.ReLU(),
                nn.Conv1d(channels[i-1], channels[i-1], 3, padding=1),
                nn.ReLU()
            ))
        
        # 输出层
        self.output_conv = nn.Conv1d(channels[0], 1, 1)
        
    def forward(self, x):
        # 重塑输入
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, -1)
        
        # 编码
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            if x.shape[-1] > 1:
                x = nn.functional.max_pool1d(x, 2)
        
        # 解码
        skip_connections = skip_connections[:-1][::-1]
        for i, decoder in enumerate(self.decoder):
            x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            if i < len(skip_connections):
                x = torch.cat([x, skip_connections[i]], dim=1)
            x = decoder(x)
        
        # 输出
        x = self.output_conv(x)
        return x.view(batch_size, -1)

class SimpleFNO(nn.Module):
    """简单的FNO实现，支持可配置的参数"""
    def __init__(self, input_dim, output_dim, modes=[8, 8], width=32, num_layers=2, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modes = modes[0] if isinstance(modes, list) else modes
        self.width = width
        self.num_layers = num_layers
        
        # 输入投影
        self.input_proj = nn.Linear(input_dim, width)
        
        # FNO层
        self.fno_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.fno_layers.append(nn.Sequential(
                nn.Linear(width, width * 2),
                nn.ReLU(),
                nn.Linear(width * 2, width)
            ))
        
        # 输出投影
        self.output_proj = nn.Linear(width, output_dim)
        
    def forward(self, x):
        x = self.input_proj(x)
        
        for layer in self.fno_layers:
            residual = x
            x = layer(x)
            x = x + residual  # 残差连接
        
        x = self.output_proj(x)
        return x

def create_model_from_config(model_type: str, model_config: Dict[str, Any], input_dim: int = 1024, output_dim: int = 1024):
    """根据配置创建模型"""
    try:
        if model_type == 'mlp':
            model = SimpleMLP(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_dims=model_config.get('hidden_dims', [256]),
                activation=model_config.get('activation', 'relu'),
                dropout=model_config.get('dropout', 0.1),
                use_batch_norm=model_config.get('use_batch_norm', True),
                use_residual=model_config.get('use_residual', False)
            )
        elif model_type == 'transformer':
            model = SimpleTransformer(
                input_dim=input_dim,
                output_dim=output_dim,
                d_model=model_config.get('d_model', 128),
                num_heads=model_config.get('num_heads', 4),
                num_layers=model_config.get('num_layers', 2),
                dropout=model_config.get('dropout', 0.1),
                seq_len=model_config.get('seq_len', 32),
                pe_type=model_config.get('pe_type', 'sinusoidal_1d'),
                use_memory_film=model_config.get('use_memory_film', False),
                max_time_steps=model_config.get('max_time_steps', 100),
                output_head_type=model_config.get('output_head_type', 'global'),
                time_encoding=model_config.get('time_encoding', 'embedding')
            )
        elif model_type == 'unet':
            model = SimpleUNet(
                input_dim=input_dim,
                output_dim=output_dim,
                channels=model_config.get('channels', [32, 64, 128])
            )
        elif model_type == 'fno':
            model = SimpleFNO(
                input_dim=input_dim,
                output_dim=output_dim,
                modes=model_config.get('modes', [8, 8]),
                width=model_config.get('width', 32),
                num_layers=model_config.get('num_layers', 2)
            )
        else:
            model = nn.Linear(input_dim, output_dim)
        
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
    print("🔍 验证参数量配置（改进版）")
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