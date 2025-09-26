#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试模型结构和参数
"""

import torch
import torch.nn as nn
from models.enhanced_transformer import EnhancedTransformer1d

def debug_model_structure():
    print("=== 调试模型结构 ===")
    
    # 使用与run_crop_model_test.py相同的参数
    model = EnhancedTransformer1d(
        input_channels=1,
        output_channels=1,
        d_model=128,
        num_heads=4,
        num_layers=3,
        input_resolution=32,
        output_resolution=128,
        attention_type='simplified_self_attention',
        pe_type='learnable_1d',
        time_encoding='embedding',
        max_time_steps=100,
        use_upsampling=True,
        input_dim=1024,  # 从配置文件传入的实际维度
        output_dim=16384  # 从配置文件传入的实际维度
    )
    
    print(f"模型类型: {type(model)}")
    print(f"input_dim: {model.input_dim}")
    print(f"final_output_dim: {model.final_output_dim}")
    print(f"transformer_output_dim: {model.transformer_output_dim}")
    print(f"output_dim: {model.output_dim}")
    
    print(f"\nTransformer类型: {type(model.transformer)}")
    print(f"Transformer input_dim: {model.transformer.input_dim}")
    print(f"Transformer output_dim: {model.transformer.output_dim}")
    print(f"Transformer fc层: {model.transformer.fc}")
    
    # 检查所有Linear层
    print("\n=== 所有Linear层 ===")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"{name}: {module}")
    
    # 测试前向传播
    print("\n=== 测试前向传播 ===")
    try:
        # 创建测试输入 - 注意这里使用的是1024维的输入
        x = torch.randn(1, 1024)  # [batch_size, input_dim]
        print(f"输入形状: {x.shape}")
        
        # 直接测试transformer
        time_steps = torch.zeros(1, 1, dtype=torch.long)
        transformer_output = model.transformer(x, time_steps)
        print(f"Transformer输出形状: {transformer_output.shape}")
        
    except Exception as e:
        print(f"前向传播错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_model_structure()