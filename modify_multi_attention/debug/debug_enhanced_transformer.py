#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试EnhancedTransformer1d的实际行为
"""

import torch
import torch.nn as nn
import numpy as np
from models.enhanced_transformer import EnhancedTransformer1d

def debug_enhanced_transformer():
    print("=== 调试EnhancedTransformer1d ===")
    
    # 创建模型实例，使用与run_crop_model_test.py相同的参数
    model = EnhancedTransformer1d(
        input_channels=1,
        output_channels=1,
        d_model=128,  # 默认值
        num_heads=4,
        num_layers=3,
        input_resolution=32,  # int(np.sqrt(1024)) = 32
        output_resolution=128,  # int(np.sqrt(16384)) = 128
        attention_type='simplified_self_attention',
        pe_type='learnable_1d'
    )
    
    print(f"模型创建成功")
    print(f"input_dim: {model.input_dim}")
    print(f"final_output_dim: {model.final_output_dim}")
    print(f"transformer_output_dim: {model.transformer_output_dim}")
    print(f"output_dim: {model.output_dim}")
    
    # 检查内部transformer的参数
    print(f"\n内部transformer参数:")
    print(f"  input_dim: {model.transformer.input_dim}")
    print(f"  output_dim: {model.transformer.output_dim}")
    
    # 检查fc_out层
    if hasattr(model.transformer, 'fc_out'):
        print(f"  fc_out: {model.transformer.fc_out}")
    if hasattr(model.transformer, 'token_head'):
        print(f"  token_head: {model.transformer.token_head}")
    if hasattr(model.transformer, 'fc'):
        print(f"  fc: {model.transformer.fc}")
    
    # 检查upsampler
    if hasattr(model, 'upsampler'):
        print(f"\nUpsampler:")
        for i, layer in enumerate(model.upsampler):
            print(f"  Layer {i}: {layer}")
    
    # 测试前向传播
    print(f"\n=== 测试前向传播 ===")
    
    # 创建测试输入 [batch_size, input_resolution, input_channels]
    batch_size = 1
    test_input = torch.randn(batch_size, 32, 1)  # [1, 32, 1]
    print(f"输入形状: {test_input.shape}")
    
    try:
        output = model(test_input)
        print(f"输出形状: {output.shape}")
        print(f"前向传播成功！")
    except Exception as e:
        print(f"前向传播失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_enhanced_transformer()