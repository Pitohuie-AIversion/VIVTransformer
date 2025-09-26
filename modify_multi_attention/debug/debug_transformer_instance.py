#!/usr/bin/env python3
"""
调试TransformerFlowReconstructionModel实例的具体参数
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

from mymodels.transformer import TransformerFlowReconstructionModel

def debug_transformer_instance():
    """调试TransformerFlowReconstructionModel实例"""
    print("=== 调试TransformerFlowReconstructionModel实例 ===")
    
    # 使用与run_crop_model_test.py相同的参数
    input_dim = 1024
    output_dim = 16384
    d_model = 256
    num_heads = 8
    num_layers = 3
    
    print(f"创建参数: input_dim={input_dim}, output_dim={output_dim}, d_model={d_model}")
    
    # 创建模型实例
    model = TransformerFlowReconstructionModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        d_model=d_model,
        max_time_steps=100,
        attention_type="relative",
        seq_len=None,
        pe_type='learnable_1d',
        output_head_type='global',
        time_encoding='embedding',
        use_memory_film=True,
        use_memory_concat=False
    )
    
    print(f"模型类型: {type(model)}")
    print(f"模型d_model: {model.d_model}")
    print(f"模型input_dim: {model.input_dim}")
    print(f"模型output_dim: {model.output_dim}")
    print(f"模型seq_len: {model.seq_len}")
    print(f"模型output_head_type: {model.output_head_type}")
    
    # 检查fc_out层
    if hasattr(model, 'fc_out'):
        print(f"fc_out层: {model.fc_out}")
        print(f"fc_out输入维度: {model.fc_out.in_features}")
        print(f"fc_out输出维度: {model.fc_out.out_features}")
    else:
        print("模型没有fc_out层")
    
    # 检查token_head层
    if hasattr(model, 'token_head'):
        print(f"token_head层: {model.token_head}")
        print(f"token_head输入维度: {model.token_head.in_features}")
        print(f"token_head输出维度: {model.token_head.out_features}")
    else:
        print("模型没有token_head层")
    
    # 检查embedding层
    print(f"embedding层: {model.embedding}")
    print(f"embedding输入维度: {model.embedding.in_features}")
    print(f"embedding输出维度: {model.embedding.out_features}")
    
    # 测试前向传播
    print("\n=== 测试前向传播 ===")
    test_input = torch.randn(1, input_dim)
    test_time_steps = torch.tensor([0], dtype=torch.long)
    
    try:
        with torch.no_grad():
            output = model(test_input, test_time_steps)
        print(f"前向传播成功!")
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")
        print(f"期望输出形状: [1, {output_dim}]")
        
        if output.shape[1] == output_dim:
            print("✅ 输出维度正确!")
        else:
            print(f"❌ 输出维度错误! 期望{output_dim}, 实际{output.shape[1]}")
            
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        print(f"错误类型: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_transformer_instance()