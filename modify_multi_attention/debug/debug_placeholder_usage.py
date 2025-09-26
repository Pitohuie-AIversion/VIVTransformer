#!/usr/bin/env python3
"""
调试占位符类是否被正确使用
"""

import sys
import torch
import torch.nn as nn
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

# 强制导入失败来测试占位符类
sys.modules['transformer'] = None

from models.enhanced_transformer import EnhancedTransformer2d

def debug_placeholder_usage():
    """调试占位符类是否被正确使用"""
    print("=== 调试占位符类使用情况 ===")
    
    # 创建EnhancedTransformer2d实例
    model = EnhancedTransformer2d(
        input_channels=1,
        output_channels=1,
        d_model=256,
        num_heads=8,
        num_layers=3,
        input_resolution=(32, 32),
        output_resolution=(128, 128)
    )
    
    print(f"模型类型: {type(model)}")
    print(f"transformer类型: {type(model.transformer)}")
    print(f"transformer模块: {model.transformer.__class__.__module__}")
    
    # 检查transformer的fc层
    if hasattr(model.transformer, 'fc'):
        print(f"transformer.fc层: {model.transformer.fc}")
        print(f"fc输入维度: {model.transformer.fc.in_features}")
        print(f"fc输出维度: {model.transformer.fc.out_features}")
    else:
        print("transformer没有fc层")
    
    # 检查transformer的其他属性
    print(f"transformer.input_dim: {getattr(model.transformer, 'input_dim', 'N/A')}")
    print(f"transformer.output_dim: {getattr(model.transformer, 'output_dim', 'N/A')}")
    
    # 测试前向传播
    print("\n=== 测试前向传播 ===")
    test_input = torch.randn(1, 32, 32, 1)
    
    try:
        with torch.no_grad():
            output = model(test_input)
        print(f"前向传播成功!")
        print(f"输入形状: {test_input.shape}")
        print(f"输出形状: {output.shape}")
        
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
        print(f"错误类型: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_placeholder_usage()