#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试CUDA错误修复

这个脚本用于验证动态分辨率训练器的CUDA错误是否已经修复
"""

import torch
import sys
from pathlib import Path

# 添加路径
# 原有：sys.path.append(str(Path(__file__).parent.parent / 'modify_multi_attention'))
# 修复：定位到项目根目录，再追加 modify_multi_attention 以便导入 mymodels
project_root = Path(__file__).parents[2]
modify_multi_attention_path = project_root / 'modify_multi_attention'
sys.path.insert(0, str(modify_multi_attention_path))
sys.path.insert(0, str(project_root))

from mymodels.transformer import TransformerFlowReconstructionModel

def test_model_forward():
    """
    测试模型前向传播是否正常
    """
    print("=== 测试模型前向传播 ===")
    
    # 设置参数
    input_dim = 1024  # 32x32
    output_dim = 16384  # 128x128
    seq_len = 32  # sqrt(1024)
    batch_size = 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = TransformerFlowReconstructionModel(
        input_dim=input_dim,
        output_dim=output_dim,
        num_heads=8,
        num_layers=2,  # 减少层数以加快测试
        d_model=512,
        max_time_steps=100,
        attention_type="sge",
        seq_len=seq_len
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")
    
    # 创建测试数据
    x_in = torch.randn(batch_size, input_dim).to(device)
    x_time = torch.randint(0, 100, (batch_size,)).to(device)
    
    print(f"输入形状: {x_in.shape}")
    print(f"时间步形状: {x_time.shape}")
    
    try:
        # 前向传播
        with torch.no_grad():
            output = model(x_in, x_time)
        
        print(f"输出形状: {output.shape}")
        print(f"期望输出形状: ({batch_size}, {output_dim})")
        
        if output.shape == (batch_size, output_dim):
            print("✅ 测试通过！模型前向传播正常")
            return True
        else:
            print("❌ 测试失败！输出形状不匹配")
            return False
            
    except Exception as e:
        print(f"❌ 测试失败！错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_different_attention_types():
    """
    测试不同注意力机制
    """
    print("\n=== 测试不同注意力机制 ===")
    
    attention_types = ['sge', 'se', 'cbam', 'eca', 'external']
    input_dim = 1024
    output_dim = 16384
    seq_len = 32
    batch_size = 2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    for attention_type in attention_types:
        print(f"\n测试注意力机制: {attention_type}")
        
        try:
            model = TransformerFlowReconstructionModel(
                input_dim=input_dim,
                output_dim=output_dim,
                num_heads=8,
                num_layers=1,  # 单层测试
                d_model=256,   # 减小模型以加快测试
                max_time_steps=100,
                attention_type=attention_type,
                seq_len=seq_len
            ).to(device)
            
            x_in = torch.randn(batch_size, input_dim).to(device)
            x_time = torch.randint(0, 100, (batch_size,)).to(device)
            
            with torch.no_grad():
                output = model(x_in, x_time)
            
            if output.shape == (batch_size, output_dim):
                print(f"✅ {attention_type}: 通过")
                results[attention_type] = True
            else:
                print(f"❌ {attention_type}: 输出形状错误")
                results[attention_type] = False
                
        except Exception as e:
            print(f"❌ {attention_type}: 错误 - {str(e)}")
            results[attention_type] = False
    
    return results

if __name__ == "__main__":
    print("CUDA错误修复测试")
    print("=" * 50)
    
    # 测试基本前向传播
    basic_test = test_model_forward()
    
    # 测试不同注意力机制
    attention_results = test_different_attention_types()
    
    # 总结结果
    print("\n" + "=" * 50)
    print("测试结果总结:")
    print(f"基本前向传播: {'✅ 通过' if basic_test else '❌ 失败'}")
    
    print("\n注意力机制测试结果:")
    for attention_type, result in attention_results.items():
        status = '✅ 通过' if result else '❌ 失败'
        print(f"  {attention_type}: {status}")
    
    passed_count = sum(attention_results.values())
    total_count = len(attention_results)
    print(f"\n注意力机制通过率: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)")
    
    if basic_test and passed_count > 0:
        print("\n🎉 CUDA错误修复成功！")
    else:
        print("\n⚠️ 仍有问题需要解决")