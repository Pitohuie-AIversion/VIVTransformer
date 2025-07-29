#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多GPU训练测试脚本

用于测试dynamic_resolution_trainer.py中的多GPU功能
"""

import torch
import sys
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent))

def test_gpu_availability():
    """
    测试GPU可用性
    """
    print("=== GPU可用性测试 ===")
    print(f"CUDA是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        
        # 测试GPU内存
        print("\n=== GPU内存测试 ===")
        for i in range(gpu_count):
            torch.cuda.set_device(i)
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i} - 已分配: {allocated:.2f}GB, 已缓存: {cached:.2f}GB")
    else:
        print("❌ CUDA不可用，无法进行多GPU训练")
        return False
    
    return gpu_count > 1

def test_dataparallel():
    """
    测试DataParallel功能
    """
    print("\n=== DataParallel测试 ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过DataParallel测试")
        return False
    
    gpu_count = torch.cuda.device_count()
    if gpu_count <= 1:
        print(f"❌ 只有 {gpu_count} 张GPU，无法测试DataParallel")
        return False
    
    try:
        # 创建简单的测试模型
        class SimpleModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(100, 50)
                self.output = torch.nn.Linear(50, 10)
            
            def forward(self, x):
                x = torch.relu(self.linear(x))
                return self.output(x)
        
        # 创建模型并移动到GPU
        model = SimpleModel().cuda()
        print(f"✅ 模型创建成功，参数数量: {sum(p.numel() for p in model.parameters())}")
        
        # 应用DataParallel
        model = torch.nn.DataParallel(model)
        print(f"✅ DataParallel应用成功，使用GPU: {list(range(gpu_count))}")
        
        # 测试前向传播
        batch_size = 32
        test_input = torch.randn(batch_size, 100).cuda()
        
        with torch.no_grad():
            output = model(test_input)
            print(f"✅ 前向传播测试成功，输入形状: {test_input.shape}, 输出形状: {output.shape}")
        
        # 测试反向传播
        criterion = torch.nn.MSELoss()
        target = torch.randn(batch_size, 10).cuda()
        loss = criterion(output, target)
        loss.backward()
        print(f"✅ 反向传播测试成功，损失值: {loss.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ DataParallel测试失败: {e}")
        return False

def test_memory_usage():
    """
    测试多GPU内存使用情况
    """
    print("\n=== 多GPU内存使用测试 ===")
    
    if not torch.cuda.is_available():
        print("❌ CUDA不可用，跳过内存测试")
        return
    
    gpu_count = torch.cuda.device_count()
    
    # 在每个GPU上分配一些内存
    tensors = []
    for i in range(gpu_count):
        try:
            # 分配100MB的张量
            tensor = torch.randn(1000, 1000, 25).cuda(i)
            tensors.append(tensor)
            
            allocated = torch.cuda.memory_allocated(i) / 1024**2
            print(f"GPU {i} - 分配内存: {allocated:.1f}MB")
        except Exception as e:
            print(f"GPU {i} - 内存分配失败: {e}")
    
    # 清理内存
    del tensors
    torch.cuda.empty_cache()
    print("✅ 内存清理完成")

def main():
    """
    主测试函数
    """
    print("多GPU训练功能测试")
    print("=" * 50)
    
    # 测试GPU可用性
    multi_gpu_available = test_gpu_availability()
    
    if multi_gpu_available:
        # 测试DataParallel
        dataparallel_success = test_dataparallel()
        
        # 测试内存使用
        test_memory_usage()
        
        print("\n=== 测试总结 ===")
        if dataparallel_success:
            print("✅ 多GPU训练功能测试通过")
            print("\n可以使用以下命令启用多GPU训练:")
            print("python dynamic_resolution_trainer.py --use_dataparallel")
            print("或在配置文件中设置: use_dataparallel: true")
        else:
            print("❌ 多GPU训练功能测试失败")
    else:
        print("\n⚠️ 多GPU环境不可用，建议:")
        print("1. 确保安装了支持CUDA的PyTorch")
        print("2. 确保系统有多张GPU")
        print("3. 确保GPU驱动正常")

if __name__ == "__main__":
    main()