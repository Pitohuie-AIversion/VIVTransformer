#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
演示如何使用修复后的鲁棒SVD损失函数
"""

import torch
import torch.nn as nn
import numpy as np
from svd10_loss import TotalLossWithSVD

def create_sample_model():
    """创建一个简单的示例模型"""
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 1, 3, padding=1)
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.conv2(x)
            return x
    
    return SimpleModel()

def create_sample_data(batch_size=4, size=32):
    """创建示例训练数据"""
    # 创建输入数据
    input_data = torch.randn(batch_size, 1, size, size)
    
    # 创建目标数据（添加一些噪声模拟真实情况）
    target_data = torch.randn(batch_size, 1, size, size)
    
    return input_data, target_data

def demo_basic_usage():
    """演示基本使用方法"""
    print("=== 基本使用演示 ===")
    
    # 创建模型和数据
    model = create_sample_model()
    input_data, target_data = create_sample_data()
    
    # 创建鲁棒SVD损失函数
    loss_fn = TotalLossWithSVD(
        base_weight=0.6,    # 60% 基础MSE损失
        topk=5             # 使用前5个SVD模态
    )
    
    # 显示权重信息
    print("\n损失函数权重配置:")
    loss_fn.print_weight_info()
    
    # 前向传播
    model.eval()
    with torch.no_grad():
        pred = model(input_data)
        # 调整形状以匹配损失函数要求
        pred_flat = pred.view(pred.size(0), -1)
        target_flat = target_data.view(target_data.size(0), -1)
        
        # 计算损失
        loss = loss_fn(pred_flat, target_flat)
        print(f"\n计算得到的损失值: {loss.item():.6f}")

def demo_training_loop():
    """演示训练循环中的使用"""
    print("\n=== 训练循环演示 ===")
    
    # 创建模型和优化器
    model = create_sample_model()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 创建损失函数
    loss_fn = TotalLossWithSVD(
        base_weight=0.5,
        svd_weights=[0.1, 0.08, 0.06, 0.04, 0.02],  # 自定义权重
        topk=5
    )
    
    print("开始训练演示...")
    
    # 模拟训练循环
    for epoch in range(3):
        model.train()
        epoch_loss = 0.0
        
        # 模拟多个batch
        for batch_idx in range(2):
            # 创建batch数据
            input_data, target_data = create_sample_data(batch_size=2)
            
            # 前向传播
            pred = model(input_data)
            pred_flat = pred.view(pred.size(0), -1)
            target_flat = target_data.view(target_data.size(0), -1)
            
            # 计算损失
            loss = loss_fn(pred_flat, target_flat)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss / 2
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.6f}\n")

def demo_problematic_data():
    """演示处理问题数据的能力"""
    print("=== 问题数据处理演示 ===")
    
    # 创建损失函数
    loss_fn = TotalLossWithSVD(base_weight=0.5, topk=3)
    
    print("\n1. 测试病态矩阵:")
    # 创建病态矩阵（接近奇异）
    pred = torch.ones(2, 64) * 1e-8
    target = torch.ones(2, 64) * 1e-8
    pred[0, 0] = 1.0
    target[0, 0] = 1.1
    
    loss = loss_fn(pred, target)
    print(f"病态矩阵损失: {loss.item():.6f}")
    
    print("\n2. 测试包含NaN的数据:")
    pred = torch.randn(2, 64)
    target = torch.randn(2, 64)
    pred[0, 0] = float('nan')
    
    loss = loss_fn(pred, target)
    print(f"NaN数据损失: {loss.item():.6f}")
    
    print("\n3. 测试极值数据:")
    pred = torch.ones(2, 64) * 1e10
    target = torch.ones(2, 64) * 1e10
    
    loss = loss_fn(pred, target)
    print(f"极值数据损失: {loss.item():.6f}")

def demo_gpu_usage():
    """演示GPU使用"""
    print("\n=== GPU使用演示 ===")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name()}")
        
        # 创建GPU上的模型和数据
        model = create_sample_model().to(device)
        input_data, target_data = create_sample_data()
        input_data = input_data.to(device)
        target_data = target_data.to(device)
        
        # 创建损失函数
        loss_fn = TotalLossWithSVD(base_weight=0.5, topk=3)
        
        # 计算损失
        model.eval()
        with torch.no_grad():
            pred = model(input_data)
            pred_flat = pred.view(pred.size(0), -1)
            target_flat = target_data.view(target_data.size(0), -1)
            
            loss = loss_fn(pred_flat, target_flat)
            print(f"GPU损失计算成功: {loss.item():.6f}")
            
        print(f"GPU内存使用: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
    else:
        print("CUDA不可用，跳过GPU演示")

def demo_weight_analysis():
    """演示权重分析功能"""
    print("\n=== 权重分析演示 ===")
    
    # 创建不同权重配置的损失函数
    configs = [
        {"name": "均衡配置", "base_weight": 0.5, "svd_weights": None},
        {"name": "MSE主导", "base_weight": 0.8, "svd_weights": [0.04, 0.04, 0.04, 0.04, 0.04]},
        {"name": "SVD主导", "base_weight": 0.2, "svd_weights": [0.16, 0.16, 0.16, 0.16, 0.16]},
        {"name": "递减权重", "base_weight": 0.4, "svd_weights": [0.2, 0.15, 0.1, 0.1, 0.05]}
    ]
    
    for config in configs:
        print(f"\n{config['name']}:")
        loss_fn = TotalLossWithSVD(
            base_weight=config["base_weight"],
            svd_weights=config["svd_weights"],
            topk=5
        )
        
        # 显示权重信息
        info = loss_fn.get_weight_info()
        print(f"  基础权重: {info['normalized_base_weight']:.3f}")
        print(f"  SVD权重: {[f'{w:.3f}' for w in info['normalized_svd_weights']]}")

def main():
    """主演示函数"""
    print("鲁棒SVD损失函数使用演示\n")
    print("=" * 50)
    
    # 设置随机种子以获得可重复的结果
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        demo_basic_usage()
        demo_training_loop()
        demo_problematic_data()
        demo_gpu_usage()
        demo_weight_analysis()
        
        print("\n" + "=" * 50)
        print("✅ 所有演示完成！")
        print("\n主要特点:")
        print("- 🛡️  数值稳定性：自动处理病态矩阵、NaN/Inf值")
        print("- 🔄 自动降级：SVD失败时自动回退到MSE损失")
        print("- ⚙️  灵活配置：支持自定义权重分布")
        print("- 🖥️  GPU兼容：完全支持CUDA加速")
        print("- 📊 监控友好：详细的警告和调试信息")
        
    except Exception as e:
        print(f"❌ 演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()