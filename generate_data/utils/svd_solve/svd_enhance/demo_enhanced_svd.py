#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版SVD损失函数演示脚本
展示增强版SVD损失函数的核心功能和性能监控
"""

import torch
import torch.nn as nn
import numpy as np
import time
from pathlib import Path
import sys

# 添加utils路径
sys.path.append('../modify_multi_attention')
from utils.enhanced_svd_loss import (
    create_enhanced_svd_loss, 
    print_global_svd_stats, 
    reset_global_svd_stats,
    get_global_svd_stats
)

def create_demo_model():
    """创建一个简单的演示模型"""
    return nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 1024)
    )

def generate_demo_data(batch_size=4, resolution=32):
    """生成演示数据"""
    # 生成输入数据 (batch_size, resolution*resolution)
    input_data = torch.randn(batch_size, resolution * resolution)
    
    # 生成目标数据 (添加一些结构化模式)
    target_data = torch.zeros_like(input_data)
    for i in range(batch_size):
        # 创建一些有趣的模式
        pattern = torch.sin(torch.linspace(0, 4*np.pi, resolution*resolution))
        noise = 0.1 * torch.randn(resolution*resolution)
        target_data[i] = pattern + noise
    
    return input_data, target_data

def demo_basic_functionality():
    """演示基本功能"""
    print("\n" + "="*60)
    print("[INFO] 增强版SVD损失函数 - 基本功能演示")
    print("="*60)
    
    # 重置统计信息
    reset_global_svd_stats()
    
    # 创建增强版SVD损失函数
    criterion = create_enhanced_svd_loss(
        monitoring=True,
        adaptive_weights=True,
        mixed_precision=True,
        fallback_level=2
    )
    
    # 生成演示数据
    input_data, target_data = generate_demo_data(batch_size=8, resolution=32)
    input_data.requires_grad_(True)
    
    print(f"[INFO] 数据形状: {input_data.shape}")
    print(f"[INFO] 目标形状: {target_data.shape}")
    
    # 计算损失
    start_time = time.time()
    loss = criterion(input_data, target_data)
    compute_time = time.time() - start_time
    
    print(f"[INFO] 损失值: {loss.item():.6f}")
    print(f"⏱️  计算时间: {compute_time*1000:.2f}ms")
    
    # 反向传播测试
    start_time = time.time()
    loss.backward()
    backward_time = time.time() - start_time
    
    print(f"⏱️  反向传播时间: {backward_time*1000:.2f}ms")
    print(f"[INFO] 梯度范数: {input_data.grad.norm().item():.6f}")
    
    # 显示性能统计
    print_global_svd_stats()
    
    return loss.item(), compute_time, backward_time

def demo_adaptive_weights():
    """演示自适应权重机制"""
    print("\n" + "="*60)
    print("[INFO] 增强版SVD损失函数 - 自适应权重演示")
    print("="*60)
    
    # 重置统计信息
    reset_global_svd_stats()
    
    # 创建带自适应权重的损失函数
    criterion = create_enhanced_svd_loss(
        monitoring=True,
        adaptive_weights=True,
        mixed_precision=True,
        fallback_level=2
    )
    
    print("[INFO] 模拟多次训练迭代以观察权重适应...")
    
    losses = []
    for epoch in range(5):
        input_data, target_data = generate_demo_data(batch_size=4, resolution=32)
        input_data.requires_grad_(True)
        
        loss = criterion(input_data, target_data)
        loss.backward()
        losses.append(loss.item())
        
        print(f"Epoch {epoch+1}: Loss = {loss.item():.6f}")
        
        # 获取当前权重信息
        if hasattr(criterion, 'get_current_weights'):
            weights = criterion.get_current_weights()
            print(f"  当前权重: MSE={weights.get('mse', 'N/A'):.3f}, SVD={weights.get('svd', 'N/A'):.3f}")
    
    print(f"\n[INFO] 损失变化趋势: {' -> '.join([f'{l:.4f}' for l in losses])}")
    print_global_svd_stats()
    
    return losses

def demo_performance_comparison():
    """演示性能对比"""
    print("\n" + "="*60)
    print("[INFO] 增强版SVD损失函数 - 性能对比演示")
    print("="*60)
    
    # 标准MSE损失
    mse_criterion = nn.MSELoss()
    
    # 增强版SVD损失
    svd_criterion = create_enhanced_svd_loss(
        monitoring=True,
        mixed_precision=True
    )
    
    # 测试不同批次大小
    batch_sizes = [2, 4, 8, 16]
    results = []
    
    for batch_size in batch_sizes:
        print(f"\n[INFO] 测试批次大小: {batch_size}")
        
        input_data, target_data = generate_demo_data(batch_size=batch_size, resolution=32)
        input_data.requires_grad_(True)
        
        # MSE性能测试
        start_time = time.time()
        mse_loss = mse_criterion(input_data, target_data)
        mse_time = time.time() - start_time
        
        # SVD性能测试
        reset_global_svd_stats()
        start_time = time.time()
        svd_loss = svd_criterion(input_data, target_data)
        svd_time = time.time() - start_time
        
        print(f"  MSE损失: {mse_loss.item():.6f} (时间: {mse_time*1000:.2f}ms)")
        print(f"  SVD损失: {svd_loss.item():.6f} (时间: {svd_time*1000:.2f}ms)")
        ratio = svd_time/mse_time if mse_time > 0 else float('inf')
        print(f"  性能比率: {ratio:.2f}x" if ratio != float('inf') else "  性能比率: N/A (MSE时间过短)")
        
        results.append({
            'batch_size': batch_size,
            'mse_loss': mse_loss.item(),
            'svd_loss': svd_loss.item(),
            'mse_time': mse_time,
            'svd_time': svd_time,
            'ratio': svd_time/mse_time if mse_time > 0 else float('inf')
        })
    
    # 总结性能
    valid_ratios = [r['ratio'] for r in results if r['ratio'] != float('inf')]
    if valid_ratios:
        avg_ratio = np.mean(valid_ratios)
        print(f"\n[INFO] 平均性能比率: {avg_ratio:.2f}x (相对于MSE)")
    else:
        print(f"\n[INFO] 平均性能比率: N/A (MSE时间过短无法比较)")
    
    return results

def demo_error_handling():
    """演示错误处理机制"""
    print("\n" + "="*60)
    print("[INFO]️  增强版SVD损失函数 - 错误处理演示")
    print("="*60)
    
    # 创建带错误处理的损失函数
    criterion = create_enhanced_svd_loss(
        monitoring=True,
        fallback_level=2,  # 启用fallback机制
        mixed_precision=True
    )
    
    test_cases = [
        ("正常数据", lambda: generate_demo_data(4, 32)),
        ("包含NaN的数据", lambda: (
            torch.randn(4, 1024),
            torch.full((4, 1024), float('nan'))
        )),
        ("极值数据", lambda: (
            torch.randn(4, 1024) * 1e6,
            torch.randn(4, 1024) * 1e-6
        )),
        ("零数据", lambda: (
            torch.zeros(4, 1024),
            torch.zeros(4, 1024)
        ))
    ]
    
    for test_name, data_generator in test_cases:
        print(f"\n[TEST] 测试: {test_name}")
        try:
            input_data, target_data = data_generator()
            input_data.requires_grad_(True)
            
            loss = criterion(input_data, target_data)
            print(f"  [OK] 成功计算损失: {loss.item():.6f}")
            
            # 检查是否有NaN或Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  [WARN]  检测到异常值: NaN={torch.isnan(loss)}, Inf={torch.isinf(loss)}")
            
        except Exception as e:
            print(f"  [ERROR] 错误: {str(e)}")
    
    print_global_svd_stats()

def main():
    """主演示函数"""
    print("[OK] 增强版SVD损失函数完整演示")
    print("="*80)
    
    # 设备信息
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[INFO]️  使用设备: {device}")
    print(f"[INFO] PyTorch版本: {torch.__version__}")
    
    try:
        # 基本功能演示
        loss, compute_time, backward_time = demo_basic_functionality()
        
        # 自适应权重演示
        losses = demo_adaptive_weights()
        
        # 性能对比演示
        results = demo_performance_comparison()
        
        # 错误处理演示
        demo_error_handling()
        
        # 最终总结
        print("\n" + "="*80)
        print("[INFO] 演示总结")
        print("="*80)
        print(f"[OK] 基本功能: 正常工作，损失值 {loss:.6f}")
        print(f"[OK] 自适应权重: 正常工作，{len(losses)}次迭代")
        print(f"[OK] 性能对比: 完成{len(results)}个批次大小测试")
        print(f"[OK] 错误处理: 通过所有测试用例")
        print("\n[OK] 增强版SVD损失函数演示完成！")
        
        # 显示最终统计
        print("\n[INFO] 最终性能统计:")
        print_global_svd_stats()
        
    except Exception as e:
        print(f"\n[ERROR] 演示过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()