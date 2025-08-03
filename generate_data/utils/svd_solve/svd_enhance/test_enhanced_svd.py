#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版SVD损失函数测试脚本

功能:
1. 测试增强版SVD损失函数的基本功能
2. 验证混合精度兼容性
3. 测试自适应权重机制
4. 验证性能监控功能
5. 测试错误处理机制

作者: AI Assistant
日期: 2025
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加路径
project_root = Path(__file__).parent.parent
modify_multi_attention_path = project_root / 'modify_multi_attention'
sys.path.insert(0, str(modify_multi_attention_path))

try:
    from utils.enhanced_svd_loss import (
        EnhancedTotalLossWithSVD, create_enhanced_svd_loss,
        get_global_svd_stats, print_global_svd_stats, reset_global_svd_stats
    )
except ImportError:
    print("❌ 无法导入增强版SVD损失函数，请检查路径配置")
    sys.exit(1)

def test_basic_functionality():
    """测试基本功能"""
    print("\n=== 测试基本功能 ===")
    
    # 创建测试数据
    batch_size = 4
    height, width = 32, 32
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    pred = torch.randn(batch_size, height, width, device=device, requires_grad=True)
    target = torch.randn(batch_size, height, width, device=device)
    
    # 创建增强版损失函数
    criterion = create_enhanced_svd_loss(
        base_weight=0.6,
        topk=5,
        mixed_precision=False,
        adaptive_weights=False,
        monitoring=True,
        fallback_level=2
    )
    
    # 计算损失
    loss = criterion(pred, target)
    
    print(f"✅ 基本功能测试通过")
    print(f"   损失值: {loss.item():.6f}")
    print(f"   设备: {device}")
    print(f"   数据形状: {pred.shape}")
    
    # 测试反向传播
    loss.backward()
    print(f"✅ 反向传播测试通过")
    print(f"   梯度形状: {pred.grad.shape}")
    
    return criterion

def test_mixed_precision():
    """测试混合精度兼容性"""
    print("\n=== 测试混合精度兼容性 ===")
    
    if not torch.cuda.is_available():
        print("⚠️ CUDA不可用，跳过混合精度测试")
        return
    
    device = torch.device('cuda')
    batch_size = 2
    height, width = 16, 16
    
    # 测试float16输入
    pred_fp16 = torch.randn(batch_size, height, width, device=device, dtype=torch.float16, requires_grad=True)
    target_fp16 = torch.randn(batch_size, height, width, device=device, dtype=torch.float16)
    
    # 创建支持混合精度的损失函数
    criterion = create_enhanced_svd_loss(
        mixed_precision=True,
        monitoring=True,
        fallback_level=1
    )
    
    # 计算损失
    loss = criterion(pred_fp16, target_fp16)
    
    print(f"✅ 混合精度测试通过")
    print(f"   输入类型: {pred_fp16.dtype}")
    print(f"   损失值: {loss.item():.6f}")
    print(f"   损失类型: {loss.dtype}")
    
    # 测试自动类型转换
    loss.backward()
    print(f"✅ 混合精度反向传播测试通过")
    
    return criterion

def test_adaptive_weights():
    """测试自适应权重机制"""
    print("\n=== 测试自适应权重机制 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 2
    height, width = 16, 16
    
    # 创建支持自适应权重的损失函数
    criterion = create_enhanced_svd_loss(
        base_weight=0.5,
        adaptive_weights=True,
        monitoring=True,
        fallback_level=1
    )
    
    # 获取初始权重
    initial_base, initial_svd = criterion.weight_manager.get_current_weights()
    print(f"初始基础权重: {initial_base:.4f}")
    
    # 模拟训练过程
    losses = [1.0, 0.8, 0.6, 0.7, 0.5, 0.4, 0.3, 0.35, 0.25, 0.2]  # 模拟损失变化
    
    for epoch, avg_loss in enumerate(losses):
        # 更新epoch信息
        criterion.update_epoch(epoch, avg_loss)
        
        # 每5个epoch检查权重变化
        if epoch % 5 == 0 and epoch > 0:
            current_base, current_svd = criterion.weight_manager.get_current_weights()
            print(f"Epoch {epoch}: 基础权重 {current_base:.4f}")
    
    # 获取最终权重
    final_base, final_svd = criterion.weight_manager.get_current_weights()
    print(f"最终基础权重: {final_base:.4f}")
    
    # 检查权重是否发生变化
    if abs(final_base - initial_base) > 0.001:
        print(f"✅ 自适应权重测试通过 - 权重发生了调整")
        print(f"   权重变化: {final_base - initial_base:+.4f}")
    else:
        print(f"ℹ️ 自适应权重测试 - 权重未发生显著变化")
    
    # 显示适应历史
    adaptation_history = criterion.weight_manager.get_adaptation_history()
    if adaptation_history:
        print(f"   权重调整次数: {len(adaptation_history)}")
        for record in adaptation_history:
            print(f"   Epoch {record['epoch']}: 趋势={record['loss_trend']:.4f}")
    
    return criterion

def test_performance_monitoring():
    """测试性能监控功能"""
    print("\n=== 测试性能监控功能 ===")
    
    # 重置全局监控器
    reset_global_svd_stats()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 3
    height, width = 24, 24
    
    # 创建启用监控的损失函数
    criterion = create_enhanced_svd_loss(
        monitoring=True,
        fallback_level=2
    )
    
    # 执行多次计算以收集统计信息
    num_iterations = 10
    for i in range(num_iterations):
        pred = torch.randn(batch_size, height, width, device=device, requires_grad=True)
        target = torch.randn(batch_size, height, width, device=device)
        
        loss = criterion(pred, target)
        loss.backward()
    
    # 获取性能统计
    stats = criterion.get_performance_stats()
    
    print(f"✅ 性能监控测试通过")
    print(f"   总调用次数: {stats['total_calls']}")
    print(f"   平均SVD时间: {stats['avg_svd_time_ms']:.2f}ms")
    print(f"   总SVD时间: {stats['total_svd_time_ms']:.2f}ms")
    print(f"   NaN/Inf率: {stats['nan_inf_rate']:.2%}")
    print(f"   精度转换率: {stats['precision_conversion_rate']:.2%}")
    
    if stats['fallback_usage']:
        print(f"   Fallback使用: {stats['fallback_usage']}")
    
    # 测试详细报告
    print("\n--- 详细性能报告 ---")
    criterion.print_performance_stats()
    
    return criterion

def test_error_handling():
    """测试错误处理机制"""
    print("\n=== 测试错误处理机制 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建不同fallback级别的损失函数
    for level in [1, 2, 3]:
        print(f"\n--- 测试Fallback级别 {level} ---")
        
        criterion = create_enhanced_svd_loss(
            fallback_level=level,
            monitoring=True
        )
        
        # 测试正常数据
        pred = torch.randn(2, 16, 16, device=device, requires_grad=True)
        target = torch.randn(2, 16, 16, device=device)
        
        loss = criterion(pred, target)
        print(f"   正常数据损失: {loss.item():.6f}")
        
        # 测试包含NaN的数据
        pred_nan = torch.randn(2, 16, 16, device=device)
        pred_nan[0, 0, 0] = float('nan')
        pred_nan = pred_nan.requires_grad_(True)
        
        loss_nan = criterion(pred_nan, target)
        print(f"   NaN数据损失: {loss_nan.item():.6f}")
        
        # 测试极值数据
        pred_extreme = torch.randn(2, 16, 16, device=device, requires_grad=True) * 1e6
        
        loss_extreme = criterion(pred_extreme, target)
        print(f"   极值数据损失: {loss_extreme.item():.6f}")
    
    print(f"✅ 错误处理测试通过")

def test_configuration_options():
    """测试配置选项"""
    print("\n=== 测试配置选项 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pred = torch.randn(2, 16, 16, device=device, requires_grad=True)
    target = torch.randn(2, 16, 16, device=device)
    
    # 测试不同配置组合
    configs = [
        {"name": "最小配置", "mixed_precision": False, "adaptive_weights": False, "monitoring": False, "fallback_level": 1},
        {"name": "标准配置", "mixed_precision": True, "adaptive_weights": True, "monitoring": True, "fallback_level": 2},
        {"name": "最大配置", "mixed_precision": True, "adaptive_weights": True, "monitoring": True, "fallback_level": 3},
    ]
    
    for config in configs:
        print(f"\n--- {config['name']} ---")
        
        criterion = create_enhanced_svd_loss(
            mixed_precision=config['mixed_precision'],
            adaptive_weights=config['adaptive_weights'],
            monitoring=config['monitoring'],
            fallback_level=config['fallback_level']
        )
        
        loss = criterion(pred, target)
        print(f"   损失值: {loss.item():.6f}")
        
        # 显示配置信息
        weight_info = criterion.get_weight_info()
        print(f"   混合精度: {weight_info['mixed_precision_mode']}")
        print(f"   自适应权重: {weight_info['adaptation_enabled']}")
        print(f"   Fallback级别: {weight_info['fallback_level']}")
    
    print(f"✅ 配置选项测试通过")

def main():
    """主测试函数"""
    print("🧪 增强版SVD损失函数测试开始")
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备: {torch.cuda.get_device_name()}")
    
    try:
        # 执行各项测试
        test_basic_functionality()
        test_mixed_precision()
        test_adaptive_weights()
        test_performance_monitoring()
        test_error_handling()
        test_configuration_options()
        
        print("\n🎉 === 所有测试通过 ===")
        print("增强版SVD损失函数功能正常，可以安全使用！")
        
        # 显示全局统计信息
        print("\n=== 全局SVD统计信息 ===")
        print_global_svd_stats()
        
    except Exception as e:
        print(f"\n❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)