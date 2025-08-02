#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试SVD损失函数的数值稳定性修复
"""

import torch
import numpy as np
from svd10_loss import TotalLossWithSVD, get_svd_modes, svd_topk_losses

def test_normal_case():
    """测试正常情况"""
    print("=== 测试正常情况 ===")
    torch.manual_seed(42)
    
    # 创建正常的测试数据
    pred = torch.randn(2, 32, 32)
    target = torch.randn(2, 32, 32)
    
    # 测试SVD模态提取
    try:
        modes = get_svd_modes(pred, topk=5)
        print(f"✓ SVD模态提取成功，模态数量: {len(modes)}")
        print(f"✓ 第一个模态形状: {modes[0].shape}")
    except Exception as e:
        print(f"✗ SVD模态提取失败: {e}")
    
    # 测试损失计算
    try:
        loss_fn = TotalLossWithSVD(base_weight=0.5, topk=5)
        loss = loss_fn(pred, target)
        print(f"✓ 损失计算成功: {loss.item():.6f}")
    except Exception as e:
        print(f"✗ 损失计算失败: {e}")

def test_ill_conditioned_matrix():
    """测试病态矩阵"""
    print("\n=== 测试病态矩阵 ===")
    
    # 创建病态矩阵（接近奇异）
    pred = torch.ones(2, 32, 32) * 1e-10  # 几乎为零的矩阵
    target = torch.ones(2, 32, 32) * 1e-10
    
    # 添加一些微小的变化
    pred[0, 0, 0] = 1.0
    target[0, 0, 0] = 1.1
    
    try:
        modes = get_svd_modes(pred, topk=5)
        print(f"✓ 病态矩阵SVD处理成功")
    except Exception as e:
        print(f"✗ 病态矩阵SVD处理失败: {e}")
    
    try:
        loss_fn = TotalLossWithSVD(base_weight=0.5, topk=5)
        loss = loss_fn(pred, target)
        print(f"✓ 病态矩阵损失计算成功: {loss.item():.6f}")
    except Exception as e:
        print(f"✗ 病态矩阵损失计算失败: {e}")

def test_nan_inf_values():
    """测试NaN和Inf值"""
    print("\n=== 测试NaN和Inf值 ===")
    
    # 创建包含NaN的数据
    pred = torch.randn(2, 32, 32)
    target = torch.randn(2, 32, 32)
    pred[0, 0, 0] = float('nan')
    target[0, 0, 0] = float('inf')
    
    try:
        modes = get_svd_modes(pred, topk=5)
        print(f"✓ NaN/Inf数据SVD处理成功")
    except Exception as e:
        print(f"✗ NaN/Inf数据SVD处理失败: {e}")
    
    try:
        loss_fn = TotalLossWithSVD(base_weight=0.5, topk=5)
        loss = loss_fn(pred, target)
        print(f"✓ NaN/Inf数据损失计算成功: {loss.item():.6f}")
    except Exception as e:
        print(f"✗ NaN/Inf数据损失计算失败: {e}")

def test_repeated_singular_values():
    """测试重复奇异值情况"""
    print("\n=== 测试重复奇异值 ===")
    
    # 创建具有重复奇异值的矩阵
    pred = torch.zeros(2, 32, 32)
    target = torch.zeros(2, 32, 32)
    
    # 设置相同的行，产生重复奇异值
    for i in range(16):
        pred[0, i, :] = 1.0
        pred[1, i, :] = 1.0
        target[0, i, :] = 1.1
        target[1, i, :] = 1.1
    
    try:
        modes = get_svd_modes(pred, topk=5)
        print(f"✓ 重复奇异值SVD处理成功")
    except Exception as e:
        print(f"✗ 重复奇异值SVD处理失败: {e}")
    
    try:
        loss_fn = TotalLossWithSVD(base_weight=0.5, topk=5)
        loss = loss_fn(pred, target)
        print(f"✓ 重复奇异值损失计算成功: {loss.item():.6f}")
    except Exception as e:
        print(f"✗ 重复奇异值损失计算失败: {e}")

def test_extreme_values():
    """测试极值情况"""
    print("\n=== 测试极值情况 ===")
    
    # 创建极大值数据
    pred = torch.ones(2, 32, 32) * 1e10
    target = torch.ones(2, 32, 32) * 1e10
    
    try:
        loss_fn = TotalLossWithSVD(base_weight=0.5, topk=5)
        loss = loss_fn(pred, target)
        print(f"✓ 极值数据损失计算成功: {loss.item():.6f}")
    except Exception as e:
        print(f"✗ 极值数据损失计算失败: {e}")
    
    # 创建极小值数据
    pred = torch.ones(2, 32, 32) * 1e-10
    target = torch.ones(2, 32, 32) * 1e-10
    
    try:
        loss_fn = TotalLossWithSVD(base_weight=0.5, topk=5)
        loss = loss_fn(pred, target)
        print(f"✓ 极小值数据损失计算成功: {loss.item():.6f}")
    except Exception as e:
        print(f"✗ 极小值数据损失计算失败: {e}")

def test_gpu_compatibility():
    """测试GPU兼容性"""
    print("\n=== 测试GPU兼容性 ===")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        pred = torch.randn(2, 32, 32, device=device)
        target = torch.randn(2, 32, 32, device=device)
        
        try:
            loss_fn = TotalLossWithSVD(base_weight=0.5, topk=5)
            loss = loss_fn(pred, target)
            print(f"✓ GPU损失计算成功: {loss.item():.6f}")
        except Exception as e:
            print(f"✗ GPU损失计算失败: {e}")
    else:
        print("⚠ CUDA不可用，跳过GPU测试")

def main():
    """主测试函数"""
    print("开始SVD损失函数数值稳定性测试...\n")
    
    test_normal_case()
    test_ill_conditioned_matrix()
    test_nan_inf_values()
    test_repeated_singular_values()
    test_extreme_values()
    test_gpu_compatibility()
    
    print("\n=== 测试完成 ===")
    print("如果所有测试都显示✓，说明SVD损失函数修复成功！")

if __name__ == "__main__":
    main()