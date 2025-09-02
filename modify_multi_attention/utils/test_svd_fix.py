#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试SVD损失函数的数值稳定性修复
"""

import torch
import numpy as np
from svd10_loss import TotalLossWithSVD, get_svd_modes, svd_topk_losses

def create_problematic_tensors():
    """创建可能导致SVD失败的测试张量"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    test_cases = {
        'normal': torch.randn(2, 32, 32, device=device),
        'zeros': torch.zeros(2, 32, 32, device=device),
        'ones': torch.ones(2, 32, 32, device=device),
        'large_values': torch.randn(2, 32, 32, device=device) * 1e6,
        'small_values': torch.randn(2, 32, 32, device=device) * 1e-8,
        'inf_values': torch.full((2, 32, 32), float('inf'), device=device),
        'nan_values': torch.full((2, 32, 32), float('nan'), device=device),
        'rank_deficient': torch.zeros(2, 32, 32, device=device),
        'repeated_rows': None,  # 将在下面设置
        'ill_conditioned': None  # 将在下面设置
    }
    
    # 创建重复行的矩阵
    repeated = torch.randn(2, 32, 32, device=device)
    repeated[:, 1:16, :] = repeated[:, 0:1, :].repeat(1, 15, 1)  # 重复前16行
    test_cases['repeated_rows'] = repeated
    
    # 创建病态矩阵
    ill_cond = torch.randn(2, 32, 32, device=device)
    # 添加非常小的奇异值
    u, s, vh = torch.linalg.svd(ill_cond[0], full_matrices=False)
    s[20:] = 1e-15  # 使后面的奇异值非常小
    ill_cond[0] = u @ torch.diag(s) @ vh
    test_cases['ill_conditioned'] = ill_cond
    
    return test_cases

def test_svd_modes():
    """测试get_svd_modes函数的稳定性"""
    print("=== 测试SVD模态提取 ===")
    test_cases = create_problematic_tensors()
    
    for name, tensor in test_cases.items():
        print(f"\n测试案例: {name}")
        try:
            modes = get_svd_modes(tensor, topk=10)
            print(f"  [OK] 成功提取{len(modes)}个模态")
            
            # 检查模态的有效性
            for i, mode in enumerate(modes):
                if torch.isnan(mode).any():
                    print(f"  [WARN] 模态{i+1}包含NaN")
                elif torch.isinf(mode).any():
                    print(f"  [WARN] 模态{i+1}包含Inf")
                else:
                    print(f"  [OK] 模态{i+1}正常")
                    
        except Exception as e:
            print(f"  [FAIL] 失败: {str(e)}")

def test_svd_losses():
    """测试SVD损失计算的稳定性"""
    print("\n=== 测试SVD损失计算 ===")
    test_cases = create_problematic_tensors()
    
    for name, tensor in test_cases.items():
        print(f"\n测试案例: {name}")
        try:
            # 使用相同的张量作为预测和目标
            losses = svd_topk_losses(tensor, tensor, topk=10)
            print(f"  [OK] 成功计算{len(losses)}个SVD损失")
            
            # 检查损失的有效性
            for i, loss in enumerate(losses):
                if torch.isnan(loss):
                    print(f"  [WARN] 损失{i+1}为NaN")
                elif torch.isinf(loss):
                    print(f"  [WARN] 损失{i+1}为Inf")
                else:
                    print(f"  [OK] 损失{i+1}: {loss.item():.6f}")
                    
        except Exception as e:
            print(f"  [FAIL] 失败: {str(e)}")

def test_total_loss():
    """测试总损失函数的稳定性"""
    print("\n=== 测试总损失函数 ===")
    test_cases = create_problematic_tensors()
    
    # 创建损失函数
    criterion = TotalLossWithSVD(base_weight=0.5, topk=10)
    
    for name, tensor in test_cases.items():
        print(f"\n测试案例: {name}")
        try:
            # 使用相同的张量作为预测和目标
            total_loss = criterion(tensor, tensor)
            
            if torch.isnan(total_loss):
                print(f"  [WARN] 总损失为NaN")
            elif torch.isinf(total_loss):
                print(f"  [WARN] 总损失为Inf")
            else:
                print(f"  [OK] 总损失: {total_loss.item():.6f}")
                
        except Exception as e:
            print(f"  [FAIL] 失败: {str(e)}")

def test_gradient_flow():
    """测试梯度流的稳定性"""
    print("\n=== 测试梯度流 ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建需要梯度的张量
    pred = torch.randn(2, 32, 32, device=device, requires_grad=True)
    target = torch.randn(2, 32, 32, device=device)
    
    criterion = TotalLossWithSVD(base_weight=0.5, topk=10)
    
    try:
        loss = criterion(pred, target)
        loss.backward()
        
        if pred.grad is not None:
            if torch.isnan(pred.grad).any():
                print("  [WARN] 梯度包含NaN")
            elif torch.isinf(pred.grad).any():
                print("  [WARN] 梯度包含Inf")
            else:
                grad_norm = torch.norm(pred.grad)
                print(f"  [OK] 梯度正常，范数: {grad_norm.item():.6f}")
        else:
            print("  [WARN] 没有计算梯度")
            
    except Exception as e:
        print(f"  [FAIL] 梯度计算失败: {str(e)}")

def main():
    """主测试函数"""
    print("开始SVD损失函数稳定性测试...")
    print(f"使用设备: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    test_svd_modes()
    test_svd_losses()
    test_total_loss()
    test_gradient_flow()
    
    print("\n=== 测试完成 ===")
    print("如果看到大量警告信息，这是正常的，说明修复机制正在工作。")
    print("重要的是没有程序崩溃，并且能够返回有效的损失值。")

if __name__ == "__main__":
    main()