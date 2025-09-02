#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试强制SVD功能

这个脚本用于验证force_svd参数是否能够正确工作，
强制使用SVD计算而不使用fallback策略。
"""

import sys
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
modify_multi_attention_path = project_root / 'modify_multi_attention'
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(modify_multi_attention_path))
sys.path.insert(0, str(Path(__file__).parent))

try:
    from modify_multi_attention.utils.enhanced_svd_loss import (
        create_enhanced_svd_loss, get_global_svd_stats, reset_global_svd_stats
    )
except ImportError:
    from utils.enhanced_svd_loss import (
        create_enhanced_svd_loss, get_global_svd_stats, reset_global_svd_stats
    )

def test_force_svd():
    """测试强制SVD功能"""
    print("=== 测试强制SVD功能 ===")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建测试数据
    batch_size = 4
    height, width = 64, 64
    
    # 创建一些具有挑战性的数据（可能导致SVD失败）
    predictions = torch.randn(batch_size, height, width, device=device, dtype=torch.float32)
    targets = torch.randn(batch_size, height, width, device=device, dtype=torch.float32)
    
    # 添加一些极值来增加SVD计算的难度
    predictions[0, :10, :10] = 1e6  # 极大值
    predictions[1, :10, :10] = 1e-6  # 极小值
    targets[0, :10, :10] = -1e6
    targets[1, :10, :10] = -1e-6
    
    print(f"测试数据形状: {predictions.shape}")
    print(f"预测值范围: [{predictions.min():.2e}, {predictions.max():.2e}]")
    print(f"目标值范围: [{targets.min():.2e}, {targets.max():.2e}]")
    
    # 测试1: 不强制SVD (force_svd=False)
    print("\n--- 测试1: 不强制SVD (force_svd=False) ---")
    reset_global_svd_stats()
    
    criterion_normal = create_enhanced_svd_loss(
        base_weight=0.8,
        svd_weights=[0.1, 0.05, 0.05],
        topk=3,
        mixed_precision=True,
        adaptive_weights=False,
        monitoring=True,
        fallback_level=2,
        force_svd=False
    )
    
    try:
        loss_normal = criterion_normal(predictions, targets)
        print(f"正常模式损失: {loss_normal.item():.6f}")
        
        stats_normal = get_global_svd_stats()
        print(f"SVD调用次数: {stats_normal.get('total_calls', 0)}")
        print(f"Fallback使用次数: {stats_normal.get('fallback_count', 0)}")
        print(f"数值问题次数: {stats_normal.get('numerical_issues', 0)}")
        
    except Exception as e:
        print(f"正常模式出错: {e}")
    
    # 测试2: 强制SVD (force_svd=True)
    print("\n--- 测试2: 强制SVD (force_svd=True) ---")
    reset_global_svd_stats()
    
    criterion_force = create_enhanced_svd_loss(
        base_weight=0.8,
        svd_weights=[0.1, 0.05, 0.05],
        topk=3,
        mixed_precision=True,
        adaptive_weights=False,
        monitoring=True,
        fallback_level=2,
        force_svd=True
    )
    
    try:
        loss_force = criterion_force(predictions, targets)
        print(f"强制SVD模式损失: {loss_force.item():.6f}")
        
        stats_force = get_global_svd_stats()
        print(f"SVD调用次数: {stats_force.get('total_calls', 0)}")
        print(f"Fallback使用次数: {stats_force.get('fallback_count', 0)}")
        print(f"数值问题次数: {stats_force.get('numerical_issues', 0)}")
        
        # 验证强制SVD模式下不应该有fallback
        if stats_force.get('fallback_count', 0) == 0:
            print("[OK] 强制SVD模式正常工作 - 没有使用fallback策略")
        else:
            print("[ERROR] 强制SVD模式异常 - 仍然使用了fallback策略")
            
    except Exception as e:
        print(f"强制SVD模式出错: {e}")
        print("这可能是预期的，因为强制SVD可能在数值不稳定时失败")
    
    print("\n=== 测试完成 ===")

if __name__ == "__main__":
    test_force_svd()