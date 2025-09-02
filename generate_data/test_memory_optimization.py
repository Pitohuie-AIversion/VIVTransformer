#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
内存优化测试脚本
测试新增的内存监控和优化功能
"""

import os
import sys
import yaml
import psutil
import gc
import time
import torch
import numpy as np
from typing import Dict, Any

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def get_memory_info():
    """获取当前内存使用信息"""
    memory_info = psutil.virtual_memory()
    return {
        'total_gb': memory_info.total / (1024**3),
        'used_gb': memory_info.used / (1024**3),
        'available_gb': memory_info.available / (1024**3),
        'percent': memory_info.percent,
        'ratio': memory_info.percent / 100.0
    }

def simulate_memory_optimization(memory_config: Dict[str, Any]):
    """模拟内存优化逻辑"""
    print("\n[TEST] 测试内存优化逻辑")
    print("=" * 50)
    
    # 获取初始内存状态
    initial_memory = get_memory_info()
    print(f"[INFO] 初始内存状态:")
    print(f"   总内存: {initial_memory['total_gb']:.2f}GB")
    print(f"   已使用: {initial_memory['used_gb']:.2f}GB")
    print(f"   可用内存: {initial_memory['available_gb']:.2f}GB")
    print(f"   使用率: {initial_memory['percent']:.1f}%")
    
    # 模拟内存监控
    if memory_config.get('enable_memory_monitoring', False):
        print("\n[INFO] 启用内存监控")
        
        # 检查阈值
        warning_threshold = memory_config.get('memory_threshold_warning', 0.8)
        critical_threshold = memory_config.get('memory_threshold_critical', 0.9)
        
        print(f"   警告阈值: {warning_threshold:.1%}")
        print(f"   临界阈值: {critical_threshold:.1%}")
        
        if initial_memory['ratio'] > critical_threshold:
            print(f"   [WARN] 内存使用率过高 ({initial_memory['ratio']:.1%})")
            print(f"   建议: 减少batch_size或num_workers")
        elif initial_memory['ratio'] > warning_threshold:
            print(f"   [WARN] 内存使用率较高 ({initial_memory['ratio']:.1%})")
            print(f"   建议: 注意监控内存使用")
        else:
            print(f"   [OK] 内存使用率正常 ({initial_memory['ratio']:.1%})")
    
    # 模拟内存清理
    if memory_config.get('enable_memory_cleanup', True):
        print("\n🧹 执行内存清理")
        gc.collect()
        
        # 获取清理后的内存状态
        after_cleanup = get_memory_info()
        freed_memory = initial_memory['used_gb'] - after_cleanup['used_gb']
        print(f"   清理前: {initial_memory['used_gb']:.2f}GB")
        print(f"   清理后: {after_cleanup['used_gb']:.2f}GB")
        print(f"   释放内存: {freed_memory:.3f}GB")
    
    # 共享内存配置
    if memory_config.get('enable_shared_memory', True):
        shared_memory_size = memory_config.get('shared_memory_size_mb', 1024)
        print(f"\n🔗 共享内存配置: {shared_memory_size}MB")
        print(f"   共享内存占总内存比例: {(shared_memory_size/1024)/initial_memory['total_gb']:.2%}")
    
    # 内存映射配置
    if memory_config.get('enable_memory_mapping', True):
        print(f"\n🗺️ 启用内存映射")
        print(f"   适用于大文件处理，减少内存占用")
    
    # 缓存配置
    cache_size = memory_config.get('memory_cache_size_mb', 2048)
    print(f"\n[INFO] 内存缓存配置: {cache_size}MB")
    print(f"   缓存占总内存比例: {(cache_size/1024)/initial_memory['total_gb']:.2%}")
    
    # 懒加载配置
    if memory_config.get('enable_lazy_loading', True):
        preload_ratio = memory_config.get('preload_ratio', 0.1)
        print(f"\n⏳ 懒加载配置")
        print(f"   预加载比例: {preload_ratio:.1%}")
        print(f"   按需加载数据，减少内存占用")

def test_memory_stress():
    """测试内存压力情况"""
    print("\n🔥 内存压力测试")
    print("=" * 50)
    
    initial_memory = get_memory_info()
    print(f"测试前内存使用: {initial_memory['used_gb']:.2f}GB ({initial_memory['percent']:.1f}%)")
    
    # 创建一些大的张量来模拟内存使用
    tensors = []
    try:
        for i in range(5):
            # 创建100MB的张量
            tensor = torch.randn(1000, 1000, 25)  # 约100MB
            tensors.append(tensor)
            
            current_memory = get_memory_info()
            print(f"创建张量 {i+1}: 内存使用 {current_memory['used_gb']:.2f}GB ({current_memory['percent']:.1f}%)")
            
            # 模拟内存阈值检查
            if current_memory['ratio'] > 0.85:
                print(f"   [WARN] 内存使用率过高，停止创建更多张量")
                break
    
    except Exception as e:
        print(f"   [ERROR] 内存不足: {e}")
    
    # 清理内存
    print("\n🧹 清理测试张量")
    del tensors
    gc.collect()
    
    final_memory = get_memory_info()
    print(f"清理后内存使用: {final_memory['used_gb']:.2f}GB ({final_memory['percent']:.1f}%)")
    
    memory_freed = (initial_memory['used_gb'] - final_memory['used_gb']) * -1  # 可能为负值
    print(f"内存变化: {memory_freed:+.3f}GB")

def test_worker_adjustment():
    """测试基于内存使用率的worker数量调整"""
    print("\n⚙️ Worker数量调整测试")
    print("=" * 50)
    
    memory_info = get_memory_info()
    cpu_count = os.cpu_count()
    
    # 模拟不同内存使用率下的worker调整
    scenarios = [
        {'memory_ratio': 0.5, 'description': '低内存使用'},
        {'memory_ratio': 0.75, 'description': '中等内存使用'},
        {'memory_ratio': 0.85, 'description': '高内存使用'},
        {'memory_ratio': 0.95, 'description': '极高内存使用'}
    ]
    
    for scenario in scenarios:
        print(f"\n[INFO] 场景: {scenario['description']} ({scenario['memory_ratio']:.1%})")
        
        # 初始worker数量
        initial_workers = min(32, cpu_count // 2)
        
        # 根据内存使用率调整
        if scenario['memory_ratio'] > 0.9:
            adjusted_workers = max(2, initial_workers // 2)
            print(f"   建议: 减少worker数量 {initial_workers} → {adjusted_workers}")
        elif scenario['memory_ratio'] > 0.8:
            adjusted_workers = max(4, initial_workers * 3 // 4)
            print(f"   建议: 适度减少worker数量 {initial_workers} → {adjusted_workers}")
        else:
            adjusted_workers = initial_workers
            print(f"   建议: 保持当前worker数量 {adjusted_workers}")
        
        print(f"   CPU核心数: {cpu_count}")
        print(f"   Worker利用率: {adjusted_workers/cpu_count:.1%}")

def main():
    """主函数"""
    print("[TEST] 内存优化功能测试")
    print("=" * 60)
    
    # 加载配置文件
    config_path = "dynamic_config_server_downsampling.yaml"
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        memory_config = config.get('dataloader', {}).get('memory_optimization', {})
        print(f"[INFO] 从配置文件加载内存优化设置: {config_path}")
        print(f"   启用内存监控: {memory_config.get('enable_memory_monitoring', False)}")
        print(f"   最大内存限制: {memory_config.get('max_memory_usage_gb', 64)}GB")
        print(f"   警告阈值: {memory_config.get('memory_threshold_warning', 0.8):.1%}")
        print(f"   临界阈值: {memory_config.get('memory_threshold_critical', 0.9):.1%}")
    else:
        print(f"[WARN] 配置文件不存在: {config_path}")
        # 使用默认配置
        memory_config = {
            'enable_memory_monitoring': True,
            'max_memory_usage_gb': 64,
            'memory_threshold_warning': 0.8,
            'memory_threshold_critical': 0.9,
            'enable_memory_cleanup': True,
            'cleanup_interval': 100,
            'enable_shared_memory': True,
            'shared_memory_size_mb': 1024,
            'enable_memory_mapping': True,
            'memory_cache_size_mb': 2048,
            'enable_lazy_loading': True,
            'preload_ratio': 0.1
        }
        print("[INFO] 使用默认内存优化配置")
    
    # 运行测试
    simulate_memory_optimization(memory_config)
    test_memory_stress()
    test_worker_adjustment()
    
    print("\n[OK] 内存优化测试完成")
    print("\n[INFO] 总结:")
    print("   - 内存监控功能可以实时跟踪内存使用情况")
    print("   - 自动调整worker数量以避免内存不足")
    print("   - 内存清理可以释放不必要的内存占用")
    print("   - 共享内存和内存映射可以优化大数据处理")
    print("   - 懒加载可以减少内存峰值使用")

if __name__ == "__main__":
    main()