#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置参数测试脚本
检查当前配置是否存在限制和阻碍
"""

import yaml
import torch
import os
import sys

def test_config_parameters():
    """测试配置参数"""
    print("=" * 60)
    print("🔍 配置参数测试报告")
    print("=" * 60)
    
    try:
        # 1. 加载配置文件
        with open('dynamic_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("✅ 配置文件加载成功")
        
        # 2. 基础配置检查
        print("\n📋 基础配置:")
        print(f"  设备配置: {config['device']}")
        print(f"  多GPU训练: {config['use_dataparallel']}")
        print(f"  批次大小: {config['data']['batch_size']}")
        print(f"  输入分辨率: {config['data']['input_resolution']}")
        print(f"  输出分辨率: {config['data']['output_resolution']}")
        print(f"  样本数量: {config['data']['num_samples']}")
        print(f"  训练轮数: {config['training']['epochs']}")
        
        # 3. 硬件环境检查
        print("\n🖥️ 硬件环境:")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name} ({props.total_memory / 1024**3:.1f}GB)")
        else:
            print("  ⚠️ CUDA不可用，将使用CPU训练")
        
        # 4. 数据路径检查
        print("\n📁 数据路径检查:")
        data_path = config['data']['path']
        if os.path.exists(data_path):
            print(f"  ✅ 数据文件存在: {data_path}")
            file_size = os.path.getsize(data_path) / 1024**3
            print(f"  📊 文件大小: {file_size:.2f}GB")
        else:
            print(f"  ❌ 数据文件不存在: {data_path}")
        
        # 5. 内存估算
        print("\n💾 内存需求估算:")
        input_size = config['data']['input_resolution'][0] * config['data']['input_resolution'][1]
        output_size = config['data']['output_resolution'][0] * config['data']['output_resolution'][1]
        batch_size = config['data']['batch_size']
        num_samples = config['data']['num_samples']
        
        # 估算单个样本内存需求（假设float32，4字节）
        sample_memory = (input_size + output_size) * 4 / 1024**2  # MB
        batch_memory = sample_memory * batch_size
        total_data_memory = sample_memory * num_samples
        
        print(f"  单样本内存: {sample_memory:.2f}MB")
        print(f"  单批次内存: {batch_memory:.2f}MB")
        print(f"  总数据内存: {total_data_memory:.2f}MB")
        
        # 6. 潜在问题检查
        print("\n⚠️ 潜在问题检查:")
        issues = []
        
        # 检查批次大小
        if batch_size < 2:
            issues.append("批次大小过小，可能影响训练稳定性")
        
        # 检查样本数量
        if num_samples < 100:
            issues.append("样本数量较少，可能导致过拟合")
        
        # 检查分辨率差异
        input_pixels = input_size
        output_pixels = output_size
        ratio = output_pixels / input_pixels
        if ratio > 16:
            issues.append(f"输出分辨率比输入分辨率大{ratio:.1f}倍，可能需要更复杂的模型")
        
        # 检查GPU内存
        if torch.cuda.is_available() and config['device'] == 'cuda':
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if batch_memory > gpu_memory * 0.5 * 1024:  # 转换为MB
                issues.append(f"批次内存需求({batch_memory:.1f}MB)可能超过GPU内存({gpu_memory:.1f}GB)的50%")
        
        # 检查数据加载器配置
        num_workers = config['dataloader']['num_workers']
        if num_workers > 8:
            issues.append(f"数据加载器工作进程数({num_workers})可能过多，建议不超过8")
        
        if not issues:
            print("  ✅ 未发现明显问题")
        else:
            for issue in issues:
                print(f"  ⚠️ {issue}")
        
        # 7. 优化建议
        print("\n💡 优化建议:")
        suggestions = []
        
        if batch_size == 1 and torch.cuda.is_available():
            suggestions.append("考虑增加批次大小到2-4以提升训练效率")
        
        if config['data']['lazy_loading'] == False and num_samples > 500:
            suggestions.append("对于大数据集，建议启用lazy_loading以节省内存")
        
        if not config['training']['enable_early_stopping']:
            suggestions.append("建议启用早停机制以防止过拟合")
        
        if config['dataloader']['persistent_workers'] == False and num_workers > 0:
            suggestions.append("建议启用persistent_workers以减少数据加载开销")
        
        if not suggestions:
            print("  ✅ 当前配置已较为优化")
        else:
            for suggestion in suggestions:
                print(f"  💡 {suggestion}")
        
        print("\n" + "=" * 60)
        print("✅ 配置测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 配置测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_config_parameters()