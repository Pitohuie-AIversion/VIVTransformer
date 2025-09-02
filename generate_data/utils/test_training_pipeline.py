#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
训练流程完整性测试脚本
检查训练管道是否存在问题和限制
"""

import os
import sys
import yaml
import torch
import traceback
from pathlib import Path

def test_training_pipeline():
    """测试训练流程完整性"""
    print("=" * 60)
    print("[START] 训练流程完整性测试")
    print("=" * 60)
    
    try:
        # 1. 导入测试
        print("\n[INFO] 模块导入测试:")
        
        # 添加必要的路径
        sys.path.append('..')
        sys.path.append('../modify_multi_attention')
        
        try:
            import dynamic_resolution_trainer as trainer
            print("  [OK] dynamic_resolution_trainer 导入成功")
        except Exception as e:
            print(f"  [ERROR] dynamic_resolution_trainer 导入失败: {e}")
            print("  [INFO] 这可能是由于依赖模块路径问题，但不影响配置测试")
            trainer = None
        
        # 2. 配置加载测试
        print("\n[CONFIG] 配置加载测试:")
        try:
            with open('dynamic_config.yaml', 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            print("  [OK] 配置文件加载成功")
        except Exception as e:
            print(f"  [ERROR] 配置文件加载失败: {e}")
            return
        
        # 3. 关键函数检查
        print("\n[CHECK] 关键函数检查:")
        key_functions = [
            'get_dynamic_loaders',
            'create_model', 
            'train_model',
            'main'
        ]
        
        if trainer is not None:
            for func_name in key_functions:
                if hasattr(trainer, func_name):
                    print(f"  [OK] {func_name} 函数存在")
                else:
                    print(f"  [ERROR] {func_name} 函数缺失")
        else:
            print("  [WARN] 跳过函数检查（模块导入失败）")
        
        # 4. 数据路径验证
        print("\n[PATH] 数据路径验证:")
        data_path = config['data']['path']
        if os.path.exists(data_path):
            print(f"  [OK] 数据文件存在: {os.path.basename(data_path)}")
            file_size = os.path.getsize(data_path) / 1024**3
            print(f"  [INFO] 文件大小: {file_size:.2f}GB")
        else:
            print(f"  [ERROR] 数据文件不存在: {data_path}")
        
        # 5. 输出目录检查
        print("\n[DIR] 输出目录检查:")
        output_dirs = [
            './results',
            './results/models',
            './results/logs',
            './results/loss_logs'
        ]
        
        for dir_path in output_dirs:
            if os.path.exists(dir_path):
                print(f"  [OK] {dir_path} 目录存在")
            else:
                print(f"  [WARN] {dir_path} 目录不存在（将自动创建）")
        
        # 6. 依赖库检查
        print("\n[DEPS] 依赖库检查:")
        required_libs = [
            ('torch', 'PyTorch'),
            ('yaml', 'PyYAML'),
            ('numpy', 'NumPy'),
            ('h5py', 'HDF5'),
            ('matplotlib', 'Matplotlib'),
            ('tqdm', 'tqdm')
        ]
        
        for lib_name, lib_desc in required_libs:
            try:
                __import__(lib_name)
                print(f"  [OK] {lib_desc} 可用")
            except ImportError:
                print(f"  [ERROR] {lib_desc} 缺失")
        
        # 7. 设备兼容性检查
        print("\n[DEVICE] 设备兼容性检查:")
        device_config = config['device']
        use_dataparallel = config['use_dataparallel']
        
        print(f"  配置设备: {device_config}")
        print(f"  多GPU训练: {use_dataparallel}")
        print(f"  CUDA可用: {torch.cuda.is_available()}")
        
        if device_config == 'cuda' and not torch.cuda.is_available():
            print("  [ERROR] 配置要求CUDA但CUDA不可用")
        elif device_config == 'cuda' and torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"  [OK] CUDA可用，GPU数量: {gpu_count}")
            
            if use_dataparallel and gpu_count < 2:
                print("  [WARN] 启用了多GPU训练但只有1张GPU")
        
        # 8. 配置参数合理性检查
        print("\n[CONFIG] 配置参数合理性检查:")
        issues = []
        
        # 批次大小检查
        batch_size = config['data']['batch_size']
        if batch_size < 1:
            issues.append("批次大小必须大于0")
        elif batch_size == 1:
            issues.append("批次大小为1可能影响训练稳定性")
        
        # 学习率检查
        lr = config['training']['learning_rate']
        if lr <= 0 or lr > 1:
            issues.append(f"学习率({lr})可能不合理")
        
        # 模型维度检查
        d_model = config['model']['d_model']
        num_heads = config['model']['num_heads']
        if d_model % num_heads != 0:
            issues.append(f"模型维度({d_model})必须能被注意力头数({num_heads})整除")
        
        # 分辨率检查
        input_res = config['data']['input_resolution']
        output_res = config['data']['output_resolution']
        if input_res[0] <= 0 or input_res[1] <= 0:
            issues.append("输入分辨率必须大于0")
        if output_res[0] <= 0 or output_res[1] <= 0:
            issues.append("输出分辨率必须大于0")
        
        # 数据加载器检查
        num_workers = config['dataloader']['num_workers']
        if num_workers < 0:
            issues.append("数据加载器工作进程数不能为负")
        elif num_workers > 16:
            issues.append(f"数据加载器工作进程数({num_workers})可能过多")
        
        if not issues:
            print("  [OK] 配置参数检查通过")
        else:
            for issue in issues:
                print(f"  [ERROR] {issue}")
        
        # 9. 内存需求评估
        print("\n[MEM] 内存需求评估:")
        
        # 数据内存
        input_size = input_res[0] * input_res[1]
        output_size = output_res[0] * output_res[1]
        num_samples = config['data']['num_samples']
        sample_memory = (input_size + output_size) * 4 / 1024**2  # MB
        total_data_memory = sample_memory * num_samples
        
        print(f"  单样本内存: {sample_memory:.2f}MB")
        print(f"  总数据内存: {total_data_memory:.2f}MB")
        
        # 模型内存（简化估算）
        model_params = (
            input_size * d_model +  # 输入投影
            config['model']['num_layers'] * d_model * d_model * 4 +  # Transformer层
            d_model * output_size  # 输出投影
        )
        model_memory = model_params * 4 / 1024**2  # MB
        print(f"  模型内存: {model_memory:.2f}MB")
        
        # GPU内存检查
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            total_memory_gb = (total_data_memory + model_memory * 3) / 1024  # 模型+梯度+优化器
            usage_ratio = total_memory_gb / gpu_memory
            print(f"  GPU内存: {gpu_memory:.1f}GB")
            print(f"  预估使用: {total_memory_gb:.2f}GB ({usage_ratio*100:.1f}%)")
            
            if usage_ratio > 0.9:
                print("  [ERROR] 内存使用率过高，可能OOM")
            elif usage_ratio > 0.7:
                print("  [WARN] 内存使用率较高，建议监控")
            else:
                print("  [OK] 内存使用率正常")
        
        # 10. 训练可行性评估
        print("\n[ASSESS] 训练可行性评估:")
        
        feasibility_score = 100
        warnings = []
        
        # 数据量评估
        if num_samples < 50:
            feasibility_score -= 30
            warnings.append("数据量过少，可能导致严重过拟合")
        elif num_samples < 200:
            feasibility_score -= 15
            warnings.append("数据量较少，注意过拟合风险")
        
        # 模型复杂度评估
        param_count = model_params / 1e6
        if param_count > 100:
            feasibility_score -= 20
            warnings.append(f"模型参数量较大({param_count:.1f}M)，训练可能较慢")
        
        # 硬件评估
        if not torch.cuda.is_available() and device_config == 'cuda':
            feasibility_score -= 40
            warnings.append("CUDA不可用但配置要求GPU训练")
        
        # 配置一致性评估
        if len(issues) > 0:
            feasibility_score -= len(issues) * 10
            warnings.append(f"存在{len(issues)}个配置问题")
        
        print(f"  可行评分: {max(0, feasibility_score)}/100")
        
        if feasibility_score >= 80:
            print("  [OK] 训练配置良好，可以开始训练")
        elif feasibility_score >= 60:
            print("  [WARN] 训练配置基本可行，建议优化")
        else:
            print("  [ERROR] 训练配置存在问题，建议修复后再训练")
        
        if warnings:
            print("\n[WARN] 警告信息:")
            for warning in warnings:
                print(f"  - {warning}")
        
        print("\n" + "=" * 60)
        print("[OK] 训练流程测试完成")
        print("=" * 60)
        
    except Exception as e:
        print(f"[ERROR] 测试过程中出现错误: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    test_training_pipeline()