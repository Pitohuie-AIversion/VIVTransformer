#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地配置测试脚本
用于验证本地环境配置是否正确，并与服务器配置进行对比
"""

import yaml
import os
import sys
import torch
import psutil
from pathlib import Path

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_system_resources():
    """检查系统资源"""
    print("=== 系统资源检查 ===")
    
    # CPU信息
    cpu_count = psutil.cpu_count(logical=True)
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU核心数: {cpu_count}")
    print(f"CPU使用率: {cpu_percent}%")
    
    # 内存信息
    memory = psutil.virtual_memory()
    print(f"总内存: {memory.total / (1024**3):.2f} GB")
    print(f"可用内存: {memory.available / (1024**3):.2f} GB")
    print(f"内存使用率: {memory.percent}%")
    
    # GPU信息
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"GPU数量: {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
    else:
        print("未检测到CUDA GPU")
    
    print()

def compare_configs(local_config, server_config):
    """对比本地和服务器配置"""
    print("=== 配置对比 ===")
    
    # 关键参数对比
    comparisons = [
        ("训练轮数", "training.epochs"),
        ("批次大小", "training.batch_size"),
        ("样本数量", "data.num_samples"),
        ("工作进程数", "dataloader.num_workers"),
        ("预取因子", "dataloader.prefetch_factor"),
        ("编码器层数", "model.encoder.num_layers"),
        ("隐藏维度", "model.encoder.hidden_dim"),
        ("注意力头数", "model.encoder.num_heads"),
        ("梯度累积步数", "gradient.accumulation_steps"),
        ("显存使用率", "max_memory_fraction"),
    ]
    
    def get_nested_value(config, key_path):
        """获取嵌套字典的值"""
        keys = key_path.split('.')
        value = config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return "未设置"
        return value
    
    print(f"{'参数':<15} {'本地配置':<15} {'服务器配置':<15} {'差异说明':<30}")
    print("-" * 80)
    
    for name, key_path in comparisons:
        local_val = get_nested_value(local_config, key_path)
        server_val = get_nested_value(server_config, key_path)
        
        # 差异说明
        if local_val == server_val:
            diff = "相同"
        elif isinstance(local_val, (int, float)) and isinstance(server_val, (int, float)):
            ratio = local_val / server_val if server_val != 0 else float('inf')
            if ratio < 1:
                diff = f"减少{1/ratio:.1f}倍"
            else:
                diff = f"增加{ratio:.1f}倍"
        else:
            diff = "不同类型"
        
        print(f"{name:<15} {str(local_val):<15} {str(server_val):<15} {diff:<30}")
    
    print()

def validate_local_config(config):
    """验证本地配置的合理性"""
    print("=== 本地配置验证 ===")
    
    issues = []
    warnings = []
    
    # 检查数据文件是否存在
    data_path = config.get('data', {}).get('data_path', '')
    if data_path and not os.path.exists(data_path):
        issues.append(f"数据文件不存在: {data_path}")
    
    # 检查批次大小是否合理
    batch_size = config.get('training', {}).get('batch_size', 0)
    if batch_size > 32:
        warnings.append(f"批次大小较大({batch_size})，可能导致显存不足")
    
    # 检查工作进程数是否合理
    num_workers = config.get('dataloader', {}).get('num_workers', 0)
    cpu_count = psutil.cpu_count(logical=True)
    if num_workers > cpu_count:
        warnings.append(f"工作进程数({num_workers})超过CPU核心数({cpu_count})")
    
    # 检查GPU设置
    if config.get('device') == 'cuda' and not torch.cuda.is_available():
        issues.append("配置使用CUDA但系统未检测到GPU")
    
    # 检查数据并行设置
    use_dataparallel = config.get('use_dataparallel', False)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if use_dataparallel and gpu_count < 2:
        warnings.append(f"启用了数据并行但只有{gpu_count}个GPU")
    
    # 输出结果
    if issues:
        print("[ERROR] 发现问题:")
        for issue in issues:
            print(f"  - {issue}")
    
    if warnings:
        print("[WARN]  警告:")
        for warning in warnings:
            print(f"  - {warning}")
    
    if not issues and not warnings:
        print("[OK] 配置验证通过")
    
    print()
    
    return len(issues) == 0

def recommend_optimizations(config):
    """推荐优化建议"""
    print("=== 优化建议 ===")
    
    cpu_count = psutil.cpu_count(logical=True)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    
    recommendations = []
    
    # CPU优化建议
    num_workers = config.get('dataloader', {}).get('num_workers', 0)
    optimal_workers = min(cpu_count // 2, 8)  # 一般不超过CPU核心数的一半，且不超过8
    if num_workers != optimal_workers:
        recommendations.append(f"建议设置 num_workers = {optimal_workers} (当前: {num_workers})")
    
    # 内存优化建议
    batch_size = config.get('training', {}).get('batch_size', 0)
    if memory_gb < 16 and batch_size > 8:
        recommendations.append(f"内存较小({memory_gb:.1f}GB)，建议减少批次大小到4-8")
    
    # GPU优化建议
    if gpu_count == 0:
        recommendations.append("未检测到GPU，建议设置 device = 'cpu'")
    elif gpu_count > 1:
        use_dp = config.get('use_dataparallel', False)
        if not use_dp:
            recommendations.append(f"检测到{gpu_count}个GPU，可考虑启用数据并行")
    
    # 训练优化建议
    epochs = config.get('training', {}).get('epochs', 0)
    if epochs > 100:
        recommendations.append("训练轮数较多，建议启用早停机制")
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"{i}. {rec}")
    else:
        print("当前配置已经比较合理")
    
    print()

def main():
    """主函数"""
    print("本地配置测试和验证工具")
    print("=" * 50)
    
    # 配置文件路径
    base_dir = Path(__file__).parent
    local_config_path = base_dir / "dynamic_config_local.yaml"
    server_config_path = base_dir / "dynamic_config_server_downsampling.yaml"
    
    # 检查配置文件是否存在
    if not local_config_path.exists():
        print(f"[ERROR] 本地配置文件不存在: {local_config_path}")
        return
    
    if not server_config_path.exists():
        print(f"[WARN]  服务器配置文件不存在: {server_config_path}")
        server_config = None
    else:
        server_config = load_config(server_config_path)
    
    # 加载本地配置
    local_config = load_config(local_config_path)
    
    # 执行检查
    check_system_resources()
    
    if server_config:
        compare_configs(local_config, server_config)
    
    is_valid = validate_local_config(local_config)
    
    recommend_optimizations(local_config)
    
    # 总结
    print("=== 总结 ===")
    if is_valid:
        print("[OK] 本地配置可以使用")
        print("[TIP] 建议先用小数据集测试训练流程")
        print("[INFO] 可以根据实际性能调整参数")
    else:
        print("[ERROR] 本地配置存在问题，请修复后再使用")
    
    print("\n使用方法:")
    print(f"python dynamic_resolution_trainer.py --config {local_config_path.name}")

if __name__ == "__main__":
    main()