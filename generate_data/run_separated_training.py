#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分离式训练启动脚本

功能:
1. 自动执行数据预处理和训练的完整流程
2. 智能检测是否需要重新预处理数据
3. 提供多种运行模式选择
4. 自动资源检测和配置优化

作者: AI Assistant
日期: 2025
"""

import os
import sys
import subprocess
import argparse
import yaml
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional

def print_banner():
    """
    打印启动横幅
    """
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                    分离式训练启动器                          ║
║                                                              ║
║  🚀 自动化数据预处理和模型训练流程                           ║
║  💡 解决服务器CPU锁定问题的最佳方案                          ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_system_resources():
    """
    检查系统资源
    
    Returns:
        资源信息字典
    """
    import psutil
    import torch
    
    resources = {
        'cpu_count': psutil.cpu_count(),
        'memory_gb': psutil.virtual_memory().total / (1024**3),
        'available_memory_gb': psutil.virtual_memory().available / (1024**3),
        'disk_free_gb': psutil.disk_usage('.').free / (1024**3),
        'has_gpu': torch.cuda.is_available(),
        'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        'gpu_memory_gb': []
    }
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            resources['gpu_memory_gb'].append(gpu_memory)
    
    return resources

def print_system_info(resources: Dict[str, Any]):
    """
    打印系统信息
    
    Args:
        resources: 资源信息
    """
    print("\n📊 系统资源检测:")
    print(f"  CPU核心数: {resources['cpu_count']}")
    print(f"  总内存: {resources['memory_gb']:.1f}GB")
    print(f"  可用内存: {resources['available_memory_gb']:.1f}GB")
    print(f"  磁盘可用空间: {resources['disk_free_gb']:.1f}GB")
    
    if resources['has_gpu']:
        print(f"  GPU数量: {resources['gpu_count']}")
        for i, gpu_mem in enumerate(resources['gpu_memory_gb']):
            print(f"  GPU {i} 内存: {gpu_mem:.1f}GB")
    else:
        print("  GPU: 未检测到")

def suggest_config(resources: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据系统资源建议配置
    
    Args:
        resources: 资源信息
        
    Returns:
        建议的配置
    """
    config_suggestions = {
        'dataloader': {},
        'model': {},
        'training': {},
        'resource_level': 'low'
    }
    
    # 根据内存大小调整配置
    if resources['memory_gb'] >= 32:
        # 高配置
        config_suggestions['resource_level'] = 'high'
        config_suggestions['dataloader'] = {
            'batch_size': 8,
            'num_workers': min(4, resources['cpu_count'] // 2),
            'pin_memory': True,
            'prefetch_factor': 4
        }
        config_suggestions['model'] = {
            'd_model': 512,
            'num_layers': 6,
            'dim_feedforward': 2048
        }
    elif resources['memory_gb'] >= 16:
        # 中等配置
        config_suggestions['resource_level'] = 'medium'
        config_suggestions['dataloader'] = {
            'batch_size': 4,
            'num_workers': min(2, resources['cpu_count'] // 2),
            'pin_memory': False,
            'prefetch_factor': 2
        }
        config_suggestions['model'] = {
            'd_model': 256,
            'num_layers': 4,
            'dim_feedforward': 1024
        }
    else:
        # 低配置
        config_suggestions['resource_level'] = 'low'
        config_suggestions['dataloader'] = {
            'batch_size': 2,
            'num_workers': 1,
            'pin_memory': False,
            'prefetch_factor': 1
        }
        config_suggestions['model'] = {
            'd_model': 128,
            'num_layers': 2,
            'dim_feedforward': 512
        }
    
    # GPU配置调整
    if resources['has_gpu'] and resources['gpu_memory_gb']:
        gpu_memory = max(resources['gpu_memory_gb'])
        if gpu_memory < 4:
            # 低GPU内存
            config_suggestions['dataloader']['batch_size'] = min(
                config_suggestions['dataloader']['batch_size'], 2
            )
            config_suggestions['model']['d_model'] = min(
                config_suggestions['model']['d_model'], 256
            )
    
    return config_suggestions

def check_preprocessed_data(data_dir: str, data_path: str, config: Dict[str, Any]) -> bool:
    """
    检查预处理数据是否存在且有效
    
    Args:
        data_dir: 预处理数据目录
        data_path: 原始数据路径
        config: 当前配置
        
    Returns:
        是否需要重新预处理
    """
    data_path_obj = Path(data_dir)
    
    # 检查必需文件是否存在
    required_files = ['train_data.h5', 'valid_data.h5', 'test_data.h5']
    for file in required_files:
        if not (data_path_obj / file).exists():
            return False
    
    # 检查配置是否匹配
    config_file = data_path_obj / 'preprocessing_config.yaml'
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                old_config = yaml.safe_load(f)
            
            # 比较关键配置项
            current_data_config = config.get('data', {})
            old_data_config = old_config.get('data', {})
            
            key_params = ['input_resolution', 'output_resolution', 'num_samples']
            for param in key_params:
                if current_data_config.get(param) != old_data_config.get(param):
                    return False
            
            # 检查原始数据文件是否更新
            if Path(data_path).stat().st_mtime > config_file.stat().st_mtime:
                return False
            
            return True
        except Exception:
            return False
    
    return False

def run_preprocessing(data_path: str, output_dir: str, config_path: str, num_samples: Optional[int] = None) -> bool:
    """
    运行数据预处理
    
    Args:
        data_path: 原始数据路径
        output_dir: 输出目录
        config_path: 配置文件路径
        num_samples: 样本数量
        
    Returns:
        是否成功
    """
    print("\n🔄 开始数据预处理...")
    
    cmd = [
        sys.executable, '-m', 'generate_data.data_preprocessor',
        '--config', config_path,
        '--data_path', data_path,
        '--output_dir', output_dir
    ]
    
    if num_samples:
        cmd.extend(['--num_samples', str(num_samples)])
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 数据预处理完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 数据预处理失败: {e}")
        print(f"错误输出: {e.stderr}")
        return False

def run_onnx_verification(output_head_type: Optional[str] = None,
                          out_channels_per_token: Optional[int] = None) -> bool:
    print("\n🔎 运行 ONNX 导出与验证...")
    try:
        # 导出 ONNX（使用 modify_multi_attention/mymodels/export_to_onnx.py）
        onnx_cmd = [
            sys.executable,
            '-m', 'modify_multi_attention.mymodels.export_to_onnx'
        ]
        if output_head_type:
            onnx_cmd.extend(['--output-head-type', output_head_type])
        if out_channels_per_token is not None:
            onnx_cmd.extend(['--out-channels-per-token', str(out_channels_per_token)])
        subprocess.run(onnx_cmd, check=True)

        # 运行 ONNX 形状验证（定位到仓库根目录下的 verify_onnx_runtime.py）
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        verify_script = os.path.join(repo_root, 'verify_onnx_runtime.py')
        verify_cmd = [sys.executable, verify_script]
        env = os.environ.copy()
        subprocess.run(verify_cmd, check=True, env=env)
        print("✅ ONNX 验证通过")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ ONNX 验证失败: {e}")
        return False

def run_training(data_dir: str, config_path: str, resume: Optional[str] = None,
                 output_head_type: Optional[str] = None,
                 out_channels_per_token: Optional[int] = None,
                 epochs: Optional[int] = None) -> bool:
    """
    运行模型训练
    
    Args:
        data_dir: 预处理数据目录
        config_path: 配置文件路径
        resume: 恢复训练的检查点
        output_head_type: 覆盖模型输出头类型（global/per_token）
        out_channels_per_token: 覆盖逐token通道数 C
        
    Returns:
        是否成功
    """
    print("\n🚀 开始模型训练...")
    
    cmd = [
        sys.executable, '-m', 'generate_data.simple_trainer',
        '--config', config_path,
        '--data_dir', data_dir
    ]
    
    if resume:
        cmd.extend(['--resume', resume])
    if output_head_type:
        cmd.extend(['--output-head-type', output_head_type])
    if out_channels_per_token is not None:
        cmd.extend(['--out-channels-per-token', str(out_channels_per_token)])
    if epochs is not None:
        cmd.extend(['--epochs', str(epochs)])
    
    try:
        # 使用实时输出
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, 
                                 text=True, bufsize=1, universal_newlines=True)
        
        for line in process.stdout:
            print(line.rstrip())
        
        process.wait()
        
        if process.returncode == 0:
            print("✅ 模型训练完成")
            return True
        else:
            print(f"❌ 模型训练失败，退出码: {process.returncode}")
            return False
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        return False

def create_optimized_config(base_config_path: str, suggestions: Dict[str, Any], output_path: str):
    """
    创建优化的配置文件
    
    Args:
        base_config_path: 基础配置文件路径
        suggestions: 优化建议
        output_path: 输出配置文件路径
    """
    # 加载基础配置
    with open(base_config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 应用优化建议
    if 'dataloader' in suggestions:
        config.setdefault('dataloader', {}).update(suggestions['dataloader'])
    
    if 'model' in suggestions:
        config.setdefault('model', {}).update(suggestions['model'])
    
    # 保存优化配置
    with open(output_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    print(f"📝 优化配置已保存: {output_path}")
    print(f"🎯 资源级别: {suggestions['resource_level']}")

def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='分离式训练启动器')
    
    # 必需参数
    parser.add_argument('--data_path', type=str, required=True,
                       help='原始数据文件路径')
    
    # 可选参数
    parser.add_argument('--config', type=str, default='dynamic_config_server_optimized_final.yaml',
                       help='配置文件路径')
    parser.add_argument('--output_dir', type=str, default='./preprocessed_data',
                       help='预处理数据输出目录')
    parser.add_argument('--num_samples', type=int,
                       help='处理的样本数量')
    parser.add_argument('--resume', type=str,
                       help='恢复训练的检查点路径')
    
    # 运行模式
    parser.add_argument('--mode', type=str, choices=['auto', 'preprocess', 'train'], 
                       default='auto', help='运行模式')
    parser.add_argument('--force_preprocess', action='store_true',
                       help='强制重新预处理数据')
    parser.add_argument('--auto_optimize', action='store_true',
                       help='自动优化配置')
    parser.add_argument('--skip_resource_check', action='store_true',
                       help='跳过资源检测')
    # 新增：ONNX 验证
    parser.add_argument('--verify-onnx', action='store_true',
                       help='在训练前导出并验证 ONNX')
    
    # 新增：透传训练脚本的输出头配置
    parser.add_argument('--output-head-type', dest='output_head_type', choices=['global', 'per_token'],
                        help='覆盖模型输出头类型')
    parser.add_argument('--out-channels-per-token', dest='out_channels_per_token', type=int,
                        help='覆盖逐token通道数C')
    # 新增：训练轮数
    parser.add_argument('--epochs', type=int, help='训练轮数')
    
    return parser.parse_args()

def main():
    """
    主函数
    """
    # 打印横幅
    print_banner()
    
    # 解析参数
    args = parse_arguments()
    
    try:
        # 检查文件存在性
        if not Path(args.data_path).exists():
            print(f"❌ 数据文件不存在: {args.data_path}")
            return 1
        
        if not Path(args.config).exists():
            print(f"❌ 配置文件不存在: {args.config}")
            return 1
        
        # 系统资源检测
        if not args.skip_resource_check:
            try:
                resources = check_system_resources()
                print_system_info(resources)
                
                # 自动优化配置
                if args.auto_optimize:
                    suggestions = suggest_config(resources)
                    optimized_config_path = 'optimized_config.yaml'
                    create_optimized_config(args.config, suggestions, optimized_config_path)
                    args.config = optimized_config_path
                    
                    print(f"\n💡 配置优化建议:")
                    print(f"  批次大小: {suggestions['dataloader']['batch_size']}")
                    print(f"  工作进程: {suggestions['dataloader']['num_workers']}")
                    print(f"  模型维度: {suggestions['model']['d_model']}")
                    print(f"  模型层数: {suggestions['model']['num_layers']}")
                    
            except ImportError:
                print("⚠️ 无法导入psutil，跳过资源检测")
        
        # 加载配置
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        success = True
        
        # 根据模式执行
        if args.mode in ['auto', 'preprocess']:
            # 检查是否需要预处理
            need_preprocess = (args.force_preprocess or 
                             not check_preprocessed_data(args.output_dir, args.data_path, config))
            
            if need_preprocess:
                print("\n🔍 检测到需要预处理数据")
                success = run_preprocessing(args.data_path, args.output_dir, args.config, args.num_samples)
            else:
                print("\n✅ 发现有效的预处理数据，跳过预处理步骤")
        
        if success and args.mode in ['auto', 'train']:
            # 可选：ONNX 验证
            if args.verify_onnx:
                ok = run_onnx_verification(args.output_head_type, args.out_channels_per_token)
                if not ok:
                    print("❌ ONNX 验证未通过，中止训练")
                    return 1
            # 确保预处理数据存在
            if not Path(args.output_dir).exists():
                print(f"❌ 预处理数据目录不存在: {args.output_dir}")
                print("请先运行预处理步骤")
                return 1
            
            success = run_training(args.output_dir, args.config, args.resume,
                                   output_head_type=args.output_head_type,
                                   out_channels_per_token=args.out_channels_per_token,
                                   epochs=args.epochs)
        
        if success:
            print("\n🎉 分离式训练流程完成！")
            print("\n📋 后续操作建议:")
            print("  - 查看训练日志了解训练进度")
            print("  - 使用 nvidia-smi 监控GPU使用情况")
            print("  - 检查生成的模型检查点文件")
            return 0
        else:
            print("\n❌ 训练流程失败")
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️ 程序被用户中断")
        return 1
    except Exception as e:
        print(f"\n❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)