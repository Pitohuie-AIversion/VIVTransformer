#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版PDEBench模型对比实验运行脚本

功能:
1. 自动运行稀疏到稠密流场重建任务的横向对比实验
2. 支持多种模型配置和参数规模
3. 生成详细的对比报告和可视化结果
4. 提供公平的实验环境和统计验证

使用方法:
    python run_enhanced_comparison.py --config enhanced_config.yaml
    python run_enhanced_comparison.py --quick_test  # 快速测试模式
    python run_enhanced_comparison.py --models fno1d,unet1d,transformer  # 指定模型

作者: AI Assistant
日期: 2025
"""

import os
import sys
import argparse
import yaml
import logging
import time
import json
from pathlib import Path
from typing import Dict, List, Any

# 设置环境变量
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''
os.environ['HEADLESS'] = '1'

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

# 导入增强版训练器
try:
    from enhanced_pdebench_trainer import EnhancedTrainer, create_enhanced_config
    logger = logging.getLogger(__name__)
except ImportError as e:
    print(f"❌ 导入增强版训练器失败: {e}")
    sys.exit(1)

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """
    设置日志配置
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    # 基础配置
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # 添加文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
    
    # 设置第三方库日志级别
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('PIL').setLevel(logging.WARNING)
    logging.getLogger('h5py').setLevel(logging.WARNING)

def create_quick_test_config() -> Dict[str, Any]:
    """
    创建快速测试配置
    """
    config = create_enhanced_config()
    
    # 减少数据量和训练轮数
    config['data']['num_samples'] = 100
    config['data']['batch_size'] = 4
    config['training']['epochs'] = 10
    
    # 只启用几个代表性模型
    for model_name in config['models']:
        config['models'][model_name]['enabled'] = False
    
    # 启用小型模型进行快速测试
    test_models = ['fno1d', 'unet1d', 'mlp_small']  # 使用实际存在的模型名称
    for model_name in test_models:
        if model_name in config['models']:
            config['models'][model_name]['enabled'] = True
    
    return config

def create_custom_config(selected_models: List[str]) -> Dict[str, Any]:
    """
    创建自定义模型配置
    
    Args:
        selected_models: 选择的模型列表
        
    Returns:
        自定义配置
    """
    config = create_enhanced_config()
    
    # 禁用所有模型
    for model_name in config['models']:
        config['models'][model_name]['enabled'] = False
    
    # 启用选择的模型
    for model_name in selected_models:
        # 尝试匹配模型名称（支持模糊匹配）
        matched_models = []
        for config_model in config['models']:
            if model_name.lower() in config_model.lower():
                matched_models.append(config_model)
        
        if matched_models:
            # 如果有多个匹配，选择第一个或者选择所有规模的模型
            for matched_model in matched_models:
                config['models'][matched_model]['enabled'] = True
                print(f"✅ 启用模型: {matched_model}")
        else:
            print(f"⚠️ 未找到匹配的模型: {model_name}")
    
    return config

def validate_data_path(data_path: str) -> bool:
    """
    验证数据路径是否存在
    
    Args:
        data_path: 数据文件路径
        
    Returns:
        是否存在
    """
    if not Path(data_path).exists():
        print(f"❌ 数据文件不存在: {data_path}")
        
        # 尝试查找可能的数据文件
        possible_paths = [
            './preprocessed_data/sample_data.h5',
            '../preprocessed_data/sample_data.h5',
            './data/sample_data.h5',
            '../data/sample_data.h5',
            './sample_data.h5'
        ]
        
        print("\n🔍 尝试查找可能的数据文件:")
        for path in possible_paths:
            if Path(path).exists():
                print(f"✅ 找到数据文件: {path}")
                return path
            else:
                print(f"❌ 不存在: {path}")
        
        print("\n💡 请确保数据文件存在，或者使用以下命令生成示例数据:")
        print("   python generate_sample_data.py")
        return False
    
    return True

def create_sample_data(output_path: str, num_samples: int = 100):
    """
    创建示例数据文件
    
    Args:
        output_path: 输出路径
        num_samples: 样本数量
    """
    import numpy as np
    import h5py
    
    print(f"🔧 创建示例数据文件: {output_path}")
    
    # 确保输出目录存在
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 生成示例数据
    input_resolution = (32, 32)
    output_resolution = (128, 128)
    
    # 生成随机流场数据
    np.random.seed(42)
    
    input_data = np.random.randn(num_samples, 1, *input_resolution).astype(np.float32)
    output_data = np.random.randn(num_samples, 1, *output_resolution).astype(np.float32)
    
    # 添加一些结构化模式
    for i in range(num_samples):
        # 输入数据：简单的高斯分布
        x = np.linspace(-2, 2, input_resolution[0])
        y = np.linspace(-2, 2, input_resolution[1])
        X, Y = np.meshgrid(x, y)
        input_data[i, 0] = np.exp(-(X**2 + Y**2) / (0.5 + 0.5 * np.random.rand()))
        
        # 输出数据：更复杂的模式
        x_out = np.linspace(-2, 2, output_resolution[0])
        y_out = np.linspace(-2, 2, output_resolution[1])
        X_out, Y_out = np.meshgrid(x_out, y_out)
        output_data[i, 0] = np.sin(X_out) * np.cos(Y_out) * np.exp(-(X_out**2 + Y_out**2) / 2)
    
    # 保存到HDF5文件
    with h5py.File(output_path, 'w') as f:
        # DynamicResolutionDataset期望'tensor'键
        f.create_dataset('tensor', data=output_data)  # 使用输出数据作为主要数据
        f.create_dataset('input_data', data=input_data)
        f.create_dataset('output_data', data=output_data)
        f.create_dataset('input_resolution', data=input_resolution)
        f.create_dataset('output_resolution', data=output_resolution)
        
        # 添加元数据
        f.attrs['num_samples'] = num_samples
        f.attrs['description'] = 'Sample data for sparse-to-dense flow reconstruction'
        f.attrs['created_by'] = 'run_enhanced_comparison.py'
        f.attrs['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"✅ 示例数据创建完成: {num_samples} 个样本")
    print(f"   输入分辨率: {input_resolution}")
    print(f"   输出分辨率: {output_resolution}")

def print_available_models():
    """
    打印可用的模型列表
    """
    config = create_enhanced_config()
    
    print("\n📋 可用的模型列表:")
    print("=" * 50)
    
    categories = {
        'PDEBench原生模型': ['fno1d', 'unet1d'],
        '增强模型': ['enhanced_fno1d', 'enhanced_unet1d', 'enhanced_pinn'],
        '注意力模型': ['transformer']
    }
    
    for category, model_types in categories.items():
        print(f"\n{category}:")
        for model_type in model_types:
            matching_models = [name for name in config['models'] if model_type in name]
            for model_name in matching_models:
                model_config = config['models'][model_name]
                target_params = model_config.get('target_params', 'N/A')
                print(f"  - {model_name} (目标参数: {target_params})")

def run_experiment(config: Dict[str, Any], output_dir: str = "./results"):
    """
    运行对比实验
    
    Args:
        config: 实验配置
        output_dir: 输出目录
    """
    print("\n🚀 开始运行增强版PDEBench模型对比实验")
    print("=" * 60)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存配置
    config_path = output_path / 'experiment_config.yaml'
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    print(f"📄 实验配置已保存: {config_path}")
    
    # 设置日志
    log_path = output_path / 'experiment.log'
    setup_logging(log_file=str(log_path))
    
    # 验证数据路径
    data_path = config['data']['data_path']
    if not validate_data_path(data_path):
        # 尝试创建示例数据
        sample_data_path = output_path / 'sample_data.h5'
        create_sample_data(str(sample_data_path), config['data']['num_samples'])
        config['data']['data_path'] = str(sample_data_path)
    
    # 创建训练器
    trainer = EnhancedTrainer(config)
    
    # 运行实验
    start_time = time.time()
    try:
        results = trainer.run_comparison_experiment()
        
        # 保存结果
        results_path = output_path / 'experiment_results.json'
        
        # 处理结果以便JSON序列化
        serializable_results = {}
        for model_name, result in results.items():
            serializable_results[model_name] = {
                'config': result.get('config', {}),
                'results': result.get('results', {}),
                'test_results': result.get('test_results', {})
            }
            # 移除不能序列化的模型对象
            if 'model' in serializable_results[model_name]:
                del serializable_results[model_name]['model']
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ 实验完成! 总用时: {time.time() - start_time:.2f}s")
        print(f"📊 结果已保存到: {output_path}")
        print(f"📄 详细结果: {results_path}")
        print(f"📋 实验报告: {output_path / 'enhanced_model_comparison_report.md'}")
        
        # 显示简要结果
        print("\n📈 实验结果摘要:")
        print("-" * 40)
        for model_name, result in results.items():
            if 'error' not in result:
                test_loss = result.get('test_results', {}).get('test_loss', 'N/A')
                params = result.get('results', {}).get('model_params', 'N/A')
                # 处理test_loss可能是字符串的情况
                if isinstance(test_loss, (int, float)):
                    test_loss_str = f"{test_loss:8.6f}"
                else:
                    test_loss_str = f"{str(test_loss):>8}"
                print(f"{model_name:25} | 测试损失: {test_loss_str} | 参数: {params:>8}")
            else:
                print(f"{model_name:25} | 错误: {result['error'][:30]}...")
        
        return results
        
    except Exception as e:
        print(f"❌ 实验运行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description='增强版PDEBench模型对比实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python run_enhanced_comparison.py --config enhanced_config.yaml
  python run_enhanced_comparison.py --quick_test
  python run_enhanced_comparison.py --models fno1d,unet1d,transformer
  python run_enhanced_comparison.py --list_models
        """
    )
    
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--quick_test', action='store_true', help='快速测试模式')
    parser.add_argument('--models', type=str, help='指定要测试的模型（逗号分隔）')
    parser.add_argument('--output_dir', type=str, default='./results', help='输出目录')
    parser.add_argument('--data_path', type=str, help='数据文件路径')
    parser.add_argument('--epochs', type=int, help='训练轮数')
    parser.add_argument('--batch_size', type=int, help='批次大小')
    parser.add_argument('--list_models', action='store_true', help='列出可用模型')
    parser.add_argument('--create_sample_data', type=str, help='创建示例数据文件')
    
    args = parser.parse_args()
    
    # 列出可用模型
    if args.list_models:
        print_available_models()
        return
    
    # 创建示例数据
    if args.create_sample_data:
        create_sample_data(args.create_sample_data)
        return
    
    # 创建配置
    if args.quick_test:
        print("🏃 快速测试模式")
        config = create_quick_test_config()
    elif args.models:
        print(f"🎯 自定义模型模式: {args.models}")
        selected_models = [m.strip() for m in args.models.split(',')]
        config = create_custom_config(selected_models)
    elif args.config and Path(args.config).exists():
        print(f"📄 使用配置文件: {args.config}")
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        print("📋 使用默认配置")
        config = create_enhanced_config()
    
    # 更新配置参数
    if args.data_path:
        config['data']['data_path'] = args.data_path
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.batch_size:
        config['data']['batch_size'] = args.batch_size
    
    # 显示实验信息
    enabled_models = [name for name, cfg in config['models'].items() if cfg.get('enabled', False)]
    print(f"\n📊 实验配置:")
    print(f"   数据路径: {config['data']['data_path']}")
    print(f"   训练轮数: {config['training']['epochs']}")
    print(f"   批次大小: {config['data']['batch_size']}")
    print(f"   启用模型: {len(enabled_models)} 个")
    for model in enabled_models:
        print(f"     - {model}")
    
    # 运行实验
    results = run_experiment(config, args.output_dir)
    
    if results:
        print("\n🎉 实验成功完成!")
    else:
        print("\n💥 实验失败!")
        sys.exit(1)

if __name__ == "__main__":
    main()