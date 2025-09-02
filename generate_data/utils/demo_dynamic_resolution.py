#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动态分辨率训练器使用示例

展示如何使用动态分辨率训练器进行不同分辨率的训练

作者: AI Assistant
日期: 2025
"""

import os
import subprocess
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """
    运行命令并记录日志
    
    Args:
        cmd: 命令列表
        description: 命令描述
    """
    logger.info(f"=== {description} ===")
    logger.info(f"执行命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            logger.info("[OK] {description} 成功完成")
            if result.stdout:
                logger.info(f"输出: {result.stdout[-500:]}...")  # 显示最后500字符
        else:
            logger.error("[ERROR] {description} 失败")
            logger.error(f"错误: {result.stderr}")
            
    except Exception as e:
        logger.error("[ERROR] 执行 {description} 时出错: {str(e)}")

def demo_command_line_usage():
    """
    演示命令行参数使用方式
    """
    logger.info("\n" + "="*60)
    logger.info("演示1: 使用命令行参数")
    logger.info("="*60)
    
    # 示例1: 基本用法
    cmd1 = [
        "python", "dynamic_resolution_trainer.py",
        "--input_resolution", "32", "32",
        "--output_resolution", "128", "128",
        "--epochs", "2",
        "--batch_size", "4",
        "--num_samples", "50"
    ]
    run_command(cmd1, "基本命令行训练 (32x32 -> 128x128)")
    
    # 示例2: 不同分辨率
    cmd2 = [
        "python", "dynamic_resolution_trainer.py",
        "--input_resolution", "16", "16",
        "--output_resolution", "64", "64",
        "--epochs", "2",
        "--batch_size", "4",
        "--num_samples", "30",
        "--attention_type", "cbam",
        "--crop_mode", "random"
    ]
    run_command(cmd2, "超分辨率训练 (16x16 -> 64x64)")

def demo_yaml_config_usage():
    """
    演示YAML配置文件使用方式
    """
    logger.info("\n" + "="*60)
    logger.info("演示2: 使用YAML配置文件")
    logger.info("="*60)
    
    # 创建自定义配置文件
    custom_config = """
# 自定义配置示例
data:
  path: "X:\\2025\\Graduation_project\\Pdebench_input_Transformer\\VIVTransformer-1\\PDEBench\\pdebench\\data_download\\2D_DarcyFlow_beta0.1_Train.hdf5"
  input_resolution: [64, 64]
  output_resolution: [64, 64]
  num_samples: 40
  crop_mode: "center"
  batch_size: 8
  train_ratio: 0.7
  valid_ratio: 0.15
  test_ratio: 0.15

training:
  epochs: 3
  learning_rate: 0.0005
  weight_decay: 0.0001
  patience: 10
  min_delta: 0.000001
  save_best_model: true
  model_save_path: "./models/custom_model.pth"
  log_interval: 5

model:
  num_layers: 3
  d_model: 256
  num_heads: 4
  max_time_steps: 100
  attention_type: "eca"

device: "auto"
seed: 123

visualization:
  enabled: true
  interval: 5
  max_samples: 3
"""
    
    # 保存自定义配置
    config_path = "custom_demo_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(custom_config)
    
    logger.info(f"创建自定义配置文件: {config_path}")
    
    # 使用配置文件训练
    cmd = [
        "python", "dynamic_resolution_trainer.py",
        "--config", config_path
    ]
    run_command(cmd, "使用YAML配置文件训练 (64x64 -> 64x64)")

def demo_different_resolutions():
    """
    演示不同分辨率组合
    """
    logger.info("\n" + "="*60)
    logger.info("演示3: 不同分辨率组合")
    logger.info("="*60)
    
    # 分辨率组合列表
    resolution_configs = [
        {
            'name': '降采样任务',
            'input': [128, 128],
            'output': [32, 32],
            'description': '从高分辨率输入预测低分辨率输出'
        },
        {
            'name': '非正方形分辨率',
            'input': [32, 64],
            'output': [64, 128],
            'description': '处理非正方形的输入输出'
        },
        {
            'name': '极小输入',
            'input': [8, 8],
            'output': [32, 32],
            'description': '从极小输入重建较大输出'
        }
    ]
    
    for config in resolution_configs:
        logger.info(f"\n--- {config['name']} ---")
        logger.info(f"描述: {config['description']}")
        logger.info(f"输入: {config['input']}, 输出: {config['output']}")
        
        cmd = [
            "python", "dynamic_resolution_trainer.py",
            "--input_resolution", str(config['input'][0]), str(config['input'][1]),
            "--output_resolution", str(config['output'][0]), str(config['output'][1]),
            "--epochs", "1",
            "--batch_size", "4",
            "--num_samples", "20"
        ]
        run_command(cmd, config['name'])

def show_usage_examples():
    """
    显示使用示例
    """
    logger.info("\n" + "="*80)
    logger.info("动态分辨率训练器使用指南")
    logger.info("="*80)
    
    examples = [
        {
            'title': '1. 基本命令行使用',
            'command': 'python dynamic_resolution_trainer.py --input_resolution 32 32 --output_resolution 128 128 --epochs 10',
            'description': '使用32x32输入，128x128输出，训练10个epoch'
        },
        {
            'title': '2. 使用配置文件',
            'command': 'python dynamic_resolution_trainer.py --config dynamic_config.yaml',
            'description': '使用YAML配置文件进行训练'
        },
        {
            'title': '3. 超分辨率重建',
            'command': 'python dynamic_resolution_trainer.py --input_resolution 16 16 --output_resolution 64 64 --attention_type cbam',
            'description': '4倍超分辨率重建，使用CBAM注意力机制'
        },
        {
            'title': '4. 自定义参数组合',
            'command': 'python dynamic_resolution_trainer.py --input_resolution 32 64 --output_resolution 64 128 --crop_mode random --batch_size 8',
            'description': '非正方形分辨率，随机裁剪模式'
        },
        {
            'title': '5. 快速测试',
            'command': 'python dynamic_resolution_trainer.py --input_resolution 16 16 --output_resolution 32 32 --epochs 2 --num_samples 20',
            'description': '快速测试，少量样本和epoch'
        }
    ]
    
    for example in examples:
        logger.info(f"\n{example['title']}:")
        logger.info(f"命令: {example['command']}")
        logger.info(f"说明: {example['description']}")
    
    logger.info("\n" + "="*80)
    logger.info("可用参数说明")
    logger.info("="*80)
    
    parameters = [
        ('--input_resolution H W', '输入分辨率，例如: --input_resolution 32 32'),
        ('--output_resolution H W', '输出分辨率，例如: --output_resolution 128 128'),
        ('--num_samples N', '样本数量，例如: --num_samples 100'),
        ('--crop_mode MODE', '裁剪模式: center, random, corner'),
        ('--epochs N', '训练轮数，例如: --epochs 10'),
        ('--batch_size N', '批次大小，例如: --batch_size 16'),
        ('--learning_rate LR', '学习率，例如: --learning_rate 0.001'),
        ('--attention_type TYPE', '注意力类型: sge, cbam, eca, se, relative, external'),
        ('--config PATH', 'YAML配置文件路径')
    ]
    
    for param, desc in parameters:
        logger.info(f"{param:<25} : {desc}")

def main():
    """
    主函数
    """
    logger.info("[INFO] 动态分辨率训练器演示开始")
    
    # 显示使用示例
    show_usage_examples()
    
    # 询问用户是否要运行演示
    print("\n是否要运行实际演示? (y/n): ", end='')
    choice = input().lower().strip()
    
    if choice == 'y' or choice == 'yes':
        logger.info("开始运行演示...")
        
        # 演示1: 命令行参数
        demo_command_line_usage()
        
        # 演示2: YAML配置
        demo_yaml_config_usage()
        
        # 演示3: 不同分辨率
        demo_different_resolutions()
        
        logger.info("\n[OK] 所有演示完成!")
    else:
        logger.info("演示已跳过，请参考上面的使用示例。")
    
    logger.info("\n📝 提示:")
    logger.info("1. 所有训练结果会保存在 ./models/ 目录")
    logger.info("2. 日志文件会保存在 ./logs/ 目录")
    logger.info("3. 可视化结果会保存为 PNG 图片")
    logger.info("4. 修改 dynamic_config.yaml 可以自定义默认配置")

if __name__ == "__main__":
    main()