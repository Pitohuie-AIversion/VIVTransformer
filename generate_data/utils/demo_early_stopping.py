#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
早停功能演示脚本

展示如何使用新添加的早停参数进行训练
"""

import os
import subprocess
import sys
from pathlib import Path

def demo_basic_early_stopping():
    """
    演示基本早停功能
    """
    print("\n=== 演示1: 基本早停功能 ===")
    print("使用默认配置文件，启用早停功能")
    
    cmd = [
        "python", "dynamic_resolution_trainer.py",
        "--config", "dynamic_config.yaml",
        "--input_resolution", "16", "16",
        "--output_resolution", "32", "32",
        "--num_samples", "50",
        "--epochs", "10",
        "--batch_size", "8"
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("早停配置:")
    print("  - enable_early_stopping: True")
    print("  - patience: 15")
    print("  - monitor: val_loss")
    print("  - mode: min")
    print("  - restore_best_weights: True")
    
    return cmd

def demo_custom_early_stopping():
    """
    演示自定义早停配置
    """
    print("\n=== 演示2: 自定义早停配置 ===")
    print("创建自定义配置文件，修改早停参数")
    
    # 创建自定义配置
    custom_config = """
# 自定义早停配置示例
data:
  path: "X:\\2025\\Graduation_project\\Pdebench_input_Transformer\\VIVTransformer-1\\PDEBench\\pdebench\\data_download\\2D_DarcyFlow_beta0.1_Train.hdf5"
  input_resolution: [16, 16]
  output_resolution: [32, 32]
  num_samples: 50
  crop_mode: "center"
  batch_size: 8
  train_ratio: 0.7
  valid_ratio: 0.15
  test_ratio: 0.15

training:
  epochs: 10
  learning_rate: 0.001
  weight_decay: 0.0001
  
  # 自定义早停配置
  enable_early_stopping: true
  patience: 5                    # 更短的耐心值
  min_delta: 0.001              # 更大的改善阈值
  monitor: "train_loss"         # 监控训练损失
  mode: "min"
  restore_best_weights: true
  
  save_best_model: true
  model_save_path: "./results/models/custom_early_stopping_model.pth"
  log_interval: 1

model:
  num_layers: 2
  d_model: 128
  num_heads: 4
  max_time_steps: 10
  attention_type: "sge"

device: "auto"
seed: 42

visualization:
  enabled: false
  interval: 10
  max_samples: 1

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./results/logs/custom_early_stopping_training.log"
  verbose: false
"""
    
    # 保存自定义配置文件
    config_path = "custom_early_stopping_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(custom_config)
    
    cmd = [
        "python", "dynamic_resolution_trainer.py",
        "--config", config_path
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("自定义早停配置:")
    print("  - enable_early_stopping: True")
    print("  - patience: 5 (更短的耐心值)")
    print("  - monitor: train_loss (监控训练损失)")
    print("  - min_delta: 0.001 (更大的改善阈值)")
    print("  - restore_best_weights: True")
    
    return cmd, config_path

def demo_disabled_early_stopping():
    """
    演示禁用早停功能
    """
    print("\n=== 演示3: 禁用早停功能 ===")
    print("创建配置文件，禁用早停功能")
    
    # 创建禁用早停的配置
    disabled_config = """
# 禁用早停配置示例
data:
  path: "X:\\2025\\Graduation_project\\Pdebench_input_Transformer\\VIVTransformer-1\\PDEBench\\pdebench\\data_download\\2D_DarcyFlow_beta0.1_Train.hdf5"
  input_resolution: [16, 16]
  output_resolution: [32, 32]
  num_samples: 30
  crop_mode: "center"
  batch_size: 8
  train_ratio: 0.7
  valid_ratio: 0.15
  test_ratio: 0.15

training:
  epochs: 5  # 少量epoch，因为没有早停
  learning_rate: 0.001
  weight_decay: 0.0001
  
  # 禁用早停
  enable_early_stopping: false
  patience: 10              # 这些参数会被忽略
  min_delta: 0.0001
  monitor: "val_loss"
  mode: "min"
  restore_best_weights: true
  
  save_best_model: true
  model_save_path: "./results/models/no_early_stopping_model.pth"
  log_interval: 1

model:
  num_layers: 2
  d_model: 128
  num_heads: 4
  max_time_steps: 10
  attention_type: "sge"

device: "auto"
seed: 42

visualization:
  enabled: false
  interval: 10
  max_samples: 1

logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "./results/logs/no_early_stopping_training.log"
  verbose: false
"""
    
    # 保存配置文件
    config_path = "no_early_stopping_config.yaml"
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(disabled_config)
    
    cmd = [
        "python", "dynamic_resolution_trainer.py",
        "--config", config_path
    ]
    
    print(f"执行命令: {' '.join(cmd)}")
    print("早停配置:")
    print("  - enable_early_stopping: False (禁用早停)")
    print("  - 训练将运行完整的5个epoch")
    
    return cmd, config_path

def interactive_demo():
    """
    交互式演示
    """
    print("\n" + "=" * 60)
    print("🚀 早停功能演示")
    print("=" * 60)
    
    print("\n请选择要运行的演示:")
    print("1. 基本早停功能 (使用默认配置)")
    print("2. 自定义早停配置 (短耐心值 + 监控训练损失)")
    print("3. 禁用早停功能")
    print("4. 查看配置文件中的早停参数")
    print("5. 退出")
    
    while True:
        try:
            choice = input("\n请输入选择 (1-5): ").strip()
            
            if choice == '1':
                cmd = demo_basic_early_stopping()
                if input("\n是否执行此命令? (y/n): ").lower() == 'y':
                    subprocess.run(cmd)
                    
            elif choice == '2':
                cmd, config_path = demo_custom_early_stopping()
                if input("\n是否执行此命令? (y/n): ").lower() == 'y':
                    try:
                        subprocess.run(cmd)
                    finally:
                        # 清理临时配置文件
                        if os.path.exists(config_path):
                            os.remove(config_path)
                            print(f"已清理临时配置文件: {config_path}")
                            
            elif choice == '3':
                cmd, config_path = demo_disabled_early_stopping()
                if input("\n是否执行此命令? (y/n): ").lower() == 'y':
                    try:
                        subprocess.run(cmd)
                    finally:
                        # 清理临时配置文件
                        if os.path.exists(config_path):
                            os.remove(config_path)
                            print(f"已清理临时配置文件: {config_path}")
                            
            elif choice == '4':
                show_config_parameters()
                
            elif choice == '5':
                print("\n👋 退出演示")
                break
                
            else:
                print("❌ 无效选择，请输入1-5")
                
        except KeyboardInterrupt:
            print("\n\n👋 用户中断，退出演示")
            break
        except Exception as e:
            print(f"\n❌ 执行过程中出错: {str(e)}")

def show_config_parameters():
    """
    显示配置文件中的早停参数
    """
    print("\n=== 配置文件中的早停参数 ===")
    
    config_file = "dynamic_config.yaml"
    if os.path.exists(config_file):
        import yaml
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        training_config = config.get('training', {})
        
        print(f"\n📄 配置文件: {config_file}")
        print("\n早停相关参数:")
        
        early_stopping_params = {
            'enable_early_stopping': '是否启用早停',
            'patience': '早停耐心值（验证损失不改善的epoch数）',
            'min_delta': '最小改善阈值',
            'monitor': '监控指标',
            'mode': '监控模式',
            'restore_best_weights': '是否恢复最佳权重'
        }
        
        for param, description in early_stopping_params.items():
            value = training_config.get(param, '未设置')
            print(f"  📌 {param}: {value}")
            print(f"     说明: {description}")
            print()
        
        print("\n💡 参数说明:")
        print("  - monitor 可选值: 'train_loss', 'val_loss', 'test_loss'")
        print("  - mode 可选值: 'min'(损失越小越好), 'max'(指标越大越好)")
        print("  - patience: 连续多少个epoch没有改善就停止训练")
        print("  - min_delta: 认为有改善的最小变化量")
        print("  - restore_best_weights: 早停时是否恢复到最佳模型权重")
        
    else:
        print(f"❌ 配置文件 {config_file} 不存在")

def main():
    """
    主函数
    """
    # 创建必要的目录
    os.makedirs('./results/models', exist_ok=True)
    os.makedirs('./results/logs', exist_ok=True)
    
    # 运行交互式演示
    interactive_demo()

if __name__ == "__main__":
    main()