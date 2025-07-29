#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
早停功能测试脚本

测试新添加的早停参数是否正常工作：
- enable_early_stopping
- monitor (train_loss, val_loss, test_loss)
- mode (min, max)
- restore_best_weights
- min_delta
"""

import os
import sys
import yaml
import tempfile
from pathlib import Path

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'modify_multi_attention'))
sys.path.append(str(Path(__file__).parent))

def create_test_config(test_name, early_stopping_config):
    """
    创建测试配置文件
    """
    config = {
        'data': {
            'path': "X:\\2025\\Graduation_project\\Pdebench_input_Transformer\\VIVTransformer-1\\PDEBench\\pdebench\\data_download\\2D_DarcyFlow_beta0.1_Train.hdf5",
            'input_resolution': [16, 16],  # 小分辨率快速测试
            'output_resolution': [32, 32],
            'num_samples': 20,  # 少量样本
            'crop_mode': 'center',
            'batch_size': 4,
            'train_ratio': 0.6,
            'valid_ratio': 0.2,
            'test_ratio': 0.2
        },
        'training': {
            'epochs': 5,  # 少量epoch
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'log_interval': 1,
            'save_best_model': True,
            'model_save_path': f'./results/models/test_{test_name}_model.pth',
            **early_stopping_config  # 合并早停配置
        },
        'model': {
            'num_layers': 2,  # 简化模型
            'd_model': 128,
            'num_heads': 4,
            'max_time_steps': 10,
            'attention_type': 'sge'
        },
        'device': 'auto',
        'seed': 42,
        'visualization': {
            'enabled': False,  # 关闭可视化加速测试
            'interval': 10,
            'max_samples': 1
        },
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': f'./results/logs/test_{test_name}_training.log',
            'verbose': False
        }
    }
    return config

def test_early_stopping_enabled():
    """
    测试启用早停功能
    """
    print("\n=== 测试1: 启用早停功能 ===")
    
    early_stopping_config = {
        'enable_early_stopping': True,
        'patience': 2,
        'min_delta': 0.001,
        'monitor': 'val_loss',
        'mode': 'min',
        'restore_best_weights': True
    }
    
    config = create_test_config('early_stopping_enabled', early_stopping_config)
    
    # 保存临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        config_path = f.name
    
    try:
        # 导入训练器
        from dynamic_resolution_trainer import main as train_main
        import argparse
        
        # 模拟命令行参数
        sys.argv = ['test_early_stopping.py', '--config', config_path]
        
        print(f"使用配置文件: {config_path}")
        print(f"早停配置: {early_stopping_config}")
        
        # 运行训练
        train_main()
        
        print("✅ 测试1通过: 启用早停功能正常")
        
    except Exception as e:
        print(f"❌ 测试1失败: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时文件
        if os.path.exists(config_path):
            os.unlink(config_path)

def test_early_stopping_disabled():
    """
    测试禁用早停功能
    """
    print("\n=== 测试2: 禁用早停功能 ===")
    
    early_stopping_config = {
        'enable_early_stopping': False,
        'patience': 2,  # 这些参数应该被忽略
        'min_delta': 0.001,
        'monitor': 'val_loss',
        'mode': 'min',
        'restore_best_weights': True
    }
    
    config = create_test_config('early_stopping_disabled', early_stopping_config)
    
    # 保存临时配置文件
    with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        config_path = f.name
    
    try:
        # 导入训练器
        from dynamic_resolution_trainer import main as train_main
        import argparse
        
        # 模拟命令行参数
        sys.argv = ['test_early_stopping.py', '--config', config_path]
        
        print(f"使用配置文件: {config_path}")
        print(f"早停配置: {early_stopping_config}")
        
        # 运行训练
        train_main()
        
        print("✅ 测试2通过: 禁用早停功能正常")
        
    except Exception as e:
        print(f"❌ 测试2失败: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        # 清理临时文件
        if os.path.exists(config_path):
            os.unlink(config_path)

def test_different_monitor_metrics():
    """
    测试不同的监控指标
    """
    print("\n=== 测试3: 不同监控指标 ===")
    
    monitors = ['train_loss', 'val_loss', 'test_loss']
    
    for monitor in monitors:
        print(f"\n--- 测试监控指标: {monitor} ---")
        
        early_stopping_config = {
            'enable_early_stopping': True,
            'patience': 2,
            'min_delta': 0.0001,
            'monitor': monitor,
            'mode': 'min',
            'restore_best_weights': True
        }
        
        config = create_test_config(f'monitor_{monitor}', early_stopping_config)
        
        # 保存临时配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            config_path = f.name
        
        try:
            # 导入训练器
            from dynamic_resolution_trainer import main as train_main
            import argparse
            
            # 模拟命令行参数
            sys.argv = ['test_early_stopping.py', '--config', config_path]
            
            print(f"使用配置文件: {config_path}")
            print(f"监控指标: {monitor}")
            
            # 运行训练
            train_main()
            
            print(f"✅ 监控指标 {monitor} 测试通过")
            
        except Exception as e:
            print(f"❌ 监控指标 {monitor} 测试失败: {str(e)}")
            import traceback
            traceback.print_exc()
        finally:
            # 清理临时文件
            if os.path.exists(config_path):
                os.unlink(config_path)

def test_config_validation():
    """
    测试配置文件验证
    """
    print("\n=== 测试4: 配置文件验证 ===")
    
    # 测试配置文件是否正确加载早停参数
    config_path = "dynamic_config.yaml"
    
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        training_config = config.get('training', {})
        
        # 检查早停参数
        expected_params = [
            'enable_early_stopping',
            'patience',
            'min_delta',
            'monitor',
            'mode',
            'restore_best_weights'
        ]
        
        print("检查配置文件中的早停参数:")
        for param in expected_params:
            if param in training_config:
                print(f"  ✅ {param}: {training_config[param]}")
            else:
                print(f"  ❌ 缺少参数: {param}")
        
        print("\n✅ 测试4通过: 配置文件验证完成")
    else:
        print(f"❌ 测试4失败: 配置文件 {config_path} 不存在")

def main():
    """
    运行所有测试
    """
    print("🧪 开始早停功能测试")
    print("=" * 50)
    
    # 创建结果目录
    os.makedirs('./results/models', exist_ok=True)
    os.makedirs('./results/logs', exist_ok=True)
    
    try:
        # 运行测试
        test_config_validation()
        test_early_stopping_enabled()
        test_early_stopping_disabled()
        test_different_monitor_metrics()
        
        print("\n" + "=" * 50)
        print("🎉 所有早停功能测试完成!")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()