#!/usr/bin/env python3
"""
配置参数传输测试脚本
"""

import yaml
import sys
import os
from pathlib import Path

# 添加当前目录到路径
sys.path.append('.')

def test_config_loading():
    """测试配置文件加载"""
    print("=== 配置文件加载测试 ===")
    
    try:
        from run_crop_model_test import load_unified_config
        config = load_unified_config('configs/unified_training_config.yaml')
        
        if config:
            print("✅ 配置文件加载成功")
            return config
        else:
            print("❌ 配置文件加载失败")
            return None
    except Exception as e:
        print(f"❌ 配置文件加载异常: {e}")
        return None

def test_parameter_transmission(config):
    """测试关键参数传输"""
    print("\n=== 关键参数传输测试 ===")
    
    if not config:
        print("❌ 配置为空，无法测试参数传输")
        return
    
    # 测试数据参数
    data_config = config.get('data', {})
    print(f"max_samples: {data_config.get('max_samples')} (类型: {type(data_config.get('max_samples'))})")
    print(f"batch_size: {data_config.get('batch_size')} (类型: {type(data_config.get('batch_size'))})")
    print(f"num_workers: {data_config.get('num_workers')} (类型: {type(data_config.get('num_workers'))})")
    
    # 测试训练参数
    training_config = config.get('training', {})
    print(f"learning_rate: {training_config.get('learning_rate')} (类型: {type(training_config.get('learning_rate'))})")
    print(f"epochs: {training_config.get('epochs')} (类型: {type(training_config.get('epochs'))})")
    print(f"weight_decay: {training_config.get('weight_decay')} (类型: {type(training_config.get('weight_decay'))})")
    
    # 测试模型参数
    models_config = config.get('models', {})
    active_model = models_config.get('active_model', 'transformer')
    print(f"active_model: {active_model}")
    
    if active_model in models_config:
        model_config = models_config[active_model]
        print(f"模型配置: {model_config}")

def test_dataloader_transmission(config):
    """测试数据加载器参数传输"""
    print("\n=== 数据加载器参数传输测试 ===")
    
    if not config:
        print("❌ 配置为空，无法测试数据加载器")
        return
    
    try:
        # 检查增强数据加载器是否可用
        try:
            from enhanced_crop_dataloader import create_enhanced_crop_dataloader
            print("✅ 增强数据加载器可用")
            
            # 创建数据加载器
            train_loader, val_loader, test_loader, normalizer = create_enhanced_crop_dataloader(config)
            
            print(f"训练集批次大小: {train_loader.batch_size}")
            print(f"训练集数据量: {len(train_loader.dataset)}")
            print(f"验证集批次大小: {val_loader.batch_size}")
            print(f"验证集数据量: {len(val_loader.dataset)}")
            print(f"测试集批次大小: {test_loader.batch_size}")
            print(f"测试集数据量: {len(test_loader.dataset)}")
            
            if normalizer:
                print(f"归一化器: {normalizer.__class__.__name__}")
            else:
                print("归一化器: 未启用")
                
        except ImportError:
            print("⚠️ 增强数据加载器不可用，尝试标准数据加载器")
            from data.crop_dataloader import create_crop_dataloader
            train_loader, val_loader, test_loader = create_crop_dataloader(config)
            
            print(f"训练集批次大小: {train_loader.batch_size}")
            print(f"训练集数据量: {len(train_loader.dataset)}")
            
    except Exception as e:
        print(f"❌ 数据加载器测试失败: {e}")

def test_model_creation(config):
    """测试模型创建参数传输"""
    print("\n=== 模型创建参数传输测试 ===")
    
    if not config:
        print("❌ 配置为空，无法测试模型创建")
        return
    
    try:
        from run_crop_model_test import create_enhanced_model
        
        # 获取模型配置
        models_config = config.get('models', {})
        active_model = models_config.get('active_model', 'transformer')
        
        if active_model in models_config:
            model_config = models_config[active_model]
            
            # 模拟输入输出维度
            input_dim = config.get('data', {}).get('input_dim', 1024)
            output_dim = config.get('data', {}).get('output_dim', 16384)
            
            print(f"创建模型: {active_model}")
            print(f"输入维度: {input_dim}")
            print(f"输出维度: {output_dim}")
            print(f"模型配置: {model_config}")
            
            # 创建模型
            model = create_enhanced_model(model_config, input_dim, output_dim)
            
            # 计算参数量
            param_count = sum(p.numel() for p in model.parameters())
            print(f"✅ 模型创建成功，参数量: {param_count:,}")
            
        else:
            print(f"❌ 未找到活动模型配置: {active_model}")
            
    except Exception as e:
        print(f"❌ 模型创建测试失败: {e}")

def test_training_config_transmission(config):
    """测试训练配置传输"""
    print("\n=== 训练配置传输测试 ===")
    
    if not config:
        print("❌ 配置为空，无法测试训练配置")
        return
    
    try:
        from run_crop_model_test import train_unified_model
        import torch
        import torch.nn as nn
        
        # 创建简单模型用于测试
        model = nn.Linear(10, 5)
        device = torch.device('cpu')
        
        # 创建简单数据
        train_data = [(torch.randn(2, 10), torch.randn(2, 5)) for _ in range(3)]
        val_data = [(torch.randn(2, 10), torch.randn(2, 5)) for _ in range(2)]
        
        class SimpleDataLoader:
            def __init__(self, data):
                self.data = data
                self.batch_size = 2
            def __iter__(self):
                return iter(self.data)
            def __len__(self):
                return len(self.data)
        
        train_loader = SimpleDataLoader(train_data)
        val_loader = SimpleDataLoader(val_data)
        
        # 获取训练配置
        training_config = config.get('training', {})
        loss_config = config.get('loss', {})
        
        print(f"训练配置传输: {training_config}")
        print(f"损失配置传输: {loss_config}")
        
        # 测试训练函数（只运行1个epoch）
        original_epochs = training_config.get('epochs', 50)
        training_config['epochs'] = 1  # 只测试1个epoch
        
        train_losses, val_losses = train_unified_model(
            model, train_loader, val_loader, device, training_config, loss_config
        )
        
        print(f"✅ 训练配置传输成功")
        print(f"训练损失: {train_losses}")
        print(f"验证损失: {val_losses}")
        
        # 恢复原始epochs设置
        training_config['epochs'] = original_epochs
        
    except Exception as e:
        print(f"❌ 训练配置传输测试失败: {e}")

def main():
    """主函数"""
    print("🔍 配置参数传输完整性测试")
    print("=" * 50)
    
    # 1. 测试配置文件加载
    config = test_config_loading()
    
    # 2. 测试关键参数传输
    test_parameter_transmission(config)
    
    # 3. 测试数据加载器参数传输
    test_dataloader_transmission(config)
    
    # 4. 测试模型创建参数传输
    test_model_creation(config)
    
    # 5. 测试训练配置传输
    test_training_config_transmission(config)
    
    print("\n=== 配置参数传输测试完成 ===")

if __name__ == "__main__":
    main()