#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型初始化和训练循环参数传输测试
测试配置参数是否正确传递到模型创建和训练过程中
"""

import sys
import os
import torch
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 导入必要的模块
from run_crop_model_test import (
    load_unified_config,
    create_enhanced_model,
    create_simple_model,
    train_model,
    evaluate_model
)
from data.crop_dataloader import create_crop_dataloader

def test_model_initialization_params():
    """测试模型初始化参数传输"""
    print("\n=== 模型初始化参数传输测试 ===")
    
    # 加载配置
    config = load_unified_config('configs/unified_config.yaml')
    
    # 测试不同模型类型的参数传输
    model_types = ['transformer', 'mlp', 'unet', 'fno']
    input_dim = 1024
    output_dim = 16384
    
    for model_type in model_types:
        print(f"\n--- 测试 {model_type} 模型参数传输 ---")
        
        # 获取模型配置，使用默认配置如果不存在
        if 'models' in config and model_type in config['models']:
            model_config = config['models'][model_type].copy()
        else:
            # 使用默认配置
            model_config = {
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 3,
                'dropout': 0.1,
                'hidden_dims': [256, 128],
                'base_ch': 32,
                'modes': 12,
                'width': 64
            }
        
        model_config['model_type'] = model_type
        model_config['input_dim'] = input_dim
        model_config['output_dim'] = output_dim
        
        print(f"模型配置参数: {model_config}")
        
        try:
            # 尝试创建增强模型
            model = create_enhanced_model(model_config, input_dim, output_dim)
            if model is not None:
                param_count = sum(p.numel() for p in model.parameters())
                print(f"✅ 增强{model_type}模型创建成功，参数量: {param_count:,}")
                
                # 验证模型输入输出维度
                test_input = torch.randn(1, input_dim)
                with torch.no_grad():
                    output = model(test_input)
                    print(f"   输入维度: {test_input.shape}, 输出维度: {output.shape}")
                    
                    # 验证输出维度是否正确
                    if output.shape[-1] == output_dim:
                        print(f"   ✅ 输出维度正确")
                    else:
                        print(f"   ❌ 输出维度错误，期望: {output_dim}, 实际: {output.shape[-1]}")
            else:
                print(f"❌ 增强{model_type}模型创建失败")
                
        except Exception as e:
            print(f"❌ 增强{model_type}模型创建异常: {e}")
            
            # 尝试创建简单模型作为后备
            try:
                model = create_simple_model(model_type, input_dim, output_dim, model_config)
                param_count = sum(p.numel() for p in model.parameters())
                print(f"✅ 简单{model_type}模型创建成功，参数量: {param_count:,}")
            except Exception as e2:
                print(f"❌ 简单{model_type}模型也创建失败: {e2}")

def test_training_loop_params():
    """测试训练循环参数传输"""
    print("\n=== 训练循环参数传输测试 ===")
    
    # 加载配置
    config = load_unified_config('configs/unified_config.yaml')
    
    # 创建简单的测试数据
    print("\n--- 创建测试数据 ---")
    batch_size = config['data']['batch_size']
    input_dim = 1024
    output_dim = 16384
    
    # 创建模拟数据
    train_data = torch.randn(100, input_dim)
    train_targets = torch.randn(100, output_dim)
    val_data = torch.randn(20, input_dim)
    val_targets = torch.randn(20, output_dim)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.TensorDataset(train_data, train_targets)
    val_dataset = torch.utils.data.TensorDataset(val_data, val_targets)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    print(f"训练数据: {len(train_dataset)} 样本, 批次大小: {batch_size}")
    print(f"验证数据: {len(val_dataset)} 样本")
    
    # 创建简单模型进行测试
    print("\n--- 创建测试模型 ---")
    model_config = config['models']['transformer'].copy()
    model = create_simple_model('transformer', input_dim, output_dim, model_config)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {param_count:,}")
    
    # 测试训练配置参数传输
    print("\n--- 训练配置参数传输测试 ---")
    training_config = config['training']
    print(f"原始训练配置: {training_config}")
    
    # 验证关键参数类型和值
    key_params = ['epochs', 'learning_rate', 'weight_decay', 'patience', 'min_delta']
    for param in key_params:
        if param in training_config:
            value = training_config[param]
            print(f"  {param}: {value} (类型: {type(value)})")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 修改配置为快速测试
    test_config = config.copy()
    test_config['training']['epochs'] = 2  # 只训练2个epoch进行测试
    
    print("\n--- 开始训练测试 ---")
    try:
        # 调用训练函数
        results = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=test_config,
            device=device,
            normalizer=None
        )
        
        print(f"✅ 训练完成")
        print(f"   训练损失: {results['train_losses']}")
        print(f"   验证损失: {results['val_losses']}")
        print(f"   最佳验证损失: {results['best_val_loss']:.6f}")
        print(f"   最终轮次: {results['final_epoch']}")
        
        # 验证参数是否正确传递
        if len(results['train_losses']) <= test_config['training']['epochs']:
            print(f"   ✅ epochs参数传递正确")
        else:
            print(f"   ❌ epochs参数传递错误")
            
    except Exception as e:
        print(f"❌ 训练过程出错: {e}")
        import traceback
        traceback.print_exc()

def test_optimizer_params():
    """测试优化器参数传输"""
    print("\n=== 优化器参数传输测试 ===")
    
    # 加载配置
    config = load_unified_config('configs/unified_config.yaml')
    
    # 创建简单模型
    input_dim = 1024
    output_dim = 16384
    # 使用默认transformer配置
    model_config = config.get('models', {}).get('transformer', {
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 3,
        'dropout': 0.1
    })
    model = create_simple_model('transformer', input_dim, output_dim, model_config)
    
    # 测试优化器参数
    training_config = config['training']
    learning_rate = float(training_config['learning_rate'])
    weight_decay = float(training_config['weight_decay'])
    
    print(f"学习率: {learning_rate} (类型: {type(learning_rate)})")
    print(f"权重衰减: {weight_decay} (类型: {type(weight_decay)})")
    
    # 创建优化器
    try:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        print(f"✅ 优化器创建成功")
        print(f"   优化器类型: {type(optimizer).__name__}")
        print(f"   参数组数量: {len(optimizer.param_groups)}")
        print(f"   实际学习率: {optimizer.param_groups[0]['lr']}")
        print(f"   实际权重衰减: {optimizer.param_groups[0]['weight_decay']}")
        
    except Exception as e:
        print(f"❌ 优化器创建失败: {e}")

def test_loss_function_params():
    """测试损失函数参数传输"""
    print("\n=== 损失函数参数传输测试 ===")
    
    # 加载配置
    config = load_unified_config('configs/unified_config.yaml')
    
    # 获取损失配置
    loss_config = config.get('loss', {})
    print(f"损失配置: {loss_config}")
    
    # 测试MSE损失
    criterion = torch.nn.MSELoss()
    
    # 创建测试数据
    pred = torch.randn(10, 100)
    target = torch.randn(10, 100)
    
    try:
        loss = criterion(pred, target)
        print(f"✅ MSE损失计算成功: {loss.item():.6f}")
        
        # 测试损失权重
        mse_weight = loss_config.get('mse_weight', 1.0)
        weighted_loss = loss * mse_weight
        print(f"   MSE权重: {mse_weight}")
        print(f"   加权损失: {weighted_loss.item():.6f}")
        
    except Exception as e:
        print(f"❌ 损失函数测试失败: {e}")

def main():
    """主测试函数"""
    print("开始模型初始化和训练循环参数传输测试")
    
    # 运行各项测试
    test_model_initialization_params()
    test_training_loop_params()
    test_optimizer_params()
    test_loss_function_params()
    
    print("\n=== 模型初始化和训练循环参数传输测试完成 ===")

if __name__ == "__main__":
    main()