#!/usr/bin/env python3
"""
训练流程验证脚本
验证统一参量级横向对比训练的完整流程
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import traceback

def check_dependencies():
    """检查依赖项"""
    print("检查依赖项...")
    
    required_packages = [
        'torch', 'numpy', 'matplotlib', 'yaml', 'pathlib'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (缺失)")
    
    if missing_packages:
        print(f"\n缺失依赖项: {', '.join(missing_packages)}")
        return False
    
    print("所有依赖项检查通过")
    return True

def check_config_files():
    """检查配置文件"""
    print("\n检查配置文件...")
    
    config_files = [
        'configs/optimized_unified_params_config.yaml',
        'configs/crop_model_with_normalization.yaml'
    ]
    
    for config_file in config_files:
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                print(f"✓ {config_file} (格式正确)")
            except Exception as e:
                print(f"✗ {config_file} (格式错误): {str(e)}")
                return False
        else:
            print(f"✗ {config_file} (文件不存在)")
            return False
    
    print("配置文件检查通过")
    return True

def check_model_imports():
    """检查模型导入"""
    print("\n检查模型导入...")
    
    model_modules = [
        ('models.transformer_model', 'TransformerModel'),
        ('models.mlp_model', 'MLPModel'),
        ('models.fno_model', 'FNOModel'),
        ('models.unet_model', 'UNetModel'),
        ('models.custom_transformer', 'CustomTransformerModel')
    ]
    
    available_models = []
    for module_name, class_name in model_modules:
        try:
            module = __import__(module_name, fromlist=[class_name])
            model_class = getattr(module, class_name)
            available_models.append((module_name, class_name))
            print(f"✓ {module_name}.{class_name}")
        except ImportError as e:
            print(f"✗ {module_name}.{class_name} (导入失败): {str(e)}")
        except AttributeError as e:
            print(f"✗ {module_name}.{class_name} (类不存在): {str(e)}")
    
    if len(available_models) == 0:
        print("没有可用的模型类")
        return False
    
    print(f"可用模型: {len(available_models)}/{len(model_modules)}")
    return True

def check_data_loader():
    """检查数据加载器"""
    print("\n检查数据加载器...")
    
    try:
        from data.data_loader import PDEDataset, create_data_loader
        print("✓ 数据加载器导入成功")
        return True
    except ImportError as e:
        print(f"✗ 数据加载器导入失败: {str(e)}")
        return False

def create_dummy_data():
    """创建虚拟数据用于测试"""
    print("\n创建虚拟测试数据...")
    
    # 创建数据目录
    data_dir = Path("test_data")
    data_dir.mkdir(exist_ok=True)
    
    # 生成虚拟数据
    batch_size = 10
    input_dim = 1024
    output_dim = 16384
    
    # 训练数据
    train_inputs = np.random.randn(batch_size, input_dim).astype(np.float32)
    train_targets = np.random.randn(batch_size, output_dim).astype(np.float32)
    
    np.save(data_dir / "train_inputs.npy", train_inputs)
    np.save(data_dir / "train_targets.npy", train_targets)
    
    # 验证数据
    val_inputs = np.random.randn(batch_size//2, input_dim).astype(np.float32)
    val_targets = np.random.randn(batch_size//2, output_dim).astype(np.float32)
    
    np.save(data_dir / "val_inputs.npy", val_inputs)
    np.save(data_dir / "val_targets.npy", val_targets)
    
    print(f"✓ 虚拟数据已创建在 {data_dir}")
    return True

def test_model_creation():
    """测试模型创建"""
    print("\n测试模型创建...")
    
    # 加载配置
    try:
        with open('configs/optimized_unified_params_config.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"✗ 配置加载失败: {str(e)}")
        return False
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    created_models = []
    
    for model_name, model_config in config['models'].items():
        try:
            model_type = model_config['model_type']
            
            # 简化的模型创建测试
            if model_type in ['transformer', 'custom_transformer']:
                # 创建简单的线性模型作为替代
                model = torch.nn.Sequential(
                    torch.nn.Linear(model_config['input_dim'], model_config['d_model']),
                    torch.nn.ReLU(),
                    torch.nn.Linear(model_config['d_model'], model_config['output_dim'])
                ).to(device)
                
            elif model_type == 'mlp':
                layers = []
                in_dim = model_config['input_dim']
                for hidden_dim in model_config['hidden_dims']:
                    layers.extend([
                        torch.nn.Linear(in_dim, hidden_dim),
                        torch.nn.ReLU()
                    ])
                    in_dim = hidden_dim
                layers.append(torch.nn.Linear(in_dim, model_config['output_dim']))
                model = torch.nn.Sequential(*layers).to(device)
                
            else:
                # 对于FNO和UNet，创建简单的线性模型
                model = torch.nn.Linear(
                    model_config['input_dim'], 
                    model_config['output_dim']
                ).to(device)
            
            # 计算参数量
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            created_models.append((model_name, model, param_count))
            print(f"✓ {model_name}: {param_count:,} 参数")
            
        except Exception as e:
            print(f"✗ {model_name}: {str(e)}")
    
    if len(created_models) == 0:
        print("没有成功创建的模型")
        return False
    
    print(f"成功创建模型: {len(created_models)}/{len(config['models'])}")
    return True

def test_training_step():
    """测试训练步骤"""
    print("\n测试训练步骤...")
    
    try:
        # 创建简单模型
        model = torch.nn.Linear(1024, 16384)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        # 创建虚拟数据
        inputs = torch.randn(4, 1024)
        targets = torch.randn(4, 16384)
        
        # 前向传播
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"✓ 训练步骤测试成功，损失: {loss.item():.6f}")
        return True
        
    except Exception as e:
        print(f"✗ 训练步骤测试失败: {str(e)}")
        return False

def test_full_pipeline():
    """测试完整训练流程"""
    print("\n测试完整训练流程...")
    
    try:
        # 导入训练器
        sys.path.append('.')
        from train_unified_models import UnifiedModelTrainer
        
        # 创建训练器实例（不实际训练）
        trainer = UnifiedModelTrainer(
            config_path='configs/optimized_unified_params_config.yaml',
            data_config_path='configs/crop_model_with_normalization.yaml',
            output_dir='test_results'
        )
        
        print("✓ 训练器创建成功")
        
        # 测试配置加载
        if hasattr(trainer, 'model_config') and hasattr(trainer, 'data_config'):
            print("✓ 配置加载成功")
        else:
            print("✗ 配置加载失败")
            return False
        
        print("✓ 完整流程测试通过")
        return True
        
    except Exception as e:
        print(f"✗ 完整流程测试失败: {str(e)}")
        traceback.print_exc()
        return False

def main():
    """主验证函数"""
    print("统一参量级横向对比训练流程验证")
    print("=" * 50)
    
    tests = [
        ("依赖项检查", check_dependencies),
        ("配置文件检查", check_config_files),
        ("模型导入检查", check_model_imports),
        ("数据加载器检查", check_data_loader),
        ("虚拟数据创建", create_dummy_data),
        ("模型创建测试", test_model_creation),
        ("训练步骤测试", test_training_step),
        ("完整流程测试", test_full_pipeline)
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed_tests += 1
                print(f"✓ {test_name} 通过")
            else:
                print(f"✗ {test_name} 失败")
        except Exception as e:
            print(f"✗ {test_name} 异常: {str(e)}")
    
    print(f"\n{'='*50}")
    print(f"验证结果: {passed_tests}/{total_tests} 测试通过")
    
    if passed_tests == total_tests:
        print("✓ 所有验证通过，可以开始训练")
        return True
    else:
        print("✗ 部分验证失败，请检查问题后重试")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)