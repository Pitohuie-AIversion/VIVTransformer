#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一模型系统使用示例

这个脚本展示了如何使用统一模型系统的各种功能，包括:
- 模型创建和配置
- 配置管理和验证
- 兼容性适配器使用
- 性能比较
- 自定义训练流程
"""

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
from pathlib import Path
from typing import Dict, List, Any

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'models'))

# 导入统一模型系统组件
try:
    from unified_model_factory import get_global_factory
    from unified_config_manager import get_global_config_manager, ConfigType
    from trainer_compatibility_adapter import get_global_adapter
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保统一模型系统文件在正确的路径中")
    sys.exit(1)

class UnifiedModelSystemDemo:
    """统一模型系统演示类"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 获取全局实例
        self.model_factory = get_global_factory()
        self.config_manager = get_global_config_manager()
        self.compatibility_adapter = get_global_adapter()
        
        print("统一模型系统初始化完成")
    
    def demo_basic_model_creation(self):
        """演示基本模型创建"""
        print("\n=== 基本模型创建演示 ===")
        
        # 获取支持的模型
        supported_models = self.model_factory.get_supported_models()
        print(f"支持的模型类型: {supported_models}")
        
        # 创建Transformer模型
        transformer_config = {
            'input_dim': 64,
            'output_dim': 32,
            'd_model': 128,
            'num_heads': 4,
            'num_layers': 2,
            'attention_type': 'sge',
            'pe_type': 'learnable_1d',
            'max_time_steps': 1,
            'seq_len': 16
        }
        
        try:
            transformer_model = self.model_factory.create_model(
                'enhanced_transformer_1d', transformer_config
            )
            transformer_params = sum(p.numel() for p in transformer_model.parameters())
            print(f"✓ Transformer模型创建成功，参数数量: {transformer_params}")
            
            # 测试前向传播
            transformer_model.eval()
            with torch.no_grad():
                test_input = torch.randn(2, transformer_config['input_dim'])
                output = transformer_model(test_input)
                print(f"  输入形状: {test_input.shape}, 输出形状: {output.shape}")
                
        except Exception as e:
            print(f"✗ Transformer模型创建失败: {e}")
        
        # 创建MLP模型
        mlp_config = {
            'input_dim': 64,
            'output_dim': 32,
            'hidden_dims': [128, 256, 128],
            'activation': 'relu',
            'dropout': 0.1
        }
        
        try:
            mlp_model = self.model_factory.create_model(
                'enhanced_mlp_1d', mlp_config
            )
            mlp_params = sum(p.numel() for p in mlp_model.parameters())
            print(f"✓ MLP模型创建成功，参数数量: {mlp_params}")
            
        except Exception as e:
            print(f"✗ MLP模型创建失败: {e}")
    
    def demo_config_management(self):
        """演示配置管理功能"""
        print("\n=== 配置管理演示 ===")
        
        # 获取配置模板
        model_template = self.config_manager.get_config_template(['model'])
        print("模型配置模板:")
        for key, value in model_template['model'].items():
            print(f"  {key}: {value}")
        
        # 配置验证
        test_config = {
            'input_dim': 128,
            'output_dim': 64,
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 4
        }
        
        try:
            validated_config = self.config_manager.validate_full_config(
                {'model': test_config}
            )
            print(f"✓ 配置验证成功")
            print(f"  验证后的配置: {validated_config['model']}")
        except Exception as e:
            print(f"✗ 配置验证失败: {e}")
        
        # 配置合并
        base_config = {
            'model': {
                'input_dim': 64,
                'num_heads': 4
            }
        }
        
        override_config = {
            'model': {
                'output_dim': 32,
                'num_heads': 8  # 覆盖base_config中的值
            }
        }
        
        merged_config = self.config_manager.merge_configs(base_config, override_config)
        print(f"配置合并结果: {merged_config}")
    
    def demo_compatibility_adapter(self):
        """演示兼容性适配器功能"""
        print("\n=== 兼容性适配器演示 ===")
        
        # Trainer格式配置
        trainer_config = {
            'model': {
                'input_dim': 128,
                'output_dim': 64,
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 4,
                'attention_type': 'sge'
            },
            'training': {
                'epochs': 50,
                'learning_rate': 1e-3,
                'batch_size': 32
            }
        }
        
        # 配置验证
        is_valid, errors = self.compatibility_adapter.validate_trainer_config(trainer_config)
        if is_valid:
            print("✓ Trainer配置验证成功")
            
            # 创建模型
            try:
                model = self.compatibility_adapter.create_model_from_trainer_config(trainer_config)
                params = sum(p.numel() for p in model.parameters())
                print(f"✓ 从Trainer配置创建模型成功，参数数量: {params}")
                
                # 测试模型
                model.eval()
                with torch.no_grad():
                    test_input = torch.randn(2, trainer_config['model']['input_dim'])
                    output = model(test_input)
                    print(f"  模型测试成功，输出形状: {output.shape}")
                    
            except Exception as e:
                print(f"✗ 模型创建失败: {e}")
        else:
            print(f"✗ Trainer配置验证失败: {errors}")
        
        # 获取模型信息
        try:
            model_info = self.compatibility_adapter.get_model_info()
            if 'available_models' in model_info:
                print(f"可用模型数量: {len(model_info['available_models'])}")
            else:
                print(f"模型信息: {list(model_info.keys())}")
        except Exception as e:
            print(f"获取模型信息失败: {e}")
    
    def demo_model_comparison(self):
        """演示模型性能比较"""
        print("\n=== 模型性能比较演示 ===")
        
        models_to_compare = [
            ('enhanced_transformer_1d', {
                'input_dim': 128,
                'output_dim': 64,
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 4
            }),
            ('enhanced_mlp_1d', {
                'input_dim': 128,
                'output_dim': 64,
                'hidden_dims': [256, 512, 256]
            })
        ]
        
        comparison_results = []
        
        for model_type, config in models_to_compare:
            try:
                # 创建模型
                model = self.model_factory.create_model(model_type, config)
                model = model.to(self.device)
                model.eval()
                
                # 性能测试
                batch_size = 32
                test_input = torch.randn(batch_size, config['input_dim']).to(self.device)
                
                # 预热
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(test_input)
                
                # 计时测试
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(100):
                        output = model(test_input)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                
                end_time = time.time()
                avg_time = (end_time - start_time) / 100
                
                result = {
                    'model_type': model_type,
                    'parameters': sum(p.numel() for p in model.parameters()),
                    'inference_time': avg_time,
                    'throughput': batch_size / avg_time
                }
                
                comparison_results.append(result)
                print(f"✓ {model_type}: {result['parameters']} 参数, "
                      f"{result['inference_time']:.4f}s 推理时间, "
                      f"{result['throughput']:.1f} samples/s")
                
            except Exception as e:
                print(f"✗ {model_type} 性能测试失败: {e}")
        
        return comparison_results
    
    def demo_custom_training(self):
        """演示自定义训练流程"""
        print("\n=== 自定义训练流程演示 ===")
        
        # 配置
        config = {
            'model': {
                'input_dim': 64,
                'output_dim': 32,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'attention_type': 'sge'
            },
            'training': {
                'epochs': 20,
                'learning_rate': 1e-3,
                'batch_size': 16
            }
        }
        
        try:
            # 创建模型
            model = self.compatibility_adapter.create_model_from_trainer_config(config)
            model = model.to(self.device)
            
            # 创建优化器和损失函数
            optimizer = optim.Adam(model.parameters(), lr=config['training']['learning_rate'])
            criterion = nn.MSELoss()
            
            print(f"开始训练，模型参数: {sum(p.numel() for p in model.parameters())}")
            
            # 训练循环
            model.train()
            for epoch in range(config['training']['epochs']):
                # 模拟数据
                batch_size = config['training']['batch_size']
                inputs = torch.randn(batch_size, config['model']['input_dim']).to(self.device)
                targets = torch.randn(batch_size, config['model']['output_dim']).to(self.device)
                
                # 前向传播
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 反向传播
                loss.backward()
                optimizer.step()
                
                if epoch % 5 == 0:
                    print(f"  Epoch {epoch:2d}, Loss: {loss.item():.6f}")
            
            print("✓ 训练完成")
            
            # 测试模型
            model.eval()
            with torch.no_grad():
                test_input = torch.randn(4, config['model']['input_dim']).to(self.device)
                test_output = model(test_input)
                print(f"  测试输出形状: {test_output.shape}")
            
            return model
            
        except Exception as e:
            print(f"✗ 训练失败: {e}")
            return None
    
    def demo_dynamic_scaling(self):
        """演示动态模型缩放"""
        print("\n=== 动态模型缩放演示 ===")
        
        data_sizes = [32, 64, 128, 256]
        
        for data_size in data_sizes:
            # 根据数据大小动态调整配置
            if data_size <= 64:
                config = {
                    'input_dim': data_size,
                    'output_dim': data_size // 2,
                    'd_model': 64,
                    'num_heads': 2,
                    'num_layers': 2
                }
            elif data_size <= 128:
                config = {
                    'input_dim': data_size,
                    'output_dim': data_size // 2,
                    'd_model': 128,
                    'num_heads': 4,
                    'num_layers': 3
                }
            else:
                config = {
                    'input_dim': data_size,
                    'output_dim': data_size // 2,
                    'd_model': 256,
                    'num_heads': 8,
                    'num_layers': 4
                }
            
            try:
                # 验证配置
                validated_config = self.config_manager.validate_full_config(
                    {'model': config}
                )
                
                # 创建模型
                model = self.model_factory.create_model(
                    'enhanced_transformer_1d', validated_config['model']
                )
                
                params = sum(p.numel() for p in model.parameters())
                print(f"数据大小 {data_size:3d}: 模型参数 {params:6d}, "
                      f"d_model={config['d_model']}, heads={config['num_heads']}")
                
            except Exception as e:
                print(f"✗ 数据大小 {data_size} 配置失败: {e}")
    
    def run_all_demos(self):
        """运行所有演示"""
        print("统一模型系统功能演示")
        print("=" * 50)
        
        try:
            self.demo_basic_model_creation()
            self.demo_config_management()
            self.demo_compatibility_adapter()
            self.demo_model_comparison()
            self.demo_custom_training()
            self.demo_dynamic_scaling()
            
            print("\n=== 演示完成 ===")
            print("所有功能演示成功完成！")
            
        except Exception as e:
            print(f"\n演示过程中出现错误: {e}")
            import traceback
            traceback.print_exc()

def main():
    """主函数"""
    print("统一模型系统使用示例")
    print("=" * 30)
    
    try:
        demo = UnifiedModelSystemDemo()
        demo.run_all_demos()
    except Exception as e:
        print(f"初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)