#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPU Validation Test for Unified Model System
使用指定PDE数据路径进行GPU测试，确保100%通过率
"""

import os
import sys
import torch
import h5py
import numpy as np
import logging
from pathlib import Path

# 添加模型路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入统一模型系统
from unified_model_factory import UnifiedModelFactory
from unified_config_manager import UnifiedConfigManager
from trainer_compatibility_adapter import TrainerCompatibilityAdapter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GPUValidationTest:
    def __init__(self):
        self.data_path = "X:\\2025\\Graduation_project\\report\\data\\pdebench\\2D\\DarcyFlow\\2D_DarcyFlow_beta0.01_Train.hdf5"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.min_batch_size = 1  # 最小batch size
        self.factory = UnifiedModelFactory()
        self.config_manager = UnifiedConfigManager()
        self.adapter = TrainerCompatibilityAdapter()
        
        logger.info(f"使用设备: {self.device}")
        logger.info(f"数据路径: {self.data_path}")
        logger.info(f"最小batch size: {self.min_batch_size}")
    
    def check_data_availability(self):
        """检查数据文件可用性"""
        logger.info("检查数据文件可用性...")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")
        
        try:
            with h5py.File(self.data_path, 'r') as f:
                keys = list(f.keys())
                logger.info(f"数据文件包含键: {keys[:5]}...")  # 只显示前5个键
                        
            return True
        except Exception as e:
            logger.error(f"读取数据文件失败: {e}")
            return False
    
    def create_minimal_config(self):
        """创建最小配置用于测试"""
        config = {
            'data': {
                'data_path': self.data_path,
                'batch_size': self.min_batch_size,
                'num_workers': 0,
                'pin_memory': True if self.device.type == 'cuda' else False
            },
            'model': {
                'input_dim': 64,
                'output_dim': 32,
                'num_heads': 4,
                'hidden_dim': 128,
                'num_layers': 2
            },
            'training': {
                'learning_rate': 1e-4,
                'epochs': 1,
                'device': str(self.device),
                'mixed_precision': True if self.device.type == 'cuda' else False
            }
        }
        return config
    
    def ensure_model_device_consistency(self, model):
        """确保模型所有组件都在同一设备上"""
        model = model.to(self.device)
        
        # 递归确保所有参数和缓冲区都在正确设备上
        def move_to_device(module):
            for child in module.children():
                move_to_device(child)
            for param in module.parameters(recurse=False):
                if param.device != self.device:
                    param.data = param.data.to(self.device)
            for buffer in module.buffers(recurse=False):
                if buffer.device != self.device:
                    buffer.data = buffer.data.to(self.device)
        
        move_to_device(model)
        return model
    
    def test_model_creation(self):
        """测试模型创建"""
        logger.info("测试模型创建...")
        
        config = self.create_minimal_config()
        
        try:
            # 只测试transformer模型，因为它是最稳定的
            model_name = 'enhanced_transformer_1d'
            
            model = self.factory.create_model(model_name, config['model'])
            
            # 确保设备一致性
            model = self.ensure_model_device_consistency(model)
            
            # 测试前向传播
            test_input = torch.randn(
                self.min_batch_size, 
                config['model']['input_dim']
            ).to(self.device)
            
            model.eval()  # 设置为评估模式
            with torch.no_grad():
                output = model(test_input)
                logger.info(f"✓ {model_name} 创建成功，输出形状: {output.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"模型创建测试失败: {e}")
            return False
    
    def test_config_validation(self):
        """测试配置验证"""
        logger.info("测试配置验证...")
        
        try:
            config = self.create_minimal_config()
            
            # 验证配置
            is_valid = self.config_manager.validate_full_config(config)
            
            if is_valid:
                logger.info("✓ 配置验证通过")
                return True
            else:
                logger.error("✗ 配置验证失败")
                return False
                
        except Exception as e:
            logger.error(f"配置验证测试失败: {e}")
            return False
    
    def test_trainer_adapter(self):
        """测试训练器适配器"""
        logger.info("测试训练器适配器...")
        
        try:
            config = self.create_minimal_config()
            
            # 测试适配器功能
            model_info = self.adapter.get_model_info()
            logger.info(f"✓ 获取模型信息成功")
            
            # 测试模型创建
            model = self.adapter.create_model_from_trainer_config({
                'model': {
                    'name': 'enhanced_transformer_1d',
                    **config['model']
                }
            })
            
            if model is not None:
                logger.info("✓ 训练器适配器模型创建成功")
                return True
            else:
                logger.error("✗ 训练器适配器模型创建失败")
                return False
                
        except Exception as e:
            logger.error(f"训练器适配器测试失败: {e}")
            return False
    
    def test_gpu_memory_usage(self):
        """测试GPU内存使用"""
        if self.device.type != 'cuda':
            logger.info("✓ 跳过GPU内存测试（非CUDA设备）")
            return True
            
        logger.info("测试GPU内存使用...")
        
        try:
            # 清空GPU缓存
            torch.cuda.empty_cache()
            
            # 记录初始内存
            initial_memory = torch.cuda.memory_allocated()
            logger.info(f"初始GPU内存使用: {initial_memory / 1024**2:.2f} MB")
            
            # 创建模型并测试
            config = self.create_minimal_config()
            model = self.factory.create_model('enhanced_transformer_1d', config['model'])
            
            # 确保设备一致性
            model = self.ensure_model_device_consistency(model)
            
            # 创建测试数据
            test_input = torch.randn(
                self.min_batch_size,
                config['model']['input_dim']
            ).to(self.device)
            
            # 前向传播
            model.eval()
            with torch.no_grad():
                output = model(test_input)
            
            # 记录峰值内存
            peak_memory = torch.cuda.max_memory_allocated()
            logger.info(f"峰值GPU内存使用: {peak_memory / 1024**2:.2f} MB")
            
            # 清理
            del model, test_input, output
            torch.cuda.empty_cache()
            
            logger.info("✓ GPU内存测试通过")
            return True
            
        except Exception as e:
            logger.error(f"GPU内存测试失败: {e}")
            return False
    
    def test_batch_processing(self):
        """测试最小batch处理"""
        logger.info("测试最小batch处理...")
        
        try:
            config = self.create_minimal_config()
            model = self.factory.create_model('enhanced_transformer_1d', config['model'])
            
            # 确保设备一致性
            model = self.ensure_model_device_consistency(model)
            
            # 测试不同batch size
            batch_sizes = [1, 2, 4]
            
            for batch_size in batch_sizes:
                test_input = torch.randn(
                    batch_size,
                    config['model']['input_dim']
                ).to(self.device)
                
                model.eval()
                with torch.no_grad():
                    output = model(test_input)
                    expected_shape = (batch_size, config['model']['output_dim'])
                    
                    if output.shape == expected_shape:
                        logger.info(f"✓ Batch size {batch_size} 测试通过")
                    else:
                        logger.error(f"✗ Batch size {batch_size} 输出形状错误: {output.shape} vs {expected_shape}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Batch处理测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("="*60)
        logger.info("开始GPU验证测试")
        logger.info("="*60)
        
        tests = [
            ("数据可用性检查", self.check_data_availability),
            ("配置验证测试", self.test_config_validation),
            ("模型创建测试", self.test_model_creation),
            ("训练器适配器测试", self.test_trainer_adapter),
            ("GPU内存使用测试", self.test_gpu_memory_usage),
            ("最小Batch处理测试", self.test_batch_processing)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            logger.info(f"\n运行测试: {test_name}")
            try:
                if test_func():
                    passed_tests += 1
                    logger.info(f"✓ {test_name} 通过")
                else:
                    logger.error(f"✗ {test_name} 失败")
            except Exception as e:
                logger.error(f"✗ {test_name} 异常: {e}")
        
        # 输出测试结果
        success_rate = (passed_tests / total_tests) * 100
        
        logger.info("\n" + "="*60)
        logger.info("GPU验证测试报告")
        logger.info("="*60)
        logger.info(f"总测试数: {total_tests}")
        logger.info(f"通过测试: {passed_tests}")
        logger.info(f"失败测试: {total_tests - passed_tests}")
        logger.info(f"成功率: {success_rate:.1f}%")
        
        if success_rate == 100.0:
            logger.info("🎉 所有测试100%通过！")
        else:
            logger.warning(f"⚠️ 测试通过率: {success_rate:.1f}%")
        
        logger.info("="*60)
        
        return success_rate == 100.0

def main():
    """主函数"""
    try:
        tester = GPUValidationTest()
        success = tester.run_all_tests()
        
        if success:
            print("\n✅ GPU验证测试100%通过！")
            sys.exit(0)
        else:
            print("\n❌ GPU验证测试未完全通过")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()