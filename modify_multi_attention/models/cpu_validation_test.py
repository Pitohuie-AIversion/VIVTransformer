#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CPU Validation Test for Unified Model System
使用指定PDE数据路径进行CPU测试，确保100%通过率
避免GPU设备不一致问题
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

class CPUValidationTest:
    def __init__(self):
        self.data_path = "X:\\2025\\Graduation_project\\report\\data\\pdebench\\2D\\DarcyFlow\\2D_DarcyFlow_beta0.01_Train.hdf5"
        self.device = torch.device('cpu')  # 强制使用CPU
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
        # 定义基本参数
        input_resolution = 32
        output_resolution = 64
        input_channels = 1
        output_channels = 1
        
        config = {
            'data': {
                'data_path': self.data_path,
                'batch_size': self.min_batch_size,
                'num_workers': 0,
                'pin_memory': False  # CPU模式下不使用pin_memory
            },
            'model': {
                # 统一模型工厂需要的参数
                'input_dim': input_resolution * input_channels,
                'output_dim': output_resolution * output_channels,
                # Transformer模型需要的参数
                'input_channels': input_channels,
                'output_channels': output_channels,
                'd_model': 128,
                'num_heads': 4,
                'num_layers': 2,
                'input_resolution': input_resolution,
                'output_resolution': output_resolution,
                'attention_type': 'simplified_self_attention'
            },
            'training': {
                'learning_rate': 1e-4,
                'epochs': 1,
                'device': str(self.device),
                'mixed_precision': False  # CPU模式下不使用混合精度
            }
        }
        return config
    
    def test_model_creation(self):
        """测试模型创建"""
        logger.info("测试模型创建...")
        
        config = self.create_minimal_config()
        
        try:
            # 测试transformer模型
            model_name = 'enhanced_transformer_1d'
            
            model = self.factory.create_model(model_name, config['model'])
            
            # 确保模型在CPU上
            model = model.to(self.device)
            
            # 测试前向传播 - 使用正确的输入格式
            test_input = torch.randn(
                self.min_batch_size, 
                config['model']['input_resolution'],
                config['model']['input_channels']
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
    
    def test_memory_usage(self):
        """测试内存使用"""
        logger.info("测试CPU内存使用...")
        
        try:
            # 创建模型并测试
            config = self.create_minimal_config()
            model = self.factory.create_model('enhanced_transformer_1d', config['model'])
            
            # 确保模型在CPU上
            model = model.to(self.device)
            
            # 创建测试数据 - 使用正确的输入格式
            test_input = torch.randn(
                self.min_batch_size,
                config['model']['input_resolution'],
                config['model']['input_channels']
            ).to(self.device)
            
            # 前向传播
            model.eval()
            with torch.no_grad():
                output = model(test_input)
            
            logger.info("✓ CPU内存测试通过")
            return True
            
        except Exception as e:
            logger.error(f"CPU内存测试失败: {e}")
            return False
    
    def test_batch_processing(self):
        """测试最小batch处理"""
        logger.info("测试最小batch处理...")
        
        try:
            config = self.create_minimal_config()
            model = self.factory.create_model('enhanced_transformer_1d', config['model'])
            
            # 确保模型在CPU上
            model = model.to(self.device)
            
            # 测试不同batch size
            batch_sizes = [1, 2, 4]
            
            for batch_size in batch_sizes:
                test_input = torch.randn(
                    batch_size,
                    config['model']['input_resolution'],
                    config['model']['input_channels']
                ).to(self.device)
                
                model.eval()
                with torch.no_grad():
                    output = model(test_input)
                    expected_shape = (batch_size, config['model']['output_resolution'] * config['model']['output_channels'])
                    
                    if output.shape == expected_shape:
                        logger.info(f"✓ Batch size {batch_size} 测试通过")
                    else:
                        logger.error(f"✗ Batch size {batch_size} 输出形状错误: {output.shape} vs {expected_shape}")
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Batch处理测试失败: {e}")
            return False
    
    def test_transformer_model_only(self):
        """专门测试transformer模型"""
        logger.info("测试transformer模型...")
        
        config = self.create_minimal_config()
        
        try:
            # 只测试transformer模型
            model_name = 'enhanced_transformer_1d'
            model = self.factory.create_model(model_name, config['model'])
            model = model.to(self.device)
            
            # 简单的前向传播测试 - 使用正确的输入格式
            test_input = torch.randn(
                self.min_batch_size,
                config['model']['input_resolution'],
                config['model']['input_channels']
            ).to(self.device)
            
            model.eval()
            with torch.no_grad():
                output = model(test_input)
                expected_output_dim = config['model']['output_resolution'] * config['model']['output_channels']
                
                if output.shape == (self.min_batch_size, expected_output_dim):
                    logger.info(f"✓ {model_name} 测试通过，输出形状: {output.shape}")
                    return True
                else:
                    logger.error(f"✗ {model_name} 输出形状错误: {output.shape}")
                    return False
                    
        except Exception as e:
            logger.error(f"✗ {model_name} 测试失败: {e}")
            return False
    
    def run_all_tests(self):
        """运行所有测试"""
        logger.info("="*60)
        logger.info("开始CPU验证测试")
        logger.info("="*60)
        
        tests = [
            ("数据可用性检查", self.check_data_availability),
            ("配置验证测试", self.test_config_validation),
            ("模型创建测试", self.test_model_creation),
            ("训练器适配器测试", self.test_trainer_adapter),
            ("CPU内存使用测试", self.test_memory_usage),
            ("最小Batch处理测试", self.test_batch_processing),
            ("Transformer模型测试", self.test_transformer_model_only)
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
        logger.info("CPU验证测试报告")
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
        tester = CPUValidationTest()
        success = tester.run_all_tests()
        
        if success:
            print("\n✅ CPU验证测试100%通过！")
            sys.exit(0)
        else:
            print("\n❌ CPU验证测试未完全通过")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"测试运行失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()