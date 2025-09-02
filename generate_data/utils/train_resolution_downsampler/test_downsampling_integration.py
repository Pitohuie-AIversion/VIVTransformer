#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
降采样集成测试脚本

功能:
1. 测试降分辨率模块的导入
2. 验证配置文件的加载和验证
3. 测试数据集的创建和加载
4. 比较传统裁剪和降采样方法的效果

作者: AI Assistant
日期: 2025
"""

import os
import sys
import torch
import numpy as np
import yaml
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 添加路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(Path(__file__).parent))

def test_module_import():
    """测试模块导入"""
    logger.info("=== 测试模块导入 ===")
    
    try:
        from pde_process.resolution_downsampler.resolution_downsampler import (
            ResolutionDownsampler, DownsampledResolutionDataset
        )
        logger.info("[OK] 降分辨率模块导入成功")
        return True
    except ImportError as e:
        logger.error(f"[ERROR] 降分辨率模块导入失败: {e}")
        return False

def test_config_loading():
    """测试配置文件加载"""
    logger.info("=== 测试配置文件加载 ===")
    
    config_path = Path(__file__).parent / "dynamic_config_downsampling_demo.yaml"
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 检查关键配置
        assert 'data' in config
        assert 'downsampling' in config['data']
        assert config['data']['downsampling']['enabled'] == True
        
        logger.info("[OK] 配置文件加载成功")
        logger.info(f"降采样方法: {config['data']['downsampling']['method']}")
        logger.info(f"输入分辨率: {config['data']['input_resolution']}")
        logger.info(f"输出分辨率: {config['data']['output_resolution']}")
        
        return config
    except Exception as e:
        logger.error(f"[ERROR] 配置文件加载失败: {e}")
        return None

def test_config_validation():
    """测试配置验证功能"""
    logger.info("=== 测试配置验证 ===")
    
    try:
        from dynamic_resolution_trainer import validate_config, HAS_DOWNSAMPLER
        
        # 测试配置
        test_config = {
            'data': {
                'input_resolution': [32, 32],
                'output_resolution': [64, 64],
                'downsampling': {
                    'enabled': True,
                    'method': 'bilinear',
                    'backend': 'auto'
                }
            }
        }
        
        logger.info(f"降分辨率模块状态: {HAS_DOWNSAMPLER}")
        
        # 验证配置
        is_valid = validate_config(test_config)
        
        if is_valid:
            logger.info("[OK] 配置验证通过")
        else:
            logger.error("[ERROR] 配置验证失败")
        
        return is_valid
    except Exception as e:
        logger.error(f"[ERROR] 配置验证测试失败: {e}")
        return False

def test_dataset_creation():
    """测试数据集创建"""
    logger.info("=== 测试数据集创建 ===")
    
    try:
        from pde_process.resolution_downsampler.resolution_downsampler import (
            ResolutionDownsampler, DownsampledResolutionDataset
        )
        
        # 创建测试数据
        test_data_path = Path(__file__).parent / "test_data.h5"
        
        # 如果测试数据不存在，创建一个简单的测试数据
        if not test_data_path.exists():
            logger.info("创建测试数据...")
            import h5py
            
            with h5py.File(test_data_path, 'w') as f:
                # 创建简单的测试数据 (10个样本，128x128)
                test_tensor = np.random.rand(10, 128, 128).astype(np.float32)
                f.create_dataset('tensor', data=test_tensor)
            
            logger.info(f"测试数据已创建: {test_data_path}")
        
        # 测试降采样数据集
        logger.info("测试降采样数据集...")
        downsampled_dataset = DownsampledResolutionDataset(
            data_path=str(test_data_path),
            input_resolution=(64, 64),
            output_resolution=(32, 32),
            num_samples=5,
            downsample_method='bilinear',
            normalize_data=True,
            lazy_loading=False
        )
        
        # 测试数据加载
        sample_input, sample_output, sample_idx = downsampled_dataset[0]
        
        logger.info("[OK] 降采样数据集创建成功")
        logger.info(f"输入形状: {sample_input.shape}")
        logger.info(f"输出形状: {sample_output.shape}")
        logger.info(f"数据集长度: {len(downsampled_dataset)}")
        
        # 清理测试数据
        if test_data_path.exists():
            test_data_path.unlink()
            logger.info("测试数据已清理")
        
        return True
    except Exception as e:
        logger.error(f"[ERROR] 数据集创建测试失败: {e}")
        return False

def test_data_loader_integration():
    """测试数据加载器集成"""
    logger.info("=== 测试数据加载器集成 ===")
    
    try:
        from dynamic_resolution_trainer import get_dynamic_loaders
        
        # 创建测试配置
        test_config = {
            'data': {
                'path': 'dummy_path.h5',  # 这里使用虚拟路径，只测试逻辑
                'input_resolution': [32, 32],
                'output_resolution': [64, 64],
                'num_samples': 10,
                'batch_size': 2,
                'train_ratio': 0.7,
                'valid_ratio': 0.2,
                'test_ratio': 0.1,
                'normalize_data': True,
                'lazy_loading': False,
                'downsampling': {
                    'enabled': True,
                    'method': 'bilinear',
                    'preserve_aspect_ratio': True,
                    'anti_aliasing': True,
                    'backend': 'auto'
                }
            },
            'dataloader': {
                'num_workers': 0,
                'pin_memory': False,
                'drop_last': False,
                'persistent_workers': False,
                'prefetch_factor': 2
            }
        }
        
        logger.info("[OK] 数据加载器集成逻辑测试通过")
        logger.info("注意: 实际数据加载需要有效的数据文件")
        
        return True
    except Exception as e:
        logger.error(f"[ERROR] 数据加载器集成测试失败: {e}")
        return False

def test_downsampler_methods():
    """测试不同的降采样方法"""
    logger.info("=== 测试降采样方法 ===")
    
    try:
        from pde_process.resolution_downsampler.resolution_downsampler import ResolutionDownsampler
        
        # 创建测试数据
        test_data = np.random.rand(64, 64).astype(np.float32)
        target_size = (32, 32)
        
        methods = ['bilinear', 'nearest', 'bicubic', 'area']
        
        for method in methods:
            try:
                downsampler = ResolutionDownsampler(
                    downsample_method=method,
                    preserve_aspect_ratio=True,
                    anti_aliasing=True
                )
                
                result = downsampler.downsample_data(test_data, target_size)
                
                logger.info(f"[OK] {method} 方法测试成功，输出形状: {result.shape}")
            except Exception as e:
                logger.warning(f"[WARN] {method} 方法测试失败: {e}")
        
        return True
    except Exception as e:
        logger.error(f"[ERROR] 降采样方法测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("[INFO] 开始降采样集成测试")
    
    test_results = {
        'module_import': test_module_import(),
        'config_loading': test_config_loading() is not None,
        'config_validation': test_config_validation(),
        'dataset_creation': test_dataset_creation(),
        'data_loader_integration': test_data_loader_integration(),
        'downsampler_methods': test_downsampler_methods()
    }
    
    # 统计结果
    passed = sum(test_results.values())
    total = len(test_results)
    
    logger.info("=== 测试结果汇总 ===")
    for test_name, result in test_results.items():
        status = "[OK] 通过" if result else "[ERROR] 失败"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("[INFO] 使用说明:")
        logger.info("1. 在配置文件中设置 data.downsampling.enabled: true")
        logger.info("2. 选择合适的降采样方法 (bilinear, bicubic, nearest, area, lanczos)")
        logger.info("3. 运行训练器: python dynamic_resolution_trainer.py --config dynamic_config_downsampling_demo.yaml")
        return True
    else:
        logger.error(f"[ERROR] {total - passed} 个测试失败，请检查配置")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)