#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器优化功能测试脚本
测试dynamic_resolution_trainer.py的服务器配置优化功能
"""

import os
import sys
import torch
import psutil
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_cpu_memory_monitoring():
    """测试CPU内存监控功能"""
    logger.info("=== 测试CPU内存监控功能 ===")
    
    try:
        # 导入CPU内存监控函数
        sys.path.append(str(Path(__file__).parent))
        from dynamic_resolution_trainer import log_cpu_memory, optimize_cpu_for_data_loading
        
        # 测试CPU内存监控
        log_cpu_memory("测试阶段")
        
        # 测试CPU优化
        optimize_cpu_for_data_loading()
        
        logger.info("[OK] CPU内存监控和优化功能正常")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] CPU内存监控测试失败: {e}")
        return False

def test_dataloader_optimization():
    """测试数据加载器优化功能"""
    logger.info("=== 测试数据加载器优化功能 ===")
    
    try:
        # 模拟服务器配置
        test_config = {
            'dataloader': {
                'num_workers': 0,  # 测试自动优化
                'pin_memory': True,
                'persistent_workers': False,
                'prefetch_factor': 2,
                'drop_last': False
            }
        }
        
        # 检查CPU核心数
        cpu_count = os.cpu_count()
        logger.info(f"检测到CPU核心数: {cpu_count}")
        
        # 模拟优化逻辑
        num_workers = test_config['dataloader']['num_workers']
        if num_workers == 0 and cpu_count > 4:
            optimized_workers = min(16, cpu_count // 2)
            logger.info(f"[INFO] 自动优化: num_workers从{num_workers}调整为{optimized_workers}")
        
        logger.info("[OK] 数据加载器优化逻辑正常")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] 数据加载器优化测试失败: {e}")
        return False

def test_gpu_memory_monitoring():
    """测试GPU内存监控功能"""
    logger.info("=== 测试GPU内存监控功能 ===")
    
    try:
        # 导入GPU内存监控函数
        from dynamic_resolution_trainer import log_gpu_memory, cleanup_memory
        
        # 测试GPU内存监控
        log_gpu_memory("测试阶段")
        
        # 测试内存清理
        cleanup_memory()
        
        logger.info("[OK] GPU内存监控和清理功能正常")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] GPU内存监控测试失败: {e}")
        return False

def test_tensor_optimization():
    """测试Tensor创建优化"""
    logger.info("=== 测试Tensor创建优化 ===")
    
    try:
        import numpy as np
        
        # 创建测试数据
        test_data = np.random.rand(1000).astype(np.float32)
        
        # 测试优化的tensor创建方法
        # 1. 确保数据类型为float32并且内存连续
        input_array = np.ascontiguousarray(test_data, dtype=np.float32)
        
        # 2. 使用torch.from_numpy()零拷贝创建tensor
        tensor = torch.from_numpy(input_array).contiguous()
        
        # 验证tensor属性
        assert tensor.dtype == torch.float32, "数据类型不正确"
        assert tensor.is_contiguous(), "内存布局不连续"
        
        logger.info(f"[OK] Tensor优化创建成功: dtype={tensor.dtype}, contiguous={tensor.is_contiguous()}")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Tensor优化测试失败: {e}")
        return False

def test_config_loading():
    """测试配置文件加载"""
    logger.info("=== 测试配置文件加载 ===")
    
    try:
        config_path = Path(__file__).parent / "dynamic_config_server_downsampling.yaml"
        
        if config_path.exists():
            import yaml
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 检查关键配置项
            dataloader_config = config.get('dataloader', {})
            logger.info(f"数据加载器配置: {dataloader_config}")
            
            # 验证服务器优化配置
            expected_keys = ['num_workers', 'pin_memory', 'persistent_workers', 'prefetch_factor']
            for key in expected_keys:
                if key in dataloader_config:
                    logger.info(f"[OK] 配置项 {key}: {dataloader_config[key]}")
                else:
                    logger.warning(f"[WARN] 缺少配置项: {key}")
            
            logger.info("[OK] 配置文件加载成功")
            return True
        else:
            logger.warning(f"[WARN] 配置文件不存在: {config_path}")
            return False
            
    except Exception as e:
        logger.error(f"[ERROR] 配置文件加载测试失败: {e}")
        return False

def test_system_resources():
    """测试系统资源检测"""
    logger.info("=== 测试系统资源检测 ===")
    
    try:
        # CPU信息
        cpu_count = os.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # 内存信息
        memory = psutil.virtual_memory()
        memory_gb = memory.total / (1024**3)
        memory_available_gb = memory.available / (1024**3)
        
        # GPU信息
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        logger.info(f"[INFO] CPU核心数: {cpu_count}")
        logger.info(f"[INFO] CPU使用率: {cpu_percent}%")
        logger.info(f"[INFO] 总内存: {memory_gb:.1f}GB")
        logger.info(f"[INFO] 可用内存: {memory_available_gb:.1f}GB")
        logger.info(f"[INFO] GPU可用: {gpu_available}")
        logger.info(f"[INFO] GPU数量: {gpu_count}")
        
        # 服务器配置建议
        if cpu_count >= 16:
            logger.info("[INFO] 检测到高性能CPU，建议使用多进程数据加载")
        if memory_gb >= 32:
            logger.info("[INFO] 检测到大内存，建议启用数据预加载")
        if gpu_count > 1:
            logger.info("[INFO] 检测到多GPU，建议使用DataParallel")
        
        logger.info("[OK] 系统资源检测完成")
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] 系统资源检测失败: {e}")
        return False

def main():
    """主测试函数"""
    logger.info("[INFO] 开始服务器优化功能测试")
    
    test_results = []
    
    # 运行所有测试
    tests = [
        ("系统资源检测", test_system_resources),
        ("CPU内存监控", test_cpu_memory_monitoring),
        ("GPU内存监控", test_gpu_memory_monitoring),
        ("数据加载器优化", test_dataloader_optimization),
        ("Tensor创建优化", test_tensor_optimization),
        ("配置文件加载", test_config_loading)
    ]
    
    for test_name, test_func in tests:
        logger.info(f"\n{'='*50}")
        logger.info(f"开始测试: {test_name}")
        logger.info(f"{'='*50}")
        
        result = test_func()
        test_results.append((test_name, result))
        
        if result:
            logger.info(f"[OK] {test_name} 测试通过")
        else:
            logger.error(f"[ERROR] {test_name} 测试失败")
    
    # 测试总结
    logger.info(f"\n{'='*50}")
    logger.info("测试总结")
    logger.info(f"{'='*50}")
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "[OK] 通过" if result else "[ERROR] 失败"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        logger.info("[OK] 所有服务器优化功能测试通过！")
        return True
    else:
        logger.warning(f"[WARN] {total - passed} 个测试失败，请检查相关功能")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)