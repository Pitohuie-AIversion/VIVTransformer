#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速服务器测试脚本
用于快速验证服务器环境下的数据加载问题

使用方法:
python quick_server_test.py
"""

import os
import sys
import time
import logging
import yaml
import h5py
import numpy as np
import torch
from pathlib import Path

# 设置简单日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def test_basic_environment():
    """测试基本环境"""
    logger.info("🔍 基本环境检查")
    logger.info(f"Python版本: {sys.version}")
    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU数量: {torch.cuda.device_count()}")
    logger.info(f"CPU核心数: {os.cpu_count()}")
    
def test_data_file_quick(data_path: str):
    """快速测试数据文件"""
    logger.info(f"📁 快速数据文件测试: {data_path}")
    
    if not Path(data_path).exists():
        logger.error(f"❌ 文件不存在: {data_path}")
        return False
    
    file_size = Path(data_path).stat().st_size / 1024**3
    logger.info(f"📊 文件大小: {file_size:.2f} GB")
    
    try:
        start_time = time.time()
        with h5py.File(data_path, 'r') as f:
            open_time = time.time() - start_time
            logger.info(f"⏱️  文件打开时间: {open_time:.3f} 秒")
            
            if 'tensor' in f:
                shape = f['tensor'].shape
                dtype = f['tensor'].dtype
                logger.info(f"📊 数据形状: {shape}")
                logger.info(f"📊 数据类型: {dtype}")
                
                # 快速读取第一个样本
                start_time = time.time()
                first_sample = np.array(f['tensor'][0], dtype=np.float32)
                read_time = time.time() - start_time
                logger.info(f"⏱️  首个样本读取时间: {read_time:.3f} 秒")
                
                return True
            else:
                logger.error("❌ 未找到'tensor'数据集")
                return False
                
    except Exception as e:
        logger.error(f"❌ 文件访问错误: {e}")
        return False

def test_lazy_vs_eager_loading(data_path: str, num_samples: int = 5):
    """对比懒加载和预加载性能"""
    logger.info(f"⚡ 懒加载 vs 预加载测试 ({num_samples} 样本)")
    
    try:
        with h5py.File(data_path, 'r') as f:
            dataset = f['tensor']
            total_samples = min(num_samples, dataset.shape[0])
            
            # 测试预加载
            logger.info("📥 测试预加载模式...")
            start_time = time.time()
            preloaded_data = []
            for i in range(total_samples):
                sample = np.array(dataset[i], dtype=np.float32)
                preloaded_data.append(sample)
            preload_time = time.time() - start_time
            logger.info(f"📥 预加载 {total_samples} 样本耗时: {preload_time:.3f} 秒")
            
            # 测试懒加载
            logger.info("🔄 测试懒加载模式...")
            start_time = time.time()
            for i in range(total_samples):
                # 模拟懒加载：每次访问时读取
                sample = np.array(dataset[i], dtype=np.float32)
                # 简单处理
                processed = sample.flatten()[:1000]  # 只处理前1000个元素
            lazy_time = time.time() - start_time
            logger.info(f"🔄 懒加载 {total_samples} 样本耗时: {lazy_time:.3f} 秒")
            
            # 比较
            if lazy_time < preload_time:
                logger.info("✅ 懒加载更快，建议使用懒加载模式")
            else:
                logger.info("✅ 预加载更快，但懒加载节省内存")
                
    except Exception as e:
        logger.error(f"❌ 加载测试失败: {e}")

def test_normalization_impact(data_path: str, sample_size: int = 10):
    """测试归一化计算影响"""
    logger.info(f"🔢 归一化计算影响测试 ({sample_size} 样本)")
    
    try:
        with h5py.File(data_path, 'r') as f:
            dataset = f['tensor']
            total_samples = min(sample_size, dataset.shape[0])
            
            # 不计算归一化
            logger.info("🚫 无归一化处理...")
            start_time = time.time()
            for i in range(total_samples):
                sample = np.array(dataset[i], dtype=np.float32)
                # 直接使用原始数据
                processed = sample.flatten()
            no_norm_time = time.time() - start_time
            logger.info(f"🚫 无归一化耗时: {no_norm_time:.3f} 秒")
            
            # 计算归一化参数
            logger.info("🔢 计算归一化参数...")
            start_time = time.time()
            all_data = []
            for i in range(total_samples):
                sample = np.array(dataset[i], dtype=np.float32)
                all_data.append(sample.flatten())
            
            combined_data = np.concatenate(all_data)
            global_min = np.min(combined_data)
            global_max = np.max(combined_data)
            
            # 应用归一化
            for data in all_data:
                normalized = (data - global_min) / (global_max - global_min)
            
            norm_time = time.time() - start_time
            logger.info(f"🔢 归一化处理耗时: {norm_time:.3f} 秒")
            logger.info(f"🔢 归一化范围: [{global_min:.6f}, {global_max:.6f}]")
            
            # 比较
            overhead = (norm_time - no_norm_time) / no_norm_time * 100
            logger.info(f"📊 归一化开销: {overhead:.1f}%")
            
            if overhead > 50:
                logger.warning("⚠️  归一化开销较大，建议禁用或使用缓存")
            
    except Exception as e:
        logger.error(f"❌ 归一化测试失败: {e}")

def test_dataloader_workers(sample_data_size: int = 100):
    """测试不同worker数量的影响"""
    logger.info(f"👷 DataLoader workers测试")
    
    # 创建测试数据
    test_data = torch.randn(sample_data_size, 32, 32)
    test_dataset = torch.utils.data.TensorDataset(test_data, test_data)
    
    cpu_count = os.cpu_count()
    test_workers = [0, 1, 4, min(8, cpu_count//2), min(16, cpu_count)]
    
    for num_workers in test_workers:
        if num_workers > cpu_count:
            continue
            
        logger.info(f"👷 测试 {num_workers} workers...")
        
        try:
            start_time = time.time()
            
            dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=16,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=num_workers > 0,
                persistent_workers=num_workers > 0
            )
            
            # 测试第一个batch
            first_batch = next(iter(dataloader))
            
            total_time = time.time() - start_time
            logger.info(f"👷 {num_workers} workers: {total_time:.3f} 秒")
            
            del dataloader
            
        except Exception as e:
            logger.error(f"❌ {num_workers} workers 测试失败: {e}")

def generate_quick_fix_suggestions():
    """生成快速修复建议"""
    logger.info("\n" + "=" * 50)
    logger.info("🛠️  快速修复建议")
    logger.info("=" * 50)
    
    suggestions = [
        "1. 启用懒加载: lazy_loading: true",
        "2. 禁用归一化测试: normalize_data: false", 
        "3. 减少样本数量: num_samples: 100",
        "4. 减少批次大小: batch_size: 8",
        "5. 减少workers: num_workers: 8",
        "6. 降低prefetch: prefetch_factor: 2",
        "7. 禁用persistent_workers: false",
        "8. 减少训练轮数: epochs: 2"
    ]
    
    for suggestion in suggestions:
        logger.info(f"  {suggestion}")
    
    # 生成快速配置
    quick_config = {
        'data': {
            'path': '/path/to/your/data.h5',  # 需要用户修改
            'input_resolution': [32, 32],
            'output_resolution': [64, 64],
            'num_samples': 100,
            'batch_size': 8,
            'crop_mode': 'center',
            'normalize_data': False,
            'lazy_loading': True,
            'train_ratio': 0.7,
            'valid_ratio': 0.2
        },
        'dataloader': {
            'num_workers': 8,
            'pin_memory': True,
            'drop_last': False,
            'persistent_workers': False,
            'prefetch_factor': 2
        },
        'training': {
            'epochs': 2,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'patience': 5
        }
    }
    
    with open('quick_test_config.yaml', 'w', encoding='utf-8') as f:
        yaml.dump(quick_config, f, default_flow_style=False)
    
    logger.info("\n💾 快速测试配置已保存到: quick_test_config.yaml")
    logger.info("🔧 请修改其中的数据路径，然后使用:")
    logger.info("   python dynamic_resolution_trainer.py --config quick_test_config.yaml")

def main():
    logger.info("🚀 快速服务器测试开始")
    logger.info("=" * 50)
    
    # 基本环境测试
    test_basic_environment()
    
    # 查找配置文件
    config_files = [
        'dynamic_config_server_downsampling.yaml',
        'dynamic_config_server.yaml',
        'dynamic_config.yaml'
    ]
    
    config_path = None
    for config_file in config_files:
        if Path(config_file).exists():
            config_path = config_file
            break
    
    if config_path:
        logger.info(f"📋 找到配置文件: {config_path}")
        
        # 读取配置获取数据路径
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            data_path = config['data']['path']
            logger.info(f"📁 数据路径: {data_path}")
            
            # 运行测试
            if test_data_file_quick(data_path):
                test_lazy_vs_eager_loading(data_path)
                test_normalization_impact(data_path)
            
        except Exception as e:
            logger.error(f"❌ 配置文件读取失败: {e}")
    else:
        logger.warning("⚠️  未找到配置文件，跳过数据文件测试")
    
    # DataLoader测试
    test_dataloader_workers()
    
    # 生成建议
    generate_quick_fix_suggestions()
    
    logger.info("\n✅ 快速测试完成！")

if __name__ == '__main__':
    main()