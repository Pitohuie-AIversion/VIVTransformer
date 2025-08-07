#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多核数据加载器测试脚本
用于验证数据加载器是否正确启用多核处理
"""

import os
import sys
import time
import psutil
import torch
import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
import numpy as np
from threading import Thread
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DummyDataset(Dataset):
    """模拟数据集，用于测试数据加载性能"""
    
    def __init__(self, size=1000, data_shape=(32, 32)):
        self.size = size
        self.data_shape = data_shape
        
    def __len__(self):
        return self.size
        
    def __getitem__(self, idx):
        # 模拟一些计算密集的数据处理
        time.sleep(0.01)  # 模拟数据加载延迟
        
        # 生成随机数据
        input_data = np.random.randn(*self.data_shape).astype(np.float32)
        output_data = np.random.randn(128, 128).astype(np.float32)
        
        return torch.from_numpy(input_data), torch.from_numpy(output_data)

def monitor_cpu_usage(duration=30, interval=1):
    """监控CPU使用率"""
    cpu_usage_history = []
    start_time = time.time()
    
    while time.time() - start_time < duration:
        cpu_percent = psutil.cpu_percent(interval=interval, percpu=True)
        cpu_usage_history.append(cpu_percent)
        
        # 实时显示CPU使用情况
        avg_cpu = np.mean(cpu_percent)
        active_cores = sum(1 for usage in cpu_percent if usage > 10)
        logger.info(f"CPU使用率: 平均 {avg_cpu:.1f}%, 活跃核心数: {active_cores}/{len(cpu_percent)}")
    
    return cpu_usage_history

def test_dataloader_performance(num_workers_list=[0, 1, 4, 8, 16, 32, 64]):
    """测试不同num_workers设置下的数据加载性能"""
    
    logger.info(f"🖥️ 系统信息: CPU核心数={os.cpu_count()}, 内存={psutil.virtual_memory().total // (1024**3)}GB")
    
    # 创建测试数据集
    dataset = DummyDataset(size=200, data_shape=(32, 32))
    
    results = {}
    
    for num_workers in num_workers_list:
        if num_workers > os.cpu_count():
            logger.warning(f"⚠️ 跳过num_workers={num_workers}，超过CPU核心数{os.cpu_count()}")
            continue
            
        logger.info(f"\n🧪 测试 num_workers={num_workers}")
        
        # 创建数据加载器
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            prefetch_factor=4 if num_workers > 0 else None
        )
        
        # 启动CPU监控
        cpu_monitor_thread = Thread(
            target=lambda: monitor_cpu_usage(duration=15, interval=0.5),
            daemon=True
        )
        cpu_monitor_thread.start()
        
        # 测试数据加载性能
        start_time = time.time()
        batch_count = 0
        
        try:
            for batch_idx, (inputs, targets) in enumerate(dataloader):
                batch_count += 1
                if batch_count >= 10:  # 只测试前10个批次
                    break
                    
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            results[num_workers] = {
                'elapsed_time': elapsed_time,
                'batches_per_second': batch_count / elapsed_time,
                'samples_per_second': (batch_count * 16) / elapsed_time
            }
            
            logger.info(f"✅ num_workers={num_workers}: {elapsed_time:.2f}秒, {results[num_workers]['batches_per_second']:.2f} batches/s")
            
        except Exception as e:
            logger.error(f"❌ num_workers={num_workers} 测试失败: {e}")
            results[num_workers] = {'error': str(e)}
        
        # 等待CPU监控线程结束
        time.sleep(1)
    
    return results

def test_automatic_optimization():
    """测试自动优化逻辑"""
    logger.info("\n🔧 测试自动优化逻辑")
    
    # 模拟配置
    config = {
        'dataloader': {
            'num_workers': 0,  # 设为0触发自动优化
            'pin_memory': True,
            'persistent_workers': False,
            'prefetch_factor': 4,
            'drop_last': True
        }
    }
    
    # 模拟自动优化逻辑
    dataloader_config = config.get('dataloader', {})
    num_workers = dataloader_config.get('num_workers', 0)
    cpu_count = os.cpu_count()
    
    logger.info(f"初始配置: num_workers={num_workers}, CPU核心数={cpu_count}")
    
    # 应用自动优化逻辑
    if num_workers == 0 and cpu_count > 4:
        if cpu_count >= 64:  # 超级服务器
            num_workers = min(64, cpu_count // 3)
            logger.info(f"🚀 超级服务器优化: 设置num_workers={num_workers} (使用1/3核心)")
        else:  # 普通服务器
            num_workers = min(16, cpu_count // 2)
            logger.info(f"🚀 普通服务器优化: 设置num_workers={num_workers} (使用1/2核心)")
    
    return num_workers

def main():
    """主函数"""
    logger.info("🚀 开始多核数据加载器测试")
    
    # 设置多进程启动方法
    try:
        if mp.get_start_method(allow_none=True) != 'spawn':
            mp.set_start_method('spawn', force=True)
            logger.info("🔧 设置多进程启动方法为spawn")
    except RuntimeError:
        logger.warning("⚠️ 无法设置多进程启动方法")
    
    # 测试自动优化逻辑
    optimal_workers = test_automatic_optimization()
    
    # 测试不同num_workers设置的性能
    test_workers = [0, 1, 4, 8, 16]
    if optimal_workers not in test_workers:
        test_workers.append(optimal_workers)
    
    # 如果是超级服务器，添加更多测试点
    if os.cpu_count() >= 64:
        test_workers.extend([32, 64])
    
    test_workers = sorted(list(set(test_workers)))
    
    logger.info(f"\n📊 将测试以下num_workers设置: {test_workers}")
    
    results = test_dataloader_performance(test_workers)
    
    # 输出结果总结
    logger.info("\n📈 性能测试结果总结:")
    logger.info("-" * 60)
    logger.info(f"{'num_workers':<12} {'时间(秒)':<10} {'批次/秒':<10} {'样本/秒':<10}")
    logger.info("-" * 60)
    
    for num_workers, result in results.items():
        if 'error' in result:
            logger.info(f"{num_workers:<12} {'错误':<10} {result['error'][:20]:<20}")
        else:
            logger.info(f"{num_workers:<12} {result['elapsed_time']:<10.2f} {result['batches_per_second']:<10.2f} {result['samples_per_second']:<10.1f}")
    
    # 找出最佳配置
    valid_results = {k: v for k, v in results.items() if 'error' not in v}
    if valid_results:
        best_config = max(valid_results.items(), key=lambda x: x[1]['batches_per_second'])
        logger.info(f"\n🏆 最佳配置: num_workers={best_config[0]} (批次/秒: {best_config[1]['batches_per_second']:.2f})")
        
        # 给出建议
        if best_config[0] == 0:
            logger.info("💡 建议: 单进程模式最优，可能是数据处理较轻或存在进程开销")
        elif best_config[0] == optimal_workers:
            logger.info("💡 建议: 自动优化设置已是最佳配置")
        else:
            logger.info(f"💡 建议: 考虑将配置文件中的num_workers设为{best_config[0]}")

if __name__ == "__main__":
    main()