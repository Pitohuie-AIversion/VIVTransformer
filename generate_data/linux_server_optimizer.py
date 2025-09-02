#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Linux服务器环境优化脚本
用于检测和解决Linux服务器上的系统限制问题
特别针对高并发数据加载的限制进行优化

作者: AI Assistant
日期: 2024
"""

import os
import sys
import subprocess
import resource
import multiprocessing
import psutil
import yaml
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('linux_server_optimization.log')
    ]
)
logger = logging.getLogger(__name__)

class LinuxServerOptimizer:
    """Linux服务器环境优化器"""
    
    def __init__(self):
        self.system_info = {}
        self.limitations = {}
        self.recommendations = {}
        
    def detect_system_info(self):
        """检测系统信息"""
        logger.info("[INFO] 检测系统信息...")
        
        # 基本系统信息
        self.system_info.update({
            'platform': sys.platform,
            'cpu_count': os.cpu_count(),
            'memory_total': psutil.virtual_memory().total // (1024**3),  # GB
            'memory_available': psutil.virtual_memory().available // (1024**3),  # GB
            'python_version': sys.version,
        })
        
        # 检测GPU信息
        try:
            import torch
            self.system_info['cuda_available'] = torch.cuda.is_available()
            if torch.cuda.is_available():
                self.system_info['gpu_count'] = torch.cuda.device_count()
                self.system_info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            else:
                self.system_info['gpu_count'] = 0
                self.system_info['gpu_names'] = []
        except ImportError:
            self.system_info['cuda_available'] = False
            self.system_info['gpu_count'] = 0
            self.system_info['gpu_names'] = []
        
        # 打印系统信息
        logger.info(f"[INFO]️ 平台: {self.system_info['platform']}")
        logger.info(f"🔢 CPU核心数: {self.system_info['cpu_count']}")
        logger.info(f"[INFO] 总内存: {self.system_info['memory_total']} GB")
        logger.info(f"[INFO] 可用内存: {self.system_info['memory_available']} GB")
        logger.info(f"[INFO] Python版本: {self.system_info['python_version'].split()[0]}")
        logger.info(f"[INFO] CUDA可用: {self.system_info['cuda_available']}")
        logger.info(f"[INFO] GPU数量: {self.system_info['gpu_count']}")
        if self.system_info['gpu_names']:
            for i, name in enumerate(self.system_info['gpu_names']):
                logger.info(f"[INFO] GPU {i}: {name}")
    
    def check_system_limits(self):
        """检查系统限制"""
        logger.info("\n[INFO] 检查系统限制...")
        
        # 检查文件描述符限制
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
            self.limitations['file_descriptors'] = {
                'soft_limit': soft_limit,
                'hard_limit': hard_limit,
                'recommended': max(4096, self.system_info['cpu_count'] * 64)
            }
            logger.info(f"[INFO] 文件描述符限制: {soft_limit} (软限制) / {hard_limit} (硬限制)")
            
            if soft_limit < self.limitations['file_descriptors']['recommended']:
                logger.warning(f"[WARN] 文件描述符限制过低，建议至少 {self.limitations['file_descriptors']['recommended']}")
        except Exception as e:
            logger.error(f"[ERROR] 无法检查文件描述符限制: {e}")
        
        # 检查进程限制
        try:
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NPROC)
            self.limitations['processes'] = {
                'soft_limit': soft_limit,
                'hard_limit': hard_limit,
                'recommended': max(2048, self.system_info['cpu_count'] * 16)
            }
            logger.info(f"[INFO] 进程数限制: {soft_limit} (软限制) / {hard_limit} (硬限制)")
            
            if soft_limit < self.limitations['processes']['recommended']:
                logger.warning(f"[WARN] 进程数限制过低，建议至少 {self.limitations['processes']['recommended']}")
        except Exception as e:
            logger.error(f"[ERROR] 无法检查进程限制: {e}")
        
        # 检查内存限制
        try:
            # 检查虚拟内存限制
            soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_AS)
            if soft_limit != resource.RLIM_INFINITY:
                self.limitations['virtual_memory'] = {
                    'soft_limit': soft_limit // (1024**3),  # GB
                    'hard_limit': hard_limit // (1024**3) if hard_limit != resource.RLIM_INFINITY else 'unlimited'
                }
                logger.info(f"[INFO] 虚拟内存限制: {self.limitations['virtual_memory']['soft_limit']} GB")
            else:
                logger.info(f"[INFO] 虚拟内存限制: 无限制")
        except Exception as e:
            logger.error(f"[ERROR] 无法检查内存限制: {e}")
    
    def test_multiprocessing_performance(self):
        """测试多进程性能"""
        logger.info("\n[TEST] 测试多进程性能...")
        
        import time
        import torch.utils.data as data
        import numpy as np
        
        # 创建测试数据集
        class TestDataset(data.Dataset):
            def __init__(self, size=1000):
                self.size = size
            
            def __len__(self):
                return self.size
            
            def __getitem__(self, idx):
                # 模拟数据加载
                time.sleep(0.001)  # 模拟I/O延迟
                return np.random.randn(32, 32), np.random.randn(128, 128)
        
        dataset = TestDataset(100)
        
        # 测试不同的worker数量
        worker_counts = [0, 1, 2, 4, 8, min(16, self.system_info['cpu_count'] // 2)]
        results = {}
        
        for num_workers in worker_counts:
            try:
                logger.info(f"[INFO] 测试 num_workers={num_workers}...")
                
                dataloader = data.DataLoader(
                    dataset,
                    batch_size=8,
                    num_workers=num_workers,
                    pin_memory=True if num_workers > 0 else False,
                    persistent_workers=True if num_workers > 0 else False,
                    prefetch_factor=2 if num_workers > 0 else None
                )
                
                start_time = time.time()
                batch_count = 0
                
                for batch in dataloader:
                    batch_count += 1
                    if batch_count >= 10:  # 只测试前10个批次
                        break
                
                end_time = time.time()
                duration = end_time - start_time
                results[num_workers] = duration
                
                logger.info(f"[OK] num_workers={num_workers}: {duration:.2f}秒")
                
            except Exception as e:
                logger.error(f"[ERROR] num_workers={num_workers} 测试失败: {e}")
                results[num_workers] = float('inf')
        
        # 找到最佳配置
        if results:
            best_workers = min(results.keys(), key=lambda k: results[k])
            self.recommendations['optimal_num_workers'] = best_workers
            logger.info(f"🏆 最佳num_workers配置: {best_workers} (耗时: {results[best_workers]:.2f}秒)")
        
        return results
    
    def generate_optimized_config(self, base_config_path, output_path):
        """生成优化后的配置文件"""
        logger.info(f"\n[INFO] 生成优化配置文件: {output_path}")
        
        # 读取基础配置
        try:
            with open(base_config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"[ERROR] 无法读取基础配置文件: {e}")
            return False
        
        # 应用优化建议
        if 'dataloader' not in config:
            config['dataloader'] = {}
        
        # 设置最佳num_workers
        optimal_workers = self.recommendations.get('optimal_num_workers', 0)
        config['dataloader']['num_workers'] = optimal_workers
        
        # 根据系统资源调整其他参数
        if optimal_workers > 0:
            config['dataloader']['persistent_workers'] = True
            config['dataloader']['prefetch_factor'] = min(4, max(2, optimal_workers // 2))
        else:
            config['dataloader']['persistent_workers'] = False
            config['dataloader']['prefetch_factor'] = None
        
        # 根据内存大小调整批次大小
        memory_gb = self.system_info['memory_available']
        if memory_gb >= 64:  # 64GB以上
            suggested_batch_size = 64
        elif memory_gb >= 32:  # 32-64GB
            suggested_batch_size = 32
        elif memory_gb >= 16:  # 16-32GB
            suggested_batch_size = 16
        else:  # 16GB以下
            suggested_batch_size = 8
        
        if 'data' in config:
            config['data']['batch_size'] = suggested_batch_size
        
        # 根据GPU数量调整设备配置
        if 'device' not in config:
            config['device'] = {}
        
        if self.system_info['gpu_count'] > 0:
            config['device']['use_cuda'] = True
            config['device']['device_ids'] = [0]  # 先使用单GPU
        else:
            config['device']['use_cuda'] = False
        
        # 禁用可能导致问题的功能
        if 'visualization' in config:
            config['visualization']['enabled'] = False
        
        if 'environment' not in config:
            config['environment'] = {}
        config['environment']['disable_gui'] = True
        config['environment']['qt_qpa_platform'] = 'offscreen'
        
        # 保存优化后的配置
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            logger.info(f"[OK] 优化配置已保存到: {output_path}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] 无法保存优化配置: {e}")
            return False
    
    def generate_system_report(self):
        """生成系统报告"""
        logger.info("\n[INFO] 生成系统优化报告...")
        
        report = {
            'system_info': self.system_info,
            'limitations': self.limitations,
            'recommendations': self.recommendations
        }
        
        # 保存报告
        report_path = 'linux_server_optimization_report.yaml'
        try:
            with open(report_path, 'w', encoding='utf-8') as f:
                yaml.dump(report, f, default_flow_style=False, allow_unicode=True, indent=2)
            logger.info(f"[OK] 系统报告已保存到: {report_path}")
        except Exception as e:
            logger.error(f"[ERROR] 无法保存系统报告: {e}")
        
        # 打印优化建议
        logger.info("\n[INFO] 优化建议:")
        
        # 文件描述符建议
        if 'file_descriptors' in self.limitations:
            fd_limit = self.limitations['file_descriptors']
            if fd_limit['soft_limit'] < fd_limit['recommended']:
                logger.info(f"[INFO] 增加文件描述符限制: ulimit -n {fd_limit['recommended']}")
        
        # 进程数建议
        if 'processes' in self.limitations:
            proc_limit = self.limitations['processes']
            if proc_limit['soft_limit'] < proc_limit['recommended']:
                logger.info(f"[INFO] 增加进程数限制: ulimit -u {proc_limit['recommended']}")
        
        # 数据加载器建议
        if 'optimal_num_workers' in self.recommendations:
            logger.info(f"[INFO] 推荐num_workers: {self.recommendations['optimal_num_workers']}")
        
        # 内存建议
        memory_usage_percent = (self.system_info['memory_total'] - self.system_info['memory_available']) / self.system_info['memory_total'] * 100
        if memory_usage_percent > 80:
            logger.warning(f"[INFO] 内存使用率较高 ({memory_usage_percent:.1f}%)，建议减少批次大小或启用懒加载")
        
        return report
    
    def run_optimization(self, base_config_path=None):
        """运行完整的优化流程"""
        logger.info("[INFO] 开始Linux服务器环境优化...")
        
        # 1. 检测系统信息
        self.detect_system_info()
        
        # 2. 检查系统限制
        self.check_system_limits()
        
        # 3. 测试多进程性能
        self.test_multiprocessing_performance()
        
        # 4. 生成系统报告
        self.generate_system_report()
        
        # 5. 生成优化配置（如果提供了基础配置）
        if base_config_path and os.path.exists(base_config_path):
            output_path = base_config_path.replace('.yaml', '_linux_optimized.yaml')
            self.generate_optimized_config(base_config_path, output_path)
        
        logger.info("\n[OK] Linux服务器环境优化完成！")
        logger.info("[INFO] 请查看生成的报告和优化配置文件")
        logger.info("🔧 如需应用系统限制调整，请运行建议的ulimit命令")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Linux服务器环境优化工具')
    parser.add_argument('--config', type=str, help='基础配置文件路径')
    parser.add_argument('--test-only', action='store_true', help='仅运行性能测试')
    
    args = parser.parse_args()
    
    optimizer = LinuxServerOptimizer()
    
    if args.test_only:
        optimizer.detect_system_info()
        optimizer.test_multiprocessing_performance()
    else:
        optimizer.run_optimization(args.config)

if __name__ == '__main__':
    main()