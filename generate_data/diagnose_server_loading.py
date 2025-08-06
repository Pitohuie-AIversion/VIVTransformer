#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器数据加载性能诊断脚本
用于诊断dynamic_resolution_trainer.py在服务器环境下启动缓慢的原因

使用方法:
python diagnose_server_loading.py --config dynamic_config_server_downsampling.yaml
"""

import os
import sys
import time
import argparse
import logging
import yaml
import h5py
import numpy as np
import psutil
import torch
from pathlib import Path
from typing import Dict, Any, Tuple

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server_diagnosis.log')
    ]
)
logger = logging.getLogger(__name__)

class ServerLoadingDiagnostic:
    """服务器加载性能诊断器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self.load_config()
        self.data_path = self.config['data']['path']
        self.results = {}
        
    def load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        logger.info(f"📋 加载配置文件: {self.config_path}")
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def log_system_info(self):
        """记录系统信息"""
        logger.info("🖥️  系统信息诊断")
        logger.info("=" * 50)
        
        # CPU信息
        cpu_count = os.cpu_count()
        cpu_percent = psutil.cpu_percent(interval=1)
        logger.info(f"CPU核心数: {cpu_count}")
        logger.info(f"CPU使用率: {cpu_percent:.1f}%")
        
        # 内存信息
        memory = psutil.virtual_memory()
        logger.info(f"总内存: {memory.total / 1024**3:.2f} GB")
        logger.info(f"可用内存: {memory.available / 1024**3:.2f} GB")
        logger.info(f"内存使用率: {memory.percent:.1f}%")
        
        # GPU信息
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPU数量: {gpu_count}")
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                logger.info(f"GPU {i}: {gpu_name} ({gpu_memory:.2f} GB)")
        else:
            logger.info("GPU: 不可用")
        
        # 存储信息
        data_dir = Path(self.data_path).parent
        if data_dir.exists():
            disk_usage = psutil.disk_usage(str(data_dir))
            logger.info(f"数据目录磁盘使用: {disk_usage.used / 1024**3:.2f} GB / {disk_usage.total / 1024**3:.2f} GB")
        
        logger.info("=" * 50)
    
    def test_data_file_access(self) -> Dict[str, Any]:
        """测试数据文件访问性能"""
        logger.info("📁 数据文件访问测试")
        logger.info("-" * 30)
        
        result = {}
        
        # 检查文件是否存在
        if not Path(self.data_path).exists():
            logger.error(f"❌ 数据文件不存在: {self.data_path}")
            result['file_exists'] = False
            return result
        
        result['file_exists'] = True
        file_size = Path(self.data_path).stat().st_size / 1024**3  # GB
        result['file_size_gb'] = file_size
        logger.info(f"✅ 文件大小: {file_size:.2f} GB")
        
        # 测试文件打开时间
        start_time = time.time()
        try:
            with h5py.File(self.data_path, 'r') as f:
                open_time = time.time() - start_time
                result['file_open_time'] = open_time
                logger.info(f"📂 文件打开时间: {open_time:.3f} 秒")
                
                # 获取数据集信息
                if 'tensor' in f:
                    dataset = f['tensor']
                    shape = dataset.shape
                    dtype = dataset.dtype
                    result['data_shape'] = shape
                    result['data_dtype'] = str(dtype)
                    
                    logger.info(f"📊 数据形状: {shape}")
                    logger.info(f"📊 数据类型: {dtype}")
                    
                    # 估算数据大小
                    estimated_size = np.prod(shape) * np.dtype(dtype).itemsize / 1024**3
                    result['estimated_data_size_gb'] = estimated_size
                    logger.info(f"📊 估算数据大小: {estimated_size:.2f} GB")
                else:
                    logger.error("❌ 数据文件中未找到'tensor'键")
                    result['has_tensor'] = False
                    
        except Exception as e:
            logger.error(f"❌ 文件访问失败: {e}")
            result['file_access_error'] = str(e)
        
        return result
    
    def test_sample_loading_speed(self, num_samples: int = 10) -> Dict[str, Any]:
        """测试样本加载速度"""
        logger.info(f"⚡ 样本加载速度测试 (测试 {num_samples} 个样本)")
        logger.info("-" * 30)
        
        result = {}
        
        try:
            with h5py.File(self.data_path, 'r') as f:
                dataset = f['tensor']
                total_samples = dataset.shape[0]
                test_samples = min(num_samples, total_samples)
                
                # 测试顺序读取
                start_time = time.time()
                for i in range(test_samples):
                    sample = np.array(dataset[i], dtype=np.float32)
                sequential_time = time.time() - start_time
                
                result['sequential_loading_time'] = sequential_time
                result['samples_per_second'] = test_samples / sequential_time
                
                logger.info(f"📈 顺序加载 {test_samples} 个样本耗时: {sequential_time:.3f} 秒")
                logger.info(f"📈 加载速度: {test_samples / sequential_time:.2f} 样本/秒")
                
                # 测试随机读取
                indices = np.random.choice(total_samples, test_samples, replace=False)
                start_time = time.time()
                for idx in indices:
                    sample = np.array(dataset[idx], dtype=np.float32)
                random_time = time.time() - start_time
                
                result['random_loading_time'] = random_time
                result['random_samples_per_second'] = test_samples / random_time
                
                logger.info(f"🎲 随机加载 {test_samples} 个样本耗时: {random_time:.3f} 秒")
                logger.info(f"🎲 随机加载速度: {test_samples / random_time:.2f} 样本/秒")
                
        except Exception as e:
            logger.error(f"❌ 样本加载测试失败: {e}")
            result['loading_error'] = str(e)
        
        return result
    
    def test_normalization_computation(self, sample_size: int = 100) -> Dict[str, Any]:
        """测试归一化参数计算时间"""
        logger.info(f"🔢 归一化参数计算测试 (采样 {sample_size} 个样本)")
        logger.info("-" * 30)
        
        result = {}
        
        try:
            with h5py.File(self.data_path, 'r') as f:
                dataset = f['tensor']
                total_samples = dataset.shape[0]
                actual_sample_size = min(sample_size, total_samples)
                
                # 测试采样方式计算归一化参数
                start_time = time.time()
                
                # 随机采样
                indices = np.random.choice(total_samples, actual_sample_size, replace=False)
                sample_data = []
                
                for idx in indices:
                    sample = np.array(dataset[idx], dtype=np.float32)
                    if len(sample.shape) == 3 and sample.shape[0] == 1:
                        sample = sample.squeeze(0)
                    sample_data.append(sample.flatten())
                
                # 计算全局最小最大值
                all_data = np.concatenate(sample_data)
                global_min = np.min(all_data)
                global_max = np.max(all_data)
                
                sampling_time = time.time() - start_time
                
                result['sampling_normalization_time'] = sampling_time
                result['global_min'] = float(global_min)
                result['global_max'] = float(global_max)
                result['sample_size_used'] = actual_sample_size
                
                logger.info(f"📊 采样归一化计算耗时: {sampling_time:.3f} 秒")
                logger.info(f"📊 全局范围: [{global_min:.6f}, {global_max:.6f}]")
                
                # 如果数据集不太大，测试全数据集计算时间
                if total_samples <= 1000:
                    logger.info("🔄 测试全数据集归一化计算...")
                    start_time = time.time()
                    
                    all_samples = []
                    for i in range(total_samples):
                        sample = np.array(dataset[i], dtype=np.float32)
                        if len(sample.shape) == 3 and sample.shape[0] == 1:
                            sample = sample.squeeze(0)
                        all_samples.append(sample.flatten())
                    
                    full_data = np.concatenate(all_samples)
                    full_min = np.min(full_data)
                    full_max = np.max(full_data)
                    
                    full_time = time.time() - start_time
                    
                    result['full_normalization_time'] = full_time
                    result['full_global_min'] = float(full_min)
                    result['full_global_max'] = float(full_max)
                    
                    logger.info(f"📊 全数据集归一化计算耗时: {full_time:.3f} 秒")
                    logger.info(f"📊 全数据集范围: [{full_min:.6f}, {full_max:.6f}]")
                    
                    # 比较差异
                    min_diff = abs(global_min - full_min)
                    max_diff = abs(global_max - full_max)
                    logger.info(f"📊 采样vs全数据集差异: min_diff={min_diff:.6f}, max_diff={max_diff:.6f}")
                
        except Exception as e:
            logger.error(f"❌ 归一化计算测试失败: {e}")
            result['normalization_error'] = str(e)
        
        return result
    
    def test_dataloader_creation(self) -> Dict[str, Any]:
        """测试数据加载器创建时间"""
        logger.info("🔄 数据加载器创建测试")
        logger.info("-" * 30)
        
        result = {}
        
        try:
            # 模拟数据集创建（简化版本）
            start_time = time.time()
            
            # 获取配置参数
            data_config = self.config['data']
            dataloader_config = self.config.get('dataloader', {})
            
            num_workers = dataloader_config.get('num_workers', 0)
            pin_memory = dataloader_config.get('pin_memory', True)
            persistent_workers = dataloader_config.get('persistent_workers', False)
            prefetch_factor = dataloader_config.get('prefetch_factor', 2)
            
            logger.info(f"🔧 数据加载器配置:")
            logger.info(f"   num_workers: {num_workers}")
            logger.info(f"   pin_memory: {pin_memory}")
            logger.info(f"   persistent_workers: {persistent_workers}")
            logger.info(f"   prefetch_factor: {prefetch_factor}")
            
            # 创建简单的测试数据集
            test_data = torch.randn(100, 32, 32)  # 100个32x32的样本
            test_dataset = torch.utils.data.TensorDataset(test_data, test_data)
            
            # 测试不同num_workers的创建时间
            for workers in [0, 1, 4, 8, 16, num_workers]:
                if workers > os.cpu_count():
                    continue
                    
                worker_start = time.time()
                
                dataloader_kwargs = {
                    'batch_size': 16,
                    'shuffle': True,
                    'num_workers': workers,
                    'pin_memory': pin_memory if workers > 0 else False,
                    'persistent_workers': persistent_workers and workers > 0
                }
                
                if workers > 0:
                    dataloader_kwargs['prefetch_factor'] = prefetch_factor
                
                test_loader = torch.utils.data.DataLoader(test_dataset, **dataloader_kwargs)
                
                # 测试第一个batch的加载时间
                batch_start = time.time()
                first_batch = next(iter(test_loader))
                batch_time = time.time() - batch_start
                
                worker_time = time.time() - worker_start
                
                logger.info(f"👷 workers={workers}: 创建耗时={worker_time:.3f}s, 首批次={batch_time:.3f}s")
                
                result[f'workers_{workers}_creation_time'] = worker_time
                result[f'workers_{workers}_first_batch_time'] = batch_time
                
                # 清理
                del test_loader
            
            total_time = time.time() - start_time
            result['total_dataloader_test_time'] = total_time
            
        except Exception as e:
            logger.error(f"❌ 数据加载器测试失败: {e}")
            result['dataloader_error'] = str(e)
        
        return result
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """分析性能瓶颈"""
        logger.info("🔍 性能瓶颈分析")
        logger.info("=" * 50)
        
        analysis = {}
        
        # 分析文件I/O性能
        if 'file_access' in self.results:
            file_result = self.results['file_access']
            if file_result.get('file_open_time', 0) > 1.0:
                analysis['slow_file_io'] = True
                logger.warning("⚠️  文件I/O较慢，可能是网络存储或磁盘性能问题")
        
        # 分析样本加载性能
        if 'sample_loading' in self.results:
            loading_result = self.results['sample_loading']
            if loading_result.get('samples_per_second', 0) < 10:
                analysis['slow_sample_loading'] = True
                logger.warning("⚠️  样本加载速度较慢，建议启用懒加载")
        
        # 分析归一化计算
        if 'normalization' in self.results:
            norm_result = self.results['normalization']
            if norm_result.get('sampling_normalization_time', 0) > 5.0:
                analysis['slow_normalization'] = True
                logger.warning("⚠️  归一化计算较慢，建议使用采样方式或缓存结果")
        
        # 分析数据加载器配置
        if 'dataloader' in self.results:
            dl_result = self.results['dataloader']
            high_workers = max([k for k in dl_result.keys() if 'workers_' in k and '_creation_time' in k], 
                              key=lambda x: dl_result[x], default=None)
            if high_workers and dl_result[high_workers] > 10.0:
                analysis['slow_dataloader_creation'] = True
                logger.warning("⚠️  数据加载器创建较慢，建议减少num_workers")
        
        return analysis
    
    def generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if analysis.get('slow_file_io'):
            recommendations.extend([
                "启用懒加载模式 (lazy_loading: true)",
                "考虑将数据复制到本地SSD存储",
                "使用数据预加载脚本"
            ])
        
        if analysis.get('slow_sample_loading'):
            recommendations.extend([
                "启用懒加载模式 (lazy_loading: true)",
                "减少批次大小 (batch_size)",
                "增加内存缓存"
            ])
        
        if analysis.get('slow_normalization'):
            recommendations.extend([
                "禁用归一化进行测试 (normalize_data: false)",
                "使用采样方式计算归一化参数",
                "缓存归一化参数到文件",
                "预计算并保存归一化后的数据"
            ])
        
        if analysis.get('slow_dataloader_creation'):
            recommendations.extend([
                "减少num_workers数量 (建议16-32)",
                "降低prefetch_factor (建议4-8)",
                "禁用persistent_workers进行测试"
            ])
        
        # 通用建议
        recommendations.extend([
            "使用更小的数据集进行初始测试",
            "监控系统资源使用情况",
            "考虑使用分布式训练"
        ])
        
        return recommendations
    
    def run_full_diagnosis(self):
        """运行完整诊断"""
        logger.info("🚀 开始服务器加载性能诊断")
        logger.info("=" * 60)
        
        # 系统信息
        self.log_system_info()
        
        # 数据文件访问测试
        self.results['file_access'] = self.test_data_file_access()
        
        # 样本加载速度测试
        self.results['sample_loading'] = self.test_sample_loading_speed()
        
        # 归一化计算测试
        self.results['normalization'] = self.test_normalization_computation()
        
        # 数据加载器创建测试
        self.results['dataloader'] = self.test_dataloader_creation()
        
        # 性能瓶颈分析
        analysis = self.analyze_bottlenecks()
        
        # 生成建议
        recommendations = self.generate_recommendations(analysis)
        
        # 输出总结
        logger.info("\n" + "=" * 60)
        logger.info("📋 诊断总结")
        logger.info("=" * 60)
        
        logger.info("🔧 优化建议:")
        for i, rec in enumerate(recommendations, 1):
            logger.info(f"  {i}. {rec}")
        
        # 生成快速修复配置
        self.generate_quick_fix_config()
        
        logger.info("\n✅ 诊断完成！请查看 server_diagnosis.log 获取详细信息")
    
    def generate_quick_fix_config(self):
        """生成快速修复配置文件"""
        logger.info("\n🛠️  生成快速修复配置...")
        
        # 基于原配置创建优化版本
        optimized_config = self.config.copy()
        
        # 数据配置优化
        optimized_config['data']['lazy_loading'] = True
        optimized_config['data']['normalize_data'] = False  # 临时禁用
        optimized_config['data']['batch_size'] = min(16, optimized_config['data'].get('batch_size', 32))
        
        # 数据加载器优化
        if 'dataloader' not in optimized_config:
            optimized_config['dataloader'] = {}
        
        optimized_config['dataloader']['num_workers'] = min(16, os.cpu_count() // 2)
        optimized_config['dataloader']['prefetch_factor'] = 4
        optimized_config['dataloader']['persistent_workers'] = False
        optimized_config['dataloader']['pin_memory'] = True
        
        # 训练配置优化（用于快速测试）
        optimized_config['training']['epochs'] = 2
        optimized_config['data']['num_samples'] = min(100, optimized_config['data'].get('num_samples', 1000))
        
        # 保存优化配置
        output_path = 'dynamic_config_server_quick_fix.yaml'
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(optimized_config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"💾 快速修复配置已保存到: {output_path}")
        logger.info("🚀 使用命令测试: python dynamic_resolution_trainer.py --config dynamic_config_server_quick_fix.yaml")

def main():
    parser = argparse.ArgumentParser(description='服务器数据加载性能诊断工具')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--samples', type=int, default=10, help='测试样本数量')
    parser.add_argument('--norm-samples', type=int, default=100, help='归一化测试样本数量')
    
    args = parser.parse_args()
    
    if not Path(args.config).exists():
        logger.error(f"❌ 配置文件不存在: {args.config}")
        sys.exit(1)
    
    # 运行诊断
    diagnostic = ServerLoadingDiagnostic(args.config)
    diagnostic.run_full_diagnosis()

if __name__ == '__main__':
    main()