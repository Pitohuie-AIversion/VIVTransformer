#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
服务器环境配置和验证脚本

这个脚本帮助在Linux服务器上正确配置和验证训练环境
主要功能:
1. 检测服务器环境和资源
2. 验证数据路径和文件
3. 创建适合的配置文件
4. 提供运行建议

作者: AI Assistant
日期: 2025
"""

import os
import sys
import yaml
import torch
import psutil
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServerEnvironmentSetup:
    """
    服务器环境配置和验证类
    """
    
    def __init__(self):
        self.system_info = {}
        self.data_paths = []
        self.config_recommendations = {}
        
    def detect_system_resources(self) -> Dict:
        """
        检测系统资源
        """
        logger.info("[INFO] 检测系统资源...")
        
        # CPU信息
        cpu_count = psutil.cpu_count(logical=True)
        cpu_usage = psutil.cpu_percent(interval=1)
        
        # 内存信息
        memory = psutil.virtual_memory()
        total_memory_gb = memory.total / (1024**3)
        available_memory_gb = memory.available / (1024**3)
        
        # GPU信息
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        
        self.system_info = {
            'cpu_count': cpu_count,
            'cpu_usage': cpu_usage,
            'total_memory_gb': total_memory_gb,
            'available_memory_gb': available_memory_gb,
            'gpu_available': gpu_available,
            'gpu_count': gpu_count,
            'platform': sys.platform
        }
        
        logger.info(f"[INFO] CPU核心数: {cpu_count}")
        logger.info(f"[INFO] CPU使用率: {cpu_usage:.1f}%")
        logger.info(f"[INFO] 总内存: {total_memory_gb:.1f}GB")
        logger.info(f"[INFO] 可用内存: {available_memory_gb:.1f}GB")
        logger.info(f"[INFO] GPU可用: {gpu_available}")
        logger.info(f"[INFO] GPU数量: {gpu_count}")
        logger.info(f"[INFO]️ 平台: {sys.platform}")
        
        return self.system_info
    
    def find_data_paths(self) -> List[str]:
        """
        查找可能的数据路径
        """
        logger.info("[INFO] 查找数据路径...")
        
        # 常见的数据路径模式
        possible_paths = [
            "/share/fandixiaLab/suguangsheng/PycharmProjects/Sparse_to_Dense_Transformer/VIVTransformer-4sh2r1-codex/pdebench_extended/PDEBench/pdebench/data_download",
            "/share/fandixiaLab/suguangsheng/PycharmProjects/VIVTransformer_pdebench/PDEBench/pdebench/data_download",
            "/data/PDEBench/pdebench/data_download",
            "/home/*/PDEBench/pdebench/data_download",
            "./PDEBench/pdebench/data_download",
            "../PDEBench/pdebench/data_download",
            "/tmp/PDEBench/pdebench/data_download"
        ]
        
        # 查找数据文件
        target_files = [
            "2D_DarcyFlow_beta0.1_Train.hdf5",
            "1D_Burgers_Sols_Nu0.001.hdf5",
            "1D_Advection_Sols_beta0.1.hdf5"
        ]
        
        found_paths = []
        
        for base_path in possible_paths:
            # 展开用户路径
            expanded_path = os.path.expanduser(base_path)
            
            if os.path.exists(expanded_path):
                # 检查是否包含目标文件
                files_found = []
                for target_file in target_files:
                    file_path = os.path.join(expanded_path, target_file)
                    if os.path.exists(file_path):
                        files_found.append(target_file)
                
                if files_found:
                    found_paths.append({
                        'path': expanded_path,
                        'files': files_found
                    })
                    logger.info(f"[OK] 找到数据路径: {expanded_path}")
                    logger.info(f"   包含文件: {', '.join(files_found)}")
        
        if not found_paths:
            logger.warning("[WARN] 未找到PDEBench数据文件")
            logger.info("请确保数据文件位于以下位置之一:")
            for path in possible_paths:
                logger.info(f"  - {path}")
        
        self.data_paths = found_paths
        return found_paths
    
    def generate_config_recommendations(self) -> Dict:
        """
        根据系统资源生成配置建议
        """
        logger.info("[INFO] 生成配置建议...")
        
        cpu_count = self.system_info.get('cpu_count', 1)
        total_memory_gb = self.system_info.get('total_memory_gb', 1)
        gpu_count = self.system_info.get('gpu_count', 0)
        
        # 基于系统资源的建议
        recommendations = {
            'dataloader': {
                'num_workers': min(cpu_count // 4, 64),  # 使用1/4的CPU核心
                'pin_memory': True if total_memory_gb > 16 else False,
                'persistent_workers': True if cpu_count > 8 else False,
                'prefetch_factor': min(cpu_count // 8, 16)
            },
            'training': {
                'batch_size': self._recommend_batch_size(),
                'epochs': 500 if total_memory_gb > 100 else 200,
                'num_samples': self._recommend_num_samples()
            },
            'device': {
                'device': 'cuda' if gpu_count > 0 else 'cpu',
                'use_dataparallel': True if gpu_count > 1 else False,
                'max_memory_fraction': 0.95 if gpu_count > 0 else 0.8
            },
            'environment': {
                'headless': True,  # 服务器环境
                'backend': 'Agg',
                'disable_gui': True
            }
        }
        
        self.config_recommendations = recommendations
        
        # 打印建议
        logger.info("[INFO] 配置建议:")
        logger.info(f"  数据加载器工作进程: {recommendations['dataloader']['num_workers']}")
        logger.info(f"  批次大小: {recommendations['training']['batch_size']}")
        logger.info(f"  样本数量: {recommendations['training']['num_samples']}")
        logger.info(f"  设备: {recommendations['device']['device']}")
        logger.info(f"  数据并行: {recommendations['device']['use_dataparallel']}")
        
        return recommendations
    
    def _recommend_batch_size(self) -> int:
        """
        推荐批次大小
        """
        gpu_count = self.system_info.get('gpu_count', 0)
        total_memory_gb = self.system_info.get('total_memory_gb', 1)
        
        if gpu_count == 0:
            # CPU训练
            return 4 if total_memory_gb < 32 else 8
        elif gpu_count == 1:
            # 单GPU
            return 32 if total_memory_gb > 64 else 16
        else:
            # 多GPU
            return 128 if total_memory_gb > 500 else 64
    
    def _recommend_num_samples(self) -> int:
        """
        推荐样本数量
        """
        total_memory_gb = self.system_info.get('total_memory_gb', 1)
        cpu_count = self.system_info.get('cpu_count', 1)
        
        if total_memory_gb > 500 and cpu_count > 100:
            return 20000  # 高性能服务器
        elif total_memory_gb > 100:
            return 10000  # 中等服务器
        elif total_memory_gb > 32:
            return 5000   # 普通服务器
        else:
            return 1000   # 低配置
    
    def create_server_config(self, output_path: str = "dynamic_config_server_auto.yaml") -> str:
        """
        创建适合服务器的配置文件
        """
        logger.info(f"[INFO] 创建服务器配置文件: {output_path}")
        
        if not self.data_paths:
            logger.error("[ERROR] 无法创建配置文件: 未找到数据路径")
            return None
        
        # 使用第一个找到的数据路径
        data_path = self.data_paths[0]['path']
        data_file = os.path.join(data_path, "2D_DarcyFlow_beta0.1_Train.hdf5")
        
        config = {
            'data': {
                'path': data_file,
                'input_resolution': [32, 32],
                'output_resolution': [128, 128],
                'num_samples': self.config_recommendations['training']['num_samples'],
                'crop_mode': 'center',
                'normalize': True,
                'normalization_method': 'global',
                'lazy_loading': True,
                'downsampling': {
                    'enabled': True,
                    'method': 'bilinear',
                    'antialias': True
                },
                'batch_size': self.config_recommendations['training']['batch_size'],
                'train_ratio': 0.8,
                'valid_ratio': 0.1,
                'test_ratio': 0.1
            },
            'training': {
                'epochs': self.config_recommendations['training']['epochs'],
                'batch_size': self.config_recommendations['training']['batch_size'],
                'learning_rate': 0.0005,
                'weight_decay': 0.0001
            },
            'dataloader': self.config_recommendations['dataloader'],
            'device': self.config_recommendations['device']['device'],
            'use_dataparallel': self.config_recommendations['device']['use_dataparallel'],
            'max_memory_fraction': self.config_recommendations['device']['max_memory_fraction'],
            'environment': self.config_recommendations['environment'],
            'early_stopping': {
                'enabled': True,
                'patience': 20,
                'min_delta': 0.000001,
                'restore_best_weights': True
            },
            'model_save': {
                'enabled': True,
                'save_best_only': True,
                'save_interval': 10,
                'save_path': 'results/models'
            },
            'logging': {
                'level': 'INFO',
                'save_logs': True,
                'log_interval': 10,
                'log_path': 'results/logs'
            },
            'visualization': {
                'enabled': False,
                'save_plots': True,
                'plot_interval': 50,
                'plot_path': 'results/plots'
            },
            'checkpoint': {
                'enabled': True,
                'save_interval': 20,
                'keep_last_n': 5,
                'auto_resume': True,
                'save_optimizer_state': True,
                'save_scheduler_state': True
            },
            'random_seed': 42,
            'deterministic': False
        }
        
        # 保存配置文件
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, indent=2)
            logger.info(f"[OK] 配置文件已保存: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"[ERROR] 保存配置文件失败: {e}")
            return None
    
    def validate_environment(self) -> bool:
        """
        验证环境是否准备就绪
        """
        logger.info("[INFO] 验证环境...")
        
        issues = []
        
        # 检查Python版本
        if sys.version_info < (3, 7):
            issues.append("Python版本过低，建议使用3.7+")
        
        # 检查PyTorch
        try:
            import torch
            logger.info(f"[OK] PyTorch版本: {torch.__version__}")
        except ImportError:
            issues.append("未安装PyTorch")
        
        # 检查CUDA
        if torch.cuda.is_available():
            logger.info(f"[OK] CUDA版本: {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                logger.info(f"[OK] GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
        else:
            logger.warning("[WARN] CUDA不可用，将使用CPU训练")
        
        # 检查数据路径
        if not self.data_paths:
            issues.append("未找到PDEBench数据文件")
        
        # 检查磁盘空间
        disk_usage = psutil.disk_usage('.')
        free_space_gb = disk_usage.free / (1024**3)
        if free_space_gb < 10:
            issues.append(f"磁盘空间不足: {free_space_gb:.1f}GB")
        
        if issues:
            logger.error("[ERROR] 环境验证失败:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        else:
            logger.info("[OK] 环境验证通过")
            return True
    
    def print_usage_instructions(self):
        """
        打印使用说明
        """
        logger.info("\n" + "="*60)
        logger.info("[INFO] 服务器使用说明")
        logger.info("="*60)
        
        if self.data_paths:
            config_file = "dynamic_config_server_auto.yaml"
            logger.info(f"\n1. 使用生成的配置文件运行训练:")
            logger.info(f"   python dynamic_resolution_trainer.py --config {config_file}")
            
            logger.info(f"\n2. 快速测试 (1轮训练, 100样本):")
            logger.info(f"   python dynamic_resolution_trainer.py --config {config_file} --epochs 1 --num_samples 100")
            
            logger.info(f"\n3. 后台运行 (推荐):")
            logger.info(f"   nohup python dynamic_resolution_trainer.py --config {config_file} > training.log 2>&1 &")
            
            logger.info(f"\n4. 监控训练进度:")
            logger.info(f"   tail -f training.log")
            
            logger.info(f"\n5. 检查GPU使用情况:")
            logger.info(f"   nvidia-smi")
            
        else:
            logger.info("\n[ERROR] 请先下载PDEBench数据集")
            logger.info("数据下载地址: https://github.com/pdebench/PDEBench")
        
        logger.info("\n" + "="*60)

def main():
    """
    主函数
    """
    logger.info("[INFO] 服务器环境配置和验证")
    logger.info("="*60)
    
    setup = ServerEnvironmentSetup()
    
    # 1. 检测系统资源
    setup.detect_system_resources()
    
    # 2. 查找数据路径
    setup.find_data_paths()
    
    # 3. 生成配置建议
    setup.generate_config_recommendations()
    
    # 4. 创建配置文件
    config_file = setup.create_server_config()
    
    # 5. 验证环境
    env_ok = setup.validate_environment()
    
    # 6. 打印使用说明
    setup.print_usage_instructions()
    
    if env_ok and config_file:
        logger.info("\n[OK] 服务器环境配置完成！")
        return True
    else:
        logger.error("\n[ERROR] 服务器环境配置失败，请检查上述问题")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)