#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🔍 服务器配置验证脚本
用于验证Linux服务器上的训练环境配置是否正确
"""

import os
import sys
import yaml
import torch
import h5py
import logging
import platform
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ServerConfigValidator:
    """服务器配置验证器"""
    
    def __init__(self):
        self.errors = []
        self.warnings = []
        self.success_items = []
        
    def log_error(self, message: str):
        """记录错误"""
        self.errors.append(message)
        logger.error(f"❌ {message}")
        
    def log_warning(self, message: str):
        """记录警告"""
        self.warnings.append(message)
        logger.warning(f"⚠️ {message}")
        
    def log_success(self, message: str):
        """记录成功"""
        self.success_items.append(message)
        logger.info(f"✅ {message}")
        
    def check_python_environment(self) -> bool:
        """检查Python环境"""
        logger.info("🐍 检查Python环境...")
        
        try:
            # Python版本
            python_version = sys.version
            self.log_success(f"Python版本: {python_version.split()[0]}")
            
            # 平台信息
            platform_info = platform.platform()
            self.log_success(f"平台: {platform_info}")
            
            # 检查必要的包
            required_packages = {
                'torch': 'PyTorch',
                'yaml': 'PyYAML', 
                'h5py': 'HDF5支持',
                'numpy': 'NumPy'
            }
            
            for package, name in required_packages.items():
                try:
                    __import__(package)
                    if package == 'torch':
                        version = torch.__version__
                        self.log_success(f"{name}版本: {version}")
                    else:
                        self.log_success(f"{name}: 已安装")
                except ImportError:
                    self.log_error(f"{name}未安装")
                    
            return len(self.errors) == 0
            
        except Exception as e:
            self.log_error(f"Python环境检查失败: {e}")
            return False
            
    def check_cuda_environment(self) -> bool:
        """检查CUDA环境"""
        logger.info("🚀 检查CUDA环境...")
        
        try:
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                gpu_count = torch.cuda.device_count()
                
                self.log_success(f"CUDA版本: {cuda_version}")
                self.log_success(f"GPU数量: {gpu_count}")
                
                # 检查每个GPU
                for i in range(gpu_count):
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
                    self.log_success(f"GPU {i}: {gpu_name} ({gpu_memory:.1f}GB)")
                    
                return True
            else:
                self.log_warning("CUDA不可用，将使用CPU训练")
                return False
                
        except Exception as e:
            self.log_error(f"CUDA环境检查失败: {e}")
            return False
            
    def check_data_file(self, data_path: str) -> bool:
        """检查数据文件"""
        logger.info("📁 检查数据文件...")
        
        try:
            if not os.path.exists(data_path):
                self.log_error(f"数据文件不存在: {data_path}")
                return False
                
            # 检查文件大小
            file_size = os.path.getsize(data_path) / 1024**3  # GB
            self.log_success(f"数据文件存在: {data_path} ({file_size:.2f}GB)")
            
            # 检查文件权限
            if os.access(data_path, os.R_OK):
                self.log_success("数据文件可读")
            else:
                self.log_error("数据文件不可读，请检查权限")
                return False
                
            # 检查HDF5文件结构
            try:
                with h5py.File(data_path, 'r') as f:
                    keys = list(f.keys())
                    self.log_success(f"HDF5文件结构正常，包含键: {keys[:5]}{'...' if len(keys) > 5 else ''}")
            except Exception as e:
                self.log_error(f"HDF5文件格式错误: {e}")
                return False
                
            return True
            
        except Exception as e:
            self.log_error(f"数据文件检查失败: {e}")
            return False
            
    def check_config_file(self, config_path: str) -> Tuple[bool, Optional[Dict]]:
        """检查配置文件"""
        logger.info("📋 检查配置文件...")
        
        try:
            if not os.path.exists(config_path):
                self.log_error(f"配置文件不存在: {config_path}")
                return False, None
                
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            self.log_success(f"配置文件加载成功: {config_path}")
            
            # 检查必要的配置项
            required_sections = ['data', 'training', 'model']
            for section in required_sections:
                if section in config:
                    self.log_success(f"配置节存在: {section}")
                else:
                    self.log_error(f"配置节缺失: {section}")
                    
            # 检查数据配置
            if 'data' in config:
                data_config = config['data']
                
                # 检查数据路径
                if 'data_path' in data_config:
                    data_path = data_config['data_path']
                    self.log_success(f"数据路径配置: {data_path}")
                else:
                    self.log_error("数据路径未配置")
                    
                # 检查批次大小
                if 'dataloader' in data_config and 'batch_size' in data_config['dataloader']:
                    batch_size = data_config['dataloader']['batch_size']
                    self.log_success(f"批次大小: {batch_size}")
                    
                    # 检查批次大小是否合理
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                        if batch_size > 256 and gpu_memory < 20:
                            self.log_warning(f"批次大小({batch_size})可能过大，GPU内存仅{gpu_memory:.1f}GB")
                            
            return True, config
            
        except Exception as e:
            self.log_error(f"配置文件检查失败: {e}")
            return False, None
            
    def check_system_resources(self) -> bool:
        """检查系统资源"""
        logger.info("💻 检查系统资源...")
        
        try:
            # CPU信息
            cpu_count = os.cpu_count()
            self.log_success(f"CPU核心数: {cpu_count}")
            
            # 内存信息
            try:
                with open('/proc/meminfo', 'r') as f:
                    meminfo = f.read()
                    
                for line in meminfo.split('\n'):
                    if 'MemTotal:' in line:
                        total_mem = int(line.split()[1]) / 1024**2  # GB
                        self.log_success(f"总内存: {total_mem:.1f}GB")
                    elif 'MemAvailable:' in line:
                        avail_mem = int(line.split()[1]) / 1024**2  # GB
                        self.log_success(f"可用内存: {avail_mem:.1f}GB")
                        
            except Exception:
                self.log_warning("无法读取内存信息")
                
            # 磁盘空间
            try:
                result = subprocess.run(['df', '-h', '.'], capture_output=True, text=True)
                if result.returncode == 0:
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        disk_info = lines[1].split()
                        self.log_success(f"磁盘空间: {disk_info[3]} 可用 / {disk_info[1]} 总计")
            except Exception:
                self.log_warning("无法获取磁盘信息")
                
            return True
            
        except Exception as e:
            self.log_error(f"系统资源检查失败: {e}")
            return False
            
    def check_file_permissions(self) -> bool:
        """检查文件权限"""
        logger.info("🔐 检查文件权限...")
        
        try:
            # 检查当前目录权限
            current_dir = os.getcwd()
            if os.access(current_dir, os.W_OK):
                self.log_success(f"当前目录可写: {current_dir}")
            else:
                self.log_error(f"当前目录不可写: {current_dir}")
                
            # 检查结果目录
            results_dir = "./results"
            if not os.path.exists(results_dir):
                try:
                    os.makedirs(results_dir, exist_ok=True)
                    self.log_success(f"结果目录已创建: {results_dir}")
                except Exception as e:
                    self.log_error(f"无法创建结果目录: {e}")
                    return False
            else:
                self.log_success(f"结果目录存在: {results_dir}")
                
            return True
            
        except Exception as e:
            self.log_error(f"文件权限检查失败: {e}")
            return False
            
    def generate_report(self) -> str:
        """生成验证报告"""
        report = []
        report.append("="*60)
        report.append("🔍 服务器配置验证报告")
        report.append("="*60)
        
        if self.success_items:
            report.append("\n✅ 成功项目:")
            for item in self.success_items:
                report.append(f"  - {item}")
                
        if self.warnings:
            report.append("\n⚠️ 警告项目:")
            for item in self.warnings:
                report.append(f"  - {item}")
                
        if self.errors:
            report.append("\n❌ 错误项目:")
            for item in self.errors:
                report.append(f"  - {item}")
                
        report.append("\n" + "="*60)
        
        if self.errors:
            report.append("❌ 验证失败，请修复上述错误后重试")
        elif self.warnings:
            report.append("⚠️ 验证通过但有警告，建议检查警告项目")
        else:
            report.append("🎉 验证完全通过，可以开始训练！")
            
        report.append("="*60)
        
        return "\n".join(report)

def main():
    """主函数"""
    logger.info("🚀 开始服务器配置验证")
    logger.info("="*60)
    
    validator = ServerConfigValidator()
    
    # 配置文件路径
    config_file = "dynamic_config_server_corrected.yaml"
    
    # 执行各项检查
    validator.check_python_environment()
    validator.check_cuda_environment()
    validator.check_system_resources()
    validator.check_file_permissions()
    
    # 检查配置文件
    config_valid, config = validator.check_config_file(config_file)
    
    # 如果配置文件有效，检查数据文件
    if config_valid and config:
        data_path = config.get('data', {}).get('data_path')
        if data_path:
            validator.check_data_file(data_path)
        else:
            validator.log_error("配置文件中未找到数据路径")
    
    # 生成并显示报告
    report = validator.generate_report()
    print("\n" + report)
    
    # 如果有错误，返回非零退出码
    if validator.errors:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()