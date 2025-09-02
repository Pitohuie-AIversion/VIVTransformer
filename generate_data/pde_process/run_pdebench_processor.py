#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDEBench数据处理器主运行脚本

提供命令行接口来处理PDEBench数据集，支持多种配置和PDE类型

使用方法:
    # 使用默认配置处理Darcy Flow数据
    python run_pdebench_processor.py --pde_type darcy
    
    # 使用快速测试配置
    python run_pdebench_processor.py --config quick_test --pde_type darcy
    
    # 处理多种PDE类型
    python run_pdebench_processor.py --config basic --pde_type darcy,burgers
    
    # 查看可用配置
    python run_pdebench_processor.py --list_configs
    
    # 查看数据集信息
    python run_pdebench_processor.py --dataset_info

作者: AI Assistant
日期: 2025
版本: 1.0
"""

import os
import sys
import argparse
import logging
from typing import List, Optional
from pathlib import Path

# 添加当前目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pdebench_data_processor import PDEBenchProcessor, PDEBenchConfig
from config_loader import ConfigLoader

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pdebench_processor.log')
    ]
)
logger = logging.getLogger(__name__)


class PDEBenchRunner:
    """PDEBench数据处理运行器"""
    
    def __init__(self, config_file: str = "pdebench_config.yaml"):
        """
        初始化运行器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_loader = ConfigLoader(config_file)
        self.processor = None
        
    def list_available_configs(self):
        """列出所有可用配置"""
        logger.info("=== 可用配置 ===")
        configs = self.config_loader.get_available_configs()
        
        for config_name in configs:
            logger.info(f"\n配置名称: {config_name}")
            config_dict = self.config_loader.config_data[config_name]
            
            # 显示关键参数
            key_params = ['max_samples', 'end_time_idx', 'spatial_downsample', 'sequence_length']
            for param in key_params:
                if param in config_dict:
                    logger.info(f"  {param}: {config_dict[param]}")
        
        # 显示使用建议
        logger.info("\n=== 使用建议 ===")
        recommendations = self.config_loader.get_usage_recommendations()
        for purpose, config_name in recommendations.items():
            logger.info(f"  {purpose}: {config_name}")
    
    def show_dataset_info(self, pde_type: str = None):
        """显示数据集信息"""
        logger.info("=== 数据集信息 ===")
        
        supported_pdes = self.config_loader.get_supported_pdes()
        dataset_info = self.config_loader.get_dataset_info()
        
        if pde_type:
            if pde_type in supported_pdes:
                logger.info(f"\nPDE类型: {pde_type}")
                logger.info(f"文件名: {supported_pdes[pde_type]}")
                
                if pde_type in dataset_info:
                    info = dataset_info[pde_type]
                    for key, value in info.items():
                        logger.info(f"  {key}: {value}")
            else:
                logger.error(f"不支持的PDE类型: {pde_type}")
        else:
            for pde_type, filename in supported_pdes.items():
                logger.info(f"\nPDE类型: {pde_type}")
                logger.info(f"文件名: {filename}")
                
                if pde_type in dataset_info:
                    info = dataset_info[pde_type]
                    for key, value in info.items():
                        logger.info(f"  {key}: {value}")
    
    def validate_pde_types(self, pde_types: List[str]) -> List[str]:
        """
        验证PDE类型的有效性
        
        Args:
            pde_types: PDE类型列表
            
        Returns:
            有效的PDE类型列表
        """
        supported_pdes = self.config_loader.get_supported_pdes()
        valid_types = []
        
        for pde_type in pde_types:
            if pde_type in supported_pdes:
                valid_types.append(pde_type)
            else:
                logger.warning(f"不支持的PDE类型: {pde_type}")
        
        if not valid_types:
            logger.error("没有有效的PDE类型")
            logger.info(f"支持的类型: {list(supported_pdes.keys())}")
        
        return valid_types
    
    def check_data_availability(self, pde_types: List[str], config_name: str) -> bool:
        """
        检查数据文件是否可用
        
        Args:
            pde_types: PDE类型列表
            config_name: 配置名称
            
        Returns:
            数据是否可用
        """
        config = self.config_loader.create_config(config_name)
        if not config:
            return False
        
        supported_pdes = self.config_loader.get_supported_pdes()
        missing_files = []
        
        for pde_type in pde_types:
            if pde_type in supported_pdes:
                filename = supported_pdes[pde_type]
                file_path = os.path.join(config.PDEBENCH_ROOT, filename)
                
                if not os.path.exists(file_path):
                    missing_files.append(file_path)
                else:
                    logger.info(f"[OK] 找到数据文件: {filename}")
        
        if missing_files:
            logger.error("以下数据文件不存在:")
            for file_path in missing_files:
                logger.error(f"  [FAIL] {file_path}")
            return False
        
        return True
    
    def process_data(self, pde_types: List[str], config_name: str = "basic", 
                    output_file: str = None, max_samples: int = None) -> bool:
        """
        处理数据
        
        Args:
            pde_types: PDE类型列表
            config_name: 配置名称
            output_file: 输出文件名
            
        Returns:
            处理是否成功
        """
        try:
            # 验证配置
            if not self.config_loader.validate_config(config_name):
                logger.error(f"配置验证失败: {config_name}")
                return False
            
            # 创建配置
            config = self.config_loader.create_config(config_name)
            if not config:
                logger.error(f"创建配置失败: {config_name}")
                return False
            
            # 设置输出文件
            if output_file:
                config.OUTPUT_FILE = output_file
            
            # 设置最大样本数
            if max_samples:
                config.MAX_SAMPLES = max_samples
            
            # 验证PDE类型
            valid_pde_types = self.validate_pde_types(pde_types)
            if not valid_pde_types:
                return False
            
            # 检查数据可用性
            if not self.check_data_availability(valid_pde_types, config_name):
                logger.error("数据文件检查失败")
                return False
            
            # 创建处理器
            self.processor = PDEBenchProcessor(config)
            
            # 获取序列长度
            sequence_length = self.config_loader.get_sequence_length(config_name)
            
            logger.info(f"开始处理数据...")
            logger.info(f"  配置: {config_name}")
            logger.info(f"  PDE类型: {valid_pde_types}")
            logger.info(f"  序列长度: {sequence_length}")
            logger.info(f"  输出文件: {config.OUTPUT_FILE}")
            
            # 处理每种PDE类型
            success_count = 0
            for pde_type in valid_pde_types:
                logger.info(f"\n正在处理 {pde_type} 数据...")
                
                if self.processor.process_pde_dataset(pde_type, sequence_length):
                    success_count += 1
                    logger.info(f"[OK] {pde_type} 处理成功")
                else:
                    logger.error(f"[FAIL] {pde_type} 处理失败")
            
            if success_count > 0:
                # 保存处理后的数据
                if self.processor.save_processed_data():
                    logger.info(f"\n[OK] 数据处理完成！")
                    logger.info(f"  成功处理: {success_count}/{len(valid_pde_types)} 种PDE类型")
                    logger.info(f"  输出文件: {config.OUTPUT_FILE}")
                    return True
                else:
                    logger.error("保存数据失败")
                    return False
            else:
                logger.error("所有PDE类型处理失败")
                return False
                
        except Exception as e:
            logger.error(f"处理数据时出错: {str(e)}")
            return False
    
    def quick_test(self, pde_type: str = "darcy") -> bool:
        """
        快速测试
        
        Args:
            pde_type: PDE类型
            
        Returns:
            测试是否成功
        """
        logger.info("=== 快速测试 ===")
        
        # 使用快速测试配置
        success = self.process_data([pde_type], "quick_test", "quick_test_output.pt")
        
        if success:
            # 验证输出文件
            if os.path.exists("quick_test_output.pt"):
                import torch
                try:
                    data = torch.load("quick_test_output.pt")
                    logger.info("[OK] 快速测试成功")
                    logger.info(f"  处理的PDE类型: {data['pde_types']}")
                    
                    # 清理测试文件
                    os.remove("quick_test_output.pt")
                    logger.info("  测试文件已清理")
                    
                    return True
                except Exception as e:
                    logger.error(f"验证输出文件时出错: {str(e)}")
                    return False
            else:
                logger.error("输出文件未生成")
                return False
        else:
            logger.error("快速测试失败")
            return False


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='PDEBench数据处理器',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 基本使用
  python run_pdebench_processor.py --pde_type darcy
  
  # 使用特定配置
  python run_pdebench_processor.py --config quick_test --pde_type darcy
  
  # 处理多种PDE类型
  python run_pdebench_processor.py --config basic --pde_type darcy,burgers
  
  # 查看可用配置
  python run_pdebench_processor.py --list_configs
  
  # 快速测试
  python run_pdebench_processor.py --quick_test
        """
    )
    
    # 主要操作参数
    parser.add_argument('--pde_type', type=str,
                       help='PDE类型，多个类型用逗号分隔 (darcy,burgers,advection等)')
    parser.add_argument('--config', type=str, default='basic',
                       help='配置名称 (默认: basic)')
    parser.add_argument('--output_file', type=str,
                       help='输出文件名 (覆盖配置中的设置)')
    parser.add_argument('--config_file', type=str, default='pdebench_config.yaml',
                       help='配置文件路径 (默认: pdebench_config.yaml)')
    parser.add_argument('--max_samples', type=int,
                       help='最大样本数量 (覆盖配置中的设置)')
    
    # 信息查看参数
    parser.add_argument('--list_configs', action='store_true',
                       help='列出所有可用配置')
    parser.add_argument('--dataset_info', type=str, nargs='?', const='all',
                       help='显示数据集信息，可指定特定PDE类型')
    parser.add_argument('--quick_test', action='store_true',
                       help='运行快速测试')
    
    # 日志参数
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='详细输出')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='静默模式')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()
    
    # 设置日志级别
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    elif args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # 创建运行器
    try:
        runner = PDEBenchRunner(args.config_file)
    except Exception as e:
        logger.error(f"初始化运行器失败: {str(e)}")
        return 1
    
    # 处理不同的操作
    try:
        if args.list_configs:
            runner.list_available_configs()
            return 0
        
        if args.dataset_info:
            pde_type = None if args.dataset_info == 'all' else args.dataset_info
            runner.show_dataset_info(pde_type)
            return 0
        
        if args.quick_test:
            pde_type = args.pde_type.split(',')[0] if args.pde_type else 'darcy'
            success = runner.quick_test(pde_type)
            return 0 if success else 1
        
        if not args.pde_type:
            logger.error("请指定PDE类型 (--pde_type) 或使用其他操作选项")
            logger.info("使用 --help 查看帮助信息")
            return 1
        
        # 解析PDE类型
        pde_types = [pde.strip() for pde in args.pde_type.split(',')]
        
        # 处理数据
        success = runner.process_data(pde_types, args.config, args.output_file, args.max_samples)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("\n用户中断操作")
        return 1
    except Exception as e:
        logger.error(f"运行时出错: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)