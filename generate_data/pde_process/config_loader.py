#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置加载器

用于从YAML配置文件加载PDEBench数据处理配置

作者: AI Assistant
日期: 2025
版本: 1.0
"""

import yaml
import os
from typing import Dict, Any, Optional
import logging
from generate_data.pde_process.pdebench_data_processor import PDEBenchConfig

logger = logging.getLogger(__name__)


class ConfigLoader:
    """配置加载器类"""
    
    def __init__(self, config_file: str = "pdebench_config.yaml"):
        """
        初始化配置加载器
        
        Args:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config_data = None
        self.load_config()
    
    def load_config(self) -> bool:
        """
        加载配置文件
        
        Returns:
            加载是否成功
        """
        try:
            if not os.path.exists(self.config_file):
                logger.error(f"配置文件不存在: {self.config_file}")
                return False
            
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config_data = yaml.safe_load(f)
            
            logger.info(f"成功加载配置文件: {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"加载配置文件时出错: {str(e)}")
            return False
    
    def get_available_configs(self) -> list:
        """
        获取可用的配置名称列表
        
        Returns:
            配置名称列表
        """
        if not self.config_data:
            return []
        
        # 排除非配置项
        exclude_keys = {'supported_pdes', 'dataset_info', 'usage_recommendations', 'performance_tips'}
        
        return [key for key in self.config_data.keys() if key not in exclude_keys]
    
    def create_config(self, config_name: str = "basic") -> Optional[PDEBenchConfig]:
        """
        根据配置名称创建PDEBenchConfig对象
        
        Args:
            config_name: 配置名称
            
        Returns:
            PDEBenchConfig对象，如果失败返回None
        """
        try:
            if not self.config_data:
                logger.error("配置数据未加载")
                return None
            
            if config_name not in self.config_data:
                logger.error(f"配置 '{config_name}' 不存在")
                logger.info(f"可用配置: {self.get_available_configs()}")
                return None
            
            config_dict = self.config_data[config_name]
            
            # 创建PDEBenchConfig对象
            config = PDEBenchConfig()
            
            # 应用配置
            if 'pdebench_root' in config_dict:
                config.PDEBENCH_ROOT = config_dict['pdebench_root']
            
            if 'max_samples' in config_dict:
                config.MAX_SAMPLES = config_dict['max_samples']
            
            if 'start_time_idx' in config_dict:
                config.START_TIME_IDX = config_dict['start_time_idx']
            
            if 'end_time_idx' in config_dict:
                config.END_TIME_IDX = config_dict['end_time_idx']
            
            if 'time_step_interval' in config_dict:
                config.TIME_STEP_INTERVAL = config_dict['time_step_interval']
            
            if 'spatial_crop' in config_dict and config_dict['spatial_crop']:
                # 转换列表格式为元组格式
                crop = config_dict['spatial_crop']
                if isinstance(crop, list) and len(crop) == 2:
                    config.SPATIAL_CROP = (tuple(crop[0]), tuple(crop[1]))
            
            if 'spatial_downsample' in config_dict:
                config.SPATIAL_DOWNSAMPLE = config_dict['spatial_downsample']
            
            if 'output_file' in config_dict:
                config.OUTPUT_FILE = config_dict['output_file']
            
            if 'normalize_data' in config_dict:
                config.NORMALIZE_DATA = config_dict['normalize_data']
            
            if 'flatten_spatial' in config_dict:
                config.FLATTEN_SPATIAL = config_dict['flatten_spatial']
            
            # 更新支持的PDE类型（如果配置文件中有定义）
            if 'supported_pdes' in self.config_data:
                config.SUPPORTED_PDES.update(self.config_data['supported_pdes'])
            
            logger.info(f"成功创建配置: {config_name}")
            return config
            
        except Exception as e:
            logger.error(f"创建配置时出错: {str(e)}")
            return None
    
    def get_sequence_length(self, config_name: str = "basic") -> int:
        """
        获取指定配置的序列长度
        
        Args:
            config_name: 配置名称
            
        Returns:
            序列长度
        """
        try:
            if not self.config_data or config_name not in self.config_data:
                return 1
            
            return self.config_data[config_name].get('sequence_length', 1)
            
        except Exception as e:
            logger.error(f"获取序列长度时出错: {str(e)}")
            return 1
    
    def get_supported_pdes(self) -> Dict[str, str]:
        """
        获取支持的PDE类型
        
        Returns:
            PDE类型字典
        """
        if not self.config_data or 'supported_pdes' not in self.config_data:
            return {}
        
        return self.config_data['supported_pdes']
    
    def get_dataset_info(self, pde_type: str = None) -> Dict[str, Any]:
        """
        获取数据集信息
        
        Args:
            pde_type: PDE类型，如果为None则返回所有信息
            
        Returns:
            数据集信息字典
        """
        if not self.config_data or 'dataset_info' not in self.config_data:
            return {}
        
        dataset_info = self.config_data['dataset_info']
        
        if pde_type:
            return dataset_info.get(pde_type, {})
        else:
            return dataset_info
    
    def get_usage_recommendations(self) -> Dict[str, str]:
        """
        获取使用建议
        
        Returns:
            使用建议字典
        """
        if not self.config_data or 'usage_recommendations' not in self.config_data:
            return {}
        
        return self.config_data['usage_recommendations']
    
    def get_performance_tips(self) -> list:
        """
        获取性能优化建议
        
        Returns:
            性能建议列表
        """
        if not self.config_data or 'performance_tips' not in self.config_data:
            return []
        
        return self.config_data['performance_tips']
    
    def print_config_summary(self, config_name: str = None):
        """
        打印配置摘要
        
        Args:
            config_name: 配置名称，如果为None则打印所有配置
        """
        if not self.config_data:
            logger.error("配置数据未加载")
            return
        
        if config_name:
            if config_name in self.config_data:
                logger.info(f"=== 配置: {config_name} ===")
                config_dict = self.config_data[config_name]
                for key, value in config_dict.items():
                    logger.info(f"  {key}: {value}")
            else:
                logger.error(f"配置 '{config_name}' 不存在")
        else:
            logger.info("=== 所有可用配置 ===")
            available_configs = self.get_available_configs()
            for config in available_configs:
                logger.info(f"\n配置: {config}")
                config_dict = self.config_data[config]
                for key, value in config_dict.items():
                    logger.info(f"  {key}: {value}")
    
    def validate_config(self, config_name: str) -> bool:
        """
        验证配置的有效性
        
        Args:
            config_name: 配置名称
            
        Returns:
            配置是否有效
        """
        try:
            if not self.config_data or config_name not in self.config_data:
                logger.error(f"配置 '{config_name}' 不存在")
                return False
            
            config_dict = self.config_data[config_name]
            
            # 检查必需字段
            required_fields = ['pdebench_root', 'max_samples', 'output_file']
            for field in required_fields:
                if field not in config_dict:
                    logger.error(f"配置 '{config_name}' 缺少必需字段: {field}")
                    return False
            
            # 检查数据路径是否存在
            pdebench_root = config_dict['pdebench_root']
            if not os.path.exists(pdebench_root):
                logger.warning(f"PDEBench数据路径不存在: {pdebench_root}")
            
            # 检查数值范围
            if config_dict['max_samples'] <= 0:
                logger.error(f"max_samples必须大于0: {config_dict['max_samples']}")
                return False
            
            if 'start_time_idx' in config_dict and config_dict['start_time_idx'] < 0:
                logger.error(f"start_time_idx不能为负数: {config_dict['start_time_idx']}")
                return False
            
            if ('end_time_idx' in config_dict and 
                config_dict['end_time_idx'] is not None and 
                'start_time_idx' in config_dict and 
                config_dict['end_time_idx'] <= config_dict['start_time_idx']):
                logger.error("end_time_idx必须大于start_time_idx")
                return False
            
            logger.info(f"配置 '{config_name}' 验证通过")
            return True
            
        except Exception as e:
            logger.error(f"验证配置时出错: {str(e)}")
            return False


def main():
    """主函数：演示配置加载器的使用"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建配置加载器
    loader = ConfigLoader()
    
    # 打印所有可用配置
    logger.info("可用配置:")
    for config_name in loader.get_available_configs():
        logger.info(f"  - {config_name}")
    
    # 打印使用建议
    logger.info("\n使用建议:")
    recommendations = loader.get_usage_recommendations()
    for purpose, config_name in recommendations.items():
        logger.info(f"  {purpose}: {config_name}")
    
    # 打印性能建议
    logger.info("\n性能优化建议:")
    for tip in loader.get_performance_tips():
        logger.info(f"  - {tip}")
    
    # 演示配置创建
    logger.info("\n=== 配置创建演示 ===")
    config = loader.create_config('quick_test')
    if config:
        logger.info(f"成功创建quick_test配置")
        logger.info(f"  最大样本数: {config.MAX_SAMPLES}")
        logger.info(f"  输出文件: {config.OUTPUT_FILE}")
        logger.info(f"  归一化: {config.NORMALIZE_DATA}")
    
    # 验证配置
    logger.info("\n=== 配置验证 ===")
    for config_name in ['basic', 'quick_test', 'high_resolution']:
        is_valid = loader.validate_config(config_name)
        logger.info(f"  {config_name}: {'[OK]' if is_valid else '[FAIL]'}")


if __name__ == "__main__":
    main()