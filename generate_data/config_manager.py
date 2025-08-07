#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理工具

功能:
1. 配置文件参数复用和模板管理
2. 配置验证和参数检查
3. 动态配置生成和覆盖
4. 配置文件对比和分析
5. 批量实验配置生成

作者: AI Assistant
日期: 2025
"""

import os
import sys
import yaml
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from copy import deepcopy
import json

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    配置管理器类
    
    提供配置文件的加载、验证、修改、保存等功能
    支持YAML锚点和引用的参数复用机制
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = None
        self.templates = {}
        self.presets = {}
        
        if config_path and Path(config_path).exists():
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            self.config_path = config_path
            logger.info(f"✅ 成功加载配置文件: {config_path}")
            
            # 提取模板和预设
            self._extract_templates()
            self._extract_presets()
            
            return self.config
            
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
            raise
    
    def _extract_templates(self):
        """
        提取配置模板
        """
        if not self.config:
            return
        
        # 查找以_template结尾的配置项
        for key, value in self.config.items():
            if key.endswith('_template'):
                template_name = key.replace('_template', '')
                self.templates[template_name] = value
                logger.info(f"📋 发现配置模板: {template_name}")
    
    def _extract_presets(self):
        """
        提取预设配置
        """
        if not self.config:
            return
        
        # 提取resolution_presets
        if 'resolution_presets' in self.config:
            self.presets = self.config['resolution_presets']
            logger.info(f"🎯 发现 {len(self.presets)} 个分辨率预设")
    
    def save_config(self, output_path: str, config: Optional[Dict[str, Any]] = None):
        """
        保存配置文件
        
        Args:
            output_path: 输出文件路径
            config: 要保存的配置，默认使用当前配置
        """
        config_to_save = config or self.config
        
        if not config_to_save:
            raise ValueError("没有可保存的配置")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_to_save, f, default_flow_style=False, 
                         allow_unicode=True, indent=2, sort_keys=False)
            
            logger.info(f"✅ 配置已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ 保存配置文件失败: {e}")
            raise
    
    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """
        验证配置文件的合理性
        
        Args:
            config: 要验证的配置，默认使用当前配置
            
        Returns:
            验证是否通过
        """
        config_to_validate = config or self.config
        
        if not config_to_validate:
            logger.error("❌ 没有可验证的配置")
            return False
        
        warnings = []
        errors = []
        
        # 检查必需的配置项
        required_sections = ['data', 'training', 'model']
        for section in required_sections:
            if section not in config_to_validate:
                errors.append(f"缺少必需的配置节: {section}")
        
        # 检查数据配置
        if 'data' in config_to_validate:
            data_config = config_to_validate['data']
            
            # 检查分辨率配置
            if 'input_resolution' in data_config and 'output_resolution' in data_config:
                input_res = data_config['input_resolution']
                output_res = data_config['output_resolution']
                
                if len(input_res) != 2 or len(output_res) != 2:
                    errors.append("输入和输出分辨率必须是长度为2的列表")
                
                if input_res and output_res:
                    input_dim = input_res[0] * input_res[1]
                    output_dim = output_res[0] * output_res[1]
                    
                    if input_dim > 50000 or output_dim > 50000:
                        warnings.append(f"分辨率较高，可能导致内存问题: 输入{input_dim}, 输出{output_dim}")
            
            # 检查批次大小
            batch_size = data_config.get('batch_size', 16)
            if batch_size > 512:
                warnings.append(f"批次大小({batch_size})非常大，请确保有足够的GPU内存")
        
        # 检查模型配置
        if 'model' in config_to_validate:
            model_config = config_to_validate['model']
            
            # 估算模型参数量
            num_layers = model_config.get('num_layers', 6)
            d_model = model_config.get('d_model', 512)
            dim_feedforward = model_config.get('dim_feedforward', 2048)
            
            # 简化的参数量估算
            attention_params = num_layers * 4 * d_model * d_model
            feedforward_params = num_layers * 2 * d_model * dim_feedforward
            total_params = (attention_params + feedforward_params) / 1e6  # 转换为百万参数
            
            if total_params > 100:  # 0.1B参数限制
                warnings.append(f"模型参数量({total_params:.1f}M)可能超过0.1B限制")
        
        # 检查训练配置
        if 'training' in config_to_validate:
            training_config = config_to_validate['training']
            
            learning_rate = training_config.get('learning_rate', 0.001)
            if learning_rate > 0.01:
                warnings.append(f"学习率({learning_rate})较高，可能导致训练不稳定")
            
            epochs = training_config.get('epochs', 100)
            if epochs > 1000:
                warnings.append(f"训练轮数({epochs})较多，建议启用早停机制")
        
        # 输出验证结果
        if warnings:
            logger.warning("⚠️  配置验证警告:")
            for warning in warnings:
                logger.warning(f"  - {warning}")
        
        if errors:
            logger.error("❌ 配置验证错误:")
            for error in errors:
                logger.error(f"  - {error}")
            return False
        
        if not warnings and not errors:
            logger.info("✅ 配置验证通过")
        
        return True
    
    def override_params(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        覆盖配置参数
        
        Args:
            overrides: 要覆盖的参数字典，支持嵌套路径如 'data.batch_size'
            
        Returns:
            覆盖后的配置
        """
        if not self.config:
            raise ValueError("没有加载的配置")
        
        new_config = deepcopy(self.config)
        
        for key, value in overrides.items():
            # 支持嵌套路径
            keys = key.split('.')
            current = new_config
            
            # 导航到目标位置
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            # 设置值
            current[keys[-1]] = value
            logger.info(f"🔧 覆盖参数: {key} = {value}")
        
        return new_config
    
    def create_from_template(self, template_name: str, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        从模板创建配置
        
        Args:
            template_name: 模板名称
            overrides: 要覆盖的参数
            
        Returns:
            新的配置
        """
        if template_name not in self.templates:
            raise ValueError(f"未找到模板: {template_name}")
        
        # 从基础配置开始
        new_config = deepcopy(self.config)
        
        # 应用模板
        template = self.templates[template_name]
        self._merge_config(new_config, template)
        
        # 应用覆盖参数
        if overrides:
            new_config = self.override_params_dict(new_config, overrides)
        
        logger.info(f"📋 从模板 '{template_name}' 创建配置")
        return new_config
    
    def override_params_dict(self, config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """
        在给定配置上覆盖参数
        
        Args:
            config: 基础配置
            overrides: 要覆盖的参数
            
        Returns:
            覆盖后的配置
        """
        new_config = deepcopy(config)
        
        for key, value in overrides.items():
            keys = key.split('.')
            current = new_config
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
        
        return new_config
    
    def _merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """
        合并配置字典
        
        Args:
            base: 基础配置（会被修改）
            override: 覆盖配置
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._merge_config(base[key], value)
            else:
                base[key] = value
    
    def compare_configs(self, config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
        """
        比较两个配置的差异
        
        Args:
            config1: 配置1
            config2: 配置2
            
        Returns:
            差异报告
        """
        differences = {
            'only_in_config1': {},
            'only_in_config2': {},
            'different_values': {},
            'same_values': {}
        }
        
        self._compare_dict(config1, config2, differences, '')
        
        return differences
    
    def _compare_dict(self, dict1: Dict[str, Any], dict2: Dict[str, Any], 
                     differences: Dict[str, Any], path: str):
        """
        递归比较字典
        """
        all_keys = set(dict1.keys()) | set(dict2.keys())
        
        for key in all_keys:
            current_path = f"{path}.{key}" if path else key
            
            if key not in dict1:
                differences['only_in_config2'][current_path] = dict2[key]
            elif key not in dict2:
                differences['only_in_config1'][current_path] = dict1[key]
            elif isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                self._compare_dict(dict1[key], dict2[key], differences, current_path)
            elif dict1[key] != dict2[key]:
                differences['different_values'][current_path] = {
                    'config1': dict1[key],
                    'config2': dict2[key]
                }
            else:
                differences['same_values'][current_path] = dict1[key]
    
    def generate_experiment_configs(self, base_config: Dict[str, Any], 
                                  param_grid: Dict[str, List[Any]], 
                                  output_dir: str) -> List[str]:
        """
        生成批量实验配置
        
        Args:
            base_config: 基础配置
            param_grid: 参数网格，如 {'data.batch_size': [16, 32, 64]}
            output_dir: 输出目录
            
        Returns:
            生成的配置文件路径列表
        """
        import itertools
        
        # 生成参数组合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        param_combinations = list(itertools.product(*param_values))
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        config_paths = []
        
        for i, combination in enumerate(param_combinations):
            # 创建参数覆盖字典
            overrides = dict(zip(param_names, combination))
            
            # 生成新配置
            new_config = self.override_params_dict(base_config, overrides)
            
            # 生成文件名
            param_str = '_'.join([f"{k.split('.')[-1]}{v}" for k, v in overrides.items()])
            config_name = f"experiment_{i:03d}_{param_str}.yaml"
            config_path = output_dir / config_name
            
            # 保存配置
            self.save_config(str(config_path), new_config)
            config_paths.append(str(config_path))
            
            logger.info(f"📄 生成实验配置 {i+1}/{len(param_combinations)}: {config_name}")
        
        logger.info(f"✅ 成功生成 {len(config_paths)} 个实验配置")
        return config_paths
    
    def print_config_summary(self, config: Optional[Dict[str, Any]] = None):
        """
        打印配置摘要
        
        Args:
            config: 要分析的配置，默认使用当前配置
        """
        config_to_analyze = config or self.config
        
        if not config_to_analyze:
            logger.error("❌ 没有可分析的配置")
            return
        
        print("\n" + "="*60)
        print("📊 配置文件摘要")
        print("="*60)
        
        # 数据配置
        if 'data' in config_to_analyze:
            data_config = config_to_analyze['data']
            print(f"\n📁 数据配置:")
            print(f"  输入分辨率: {data_config.get('input_resolution', 'N/A')}")
            print(f"  输出分辨率: {data_config.get('output_resolution', 'N/A')}")
            print(f"  样本数量: {data_config.get('num_samples', 'N/A')}")
            print(f"  批次大小: {data_config.get('batch_size', 'N/A')}")
            print(f"  懒加载: {data_config.get('lazy_loading', 'N/A')}")
        
        # 训练配置
        if 'training' in config_to_analyze:
            training_config = config_to_analyze['training']
            print(f"\n🎯 训练配置:")
            print(f"  训练轮数: {training_config.get('epochs', 'N/A')}")
            print(f"  学习率: {training_config.get('learning_rate', 'N/A')}")
            print(f"  权重衰减: {training_config.get('weight_decay', 'N/A')}")
            print(f"  早停: {training_config.get('enable_early_stopping', 'N/A')}")
        
        # 模型配置
        if 'model' in config_to_analyze:
            model_config = config_to_analyze['model']
            print(f"\n🧠 模型配置:")
            print(f"  层数: {model_config.get('num_layers', 'N/A')}")
            print(f"  模型维度: {model_config.get('d_model', 'N/A')}")
            print(f"  注意力头: {model_config.get('num_heads', 'N/A')}")
            print(f"  前馈维度: {model_config.get('dim_feedforward', 'N/A')}")
            print(f"  Dropout: {model_config.get('dropout', 'N/A')}")
        
        # 设备配置
        print(f"\n💻 设备配置:")
        print(f"  设备: {config_to_analyze.get('device', 'N/A')}")
        print(f"  数据并行: {config_to_analyze.get('use_dataparallel', 'N/A')}")
        print(f"  混合精度: {config_to_analyze.get('mixed_precision', {}).get('enabled', 'N/A')}")
        
        # 数据加载配置
        if 'dataloader' in config_to_analyze:
            dataloader_config = config_to_analyze['dataloader']
            print(f"\n⚡ 数据加载配置:")
            print(f"  工作进程: {dataloader_config.get('num_workers', 'N/A')}")
            print(f"  预取因子: {dataloader_config.get('prefetch_factor', 'N/A')}")
            print(f"  固定内存: {dataloader_config.get('pin_memory', 'N/A')}")
        
        print("\n" + "="*60)

def main():
    """
    命令行主函数
    """
    parser = argparse.ArgumentParser(description='配置管理工具')
    parser.add_argument('--config', '-c', type=str, help='配置文件路径')
    parser.add_argument('--action', '-a', type=str, required=True,
                       choices=['validate', 'summary', 'compare', 'override', 'template', 'experiment'],
                       help='操作类型')
    parser.add_argument('--output', '-o', type=str, help='输出文件路径')
    parser.add_argument('--template', '-t', type=str, help='模板名称')
    parser.add_argument('--overrides', type=str, help='参数覆盖（JSON格式）')
    parser.add_argument('--compare-with', type=str, help='比较的第二个配置文件')
    parser.add_argument('--param-grid', type=str, help='参数网格（JSON格式）')
    parser.add_argument('--output-dir', type=str, help='输出目录')
    
    args = parser.parse_args()
    
    # 创建配置管理器
    config_manager = ConfigManager(args.config)
    
    try:
        if args.action == 'validate':
            # 验证配置
            if not args.config:
                logger.error("❌ 验证操作需要指定配置文件")
                return
            
            is_valid = config_manager.validate_config()
            if is_valid:
                print("✅ 配置验证通过")
            else:
                print("❌ 配置验证失败")
                sys.exit(1)
        
        elif args.action == 'summary':
            # 打印配置摘要
            if not args.config:
                logger.error("❌ 摘要操作需要指定配置文件")
                return
            
            config_manager.print_config_summary()
        
        elif args.action == 'compare':
            # 比较配置
            if not args.config or not args.compare_with:
                logger.error("❌ 比较操作需要指定两个配置文件")
                return
            
            config2 = ConfigManager(args.compare_with).config
            differences = config_manager.compare_configs(config_manager.config, config2)
            
            print("\n📊 配置比较结果:")
            print(f"仅在配置1中: {len(differences['only_in_config1'])} 项")
            print(f"仅在配置2中: {len(differences['only_in_config2'])} 项")
            print(f"值不同: {len(differences['different_values'])} 项")
            print(f"值相同: {len(differences['same_values'])} 项")
            
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    json.dump(differences, f, indent=2, ensure_ascii=False)
                logger.info(f"✅ 比较结果已保存到: {args.output}")
        
        elif args.action == 'override':
            # 参数覆盖
            if not args.config or not args.overrides:
                logger.error("❌ 覆盖操作需要指定配置文件和覆盖参数")
                return
            
            overrides = json.loads(args.overrides)
            new_config = config_manager.override_params(overrides)
            
            if args.output:
                config_manager.save_config(args.output, new_config)
            else:
                print(yaml.dump(new_config, default_flow_style=False, allow_unicode=True))
        
        elif args.action == 'template':
            # 从模板创建配置
            if not args.config or not args.template:
                logger.error("❌ 模板操作需要指定配置文件和模板名称")
                return
            
            overrides = json.loads(args.overrides) if args.overrides else None
            new_config = config_manager.create_from_template(args.template, overrides)
            
            if args.output:
                config_manager.save_config(args.output, new_config)
            else:
                print(yaml.dump(new_config, default_flow_style=False, allow_unicode=True))
        
        elif args.action == 'experiment':
            # 生成实验配置
            if not args.config or not args.param_grid or not args.output_dir:
                logger.error("❌ 实验操作需要指定配置文件、参数网格和输出目录")
                return
            
            param_grid = json.loads(args.param_grid)
            config_paths = config_manager.generate_experiment_configs(
                config_manager.config, param_grid, args.output_dir
            )
            
            print(f"✅ 成功生成 {len(config_paths)} 个实验配置")
            for path in config_paths:
                print(f"  - {path}")
    
    except Exception as e:
        logger.error(f"❌ 操作失败: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()