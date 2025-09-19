#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一配置管理器

这个模块提供了一个统一的配置管理系统，用于管理所有模型、训练、数据处理等相关的配置参数。
支持配置验证、参数继承、动态调整等功能。
"""

import yaml
import json
import copy
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass, field
from enum import Enum

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConfigType(Enum):
    """配置类型枚举"""
    MODEL = "model"
    TRAINING = "training"
    DATA = "data"
    LOSS = "loss"
    OPTIMIZER = "optimizer"
    SCHEDULER = "scheduler"
    DEVICE = "device"
    LOGGING = "logging"
    VISUALIZATION = "visualization"
    MIXED_PRECISION = "mixed_precision"
    SVD_PROJECTION = "svd_projection"
    DOWNSAMPLING = "downsampling"

@dataclass
class ConfigSchema:
    """配置模式定义"""
    name: str
    type: ConfigType
    required_fields: List[str] = field(default_factory=list)
    optional_fields: Dict[str, Any] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)

class UnifiedConfigManager:
    """
    统一配置管理器
    
    提供统一的配置管理功能，包括:
    - 配置加载和保存
    - 参数验证和补全
    - 配置继承和覆盖
    - 动态参数调整
    - 兼容性检查
    """
    
    def __init__(self):
        self.config_schemas = self._init_config_schemas()
        self.config_cache = {}
        self.validation_enabled = True
        
    def _init_config_schemas(self) -> Dict[str, ConfigSchema]:
        """初始化配置模式"""
        schemas = {}
        
        # 模型配置模式
        schemas['model'] = ConfigSchema(
            name='model',
            type=ConfigType.MODEL,
            required_fields=['input_dim', 'output_dim'],
            optional_fields={
                'model_type': 'enhanced_transformer_1d',
                'd_model': 256,
                'num_heads': 8,
                'num_layers': 6,
                'attention_type': 'sge',
                'pe_type': 'learnable_1d',
                'max_time_steps': 1,
                'seq_len': 32,
                'input_hw': None,
                'output_head_type': 'global',
                'out_channels_per_token': None,
                'hidden_dims': [256, 512, 256],
                'activation': 'gelu',
                'dropout': 0.1,
                'use_fourier': True,
                'modes': 16,
                'width': 64,
                'features': [64, 128, 256, 512],
                'bilinear': True,
                'physics_loss_weight': 1.0
            },
            validation_rules={
                'input_dim': {'type': int, 'min': 1},
                'output_dim': {'type': int, 'min': 1},
                'd_model': {'type': int, 'min': 32, 'max': 2048},
                'num_heads': {'type': int, 'min': 1, 'max': 32},
                'num_layers': {'type': int, 'min': 1, 'max': 24},
                'dropout': {'type': float, 'min': 0.0, 'max': 0.9}
            }
        )
        
        # 训练配置模式
        schemas['training'] = ConfigSchema(
            name='training',
            type=ConfigType.TRAINING,
            required_fields=[],
            optional_fields={
                'epochs': 50,
                'learning_rate': 1e-3,
                'weight_decay': 0.0,
                'batch_size': 32,
                'patience': 15,
                'gradient_clip_val': 1.0,
                'accumulate_grad_batches': 1,
                'val_check_interval': 1.0,
                'log_every_n_steps': 50,
                'save_top_k': 3,
                'monitor': 'val_loss',
                'mode': 'min'
            },
            validation_rules={
                'epochs': {'type': int, 'min': 1},
                'learning_rate': {'type': float, 'min': 1e-6, 'max': 1.0},
                'batch_size': {'type': int, 'min': 1},
                'patience': {'type': int, 'min': 1}
            }
        )
        
        # 数据配置模式
        schemas['data'] = ConfigSchema(
            name='data',
            type=ConfigType.DATA,
            required_fields=['data_path'],
            optional_fields={
                'input_resolution': (64, 64),
                'output_resolution': (64, 64),
                'num_samples': 100,
                'crop_mode': 'center',
                'normalize_data': True,
                'lazy_loading': False,
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'shuffle': True,
                'num_workers': 4,
                'pin_memory': True,
                'drop_last': False
            },
            validation_rules={
                'num_samples': {'type': int, 'min': 1},
                'train_split': {'type': float, 'min': 0.1, 'max': 0.9},
                'val_split': {'type': float, 'min': 0.05, 'max': 0.5},
                'test_split': {'type': float, 'min': 0.05, 'max': 0.5}
            }
        )
        
        # 损失函数配置模式
        schemas['loss'] = ConfigSchema(
            name='loss',
            type=ConfigType.LOSS,
            required_fields=[],
            optional_fields={
                'base_weight': 0.8,
                'svd_weights': [0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.01, 0.01, 0.01, 0.01],
                'topk': 10,
                'svd_loss_enabled': True,
                'enhanced': {
                    'enabled': True,
                    'mixed_precision_mode': False,
                    'adaptive_weights': True,
                    'fallback_level': 2,
                    'enable_monitoring': True,
                    'force_svd': False
                }
            },
            validation_rules={
                'base_weight': {'type': float, 'min': 0.0, 'max': 1.0},
                'topk': {'type': int, 'min': 1, 'max': 50}
            }
        )
        
        # 优化器配置模式
        schemas['optimizer'] = ConfigSchema(
            name='optimizer',
            type=ConfigType.OPTIMIZER,
            required_fields=[],
            optional_fields={
                'type': 'adam',
                'betas': (0.9, 0.999),
                'eps': 1e-8,
                'amsgrad': False,
                'momentum': 0.9
            },
            validation_rules={
                'type': {'type': str, 'choices': ['adam', 'adamw', 'sgd']},
                'eps': {'type': float, 'min': 1e-10, 'max': 1e-3}
            }
        )
        
        # 调度器配置模式
        schemas['scheduler'] = ConfigSchema(
            name='scheduler',
            type=ConfigType.SCHEDULER,
            required_fields=[],
            optional_fields={
                'enabled': False,
                'type': 'cosine',
                'T_max': 50,
                'step_size': 20,
                'gamma': 0.5
            },
            validation_rules={
                'type': {'type': str, 'choices': ['cosine', 'step', 'exponential']},
                'T_max': {'type': int, 'min': 1},
                'step_size': {'type': int, 'min': 1},
                'gamma': {'type': float, 'min': 0.01, 'max': 0.99}
            }
        )
        
        # 设备配置模式
        schemas['device'] = ConfigSchema(
            name='device',
            type=ConfigType.DEVICE,
            required_fields=[],
            optional_fields={
                'use_cuda': True,
                'device_id': 0,
                'use_dataparallel': False,
                'mixed_precision': False
            }
        )
        
        # 混合精度配置模式
        schemas['mixed_precision'] = ConfigSchema(
            name='mixed_precision',
            type=ConfigType.MIXED_PRECISION,
            required_fields=[],
            optional_fields={
                'enabled': False,
                'opt_level': 'O1',
                'loss_scale': 'dynamic',
                'keep_batchnorm_fp32': True
            }
        )
        
        # SVD投影配置模式
        schemas['svd_projection'] = ConfigSchema(
            name='svd_projection',
            type=ConfigType.SVD_PROJECTION,
            required_fields=[],
            optional_fields={
                'enabled': False,
                'n_components': 50,
                'explained_variance_threshold': 0.95,
                'batch_size': 1000,
                'random_state': 42
            }
        )
        
        # 降采样配置模式
        schemas['downsampling'] = ConfigSchema(
            name='downsampling',
            type=ConfigType.DOWNSAMPLING,
            required_fields=[],
            optional_fields={
                'enabled': False,
                'method': 'bilinear',
                'target_resolution': (32, 32),
                'preserve_aspect_ratio': True
            }
        )
        
        # 可视化配置模式
        schemas['visualization'] = ConfigSchema(
            name='visualization',
            type=ConfigType.VISUALIZATION,
            required_fields=[],
            optional_fields={
                'enabled': True,
                'save_plots': True,
                'plot_format': 'png',
                'dpi': 300,
                'figsize': (10, 6)
            }
        )
        
        # 日志配置模式
        schemas['logging'] = ConfigSchema(
            name='logging',
            type=ConfigType.LOGGING,
            required_fields=[],
            optional_fields={
                'level': 'INFO',
                'save_to_file': True,
                'log_dir': './logs',
                'max_file_size': '10MB',
                'backup_count': 5
            }
        )
        
        return schemas
    
    def validate_config_section(self, section_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证配置段
        
        Args:
            section_name: 配置段名称
            config: 配置字典
            
        Returns:
            验证并补全后的配置
        """
        if not self.validation_enabled:
            return config
        
        if section_name not in self.config_schemas:
            logger.warning(f"未知的配置段: {section_name}")
            return config
        
        schema = self.config_schemas[section_name]
        validated_config = copy.deepcopy(config)
        
        # 检查必需字段
        for field in schema.required_fields:
            if field not in validated_config:
                raise ValueError(f"配置段 {section_name} 缺少必需字段: {field}")
        
        # 补全可选字段
        for field, default_value in schema.optional_fields.items():
            if field not in validated_config:
                validated_config[field] = copy.deepcopy(default_value)
        
        # 验证字段值
        for field, value in validated_config.items():
            if field in schema.validation_rules:
                self._validate_field(section_name, field, value, schema.validation_rules[field])
        
        return validated_config
    
    def _validate_field(self, section_name: str, field_name: str, value: Any, rules: Dict[str, Any]):
        """
        验证单个字段
        
        Args:
            section_name: 配置段名称
            field_name: 字段名称
            value: 字段值
            rules: 验证规则
        """
        # 类型检查
        if 'type' in rules:
            expected_type = rules['type']
            if not isinstance(value, expected_type):
                raise TypeError(f"配置段 {section_name}.{field_name} 期望类型 {expected_type.__name__}，实际类型 {type(value).__name__}")
        
        # 数值范围检查
        if isinstance(value, (int, float)):
            if 'min' in rules and value < rules['min']:
                raise ValueError(f"配置段 {section_name}.{field_name} 值 {value} 小于最小值 {rules['min']}")
            if 'max' in rules and value > rules['max']:
                raise ValueError(f"配置段 {section_name}.{field_name} 值 {value} 大于最大值 {rules['max']}")
        
        # 选择检查
        if 'choices' in rules:
            if value not in rules['choices']:
                raise ValueError(f"配置段 {section_name}.{field_name} 值 {value} 不在允许的选择中: {rules['choices']}")
    
    def load_config(self, config_path: Union[str, Path]) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            加载的配置字典
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 根据文件扩展名选择加载方式
        if config_path.suffix.lower() in ['.yaml', '.yml']:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix.lower() == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")
        
        # 验证配置
        validated_config = self.validate_full_config(config)
        
        # 缓存配置
        self.config_cache[str(config_path)] = validated_config
        
        logger.info(f"成功加载配置文件: {config_path}")
        return validated_config
    
    def save_config(self, config: Dict[str, Any], config_path: Union[str, Path], format: str = 'yaml'):
        """
        保存配置文件
        
        Args:
            config: 配置字典
            config_path: 保存路径
            format: 保存格式 ('yaml' 或 'json')
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() in ['yaml', 'yml']:
            with open(config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True, indent=2)
        elif format.lower() == 'json':
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, ensure_ascii=False)
        else:
            raise ValueError(f"不支持的保存格式: {format}")
        
        logger.info(f"配置已保存到: {config_path}")
    
    def validate_full_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证完整配置
        
        Args:
            config: 完整配置字典
            
        Returns:
            验证并补全后的配置
        """
        validated_config = {}
        
        # 验证每个配置段
        for section_name, section_config in config.items():
            if isinstance(section_config, dict):
                validated_config[section_name] = self.validate_config_section(section_name, section_config)
            else:
                validated_config[section_name] = section_config
        
        # 添加缺失的配置段（使用默认值）
        for schema_name, schema in self.config_schemas.items():
            if schema_name not in validated_config:
                validated_config[schema_name] = self.validate_config_section(schema_name, {})
        
        # 验证配置间的依赖关系
        self._validate_config_dependencies(validated_config)
        
        return validated_config
    
    def _validate_config_dependencies(self, config: Dict[str, Any]):
        """
        验证配置间的依赖关系
        
        Args:
            config: 完整配置字典
        """
        # 检查数据分割比例总和
        if 'data' in config:
            data_config = config['data']
            total_split = data_config.get('train_split', 0.7) + \
                         data_config.get('val_split', 0.15) + \
                         data_config.get('test_split', 0.15)
            
            if abs(total_split - 1.0) > 1e-6:
                raise ValueError(f"数据分割比例总和应为1.0，实际为: {total_split}")
        
        # 检查模型维度与数据分辨率的兼容性
        if 'model' in config and 'data' in config:
            model_config = config['model']
            data_config = config['data']
            
            input_resolution = data_config.get('input_resolution')
            output_resolution = data_config.get('output_resolution')
            
            if input_resolution and output_resolution:
                if len(input_resolution) != len(output_resolution):
                    logger.warning("输入和输出分辨率维度不匹配")
        
        # 检查混合精度配置
        if 'mixed_precision' in config and config['mixed_precision'].get('enabled', False):
            if 'device' in config and not config['device'].get('use_cuda', True):
                logger.warning("混合精度训练需要CUDA支持")
    
    def merge_configs(self, base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        合并配置（覆盖模式）
        
        Args:
            base_config: 基础配置
            override_config: 覆盖配置
            
        Returns:
            合并后的配置
        """
        merged_config = copy.deepcopy(base_config)
        
        def deep_merge(base: Dict[str, Any], override: Dict[str, Any]):
            for key, value in override.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_merge(base[key], value)
                else:
                    base[key] = copy.deepcopy(value)
        
        deep_merge(merged_config, override_config)
        return merged_config
    
    def get_config_template(self, sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        获取配置模板
        
        Args:
            sections: 需要的配置段列表，None表示所有段
            
        Returns:
            配置模板
        """
        template = {}
        
        target_sections = sections or list(self.config_schemas.keys())
        
        for section_name in target_sections:
            if section_name in self.config_schemas:
                schema = self.config_schemas[section_name]
                section_template = {}
                
                # 添加必需字段（占位符）
                for field in schema.required_fields:
                    section_template[field] = f"<REQUIRED_{field.upper()}>"
                
                # 添加可选字段（默认值）
                section_template.update(copy.deepcopy(schema.optional_fields))
                
                template[section_name] = section_template
        
        return template
    
    def update_config_parameter(self, config: Dict[str, Any], parameter_path: str, value: Any) -> Dict[str, Any]:
        """
        更新配置参数
        
        Args:
            config: 配置字典
            parameter_path: 参数路径，如 'model.d_model' 或 'training.learning_rate'
            value: 新值
            
        Returns:
            更新后的配置
        """
        updated_config = copy.deepcopy(config)
        
        # 解析参数路径
        path_parts = parameter_path.split('.')
        
        # 导航到目标位置
        current = updated_config
        for part in path_parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # 设置值
        current[path_parts[-1]] = value
        
        # 验证更新后的配置
        if self.validation_enabled:
            section_name = path_parts[0]
            if section_name in self.config_schemas:
                updated_config[section_name] = self.validate_config_section(
                    section_name, updated_config[section_name]
                )
        
        logger.info(f"参数 {parameter_path} 已更新为: {value}")
        return updated_config
    
    def get_parameter_info(self, parameter_path: str) -> Dict[str, Any]:
        """
        获取参数信息
        
        Args:
            parameter_path: 参数路径
            
        Returns:
            参数信息字典
        """
        path_parts = parameter_path.split('.')
        section_name = path_parts[0]
        field_name = path_parts[1] if len(path_parts) > 1 else None
        
        if section_name not in self.config_schemas:
            return {'error': f'未知的配置段: {section_name}'}
        
        schema = self.config_schemas[section_name]
        
        if not field_name:
            return {
                'section': section_name,
                'type': schema.type.value,
                'required_fields': schema.required_fields,
                'optional_fields': list(schema.optional_fields.keys())
            }
        
        info = {
            'section': section_name,
            'field': field_name,
            'required': field_name in schema.required_fields
        }
        
        if field_name in schema.optional_fields:
            info['default_value'] = schema.optional_fields[field_name]
        
        if field_name in schema.validation_rules:
            info['validation_rules'] = schema.validation_rules[field_name]
        
        return info
    
    def disable_validation(self):
        """禁用配置验证"""
        self.validation_enabled = False
        logger.info("配置验证已禁用")
    
    def enable_validation(self):
        """启用配置验证"""
        self.validation_enabled = True
        logger.info("配置验证已启用")

# 全局配置管理器实例
_global_config_manager = None

def get_global_config_manager() -> UnifiedConfigManager:
    """获取全局配置管理器实例"""
    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = UnifiedConfigManager()
    return _global_config_manager

def load_config_file(config_path: Union[str, Path]) -> Dict[str, Any]:
    """便捷函数：加载配置文件"""
    manager = get_global_config_manager()
    return manager.load_config(config_path)

def create_config_template(sections: Optional[List[str]] = None) -> Dict[str, Any]:
    """便捷函数：创建配置模板"""
    manager = get_global_config_manager()
    return manager.get_config_template(sections)

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """便捷函数：验证配置"""
    manager = get_global_config_manager()
    return manager.validate_full_config(config)

if __name__ == "__main__":
    # 测试代码
    manager = UnifiedConfigManager()
    
    # 创建配置模板
    template = manager.get_config_template(['model', 'training', 'data'])
    print("配置模板:")
    print(yaml.dump(template, sort_keys=False, allow_unicode=True, indent=2))
    
    # 测试配置验证
    test_config = {
        'model': {
            'input_dim': 64,
            'output_dim': 32,
            'd_model': 256
        },
        'training': {
            'epochs': 100,
            'learning_rate': 1e-3
        }
    }
    
    try:
        validated = manager.validate_full_config(test_config)
        print("\n验证成功！")
    except Exception as e:
        print(f"\n验证失败: {e}")