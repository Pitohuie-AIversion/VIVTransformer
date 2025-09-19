#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一模型工厂 - 整合所有增强模型

这个模块提供了一个统一的接口来创建和管理所有增强模型，
包括Transformer、FNO、MLP、PINN、UNet等，并支持动态配置参数。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'models'))

# 导入所有增强模型
try:
    from enhanced_transformer import (
        EnhancedTransformer1d, EnhancedTransformer2d,
        create_enhanced_transformer1d, create_enhanced_transformer2d
    )
    TRANSFORMER_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Transformer模块导入失败: {e}")
    TRANSFORMER_AVAILABLE = False

try:
    from enhanced_fno import (
        EnhancedFNO1d, EnhancedFNO2d
    )
    # 导入修复后的FNO模型
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'generate_data'))
    from fixed_fno_model import FixedEnhancedFNO2d, create_fixed_fno
    FNO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"FNO模块导入失败: {e}")
    FNO_AVAILABLE = False

try:
    from enhanced_mlp import (
        EnhancedMLP, EnhancedMLP1d, EnhancedMLP2d
    )
    MLP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MLP模块导入失败: {e}")
    MLP_AVAILABLE = False

try:
    from enhanced_pinn import (
        EnhancedPINN, EnhancedPINN1d, EnhancedPINN2d
    )
    PINN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"PINN模块导入失败: {e}")
    PINN_AVAILABLE = False

try:
    from enhanced_unet import (
        EnhancedUNet1d, EnhancedUNet2d, EnhancedUNet3d
    )
    UNET_AVAILABLE = True
except ImportError as e:
    logging.warning(f"UNet模块导入失败: {e}")
    UNET_AVAILABLE = False

try:
    from unified_interface import BaseEnhancedModel, ModelRegistry
    INTERFACE_AVAILABLE = True
except ImportError as e:
    logging.warning(f"统一接口模块导入失败: {e}")
    INTERFACE_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedModelFactory:
    """
    统一模型工厂类
    
    提供统一的接口来创建和管理所有增强模型，支持:
    - 动态模型创建
    - 参数配置验证
    - 模型兼容性检查
    - 自动维度推断
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.supported_models = self._get_supported_models()
        logger.info(f"初始化统一模型工厂，支持的模型: {list(self.supported_models.keys())}")
    
    def _get_supported_models(self) -> Dict[str, Dict[str, Any]]:
        """获取所有支持的模型配置"""
        models = {}
        
        if TRANSFORMER_AVAILABLE:
            models.update({
                'enhanced_transformer_1d': {
                    'class': EnhancedTransformer1d,
                    'creator': create_enhanced_transformer1d,
                    'dimensions': 1,
                    'supports_sparse': True,
                    'default_config': {
                        'd_model': 256,
                        'num_heads': 8,
                        'num_layers': 6,
                        'attention_type': 'sge',
                        'pe_type': 'learnable_1d'
                    }
                },
                'enhanced_transformer_2d': {
                    'class': EnhancedTransformer2d,
                    'creator': create_enhanced_transformer2d,
                    'dimensions': 2,
                    'supports_sparse': True,
                    'default_config': {
                        'd_model': 256,
                        'num_heads': 8,
                        'num_layers': 6,
                        'attention_type': 'sge',
                        'pe_type': 'learnable_2d'
                    }
                }
            })
        
        if FNO_AVAILABLE:
            models.update({
                'enhanced_fno_1d': {
                    'class': EnhancedFNO1d,
                    'dimensions': 1,
                    'supports_sparse': True,
                    'default_config': {
                        'modes': 16,
                        'width': 64,
                        'num_layers': 4
                    }
                },
                'enhanced_fno_2d': {
                    'class': EnhancedFNO2d,
                    'dimensions': 2,
                    'supports_sparse': True,
                    'default_config': {
                        'modes1': 12,
                        'modes2': 12,
                        'width': 32,
                        'num_layers': 4
                    }
                },
                'fixed_enhanced_fno_2d': {
                    'class': FixedEnhancedFNO2d,
                    'creator': create_fixed_fno,
                    'dimensions': 2,
                    'supports_sparse': True,
                    'default_config': {
                        'modes1': 12,
                        'modes2': 12,
                        'width': 32,
                        'num_layers': 4
                    }
                }
            })
        
        if MLP_AVAILABLE:
            models.update({
                'enhanced_mlp': {
                    'class': EnhancedMLP,
                    'dimensions': 'any',
                    'supports_sparse': True,
                    'default_config': {
                        'hidden_dims': [256, 512, 256],
                        'activation': 'gelu',
                        'dropout': 0.1,
                        'use_fourier': True
                    }
                },
                'enhanced_mlp_1d': {
                    'class': EnhancedMLP1d,
                    'dimensions': 1,
                    'supports_sparse': True,
                    'default_config': {
                        'hidden_dims': [256, 512, 256],
                        'activation': 'gelu',
                        'dropout': 0.1
                    }
                },
                'enhanced_mlp_2d': {
                    'class': EnhancedMLP2d,
                    'dimensions': 2,
                    'supports_sparse': True,
                    'default_config': {
                        'hidden_dims': [256, 512, 256],
                        'activation': 'gelu',
                        'dropout': 0.1
                    }
                }
            })
        
        if PINN_AVAILABLE:
            models.update({
                'enhanced_pinn': {
                    'class': EnhancedPINN,
                    'dimensions': 'any',
                    'supports_sparse': True,
                    'default_config': {
                        'hidden_dims': [256, 512, 256],
                        'activation': 'tanh',
                        'dropout': 0.0,
                        'use_fourier': True,
                        'physics_loss_weight': 1.0
                    }
                },
                'enhanced_pinn_1d': {
                    'class': EnhancedPINN1d,
                    'dimensions': 1,
                    'supports_sparse': True,
                    'default_config': {
                        'hidden_dims': [256, 512, 256],
                        'activation': 'tanh',
                        'dropout': 0.0
                    }
                },
                'enhanced_pinn_2d': {
                    'class': EnhancedPINN2d,
                    'dimensions': 2,
                    'supports_sparse': True,
                    'default_config': {
                        'hidden_dims': [256, 512, 256],
                        'activation': 'tanh',
                        'dropout': 0.0
                    }
                }
            })
        
        if UNET_AVAILABLE:
            models.update({
                'enhanced_unet_1d': {
                    'class': EnhancedUNet1d,
                    'dimensions': 1,
                    'supports_sparse': True,
                    'default_config': {
                        'features': [64, 128, 256, 512],
                        'bilinear': True
                    }
                },
                'enhanced_unet_2d': {
                    'class': EnhancedUNet2d,
                    'dimensions': 2,
                    'supports_sparse': True,
                    'default_config': {
                        'features': [64, 128, 256, 512],
                        'bilinear': True
                    }
                },
                'enhanced_unet_3d': {
                    'class': EnhancedUNet3d,
                    'dimensions': 3,
                    'supports_sparse': True,
                    'default_config': {
                        'features': [64, 128, 256],
                        'bilinear': True
                    }
                }
            })
        
        return models
    
    def get_supported_models(self) -> List[str]:
        """获取所有支持的模型名称"""
        return list(self.supported_models.keys())
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """获取指定模型的信息"""
        if model_name not in self.supported_models:
            raise ValueError(f"不支持的模型: {model_name}")
        return self.supported_models[model_name]
    
    def validate_config(self, model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证并补全模型配置
        
        Args:
            model_name: 模型名称
            config: 用户提供的配置
            
        Returns:
            验证并补全后的配置
        """
        if model_name not in self.supported_models:
            raise ValueError(f"不支持的模型: {model_name}")
        
        model_info = self.supported_models[model_name]
        default_config = model_info.get('default_config', {})
        
        # 合并默认配置和用户配置
        validated_config = default_config.copy()
        validated_config.update(config)
        
        # 验证必需参数
        required_params = ['input_dim', 'output_dim']
        for param in required_params:
            if param not in validated_config:
                raise ValueError(f"缺少必需参数: {param}")
        
        return validated_config
    
    def infer_dimensions(self, input_shape: Tuple[int, ...], output_shape: Tuple[int, ...]) -> int:
        """
        根据输入输出形状推断数据维度
        
        Args:
            input_shape: 输入形状 (不包括batch维度)
            output_shape: 输出形状 (不包括batch维度)
            
        Returns:
            数据维度 (1, 2, 3等)
        """
        # 简单的维度推断逻辑
        if len(input_shape) == 1:
            return 1
        elif len(input_shape) == 2:
            return 2
        elif len(input_shape) == 3:
            return 3
        else:
            return 'any'
    
    def create_model(self, model_name: str, config: Dict[str, Any], 
                    input_shape: Optional[Tuple[int, ...]] = None,
                    output_shape: Optional[Tuple[int, ...]] = None) -> nn.Module:
        """
        创建指定的模型
        
        Args:
            model_name: 模型名称
            config: 模型配置
            input_shape: 输入形状 (可选，用于自动推断)
            output_shape: 输出形状 (可选，用于自动推断)
            
        Returns:
            创建的模型实例
        """
        if model_name not in self.supported_models:
            raise ValueError(f"不支持的模型: {model_name}")
        
        # 验证配置
        validated_config = self.validate_config(model_name, config)
        
        # 获取模型信息
        model_info = self.supported_models[model_name]
        model_class = model_info['class']
        
        # 检查维度兼容性
        if input_shape and output_shape:
            inferred_dim = self.infer_dimensions(input_shape, output_shape)
            model_dim = model_info['dimensions']
            
            if model_dim != 'any' and model_dim != inferred_dim:
                logger.warning(f"模型 {model_name} 期望 {model_dim}D 数据，但推断为 {inferred_dim}D")
        
        try:
            # 使用创建器函数（如果有）
            if 'creator' in model_info:
                creator_func = model_info['creator']
                model = creator_func(**validated_config)
            else:
                # 直接使用类构造函数
                model = model_class(**validated_config)
            
            # 移动到指定设备
            model = model.to(self.device)
            
            logger.info(f"成功创建模型 {model_name}，参数数量: {sum(p.numel() for p in model.parameters())}")
            return model
            
        except Exception as e:
            logger.error(f"创建模型 {model_name} 失败: {e}")
            raise
    
    def create_model_from_trainer_config(self, trainer_config: Dict[str, Any]) -> nn.Module:
        """
        从trainer配置创建模型（兼容dynamic_resolution_trainer.py）
        
        Args:
            trainer_config: trainer的完整配置
            
        Returns:
            创建的模型实例
        """
        model_config = trainer_config.get('model', {})
        
        # 提取基本参数
        input_dim = model_config.get('input_dim')
        output_dim = model_config.get('output_dim')
        
        if not input_dim or not output_dim:
            raise ValueError("trainer配置中缺少input_dim或output_dim")
        
        # 确定模型类型
        model_type = model_config.get('model_type', 'enhanced_transformer_1d')
        attention_type = model_config.get('attention_type', 'sge')
        
        # 构建模型配置
        config = {
            'input_dim': input_dim,
            'output_dim': output_dim,
            'd_model': model_config.get('d_model', 256),
            'num_heads': model_config.get('num_heads', 8),
            'num_layers': model_config.get('num_layers', 6),
            'attention_type': attention_type,
            'pe_type': model_config.get('pe_type', 'learnable_1d'),
            'max_time_steps': model_config.get('max_time_steps', 1),
            'seq_len': model_config.get('seq_len', 32),
            'input_hw': model_config.get('input_hw'),
            'output_head_type': model_config.get('output_head_type', 'global'),
            'out_channels_per_token': model_config.get('out_channels_per_token')
        }
        
        return self.create_model(model_type, config)
    
    def get_model_compatibility_info(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型兼容性信息
        
        Args:
            model_name: 模型名称
            
        Returns:
            兼容性信息字典
        """
        if model_name not in self.supported_models:
            raise ValueError(f"不支持的模型: {model_name}")
        
        model_info = self.supported_models[model_name]
        
        return {
            'model_name': model_name,
            'dimensions': model_info['dimensions'],
            'supports_sparse': model_info.get('supports_sparse', False),
            'default_config': model_info.get('default_config', {}),
            'available': True
        }
    
    def list_models_by_dimension(self, dimension: Union[int, str]) -> List[str]:
        """
        按维度列出可用模型
        
        Args:
            dimension: 数据维度 (1, 2, 3, 'any')
            
        Returns:
            支持该维度的模型列表
        """
        compatible_models = []
        
        for model_name, model_info in self.supported_models.items():
            model_dim = model_info['dimensions']
            if model_dim == dimension or model_dim == 'any':
                compatible_models.append(model_name)
        
        return compatible_models
    
    def get_configuration_template(self, model_name: str) -> Dict[str, Any]:
        """
        获取模型配置模板
        
        Args:
            model_name: 模型名称
            
        Returns:
            配置模板
        """
        if model_name not in self.supported_models:
            raise ValueError(f"不支持的模型: {model_name}")
        
        model_info = self.supported_models[model_name]
        template = model_info.get('default_config', {}).copy()
        
        # 添加必需参数的占位符
        template.update({
            'input_dim': '<REQUIRED>',
            'output_dim': '<REQUIRED>'
        })
        
        return template

# 全局工厂实例
_global_factory = None

def get_global_factory() -> UnifiedModelFactory:
    """获取全局模型工厂实例"""
    global _global_factory
    if _global_factory is None:
        _global_factory = UnifiedModelFactory()
    return _global_factory

def create_model_from_config(model_name: str, config: Dict[str, Any], 
                           device: Optional[torch.device] = None) -> nn.Module:
    """
    便捷函数：从配置创建模型
    
    Args:
        model_name: 模型名称
        config: 模型配置
        device: 设备
        
    Returns:
        创建的模型实例
    """
    factory = UnifiedModelFactory(device)
    return factory.create_model(model_name, config)

def list_available_models() -> List[str]:
    """便捷函数：列出所有可用模型"""
    factory = get_global_factory()
    return factory.get_supported_models()

def get_model_template(model_name: str) -> Dict[str, Any]:
    """便捷函数：获取模型配置模板"""
    factory = get_global_factory()
    return factory.get_configuration_template(model_name)

if __name__ == "__main__":
    # 测试代码
    factory = UnifiedModelFactory()
    
    print("支持的模型:")
    for model_name in factory.get_supported_models():
        print(f"  - {model_name}")
    
    # 测试创建一个简单的模型
    if 'enhanced_mlp' in factory.get_supported_models():
        config = {
            'input_dim': 64,
            'output_dim': 32,
            'hidden_dims': [128, 256, 128]
        }
        
        try:
            model = factory.create_model('enhanced_mlp', config)
            print(f"\n成功创建模型，参数数量: {sum(p.numel() for p in model.parameters())}")
        except Exception as e:
            print(f"\n创建模型失败: {e}")