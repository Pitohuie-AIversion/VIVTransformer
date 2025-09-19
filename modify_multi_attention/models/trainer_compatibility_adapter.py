#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Trainer兼容性适配器

这个模块提供了一个适配器，用于将新的统一模型工厂和配置管理器
与现有的dynamic_resolution_trainer.py系统无缝集成。
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, Tuple, List
import logging
from pathlib import Path
import sys
import copy

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'models'))

# 导入统一模块
try:
    from unified_model_factory import UnifiedModelFactory, get_global_factory
    from unified_config_manager import UnifiedConfigManager, get_global_config_manager
    UNIFIED_MODULES_AVAILABLE = True
except ImportError as e:
    logging.warning(f"统一模块导入失败: {e}")
    UNIFIED_MODULES_AVAILABLE = False

# 导入原有的Transformer模型（作为后备）
try:
    from mymodels.transformer import TransformerFlowReconstructionModel
    ORIGINAL_TRANSFORMER_AVAILABLE = True
except ImportError:
    try:
        sys.path.append(str(project_root / 'mymodels'))
        from transformer import TransformerFlowReconstructionModel
        ORIGINAL_TRANSFORMER_AVAILABLE = True
    except ImportError as e:
        logging.warning(f"原始Transformer模型导入失败: {e}")
        ORIGINAL_TRANSFORMER_AVAILABLE = False

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainerCompatibilityAdapter:
    """
    Trainer兼容性适配器
    
    提供与dynamic_resolution_trainer.py的兼容性接口，包括:
    - 模型创建的兼容性包装
    - 配置格式的转换
    - 参数映射和验证
    - 渐进式迁移支持
    """
    
    def __init__(self, use_unified_factory: bool = True, fallback_to_original: bool = True):
        self.use_unified_factory = use_unified_factory and UNIFIED_MODULES_AVAILABLE
        self.fallback_to_original = fallback_to_original
        
        if self.use_unified_factory:
            self.model_factory = get_global_factory()
            self.config_manager = get_global_config_manager()
            logger.info("使用统一模型工厂")
        else:
            self.model_factory = None
            self.config_manager = None
            logger.info("使用原始模型创建方式")
    
    def create_model_from_trainer_config(self, config: Dict[str, Any]) -> nn.Module:
        """
        从trainer配置创建模型（兼容原有接口）
        
        Args:
            config: trainer的完整配置字典
            
        Returns:
            创建的模型实例
        """
        model_config = config.get('model', {})
        
        # 提取基本参数
        input_dim = model_config.get('input_dim')
        output_dim = model_config.get('output_dim')
        
        if not input_dim or not output_dim:
            raise ValueError("配置中缺少input_dim或output_dim")
        
        # 尝试使用统一工厂
        if self.use_unified_factory:
            try:
                return self._create_model_with_unified_factory(config)
            except Exception as e:
                logger.warning(f"统一工厂创建模型失败: {e}")
                if self.fallback_to_original:
                    logger.info("回退到原始模型创建方式")
                    return self._create_model_original_way(config)
                else:
                    raise
        else:
            return self._create_model_original_way(config)
    
    def _create_model_with_unified_factory(self, config: Dict[str, Any]) -> nn.Module:
        """
        使用统一工厂创建模型
        
        Args:
            config: trainer配置
            
        Returns:
            创建的模型实例
        """
        model_config = config.get('model', {})
        
        # 映射模型类型
        model_type = self._map_model_type(model_config)
        
        # 转换配置格式
        unified_config = self._convert_config_to_unified_format(model_config)
        
        # 验证配置
        if self.config_manager:
            unified_config = self.config_manager.validate_config_section('model', unified_config)
        
        # 创建模型
        model = self.model_factory.create_model(model_type, unified_config)
        
        logger.info(f"使用统一工厂成功创建模型: {model_type}")
        return model
    
    def _create_model_original_way(self, config: Dict[str, Any]) -> nn.Module:
        """
        使用原始方式创建模型（兼容性后备）
        
        Args:
            config: trainer配置
            
        Returns:
            创建的模型实例
        """
        if not ORIGINAL_TRANSFORMER_AVAILABLE:
            raise ImportError("原始Transformer模型不可用")
        
        model_config = config.get('model', {})
        
        # 使用原始的TransformerFlowReconstructionModel
        model = TransformerFlowReconstructionModel(
            input_dim=model_config.get('input_dim'),
            output_dim=model_config.get('output_dim'),
            num_heads=model_config.get('num_heads', 8),
            num_layers=model_config.get('num_layers', 6),
            d_model=model_config.get('d_model', 256),
            max_time_steps=model_config.get('max_time_steps', 1),
            attention_type=model_config.get('attention_type', 'sge'),
            seq_len=model_config.get('seq_len', 32),
            input_hw=tuple(model_config.get('input_hw')) if model_config.get('input_hw') else None,
            pe_type=model_config.get('pe_type', 'learnable_1d'),
            output_head_type=model_config.get('output_head_type', 'global'),
            out_channels_per_token=model_config.get('out_channels_per_token')
        )
        
        logger.info("使用原始方式成功创建Transformer模型")
        return model
    
    def _map_model_type(self, model_config: Dict[str, Any]) -> str:
        """
        映射模型类型到统一工厂的模型名称
        
        Args:
            model_config: 模型配置
            
        Returns:
            统一工厂的模型名称
        """
        # 从配置中推断模型类型
        model_type = model_config.get('model_type', 'transformer')
        attention_type = model_config.get('attention_type', 'sge')
        
        # 根据输入输出维度推断数据维度
        input_hw = model_config.get('input_hw')
        if input_hw and len(input_hw) == 2:
            data_dim = '2d'
        else:
            data_dim = '1d'
        
        # 映射到统一工厂的模型名称
        type_mapping = {
            'transformer': f'enhanced_transformer_{data_dim}',
            'fno': f'enhanced_fno_{data_dim}',
            'mlp': f'enhanced_mlp_{data_dim}',
            'pinn': f'enhanced_pinn_{data_dim}',
            'unet': f'enhanced_unet_{data_dim}'
        }
        
        unified_type = type_mapping.get(model_type.lower(), f'enhanced_transformer_{data_dim}')
        
        # 检查模型是否可用
        if unified_type not in self.model_factory.get_supported_models():
            logger.warning(f"模型类型 {unified_type} 不可用，回退到 enhanced_transformer_1d")
            unified_type = 'enhanced_transformer_1d'
        
        return unified_type
    
    def _convert_config_to_unified_format(self, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        将trainer配置格式转换为统一工厂格式
        
        Args:
            model_config: trainer的模型配置
            
        Returns:
            统一工厂格式的配置
        """
        unified_config = {
            'input_dim': model_config.get('input_dim'),
            'output_dim': model_config.get('output_dim'),
            'd_model': model_config.get('d_model', 256),
            'num_heads': model_config.get('num_heads', 8),
            'num_layers': model_config.get('num_layers', 6),
            'attention_type': model_config.get('attention_type', 'sge'),
            'pe_type': model_config.get('pe_type', 'learnable_1d'),
            'max_time_steps': model_config.get('max_time_steps', 1),
            'seq_len': model_config.get('seq_len', 32),
            'output_head_type': model_config.get('output_head_type', 'global'),
            'out_channels_per_token': model_config.get('out_channels_per_token')
        }
        
        # 处理input_hw
        input_hw = model_config.get('input_hw')
        if input_hw:
            unified_config['input_hw'] = tuple(input_hw) if isinstance(input_hw, list) else input_hw
        
        # 移除None值
        unified_config = {k: v for k, v in unified_config.items() if v is not None}
        
        return unified_config
    
    def enhance_trainer_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        增强trainer配置，添加统一配置管理器的功能
        
        Args:
            config: 原始trainer配置
            
        Returns:
            增强后的配置
        """
        if not self.config_manager:
            return config
        
        enhanced_config = copy.deepcopy(config)
        
        try:
            # 验证和补全各个配置段
            for section_name in ['model', 'training', 'data', 'loss', 'optimizer', 'scheduler']:
                if section_name in enhanced_config:
                    enhanced_config[section_name] = self.config_manager.validate_config_section(
                        section_name, enhanced_config[section_name]
                    )
            
            logger.info("trainer配置已通过统一配置管理器增强")
            
        except Exception as e:
            logger.warning(f"配置增强失败: {e}，使用原始配置")
            return config
        
        return enhanced_config
    
    def get_model_info(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model_name: 模型名称，None表示获取所有可用模型
            
        Returns:
            模型信息字典
        """
        if not self.model_factory:
            return {'error': '统一模型工厂不可用'}
        
        if model_name:
            try:
                return self.model_factory.get_model_info(model_name)
            except ValueError:
                return {'error': f'模型 {model_name} 不存在'}
        else:
            return {
                'supported_models': self.model_factory.get_supported_models(),
                'factory_available': True
            }
    
    def create_config_template(self, sections: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        创建配置模板
        
        Args:
            sections: 需要的配置段
            
        Returns:
            配置模板
        """
        if not self.config_manager:
            return {'error': '统一配置管理器不可用'}
        
        return self.config_manager.get_config_template(sections)
    
    def validate_trainer_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证trainer配置
        
        Args:
            config: trainer配置
            
        Returns:
            (是否有效, 错误信息列表)
        """
        errors = []
        
        # 基本验证
        if 'model' not in config:
            errors.append("缺少model配置段")
        else:
            model_config = config['model']
            if 'input_dim' not in model_config:
                errors.append("model配置中缺少input_dim")
            if 'output_dim' not in model_config:
                errors.append("model配置中缺少output_dim")
        
        # 使用统一配置管理器验证（如果可用）
        if self.config_manager:
            try:
                self.config_manager.validate_full_config(config)
            except Exception as e:
                errors.append(f"统一配置验证失败: {e}")
        
        return len(errors) == 0, errors
    
    def get_parameter_suggestions(self, model_type: str) -> Dict[str, Any]:
        """
        获取模型参数建议
        
        Args:
            model_type: 模型类型
            
        Returns:
            参数建议字典
        """
        suggestions = {
            'transformer': {
                'd_model': [128, 256, 512],
                'num_heads': [4, 8, 16],
                'num_layers': [4, 6, 8, 12],
                'attention_type': ['sge', 'simplified_self', 'multi_head'],
                'pe_type': ['learnable_1d', 'learnable_2d', 'sinusoidal']
            },
            'fno': {
                'modes': [8, 12, 16, 24],
                'width': [32, 64, 128],
                'num_layers': [2, 4, 6]
            },
            'mlp': {
                'hidden_dims': [[128, 256, 128], [256, 512, 256], [512, 1024, 512]],
                'activation': ['relu', 'gelu', 'tanh'],
                'dropout': [0.0, 0.1, 0.2]
            },
            'unet': {
                'features': [[32, 64, 128], [64, 128, 256, 512]],
                'bilinear': [True, False]
            },
            'pinn': {
                'hidden_dims': [[128, 256, 128], [256, 512, 256]],
                'activation': ['tanh', 'relu', 'gelu'],
                'physics_loss_weight': [0.1, 1.0, 10.0]
            }
        }
        
        return suggestions.get(model_type.lower(), {})
    
    def migrate_old_config(self, old_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        迁移旧版本配置到新格式
        
        Args:
            old_config: 旧版本配置
            
        Returns:
            新格式配置
        """
        migrated_config = copy.deepcopy(old_config)
        
        # 配置格式迁移逻辑
        # 这里可以添加具体的迁移规则
        
        logger.info("配置迁移完成")
        return migrated_config

# 全局适配器实例
_global_adapter = None

def get_global_adapter() -> TrainerCompatibilityAdapter:
    """获取全局适配器实例"""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = TrainerCompatibilityAdapter()
    return _global_adapter

def create_model_compatible(config: Dict[str, Any]) -> nn.Module:
    """
    便捷函数：兼容性模型创建
    
    Args:
        config: trainer配置
        
    Returns:
        创建的模型实例
    """
    adapter = get_global_adapter()
    return adapter.create_model_from_trainer_config(config)

def enhance_config_compatible(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    便捷函数：兼容性配置增强
    
    Args:
        config: 原始配置
        
    Returns:
        增强后的配置
    """
    adapter = get_global_adapter()
    return adapter.enhance_trainer_config(config)

def validate_config_compatible(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    便捷函数：兼容性配置验证
    
    Args:
        config: 配置字典
        
    Returns:
        (是否有效, 错误信息列表)
    """
    adapter = get_global_adapter()
    return adapter.validate_trainer_config(config)

if __name__ == "__main__":
    # 测试代码
    adapter = TrainerCompatibilityAdapter()
    
    # 测试配置
    test_config = {
        'model': {
            'input_dim': 64,
            'output_dim': 32,
            'd_model': 256,
            'num_heads': 8,
            'num_layers': 6,
            'attention_type': 'sge'
        },
        'training': {
            'epochs': 50,
            'learning_rate': 1e-3
        }
    }
    
    print("测试配置验证:")
    is_valid, errors = adapter.validate_trainer_config(test_config)
    print(f"配置有效: {is_valid}")
    if errors:
        print(f"错误: {errors}")
    
    print("\n支持的模型:")
    model_info = adapter.get_model_info()
    if 'supported_models' in model_info:
        for model in model_info['supported_models']:
            print(f"  - {model}")
    
    # 测试模型创建
    try:
        model = adapter.create_model_from_trainer_config(test_config)
        print(f"\n成功创建模型，参数数量: {sum(p.numel() for p in model.parameters())}")
    except Exception as e:
        print(f"\n模型创建失败: {e}")