#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
调试配置验证问题
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Dict, Any

# 添加路径
sys.path.append(str(Path(__file__).parent.parent / 'modify_multi_attention'))
sys.path.append(str(Path(__file__).parent))

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置文件的合理性（复制自主程序）"""
    warnings = []
    errors = []
    
    # 检查数据配置
    data_config = config.get('data', {})
    
    # 检查分辨率配置
    input_res = data_config.get('input_resolution', [])
    output_res = data_config.get('output_resolution', [])
    
    if len(input_res) != 2 or len(output_res) != 2:
        errors.append("输入和输出分辨率必须是长度为2的列表")
    
    if input_res and output_res:
        input_dim = input_res[0] * input_res[1]
        output_dim = output_res[0] * output_res[1]
        
        # 检查分辨率是否过高（仅警告，不阻止训练）
        if input_dim > 50000 or output_dim > 50000:
            warnings.append(f"分辨率较高，可能导致内存问题: 输入{input_dim}, 输出{output_dim}")
        
        if output_dim > input_dim * 16:  # 超分辨率倍数过大
            warnings.append(f"输出分辨率({output_res})比输入分辨率({input_res})大很多，可能影响训练效果")
    
    # 检查批次大小
    batch_size = data_config.get('batch_size', 16)
    if batch_size > 64:
        warnings.append(f"批次大小({batch_size})较大，可能导致内存不足")
    
    # 检查高级功能配置
    gradient_config = config.get('gradient', {})
    mixed_precision_config = config.get('mixed_precision', {})
    
    if gradient_config.get('clip_enabled', False) and not gradient_config.get('clip_value'):
        errors.append("启用梯度裁剪时必须设置clip_value")
    
    import torch
    if mixed_precision_config.get('enabled', False) and not torch.cuda.is_available():
        warnings.append("混合精度训练需要GPU支持，当前为CPU模式")
    
    # 输出验证结果
    if warnings:
        logger.warning("[WARN] 配置验证警告:")
        for warning in warnings:
            logger.warning(f"  - {warning}")
    
    if errors:
        logger.error("[ERROR] 配置验证错误:")
        for error in errors:
            logger.error(f"  - {error}")
        return False
    
    if not warnings and not errors:
        logger.info("[OK] 配置验证通过")
    
    return True

def load_config_with_args(config_path, args=None):
    """加载配置文件并合并命令行参数（简化版）"""
    # 加载基础配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 计算模型维度
    input_res = config['data']['input_resolution']
    output_res = config['data']['output_resolution']
    
    config['model']['input_dim'] = input_res[0] * input_res[1]
    config['model']['output_dim'] = output_res[0] * output_res[1]
    
    return config

if __name__ == "__main__":
    config_path = "dynamic_config.yaml"
    
    try:
        # 使用主程序相同的配置加载方式
        config = load_config_with_args(config_path)
        print(f"成功加载配置文件: {config_path}")
        print(f"配置内容: {config}")
        
        result = validate_config(config)
        print(f"\n验证结果: {'通过' if result else '失败'}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()