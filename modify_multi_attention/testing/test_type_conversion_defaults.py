#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
参数类型转换和默认值处理逻辑测试
测试配置参数的类型转换、默认值设置和错误处理机制
"""

import sys
import os
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# 导入必要的模块
from run_crop_model_test import (
    load_unified_config,
    get_default_config,
    validate_config,
    merge_configs
)

def test_config_loading_fallback():
    """测试配置加载的后备机制"""
    print("\n=== 配置加载后备机制测试 ===")
    
    # 测试不存在的配置文件
    print("\n--- 测试不存在的配置文件 ---")
    config1 = load_unified_config('nonexistent_config.yaml')
    print(f"不存在文件的配置类型: {type(config1)}")
    print(f"是否包含默认配置: {'training' in config1}")
    
    # 测试存在的配置文件
    print("\n--- 测试存在的配置文件 ---")
    config2 = load_unified_config('configs/unified_config.yaml')
    print(f"存在文件的配置类型: {type(config2)}")
    print(f"是否包含训练配置: {'training' in config2}")
    
    # 测试默认配置
    print("\n--- 测试默认配置 ---")
    default_config = get_default_config()
    print(f"默认配置类型: {type(default_config)}")
    print(f"默认配置键: {list(default_config.keys())}")
    
    return config1, config2, default_config

def test_parameter_type_conversion():
    """测试参数类型转换"""
    print("\n=== 参数类型转换测试 ===")
    
    # 获取配置
    config = load_unified_config('configs/unified_config.yaml')
    training_config = config.get('training', {})
    
    print("\n--- 原始配置参数类型 ---")
    key_params = ['epochs', 'learning_rate', 'weight_decay', 'patience', 'min_delta']
    
    for param in key_params:
        if param in training_config:
            value = training_config[param]
            print(f"{param}: {value} (类型: {type(value).__name__})")
        else:
            print(f"{param}: 不存在")
    
    print("\n--- 类型转换测试 ---")
    
    # 测试整数转换
    test_values = {
        'epochs': [50, '50', 50.0, '50.5'],
        'patience': [10, '10', 10.0, 'invalid'],
        'learning_rate': [0.001, '0.001', 1e-3, 'invalid'],
        'weight_decay': [0.0001, '0.0001', 1e-4, 'invalid']
    }
    
    for param, values in test_values.items():
        print(f"\n{param} 转换测试:")
        for value in values:
            try:
                if param in ['epochs', 'patience']:
                    converted = int(float(value))  # 先转float再转int，处理字符串数字
                    print(f"  {value} ({type(value).__name__}) -> {converted} (int) ✅")
                else:
                    converted = float(value)
                    print(f"  {value} ({type(value).__name__}) -> {converted} (float) ✅")
            except (ValueError, TypeError) as e:
                print(f"  {value} ({type(value).__name__}) -> 转换失败: {e} ❌")

def test_default_value_handling():
    """测试默认值处理"""
    print("\n=== 默认值处理测试 ===")
    
    # 创建不完整的配置
    incomplete_config = {
        'training': {
            'epochs': 50,
            'learning_rate': 0.001
            # 缺少其他参数
        },
        'data': {
            'batch_size': 32
            # 缺少其他参数
        }
    }
    
    print("\n--- 不完整配置 ---")
    print(f"原始配置: {incomplete_config}")
    
    # 获取默认配置并合并
    default_config = get_default_config()
    merged_config = merge_configs(default_config, incomplete_config)
    
    print("\n--- 合并后配置 ---")
    print(f"训练配置: {merged_config.get('training', {})}")
    print(f"数据配置: {merged_config.get('data', {})}")
    
    # 测试参数获取的默认值处理
    print("\n--- 参数获取默认值测试 ---")
    
    def safe_get_param(config, section, param, default_value, param_type):
        """安全获取参数，带类型转换和默认值"""
        try:
            value = config.get(section, {}).get(param, default_value)
            if param_type == int:
                return int(float(value))
            elif param_type == float:
                return float(value)
            else:
                return value
        except (ValueError, TypeError):
            return default_value
    
    # 测试各种参数获取
    test_params = [
        ('training', 'epochs', 50, int),
        ('training', 'learning_rate', 0.001, float),
        ('training', 'patience', 10, int),
        ('training', 'min_delta', 0.0001, float),
        ('data', 'batch_size', 32, int),
        ('data', 'max_samples', None, type(None)),
        ('models', 'transformer', {}, dict)
    ]
    
    for section, param, default, param_type in test_params:
        result = safe_get_param(merged_config, section, param, default, param_type)
        print(f"{section}.{param}: {result} (类型: {type(result).__name__})")

def test_config_validation():
    """测试配置验证"""
    print("\n=== 配置验证测试 ===")
    
    # 测试有效配置
    print("\n--- 有效配置验证 ---")
    valid_config = {
        'training': {
            'epochs': 50,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'patience': 10,
            'min_delta': 0.0001
        },
        'data': {
            'batch_size': 32,
            'max_samples': 10000
        }
    }
    
    try:
        validated_config = validate_config(valid_config)
        print(f"✅ 有效配置验证通过")
        print(f"验证后配置类型: {type(validated_config)}")
    except Exception as e:
        print(f"❌ 有效配置验证失败: {e}")
    
    # 测试无效配置
    print("\n--- 无效配置验证 ---")
    invalid_configs = [
        # 缺少必要字段
        {'training': {}},
        # 错误的数据类型
        {'training': {'epochs': 'invalid', 'learning_rate': 'invalid'}},
        # 负数值
        {'training': {'epochs': -10, 'learning_rate': -0.001}},
        # 空配置
        {}
    ]
    
    for i, invalid_config in enumerate(invalid_configs):
        try:
            validated_config = validate_config(invalid_config)
            print(f"配置 {i+1}: ✅ 验证通过（可能使用了默认值）")
        except Exception as e:
            print(f"配置 {i+1}: ❌ 验证失败: {e}")

def test_yaml_parsing_errors():
    """测试YAML解析错误处理"""
    print("\n=== YAML解析错误处理测试 ===")
    
    # 创建临时的错误YAML文件
    invalid_yaml_content = """
    training:
      epochs: 50
      learning_rate: 0.001
    # 缺少缩进的错误YAML
    data:
  batch_size: 32
    """
    
    temp_file = 'temp_invalid.yaml'
    
    try:
        # 写入错误的YAML内容
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(invalid_yaml_content)
        
        print("\n--- 测试错误YAML文件解析 ---")
        config = load_unified_config(temp_file)
        print(f"错误YAML解析结果类型: {type(config)}")
        print(f"是否返回默认配置: {'training' in config}")
        
    except Exception as e:
        print(f"YAML错误处理测试失败: {e}")
    finally:
        # 清理临时文件
        if os.path.exists(temp_file):
            os.remove(temp_file)

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 边界情况测试 ===")
    
    # 测试极值
    print("\n--- 极值测试 ---")
    extreme_values = {
        'epochs': [0, 1, 10000, float('inf')],
        'learning_rate': [0.0, 1e-10, 1.0, 100.0],
        'batch_size': [1, 2, 1024, 10000]
    }
    
    for param, values in extreme_values.items():
        print(f"\n{param} 极值测试:")
        for value in values:
            try:
                if param == 'epochs':
                    if value == float('inf'):
                        print(f"  {value} -> 无穷大，应该被处理 ❌")
                    else:
                        converted = int(value)
                        print(f"  {value} -> {converted} ✅")
                elif param == 'learning_rate':
                    converted = float(value)
                    if 0 <= converted <= 1:
                        print(f"  {value} -> {converted} ✅")
                    else:
                        print(f"  {value} -> {converted} (可能超出合理范围) ⚠️")
                else:  # batch_size
                    converted = int(value)
                    if converted > 0:
                        print(f"  {value} -> {converted} ✅")
                    else:
                        print(f"  {value} -> {converted} (无效值) ❌")
            except (ValueError, OverflowError) as e:
                print(f"  {value} -> 转换失败: {e} ❌")
    
    # 测试None值处理
    print("\n--- None值处理测试 ---")
    none_config = {
        'training': {
            'epochs': None,
            'learning_rate': None
        },
        'data': {
            'batch_size': None,
            'max_samples': None
        }
    }
    
    default_config = get_default_config()
    merged_config = merge_configs(default_config, none_config)
    
    print(f"None值合并后的训练配置: {merged_config.get('training', {})}")
    print(f"None值合并后的数据配置: {merged_config.get('data', {})}")

def main():
    """主测试函数"""
    print("开始参数类型转换和默认值处理逻辑测试")
    
    # 运行各项测试
    test_config_loading_fallback()
    test_parameter_type_conversion()
    test_default_value_handling()
    test_config_validation()
    test_yaml_parsing_errors()
    test_edge_cases()
    
    print("\n=== 参数类型转换和默认值处理逻辑测试完成 ===")

if __name__ == "__main__":
    main()