#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置参数使用情况分析脚本
分析 dynamic_config.yaml 中的参数是否真正被训练代码使用
"""

import yaml
import re
from pathlib import Path
from typing import Dict, List, Set

def load_config(config_path: str) -> Dict:
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def extract_config_keys(config: Dict, prefix: str = "") -> Set[str]:
    """递归提取配置文件中的所有键"""
    keys = set()
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        keys.add(full_key)
        if isinstance(value, dict):
            keys.update(extract_config_keys(value, full_key))
    return keys

def search_code_usage(file_path: str, config_keys: Set[str]) -> Dict[str, List[str]]:
    """搜索代码文件中对配置键的使用"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    used_keys = {}
    for key in config_keys:
        # 搜索不同的使用模式
        patterns = [
            rf"config\['{key.split('.')[-1]}'\]",  # config['key']
            rf'config\["{key.split(".")[-1]}"\]',  # config["key"]
            rf"config\.get\('{key.split('.')[-1]}'",  # config.get('key')
            rf'config\.get\("{key.split(".")[-1]}"',  # config.get("key")
            rf"{key.split('.')[-1]}_config",  # key_config
            rf"\['{key}'\]",  # ['full.key']
            rf'\["{key}"\]',  # ["full.key"]
        ]
        
        # 对于嵌套键，也搜索父级访问模式
        if '.' in key:
            parts = key.split('.')
            for i in range(len(parts)):
                parent_key = '.'.join(parts[:i+1])
                patterns.extend([
                    rf"config\['{parts[0]}'\]\['{parts[i]}'\]",
                    rf'config\["{parts[0]}"\]\["{parts[i]}"\]',
                ])
        
        matches = []
        for pattern in patterns:
            found = re.findall(pattern, content, re.IGNORECASE)
            if found:
                matches.extend(found)
        
        if matches:
            used_keys[key] = matches
    
    return used_keys

def analyze_config_usage():
    """分析配置使用情况"""
    # 配置文件路径
    config_path = "dynamic_config.yaml"
    
    # 代码文件路径
    code_files = [
        "dynamic_resolution_trainer.py",
        "../modify_multi_attention/training/trainer.py",
        "../modify_multi_attention/mymodels/transformer.py",
    ]
    
    # 加载配置
    config = load_config(config_path)
    config_keys = extract_config_keys(config)
    
    print("=== 配置参数使用情况分析 ===")
    print(f"配置文件: {config_path}")
    print(f"总配置参数数量: {len(config_keys)}")
    print()
    
    # 分析每个代码文件
    all_used_keys = set()
    for code_file in code_files:
        if Path(code_file).exists():
            print(f"分析文件: {code_file}")
            used_keys = search_code_usage(code_file, config_keys)
            all_used_keys.update(used_keys.keys())
            
            if used_keys:
                print(f"  使用的配置参数 ({len(used_keys)}个):")
                for key in sorted(used_keys.keys()):
                    print(f"    - {key}")
            else:
                print("  未发现配置参数使用")
            print()
        else:
            print(f"文件不存在: {code_file}")
    
    # 总结
    unused_keys = config_keys - all_used_keys
    
    print("=== 总结 ===")
    print(f"实际使用的配置参数: {len(all_used_keys)}个")
    print(f"未使用的配置参数: {len(unused_keys)}个")
    print(f"使用率: {len(all_used_keys)/len(config_keys)*100:.1f}%")
    print()
    
    if unused_keys:
        print("未使用的配置参数列表:")
        # 按类别分组
        categories = {}
        for key in unused_keys:
            category = key.split('.')[0]
            if category not in categories:
                categories[category] = []
            categories[category].append(key)
        
        for category, keys in sorted(categories.items()):
            print(f"\n  {category} 类别:")
            for key in sorted(keys):
                print(f"    - {key}")
    
    print("\n=== 建议 ===")
    if len(unused_keys) > len(all_used_keys):
        print("⚠️  大部分配置参数未被使用，建议:")
        print("   1. 实现对应的功能代码")
        print("   2. 或者移除不需要的配置参数")
        print("   3. 添加配置验证和警告机制")
    else:
        print("✅ 大部分配置参数已被使用")

if __name__ == "__main__":
    analyze_config_usage()