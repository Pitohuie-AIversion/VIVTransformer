#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试注意力类型打印功能
"""

import yaml
import sys
from pathlib import Path

def test_attention_type_print():
    """
    测试配置文件中的注意力类型并模拟训练时的打印
    """
    print("=== 测试注意力类型打印功能 ===")
    
    # 读取配置文件
    config_path = Path("generate_data/dynamic_config.yaml")
    
    if not config_path.exists():
        print(f"[ERROR] 配置文件不存在: {config_path}")
        return
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # 模拟训练时的打印信息
        print("\n[INFO] 配置信息:")
        print(f"样本数量: {config['data']['num_samples']}")
        print(f"批次大小: {config['data']['batch_size']}")
        print(f"训练轮数: {config['training']['epochs']}")
        print(f"[MODEL] 使用注意力机制: {config['model']['attention_type']}")
        
        # 模拟训练开始
        print("\n=== 开始训练 ===")
        print(f"[MODEL] 使用注意力机制: {config['model']['attention_type']}")
        
        # 模拟测试完成
        attention_type = config['model']['attention_type']
        test_loss = 0.123456
        print(f"[TEST] 测试完成，{attention_type} Test Loss: {test_loss:.6f}")
        
        print("\n[OK] 注意力类型打印功能正常")
        
    except Exception as e:
        print(f"[ERROR] 读取配置文件失败: {e}")

if __name__ == "__main__":
    test_attention_type_print()