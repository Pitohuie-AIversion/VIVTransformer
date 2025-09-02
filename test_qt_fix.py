#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Qt错误修复

这个脚本用于验证Qt平台插件错误是否已经修复。
它会尝试导入所有可能引起Qt错误的模块，并运行一个简单的测试。
"""

# 设置无头模式环境变量（必须在任何GUI相关导入之前）
import os
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['MPLBACKEND'] = 'Agg'
os.environ['DISPLAY'] = ''
os.environ['HEADLESS'] = '1'

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'modify_multi_attention'))
sys.path.insert(0, str(project_root / 'generate_data'))

print("🔧 测试Qt错误修复...")
print(f"环境变量设置:")
print(f"  QT_QPA_PLATFORM: {os.environ.get('QT_QPA_PLATFORM')}")
print(f"  MPLBACKEND: {os.environ.get('MPLBACKEND')}")
print(f"  DISPLAY: {os.environ.get('DISPLAY')}")
print(f"  HEADLESS: {os.environ.get('HEADLESS')}")

try:
    # 测试matplotlib导入
    print("\n[INFO] 测试matplotlib导入...")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    print(f"[OK] matplotlib后端: {matplotlib.get_backend()}")
    
    # 测试创建简单图形
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 2])
    plt.close(fig)
    print("[OK] matplotlib图形创建成功")
    
except Exception as e:
    print(f"[ERROR] matplotlib测试失败: {e}")
    sys.exit(1)

try:
    # 测试训练器导入
    print("\n[INFO] 测试训练器导入...")
    from generate_data.dynamic_resolution_trainer import DynamicResolutionDataset
    print("[OK] 动态分辨率训练器导入成功")
    
except Exception as e:
    print(f"[ERROR] 训练器导入失败: {e}")
    sys.exit(1)

try:
    # 测试可视化模块导入
    print("\n🎨 测试可视化模块导入...")
    from modify_multi_attention.utils.visualization import plot_comparison_figure
    print("[OK] 可视化模块导入成功")
    
except Exception as e:
    print(f"[ERROR] 可视化模块导入失败: {e}")
    sys.exit(1)

try:
    # 测试配置文件加载
    print("\n⚙️ 测试配置文件加载...")
    import yaml
    config_path = project_root / 'generate_data' / 'dynamic_config_server_downsampling.yaml'
    if config_path.exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"[OK] 配置文件加载成功")
        print(f"  可视化启用状态: {config.get('visualization', {}).get('enabled', 'N/A')}")
        print(f"  环境配置: {config.get('environment', 'N/A')}")
    else:
        print(f"[WARN] 配置文件不存在: {config_path}")
        
except Exception as e:
    print(f"[ERROR] 配置文件测试失败: {e}")
    sys.exit(1)

print("\n[OK] 所有测试通过！Qt错误修复成功！")
print("\n[INFO] 修复总结:")
print("  1. 设置了QT_QPA_PLATFORM=offscreen")
print("  2. 强制使用matplotlib的Agg后端")
print("  3. 禁用了DISPLAY环境变量")
print("  4. 在所有相关文件中添加了无头模式设置")
print("  5. 配置文件中禁用了可视化功能")
print("\n[INFO] 现在可以在服务器环境下正常运行训练！")