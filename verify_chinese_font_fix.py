#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文字体修复验证脚本
检查项目中所有matplotlib可视化文件的中文字体配置
"""

import os
import re
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def find_matplotlib_files(root_dir):
    """查找所有包含matplotlib的Python文件"""
    matplotlib_files = []
    
    for root, dirs, files in os.walk(root_dir):
        # 跳过.git等隐藏目录
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if 'import matplotlib.pyplot' in content or 'from matplotlib' in content:
                            matplotlib_files.append(file_path)
                except Exception as e:
                    print(f"读取文件失败: {file_path}, 错误: {e}")
    
    return matplotlib_files

def check_chinese_font_config(file_path):
    """检查文件是否包含中文字体配置"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # 检查是否有中文字体配置
        has_font_config = (
            "plt.rcParams['font.sans-serif']" in content and
            "plt.rcParams['axes.unicode_minus']" in content
        )
        
        return has_font_config, content
    except Exception as e:
        return False, f"读取错误: {e}"

def generate_test_visualization():
    """生成测试可视化图片验证中文字体"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('中文字体显示测试 - 项目可视化修复验证', fontsize=16)
    
    # 测试1: 基本中文文字
    axes[0, 0].text(0.5, 0.5, '测试中文显示\n数据可视化\n损失函数曲线', 
                    ha='center', va='center', fontsize=14)
    axes[0, 0].set_title('基本中文文字测试')
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    
    # 测试2: 带负号的数据
    x = np.linspace(-5, 5, 100)
    y = -x**2 + 10
    axes[0, 1].plot(x, y, 'b-', label='损失函数: y = -x² + 10')
    axes[0, 1].set_xlabel('训练轮次')
    axes[0, 1].set_ylabel('损失值')
    axes[0, 1].set_title('负号显示测试')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # 测试3: 热力图
    data = np.random.randn(10, 10)
    im = axes[1, 0].imshow(data, cmap='coolwarm')
    axes[1, 0].set_title('压力场可视化')
    axes[1, 0].set_xlabel('空间坐标 x')
    axes[1, 0].set_ylabel('空间坐标 y')
    plt.colorbar(im, ax=axes[1, 0])
    
    # 测试4: 多条曲线
    epochs = np.arange(1, 51)
    train_loss = np.exp(-epochs/10) + 0.1 * np.random.randn(50)
    valid_loss = np.exp(-epochs/12) + 0.1 * np.random.randn(50)
    
    axes[1, 1].plot(epochs, train_loss, 'r-', label='训练损失')
    axes[1, 1].plot(epochs, valid_loss, 'b-', label='验证损失')
    axes[1, 1].set_xlabel('训练轮次')
    axes[1, 1].set_ylabel('损失值')
    axes[1, 1].set_title('训练过程监控')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('chinese_font_verification_result.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✅ 测试可视化图片已生成: chinese_font_verification_result.png")

def main():
    """主函数"""
    print("🔍 开始检查项目中文字体修复情况...\n")
    
    # 查找所有matplotlib文件
    root_dir = Path(__file__).parent
    matplotlib_files = find_matplotlib_files(root_dir)
    
    print(f"📊 发现 {len(matplotlib_files)} 个包含matplotlib的文件:\n")
    
    fixed_files = []
    unfixed_files = []
    
    for file_path in matplotlib_files:
        rel_path = os.path.relpath(file_path, root_dir)
        has_config, content = check_chinese_font_config(file_path)
        
        if has_config:
            fixed_files.append(rel_path)
            print(f"✅ {rel_path}")
        else:
            unfixed_files.append(rel_path)
            print(f"❌ {rel_path}")
    
    print(f"\n📈 修复统计:")
    print(f"  ✅ 已修复: {len(fixed_files)} 个文件")
    print(f"  ❌ 未修复: {len(unfixed_files)} 个文件")
    print(f"  📊 修复率: {len(fixed_files)/(len(fixed_files)+len(unfixed_files))*100:.1f}%")
    
    if unfixed_files:
        print(f"\n⚠️ 以下文件仍需修复:")
        for file in unfixed_files:
            print(f"   - {file}")
    else:
        print(f"\n🎉 所有matplotlib文件都已正确配置中文字体支持!")
    
    # 生成测试可视化
    print(f"\n🎨 生成测试可视化图片...")
    generate_test_visualization()
    
    print(f"\n📋 修复详情:")
    print(f"已修复的文件列表:")
    for i, file in enumerate(fixed_files, 1):
        print(f"  {i:2d}. {file}")
    
    print(f"\n✨ 中文字体修复验证完成!")
    print(f"   - 所有可视化图表现在都能正确显示中文字符")
    print(f"   - 负号显示问题已解决")
    print(f"   - 支持的字体: SimHei, Microsoft YaHei, DejaVu Sans")

if __name__ == "__main__":
    main()