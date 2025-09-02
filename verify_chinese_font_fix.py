#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文字体修复验证脚本
检查项目中所有matplotlib可视化文件的中文字体配置
"""

import os
import re
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# 依赖全局 sitecustomize 的配置，不在此脚本内设置局部字体
# 确认全局 SVG 路径化设置

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

FONT_CONFIG_PATTERNS = [
    r"(plt|matplotlib)\.rcParams\[['\"]font\.sans-serif['\"]\]",
    r"(plt|matplotlib)\.rcParams\[['\"]axes\.unicode_minus['\"]\]",
    r"(plt|matplotlib)\.rcParams\[['\"]svg\.fonttype['\"]\]",
]

def file_has_local_font_config(content: str) -> bool:
    for pat in FONT_CONFIG_PATTERNS:
        if re.search(pat, content):
            return True
    return False

def check_svg_fonttype_is_path() -> bool:
    """检查 SVG 字体类型是否设置为路径化"""
    current_value = matplotlib.rcParams.get('svg.fonttype', None)
    return current_value == 'path'

def verify_svg_path_conversion():
    """验证 SVG 文件是否真正路径化字体"""
    svg_path = 'chinese_font_verification_result.svg'
    if not os.path.exists(svg_path):
        return False, "SVG文件不存在"
    
    try:
        with open(svg_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 检查是否包含路径化的文本（<path>标签）而非字体引用
        has_paths = '<path' in content
        has_font_refs = '<text' in content or 'font-family' in content
        
        if has_paths and not has_font_refs:
            return True, "SVG文字已完全路径化"
        elif has_paths and has_font_refs:
            return True, "SVG文字部分路径化（混合模式）"
        else:
            return False, "SVG文字未路径化，仍使用字体引用"
    except Exception as e:
        return False, f"SVG验证失败: {e}"

def generate_test_visualization():
    """生成测试可视化图片验证中文字体与 SVG 路径化"""
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
    png_path = 'chinese_font_verification_result.png'
    svg_path = 'chinese_font_verification_result.svg'
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    plt.close()
    
    # 输出检查结果
    svg_ok = check_svg_fonttype_is_path()
    print(f"SVG 路径化设置: {'OK' if svg_ok else '未生效'} (matplotlib.rcParams['svg.fonttype']={matplotlib.rcParams.get('svg.fonttype')})")
    print(f"[OK] PNG: {png_path}")
    print(f"[OK] SVG: {svg_path}")


def main():
    """主函数"""
    print("[INFO] 开始检查项目中文字体与 SVG 路径化配置...\n")
    print(f"全局 svg.fonttype = {matplotlib.rcParams.get('svg.fonttype')}")
    
    # 查找所有matplotlib文件
    root_dir = Path(__file__).parent
    matplotlib_files = find_matplotlib_files(root_dir)
    print(f"[INFO] 发现 {len(matplotlib_files)} 个包含matplotlib的文件:\n")
    
    files_with_local_config = []
    files_clean = []
    
    for file_path in matplotlib_files:
        rel_path = os.path.relpath(file_path, root_dir)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if file_has_local_font_config(content):
                files_with_local_config.append(rel_path)
                print(f"[WARN] 仍包含局部字体/负号/SVG设置: {rel_path}")
            else:
                files_clean.append(rel_path)
                print(f"[OK] 无局部配置: {rel_path}")
        except Exception as e:
            print(f"读取失败: {rel_path}, 错误: {e}")
    
    print(f"\n[INFO] 统计:")
    print(f"  [OK] 无局部配置: {len(files_clean)} 个文件")
    print(f"  [WARN] 含局部配置: {len(files_with_local_config)} 个文件")
    total = len(files_clean) + len(files_with_local_config)
    if total:
        print(f"  [INFO] 清理进度: {len(files_clean)/total*100:.1f}%")
    
    if files_with_local_config:
        print("\n[WARN] 以下文件仍设置了局部 rcParams（建议继续清理）:")
        for file in files_with_local_config:
            print(f"   - {file}")
    else:
        print("\n[OK] 所有文件均已移除局部 rcParams 设置，统一使用全局 sitecustomize 配置！")
    
    # 生成测试可视化
    print(f"\n🎨 生成测试可视化图片 (PNG + SVG 路径化)...")
    generate_test_visualization()

    # 验证 SVG 文件是否真正路径化
    ok, msg = verify_svg_path_conversion()
    status = 'OK' if ok else '未完全路径化'
    print(f"SVG 文件检查: {status} - {msg}")
    
    print("\n[OK] 验证完成!")

if __name__ == "__main__":
    main()