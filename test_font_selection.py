#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
字体选择诊断脚本
检查系统中实际可用的中文字体以及 matplotlib 实际使用的字体
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.backends.backend_agg import FigureCanvasAgg
import os

def check_available_fonts():
    """检查系统中可用的字体"""
    print("=== 系统字体检查 ===")
    
    # 获取所有字体
    all_fonts = [f.name for f in fm.fontManager.ttflist]
    
    # 中文字体候选列表
    chinese_candidates = [
        'Microsoft YaHei', 'SimHei', 'SimSun', 'Source Han Sans SC',
        'Noto Sans CJK SC', 'Noto Sans SC', 'WenQuanYi Micro Hei', 
        'WenQuanYi Zen Hei', 'PingFang SC', 'Arial Unicode MS'
    ]
    
    print("候选中文字体可用性:")
    available_chinese = []
    for font in chinese_candidates:
        if font in all_fonts:
            print(f"  [OK] {font} - 可用")
            available_chinese.append(font)
        else:
            print(f"  [ERROR] {font} - 不可用")
    
    print(f"\n总计可用中文字体: {len(available_chinese)} 个")
    return available_chinese

def check_matplotlib_config():
    """检查当前 matplotlib 配置"""
    print("\n=== Matplotlib 配置检查 ===")
    
    print(f"font.family: {matplotlib.rcParams['font.family']}")
    print(f"font.sans-serif: {matplotlib.rcParams['font.sans-serif']}")
    print(f"axes.unicode_minus: {matplotlib.rcParams['axes.unicode_minus']}")
    print(f"svg.fonttype: {matplotlib.rcParams['svg.fonttype']}")

def test_actual_font_used():
    """测试实际使用的字体"""
    print("\n=== 实际字体使用测试 ===")
    
    # 创建图形但不显示
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 添加中文文本
    text_obj = ax.text(0.5, 0.5, '测试中文字体显示', ha='center', va='center', fontsize=16)
    
    # 获取实际使用的字体
    # 需要先渲染才能获取实际字体
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    
    # 获取文本对象的字体属性
    font_path = text_obj.get_fontname()
    font_family = text_obj.get_family()
    
    print(f"实际使用的字体名称: {font_path}")
    print(f"字体族: {font_family}")
    
    # 检查是否回退到 DejaVu Sans
    if 'DejaVu' in str(font_path):
        print("[WARN] 警告: 正在使用 DejaVu Sans，这意味着中文字体设置未生效")
        
        # 尝试手动设置字体
        available_chinese = check_available_fonts()
        if available_chinese:
            print(f"\n尝试手动设置为: {available_chinese[0]}")
            text_obj.set_fontfamily(available_chinese[0])
            canvas.draw()
            new_font = text_obj.get_fontname()
            print(f"手动设置后的字体: {new_font}")
    else:
        print("[OK] 成功使用中文字体")
    
    plt.close(fig)

def create_font_test_image():
    """创建字体测试图像"""
    print("\n=== 生成字体测试图像 ===")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('字体测试 - 各种中文字符', fontsize=16)
    
    # 测试不同类型的中文字符
    test_texts = [
        "基础中文: 数据可视化",
        "数学符号: α β γ δ ∑ ∫ √",  
        "特殊字符: ①②③④⑤",
        "混合文本: Data数据Loss损失"
    ]
    
    for i, (ax, text) in enumerate(zip(axes.flat, test_texts)):
        ax.text(0.5, 0.5, text, ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        ax.set_title(f"测试 {i+1}")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    png_path = 'font_test_detailed.png'
    svg_path = 'font_test_detailed.svg'
    
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_path, bbox_inches='tight')
    
    print(f"[OK] 生成测试图像: {png_path}, {svg_path}")
    
    plt.close()
    
    return png_path, svg_path

def main():
    """主函数"""
    print("[INFO] 开始详细字体诊断...\n")
    
    # 检查系统字体
    available_chinese = check_available_fonts()
    
    # 检查 matplotlib 配置
    check_matplotlib_config()
    
    # 测试实际使用的字体
    test_actual_font_used()
    
    # 生成测试图像
    png_path, svg_path = create_font_test_image()
    
    print("\n" + "="*50)
    print("[INFO] 诊断总结:")
    
    if not available_chinese:
        print("[ERROR] 系统中没有检测到常见的中文字体")
        print("[TIP] 建议:")
        print("   1. 安装 Source Han Sans SC (思源黑体)")
        print("   2. 或安装 Noto Sans CJK SC")
        print("   3. 下载地址:")
        print("      - 思源黑体: https://github.com/adobe-fonts/source-han-sans/releases")
        print("      - Noto Sans: https://fonts.google.com/noto/specimen/Noto+Sans+SC")
    else:
        print(f"[OK] 发现 {len(available_chinese)} 个可用中文字体")
        print("🔧 如果仍显示方框，可能的原因:")
        print("   1. matplotlib 字体缓存需要清理")
        print("   2. 字体优先级设置有问题")
        print("   3. 需要重启 Python 解释器")
    
    print(f"\n[INFO] 生成的测试文件:")
    print(f"   - {png_path}")
    print(f"   - {svg_path}")
    print("\n请检查这些文件中的中文字符是否正常显示。")

if __name__ == "__main__":
    main()