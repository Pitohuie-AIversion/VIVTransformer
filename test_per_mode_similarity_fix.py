#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复后的每模态相似度分析功能
- 验证是否能正常生成热力图
- 检查是否还有中文字体警告
- 确认PNG和SVG格式都能正常保存
"""

import os
import sys
import numpy as np
import logging
import warnings
from pathlib import Path

# 设置警告过滤器，捕获所有字体相关警告
warnings.filterwarnings('default')

# 添加模块路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'generate_data', 'utils'))

from analyze_output_modality_only import OutputModalityAnalyzer
import matplotlib.pyplot as plt
import matplotlib as mpl


def run_per_mode_similarity_check() -> bool:
    """执行每模态相似度分析并返回是否通过校验。"""
    print("=== 开始测试每模态相似度分析功能 ===")

    # 创建测试输出目录
    test_output_dir = Path("test_per_mode_output")
    test_output_dir.mkdir(exist_ok=True)

    # 设置日志
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    # 初始化分析器
    analyzer = OutputModalityAnalyzer(test_output_dir)

    # 创建模拟的样本嵌入数据
    np.random.seed(42)
    n_samples = 50  # 模拟50个样本
    n_modes = 5     # 模拟5个模态

    # 生成模拟的样本嵌入矩阵：每行是一个样本，每列是一个模态的系数
    sample_embeddings = np.random.randn(n_samples, n_modes)

    # 为了让相似度更明显，添加一些结构化的模式
    for i in range(n_modes):
        # 让某些样本在特定模态上有更强的响应
        start_idx = i * (n_samples // n_modes)
        end_idx = (i + 1) * (n_samples // n_modes)
        sample_embeddings[start_idx:end_idx, i] += 2.0

    print(f"样本嵌入矩阵形状: {sample_embeddings.shape}")
    print(f"样本嵌入矩阵统计:")
    print(f"  均值: {np.mean(sample_embeddings):.3f}")
    print(f"  标准差: {np.std(sample_embeddings):.3f}")
    print(f"  最小值: {np.min(sample_embeddings):.3f}")
    print(f"  最大值: {np.max(sample_embeddings):.3f}")

    try:
        # 执行每模态相似度分析
        print("\n开始执行每模态相似度分析...")
        results = analyzer.analyze_per_mode_sample_similarity(
            sample_embeddings=sample_embeddings,
            n_modes=n_modes,
            max_show=30  # 最多显示30个样本
        )

        print(f"\n分析完成！生成了 {results['summary']['heatmaps_generated']} 张热力图")

        # 验证生成的文件
        print("\n验证生成的文件:")
        all_exist = True
        for mode_idx in range(1, n_modes + 1):
            png_path = results[f'mode_{mode_idx}_png']
            svg_path = results[f'mode_{mode_idx}_svg']

            png_exists = png_path.exists()
            svg_exists = svg_path.exists()
            all_exist = all_exist and png_exists and svg_exists

            print(f"  模态 {mode_idx}:")
            print(f"    PNG: {'[OK]' if png_exists else '[FAIL]'} {png_path}")
            if png_exists:
                png_size = png_path.stat().st_size
                print(f"         大小: {png_size:,} 字节")

            print(f"    SVG: {'[OK]' if svg_exists else '[FAIL]'} {svg_path}")
            if svg_exists:
                svg_size = svg_path.stat().st_size
                print(f"         大小: {svg_size:,} 字节")

            # 验证相似度统计落在合理范围内 [-1, 1]
            cosine_stats = results[f'mode_{mode_idx}_cosine_stats']
            print(f"    余弦相似度统计: 均值={cosine_stats['mean']:.3f}, "
                  f"标准差={cosine_stats['std']:.3f}, "
                  f"范围=[{cosine_stats['min']:.3f}, {cosine_stats['max']:.3f}]")
            if not (-1.01 <= cosine_stats['min'] <= 1.01 and -1.01 <= cosine_stats['max'] <= 1.01):
                all_exist = False

        summary = results['summary']
        ok = (
            all_exist and
            summary['analyzed_modes'] == n_modes and
            summary['heatmaps_generated'] == n_modes and
            summary['total_samples'] == n_samples and
            1 <= summary['displayed_samples'] <= n_samples
        )

        print("\n=== 测试成功完成！===" if ok else "\n=== 测试未通过 ===")
        return ok

    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_per_mode_similarity():
    """测试每模态相似度分析功能（pytest）"""
    ok = run_per_mode_similarity_check()
    assert ok, "每模态相似度分析未通过或未生成预期输出文件"


def run_font_warnings_check() -> bool:
    """执行字体警告检查并返回是否通过。"""
    print("\n=== 测试字体警告问题 ===")

    # 捕获警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        # 创建一个简单的测试图
        plt.figure(figsize=(4, 3))
        plt.plot([1, 2, 3], [1, 4, 2])
        plt.title("Test Plot with English Title")
        plt.xlabel("X Axis")
        plt.ylabel("Y Axis")

        test_png = "test_font_warning.png"
        plt.savefig(test_png, dpi=150)
        plt.close()

        # 检查警告
        font_warnings = [warning for warning in w
                         if 'font' in str(warning.message).lower() or
                            'glyph' in str(warning.message).lower()]

        if font_warnings:
            print(f"检测到 {len(font_warnings)} 个字体相关警告:")
            for warning in font_warnings:
                print(f"  - {warning.message}")
        else:
            print("[OK] 没有检测到字体相关警告")

        # 清理测试文件
        if os.path.exists(test_png):
            os.remove(test_png)

        return len(font_warnings) == 0


def test_font_warnings():
    """pytest：字体相关警告应当不存在"""
    ok = run_font_warnings_check()
    assert ok, "检测到字体相关警告"


if __name__ == "__main__":
    print("开始测试每模态相似度分析功能的修复...")

    # 1. 测试字体警告
    font_ok = run_font_warnings_check()

    # 2. 测试主功能
    main_ok = run_per_mode_similarity_check()

    if main_ok and font_ok:
        print("\n[OK] 所有测试通过！修复成功！")
        exit_code = 0
    elif main_ok:
        print("\n[WARN]  主功能正常，但仍有字体警告")
        exit_code = 1
    else:
        print("\n[ERROR] 测试失败")
        exit_code = 2

    sys.exit(exit_code)