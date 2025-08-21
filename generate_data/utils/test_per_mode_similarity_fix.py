#!/usr/bin/env python3
"""
测试每模态样本相似度分析修复后的功能
"""

import numpy as np
from pathlib import Path
import logging
import warnings

# 设置日志和警告
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 设置忽略一些字体相关的警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")

def test_per_mode_similarity():
    """测试每模态样本相似度分析功能"""
    print("开始测试每模态样本相似度分析...")
    
    # 导入分析器
    from analyze_output_modality_only import OutputModalityAnalyzer
    
    # 创建输出目录
    output_dir = Path("test_per_mode_output")
    output_dir.mkdir(exist_ok=True)
    
    # 创建分析器实例
    analyzer = OutputModalityAnalyzer(output_dir=output_dir)
    
    # 生成模拟数据：50个样本，5个模态
    n_samples = 50
    n_modes = 5
    
    # 创建具有明显模态特征的数据
    np.random.seed(42)
    sample_embeddings = np.random.randn(n_samples, n_modes)
    
    # 为每个模态添加不同的特征模式
    for i in range(n_modes):
        # 让一部分样本在第i个模态上有相似的值
        group_size = n_samples // (i + 2)
        sample_embeddings[:group_size, i] += 2.0  # 增强第i个模态的信号
    
    print(f"生成模拟数据: {n_samples} 样本, {n_modes} 模态")
    print(f"样本嵌入矩阵形状: {sample_embeddings.shape}")
    
    # 执行每模态样本相似度分析
    try:
        result = analyzer.analyze_per_mode_sample_similarity(
            sample_embeddings=sample_embeddings,
            n_modes=n_modes,
            max_show=n_samples  # 显示所有样本
        )
        
        # 检查结果
        print("\n分析结果:")
        print(f"生成的热力图数量: {result['summary']['heatmaps_generated']}")
        print(f"分析的模态数: {result['summary']['analyzed_modes']}")
        print(f"总样本数: {result['summary']['total_samples']}")
        print(f"显示样本数: {result['summary']['displayed_samples']}")
        
        # 验证文件是否生成
        print("\n生成的文件:")
        png_count = 0
        svg_count = 0
        
        for mode_idx in range(1, n_modes + 1):
            png_key = f'mode_{mode_idx}_png'
            svg_key = f'mode_{mode_idx}_svg'
            
            if png_key in result:
                png_path = result[png_key]
                if png_path.exists():
                    print(f"✓ 模态 {mode_idx} PNG: {png_path.name}")
                    png_count += 1
                else:
                    print(f"✗ 模态 {mode_idx} PNG 文件不存在: {png_path}")
            
            if svg_key in result:
                svg_path = result[svg_key]
                if svg_path.exists():
                    print(f"✓ 模态 {mode_idx} SVG: {svg_path.name}")
                    svg_count += 1
                else:
                    print(f"✗ 模态 {mode_idx} SVG 文件不存在: {svg_path}")
                    
            # 显示统计信息
            cosine_stats_key = f'mode_{mode_idx}_cosine_stats'
            if cosine_stats_key in result:
                stats = result[cosine_stats_key]
                print(f"  - 余弦相似度: 均值={stats['mean']:.3f}, 标准差={stats['std']:.3f}")
        
        print(f"\n文件生成统计:")
        print(f"PNG 文件: {png_count}/{n_modes}")
        print(f"SVG 文件: {svg_count}/{n_modes}")
        
        # 测试结果判断
        success = (png_count == n_modes and svg_count == n_modes)
        
        if success:
            print("\n🎉 测试通过！所有预期的热力图都已成功生成")
        else:
            print(f"\n❌ 测试失败！预期 {n_modes} 张PNG和SVG，实际生成 {png_count} PNG, {svg_count} SVG")
            
        return success
        
    except Exception as e:
        print(f"\n❌ 测试失败，发生异常: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("每模态样本相似度分析功能测试")
    print("=" * 60)
    
    success = test_per_mode_similarity()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 所有测试通过！")
    else:
        print("❌ 测试失败！")
    print("=" * 60)