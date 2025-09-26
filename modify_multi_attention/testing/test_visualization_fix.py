#!/usr/bin/env python3
"""
测试修复后的可视化功能
验证utils.visualization模块是否能正确处理fixed_models_test_results.json数据
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent))

from utils.visualization import create_model_comparison_plots, generate_model_comparison_summary

def test_visualization_with_fixed_models_data():
    """测试使用fixed_models_test_results.json数据的可视化功能"""
    
    print("🔧 测试修复后的可视化功能...")
    
    # 检查数据文件是否存在
    data_file = "fixed_models_test_results.json"
    if not os.path.exists(data_file):
        print(f"❌ 数据文件不存在: {data_file}")
        return False
    
    try:
        # 加载数据
        with open(data_file, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        print(f"✅ 成功加载数据文件: {data_file}")
        print(f"📊 包含模型数量: {len(results)}")
        
        # 显示数据结构
        for model_name, model_data in results.items():
            print(f"   - {model_name}: 状态={model_data.get('status', 'unknown')}")
            if 'final_test_mse' in model_data:
                print(f"     MSE={model_data['final_test_mse']:.6f}, R²={model_data['final_test_r2']:.4f}")
        
        # 过滤成功的结果
        successful_results = {
            name: data for name, data in results.items() 
            if data.get('status') == 'success'
        }
        
        if not successful_results:
            print("❌ 没有成功的训练结果")
            return False
        
        print(f"✅ 成功模型数量: {len(successful_results)}")
        
        # 创建输出目录
        output_dir = "test_visualization_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # 测试可视化图表生成
        print("\n🎨 测试可视化图表生成...")
        try:
            plot_files = create_model_comparison_plots(
                successful_results, 
                output_dir=output_dir,
                timestamp="test_fix"
            )
            print(f"✅ 成功生成 {len(plot_files)} 个图表文件:")
            for plot_file in plot_files:
                if os.path.exists(plot_file):
                    print(f"   ✓ {plot_file}")
                else:
                    print(f"   ❌ {plot_file} (文件未生成)")
        except Exception as e:
            print(f"❌ 图表生成失败: {e}")
            return False
        
        # 测试汇总报告生成
        print("\n📝 测试汇总报告生成...")
        try:
            summary_file = generate_model_comparison_summary(
                successful_results,
                plot_files,
                output_dir=output_dir,
                timestamp="test_fix"
            )
            if os.path.exists(summary_file):
                print(f"✅ 成功生成汇总报告: {summary_file}")
                
                # 显示报告前几行
                with open(summary_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()[:10]
                    print("📄 报告预览:")
                    for line in lines:
                        print(f"   {line.rstrip()}")
            else:
                print(f"❌ 汇总报告未生成: {summary_file}")
                return False
        except Exception as e:
            print(f"❌ 汇总报告生成失败: {e}")
            return False
        
        print("\n🎉 可视化功能测试完成！")
        print(f"📁 输出目录: {output_dir}")
        return True
        
    except Exception as e:
        print(f"❌ 测试过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("🧪 可视化功能修复验证测试")
    print("=" * 60)
    
    success = test_visualization_with_fixed_models_data()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ 所有测试通过！可视化功能已修复")
    else:
        print("❌ 测试失败，需要进一步检查")
    print("=" * 60)

if __name__ == "__main__":
    main()