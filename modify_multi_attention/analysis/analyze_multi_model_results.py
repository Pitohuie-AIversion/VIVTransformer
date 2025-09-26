#!/usr/bin/env python3
"""
多模型训练结果分析脚本
分析JSON结果文件并生成汇总报告和可视化图表
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results(json_path):
    """加载训练结果JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def extract_model_metrics(results):
    """提取模型关键指标"""
    metrics = {}
    
    for model_name, model_data in results.items():
        if isinstance(model_data, dict) and 'param_count' in model_data:
            metrics[model_name] = {
                'param_count': model_data['param_count'],
                'final_train_loss': model_data['train_losses'][-1] if model_data['train_losses'] else None,
                'final_val_loss': model_data['val_losses'][-1] if model_data['val_losses'] else None,
                'test_loss': model_data['test_loss'],
                'test_mse': model_data['test_metrics']['mse'],
                'test_r2': model_data['test_metrics']['r2_score'],
                'train_losses': model_data['train_losses'],
                'val_losses': model_data['val_losses']
            }
    
    return metrics

def create_summary_report(metrics, output_path):
    """创建汇总报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report = f"""# 多模型训练结果汇总报告
生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 模型性能对比

| 模型 | 参数量 | 测试MSE | 测试R² | 最终训练损失 | 最终验证损失 |
|------|--------|---------|--------|-------------|-------------|
"""
    
    # 按参数量排序
    sorted_models = sorted(metrics.items(), key=lambda x: x[1]['param_count'])
    
    for model_name, model_metrics in sorted_models:
        param_count = f"{model_metrics['param_count']:,}"
        test_mse = f"{model_metrics['test_mse']:.6f}"
        test_r2 = f"{model_metrics['test_r2']:.6f}"
        train_loss = f"{model_metrics['final_train_loss']:.6f}" if model_metrics['final_train_loss'] else "N/A"
        val_loss = f"{model_metrics['final_val_loss']:.6f}" if model_metrics['final_val_loss'] else "N/A"
        
        report += f"| {model_name} | {param_count} | {test_mse} | {test_r2} | {train_loss} | {val_loss} |\n"
    
    report += f"""
## 关键发现

### 参数效率分析
"""
    
    # 找到最佳性能模型
    best_mse_model = min(metrics.items(), key=lambda x: x[1]['test_mse'])
    best_r2_model = max(metrics.items(), key=lambda x: x[1]['test_r2'])
    most_efficient_model = min(metrics.items(), key=lambda x: x[1]['param_count'])
    
    report += f"""
- **最低MSE**: {best_mse_model[0]} (MSE: {best_mse_model[1]['test_mse']:.6f})
- **最高R²**: {best_r2_model[0]} (R²: {best_r2_model[1]['test_r2']:.6f})
- **最少参数**: {most_efficient_model[0]} ({most_efficient_model[1]['param_count']:,} 参数)

### 模型特点分析
"""
    
    for model_name, model_metrics in sorted_models:
        efficiency_ratio = model_metrics['test_mse'] * model_metrics['param_count'] / 1e6
        report += f"- **{model_name}**: 参数量 {model_metrics['param_count']:,}, 效率比 {efficiency_ratio:.2f}\n"
    
    # 保存报告
    report_path = output_path / f"multi_model_analysis_report_{timestamp}.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ 汇总报告已保存到: {report_path}")
    return report_path

def create_comparison_plots(metrics, output_path):
    """创建对比图表"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 准备数据
    models = list(metrics.keys())
    param_counts = [metrics[m]['param_count'] for m in models]
    test_mses = [metrics[m]['test_mse'] for m in models]
    test_r2s = [metrics[m]['test_r2'] for m in models]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('多模型性能对比分析', fontsize=16, fontweight='bold')
    
    # 1. 参数量 vs MSE
    axes[0, 0].scatter(param_counts, test_mses, s=100, alpha=0.7, c=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    for i, model in enumerate(models):
        axes[0, 0].annotate(model, (param_counts[i], test_mses[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
    axes[0, 0].set_xlabel('参数量')
    axes[0, 0].set_ylabel('测试MSE')
    axes[0, 0].set_title('参数量 vs 测试MSE')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 参数量 vs R²
    axes[0, 1].scatter(param_counts, test_r2s, s=100, alpha=0.7, c=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    for i, model in enumerate(models):
        axes[0, 1].annotate(model, (param_counts[i], test_r2s[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=10)
    axes[0, 1].set_xlabel('参数量')
    axes[0, 1].set_ylabel('测试R²')
    axes[0, 1].set_title('参数量 vs 测试R²')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 模型性能条形图
    x_pos = np.arange(len(models))
    axes[1, 0].bar(x_pos, test_mses, alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
    axes[1, 0].set_xlabel('模型')
    axes[1, 0].set_ylabel('测试MSE')
    axes[1, 0].set_title('各模型测试MSE对比')
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels(models, rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 训练损失曲线
    for model in models:
        if metrics[model]['train_losses']:
            epochs = range(len(metrics[model]['train_losses']))
            axes[1, 1].plot(epochs, metrics[model]['train_losses'], 
                           marker='o', label=f'{model} (训练)', linewidth=2)
            if metrics[model]['val_losses']:
                axes[1, 1].plot(epochs, metrics[model]['val_losses'], 
                               marker='s', label=f'{model} (验证)', linewidth=2, linestyle='--')
    
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('损失值')
    axes[1, 1].set_title('训练损失曲线')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    plot_path = output_path / f"multi_model_comparison_plots_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 对比图表已保存到: {plot_path}")
    return plot_path

def main():
    """主函数"""
    # 设置路径
    results_path = Path("results/compatibility_test/training_results.json")
    output_dir = Path("utils/visualization_results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not results_path.exists():
        print(f"❌ 结果文件不存在: {results_path}")
        return
    
    print("🔍 开始分析多模型训练结果...")
    
    # 加载结果
    results = load_results(results_path)
    print(f"📊 加载了 {len(results)} 个模型的结果")
    
    # 提取指标
    metrics = extract_model_metrics(results)
    print(f"📈 提取了 {len(metrics)} 个模型的关键指标")
    
    # 生成汇总报告
    print("📝 生成汇总报告...")
    report_path = create_summary_report(metrics, output_dir)
    
    # 生成对比图表
    print("📊 生成对比图表...")
    plot_path = create_comparison_plots(metrics, output_dir)
    
    print("\n🎉 分析完成！")
    print(f"📄 汇总报告: {report_path}")
    print(f"📊 对比图表: {plot_path}")

if __name__ == "__main__":
    main()