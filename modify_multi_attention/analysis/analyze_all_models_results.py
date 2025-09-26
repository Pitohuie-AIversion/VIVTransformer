#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析所有模型测试结果的脚本

功能:
1. 解析训练结果JSON文件
2. 生成模型性能对比报告
3. 创建可视化图表
4. 分析错误和成功情况

作者: AI Assistant
日期: 2025-01-22
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from datetime import datetime
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results(results_path):
    """加载测试结果"""
    try:
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        logger.info(f"✅ 成功加载结果文件: {results_path}")
        return results
    except Exception as e:
        logger.error(f"❌ 加载结果文件失败: {e}")
        return None

def analyze_model_results(results):
    """分析模型结果"""
    analysis = {
        'successful_models': {},
        'failed_models': {},
        'summary': {}
    }
    
    for model_name, model_data in results.items():
        if 'error' in model_data:
            # 失败的模型
            analysis['failed_models'][model_name] = {
                'error': model_data['error'],
                'config': model_data.get('config', {})
            }
            logger.warning(f"❌ {model_name} 模型训练失败: {model_data['error']}")
        else:
            # 成功的模型
            train_losses = model_data.get('train_losses', [])
            val_losses = model_data.get('val_losses', [])
            test_metrics = model_data.get('test_metrics', {})
            
            analysis['successful_models'][model_name] = {
                'param_count': model_data.get('param_count', 0),
                'final_train_loss': train_losses[-1] if train_losses else None,
                'final_val_loss': val_losses[-1] if val_losses else None,
                'best_val_loss': min(val_losses) if val_losses else None,
                'test_mse': test_metrics.get('test_mse', None),
                'test_mae': test_metrics.get('test_mae', None),
                'test_r2': test_metrics.get('test_r2', None),
                'convergence_epochs': len(train_losses),
                'config': model_data.get('config', {})
            }
            logger.info(f"✅ {model_name} 模型训练成功")
    
    # 生成摘要
    analysis['summary'] = {
        'total_models': len(results),
        'successful_count': len(analysis['successful_models']),
        'failed_count': len(analysis['failed_models']),
        'success_rate': len(analysis['successful_models']) / len(results) * 100
    }
    
    return analysis

def create_comparison_plots(analysis, output_dir):
    """创建对比图表"""
    successful_models = analysis['successful_models']
    
    if not successful_models:
        logger.warning("⚠️ 没有成功的模型，无法生成对比图表")
        return []
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    plot_files = []
    
    # 1. 参数量对比
    fig, ax = plt.subplots(figsize=(10, 6))
    models = list(successful_models.keys())
    param_counts = [successful_models[m]['param_count'] for m in models]
    
    bars = ax.bar(models, param_counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax.set_ylabel('参数数量')
    ax.set_title('模型参数量对比')
    ax.tick_params(axis='x', rotation=45)
    
    # 添加数值标签
    for bar, count in zip(bars, param_counts):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{count:,}', ha='center', va='bottom')
    
    plt.tight_layout()
    param_plot_path = output_dir / 'parameter_comparison.png'
    plt.savefig(param_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(str(param_plot_path))
    logger.info(f"📊 参数量对比图已保存: {param_plot_path}")
    
    # 2. 性能对比（如果有测试指标）
    test_mse_values = [successful_models[m]['test_mse'] for m in models if successful_models[m]['test_mse'] is not None]
    if test_mse_values:
        fig, ax = plt.subplots(figsize=(10, 6))
        valid_models = [m for m in models if successful_models[m]['test_mse'] is not None]
        valid_mse = [successful_models[m]['test_mse'] for m in valid_models]
        
        bars = ax.bar(valid_models, valid_mse, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
        ax.set_ylabel('测试MSE损失')
        ax.set_title('模型性能对比（测试MSE）')
        ax.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, mse in zip(bars, valid_mse):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{mse:.6f}', ha='center', va='bottom')
        
        plt.tight_layout()
        mse_plot_path = output_dir / 'performance_comparison.png'
        plt.savefig(mse_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(str(mse_plot_path))
        logger.info(f"📊 性能对比图已保存: {mse_plot_path}")
    
    # 3. 训练损失曲线对比（如果有训练历史）
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, model_name in enumerate(models):
        model_data = successful_models[model_name]
        if 'final_train_loss' in model_data and model_data['final_train_loss'] is not None:
            # 这里我们只能显示最终损失，因为完整的训练历史可能太大
            ax.scatter([model_data['convergence_epochs']], [model_data['final_train_loss']], 
                      color=colors[i % len(colors)], label=f'{model_name} (最终损失)', s=100)
    
    ax.set_xlabel('训练轮数')
    ax.set_ylabel('训练损失')
    ax.set_title('模型训练收敛情况')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_plot_path = output_dir / 'training_convergence.png'
    plt.savefig(loss_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(str(loss_plot_path))
    logger.info(f"📊 训练收敛图已保存: {loss_plot_path}")
    
    return plot_files

def generate_report(analysis, plot_files, output_dir):
    """生成详细报告"""
    output_dir = Path(output_dir)
    report_path = output_dir / 'model_comparison_report.md'
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"# 多模型测试结果分析报告\n\n")
        f.write(f"**生成时间**: {timestamp}\n\n")
        
        # 执行摘要
        f.write("## 执行摘要\n\n")
        summary = analysis['summary']
        f.write(f"- **总模型数**: {summary['total_models']}\n")
        f.write(f"- **成功模型数**: {summary['successful_count']}\n")
        f.write(f"- **失败模型数**: {summary['failed_count']}\n")
        f.write(f"- **成功率**: {summary['success_rate']:.1f}%\n\n")
        
        # 成功模型详情
        if analysis['successful_models']:
            f.write("## 成功模型详情\n\n")
            f.write("| 模型名称 | 参数数量 | 最终训练损失 | 最佳验证损失 | 测试MSE | 训练轮数 |\n")
            f.write("|---------|---------|-------------|-------------|---------|----------|\n")
            
            for model_name, data in analysis['successful_models'].items():
                param_count = f"{data['param_count']:,}" if data['param_count'] else "N/A"
                train_loss = f"{data['final_train_loss']:.6f}" if data['final_train_loss'] else "N/A"
                val_loss = f"{data['best_val_loss']:.6f}" if data['best_val_loss'] else "N/A"
                test_mse = f"{data['test_mse']:.6f}" if data['test_mse'] else "N/A"
                epochs = data['convergence_epochs'] if data['convergence_epochs'] else "N/A"
                
                f.write(f"| {model_name} | {param_count} | {train_loss} | {val_loss} | {test_mse} | {epochs} |\n")
            
            f.write("\n")
        
        # 失败模型详情
        if analysis['failed_models']:
            f.write("## 失败模型详情\n\n")
            for model_name, data in analysis['failed_models'].items():
                f.write(f"### {model_name}\n")
                f.write(f"**错误信息**: {data['error']}\n\n")
                if data['config']:
                    f.write("**配置参数**:\n")
                    for key, value in data['config'].items():
                        f.write(f"- {key}: {value}\n")
                f.write("\n")
        
        # 性能排名
        if analysis['successful_models']:
            f.write("## 性能排名\n\n")
            
            # 按参数量排序
            f.write("### 按参数量排序（从少到多）\n")
            sorted_by_params = sorted(analysis['successful_models'].items(), 
                                    key=lambda x: x[1]['param_count'] or 0)
            for i, (model_name, data) in enumerate(sorted_by_params, 1):
                param_count = f"{data['param_count']:,}" if data['param_count'] else "N/A"
                f.write(f"{i}. **{model_name}**: {param_count} 参数\n")
            f.write("\n")
            
            # 按性能排序（如果有测试指标）
            models_with_mse = [(name, data) for name, data in analysis['successful_models'].items() 
                              if data['test_mse'] is not None]
            if models_with_mse:
                f.write("### 按测试MSE排序（从好到差）\n")
                sorted_by_mse = sorted(models_with_mse, key=lambda x: x[1]['test_mse'])
                for i, (model_name, data) in enumerate(sorted_by_mse, 1):
                    f.write(f"{i}. **{model_name}**: MSE = {data['test_mse']:.6f}\n")
                f.write("\n")
        
        # 可视化图表
        if plot_files:
            f.write("## 可视化图表\n\n")
            for plot_file in plot_files:
                plot_name = Path(plot_file).stem
                f.write(f"- [{plot_name}]({Path(plot_file).name})\n")
            f.write("\n")
        
        # 结论和建议
        f.write("## 结论和建议\n\n")
        
        if analysis['successful_models']:
            # 找到有测试MSE的模型
            models_with_mse = [(name, data) for name, data in analysis['successful_models'].items() 
                              if data['test_mse'] is not None]
            
            if models_with_mse:
                best_model = min(models_with_mse, key=lambda x: x[1]['test_mse'])
                f.write(f"### 最佳性能模型\n")
                f.write(f"**{best_model[0]}** 在测试集上表现最佳，MSE = {best_model[1]['test_mse']:.6f}\n\n")
            else:
                f.write(f"### 模型训练状态\n")
                f.write(f"所有成功的模型都完成了训练，但测试指标暂未计算完成\n\n")
            
            # 效率分析
            efficiency_scores = []
            for name, data in analysis['successful_models'].items():
                if data['test_mse'] and data['param_count']:
                    # 效率分数 = 1 / (MSE * log(参数量))
                    efficiency = 1 / (data['test_mse'] * np.log(data['param_count']))
                    efficiency_scores.append((name, efficiency))
            
            if efficiency_scores:
                best_efficiency = max(efficiency_scores, key=lambda x: x[1])
                f.write(f"### 最佳效率模型\n")
                f.write(f"**{best_efficiency[0]}** 在性能和参数量平衡方面表现最佳\n\n")
        
        if analysis['failed_models']:
            f.write("### 失败模型分析\n")
            f.write("以下模型需要进一步调试和优化:\n")
            for model_name in analysis['failed_models'].keys():
                f.write(f"- **{model_name}**: 需要检查模型架构和数据维度匹配\n")
            f.write("\n")
        
        f.write("### 总体建议\n")
        f.write("1. 对于失败的模型，建议检查输入输出维度匹配问题\n")
        f.write("2. 考虑使用成功模型的架构作为基准进行进一步优化\n")
        f.write("3. 可以尝试调整失败模型的超参数和架构设计\n")
        f.write("4. 建议进行更长时间的训练以获得更稳定的性能评估\n")
    
    logger.info(f"📄 详细报告已保存: {report_path}")
    return str(report_path)

def main():
    """主函数"""
    logger.info("🚀 开始分析模型测试结果...")
    
    # 查找最新的结果文件
    results_dir = Path("unified_training_results")
    if not results_dir.exists():
        logger.error("❌ 结果目录不存在")
        return
    
    # 查找training_results.json文件
    results_file = results_dir / "training_results.json"
    if not results_file.exists():
        logger.error("❌ 结果文件不存在")
        return
    
    # 加载结果
    results = load_results(results_file)
    if not results:
        return
    
    # 分析结果
    logger.info("📊 分析模型结果...")
    analysis = analyze_model_results(results)
    
    # 创建输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"model_analysis_results_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成图表
    logger.info("🎨 生成对比图表...")
    plot_files = create_comparison_plots(analysis, output_dir)
    
    # 生成报告
    logger.info("📄 生成详细报告...")
    report_path = generate_report(analysis, plot_files, output_dir)
    
    # 打印摘要
    logger.info("=" * 60)
    logger.info("📋 测试结果摘要:")
    logger.info(f"   总模型数: {analysis['summary']['total_models']}")
    logger.info(f"   成功模型: {analysis['summary']['successful_count']}")
    logger.info(f"   失败模型: {analysis['summary']['failed_count']}")
    logger.info(f"   成功率: {analysis['summary']['success_rate']:.1f}%")
    logger.info("=" * 60)
    
    if analysis['successful_models']:
        logger.info("✅ 成功的模型:")
        for model_name in analysis['successful_models'].keys():
            logger.info(f"   - {model_name}")
    
    if analysis['failed_models']:
        logger.info("❌ 失败的模型:")
        for model_name in analysis['failed_models'].keys():
            logger.info(f"   - {model_name}")
    
    logger.info(f"📁 详细结果保存在: {output_dir}")
    logger.info(f"📄 报告文件: {report_path}")
    logger.info("🎉 分析完成！")

if __name__ == "__main__":
    main()