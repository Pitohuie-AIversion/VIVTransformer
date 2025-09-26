#!/usr/bin/env python3
"""
多模型对比可视化分析脚本
生成训练损失曲线、性能对比图表等
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

def load_training_results(json_path):
    """加载训练结果JSON文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_training_curves(results, save_dir):
    """绘制训练损失曲线对比"""
    plt.figure(figsize=(15, 10))
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('多模型训练损失曲线对比', fontsize=16, fontweight='bold')
    
    successful_models = []
    
    for model_name, model_data in results.items():
        if 'error' not in model_data and 'training_losses' in model_data:
            training_losses = model_data['training_losses']
            val_losses = model_data.get('val_losses', [])
            
            epochs = range(1, len(training_losses) + 1)
            
            if model_name == 'transformer':
                ax = ax1
                ax.set_title(f'Transformer模型 (参数量: {model_data.get("num_parameters", "N/A"):,})')
            elif model_name == 'fno':
                ax = ax2
                ax.set_title(f'FNO模型 (参数量: {model_data.get("num_parameters", "N/A"):,})')
            else:
                continue
                
            # 绘制训练损失
            ax.plot(epochs, training_losses, 'b-', label='训练损失', linewidth=2)
            
            # 绘制验证损失（如果有）
            if val_losses and len(val_losses) == len(training_losses):
                ax.plot(epochs, val_losses, 'r--', label='验证损失', linewidth=2)
            
            ax.set_xlabel('训练轮数 (Epochs)')
            ax.set_ylabel('损失值 (Loss)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')  # 使用对数坐标
            
            successful_models.append(model_name)
    
    # 第三个子图：失败模型信息
    ax3.text(0.1, 0.8, '失败模型分析:', fontsize=14, fontweight='bold', transform=ax3.transAxes)
    y_pos = 0.6
    
    for model_name, model_data in results.items():
        if 'error' in model_data:
            error_msg = model_data['error'][:100] + '...' if len(model_data['error']) > 100 else model_data['error']
            ax3.text(0.1, y_pos, f'{model_name.upper()}: {error_msg}', 
                    fontsize=10, transform=ax3.transAxes, wrap=True)
            y_pos -= 0.15
    
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    ax3.axis('off')
    ax3.set_title('模型训练失败原因')
    
    # 第四个子图：性能对比
    if len(successful_models) >= 2:
        model_names = []
        test_mse = []
        test_r2 = []
        param_counts = []
        
        for model_name in successful_models:
            model_data = results[model_name]
            model_names.append(model_name.upper())
            test_mse.append(model_data.get('test_mse', 0))
            test_r2.append(model_data.get('test_r2', 0))
            param_counts.append(model_data.get('num_parameters', 0))
        
        # 双y轴图
        ax4_twin = ax4.twinx()
        
        x_pos = np.arange(len(model_names))
        width = 0.35
        
        # MSE柱状图
        bars1 = ax4.bar(x_pos - width/2, test_mse, width, label='测试MSE', color='skyblue', alpha=0.8)
        
        # R²柱状图
        bars2 = ax4_twin.bar(x_pos + width/2, test_r2, width, label='测试R²', color='lightcoral', alpha=0.8)
        
        ax4.set_xlabel('模型类型')
        ax4.set_ylabel('测试MSE', color='blue')
        ax4_twin.set_ylabel('测试R²', color='red')
        ax4.set_title('模型性能对比')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(model_names)
        
        # 添加数值标签
        for i, (mse, r2, params) in enumerate(zip(test_mse, test_r2, param_counts)):
            ax4.text(i - width/2, mse + max(test_mse)*0.01, f'{mse:.4f}', 
                    ha='center', va='bottom', fontsize=9)
            ax4_twin.text(i + width/2, r2 + max(test_r2)*0.01, f'{r2:.4f}', 
                         ha='center', va='bottom', fontsize=9)
            ax4.text(i, -max(test_mse)*0.1, f'{params:,}参数', 
                    ha='center', va='top', fontsize=8, rotation=45)
        
        ax4.legend(loc='upper left')
        ax4_twin.legend(loc='upper right')
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    save_path = os.path.join(save_dir, 'training_curves_comparison.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"训练曲线对比图已保存: {save_path}")
    plt.close()

def plot_parameter_efficiency(results, save_dir):
    """绘制参数效率分析图"""
    plt.figure(figsize=(12, 8))
    
    successful_models = []
    for model_name, model_data in results.items():
        if 'error' not in model_data and 'test_mse' in model_data:
            successful_models.append({
                'name': model_name.upper(),
                'params': model_data.get('num_parameters', 0),
                'mse': model_data.get('test_mse', 0),
                'r2': model_data.get('test_r2', 0)
            })
    
    if len(successful_models) >= 2:
        # 散点图：参数量 vs 性能
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 参数量 vs MSE
        params = [m['params'] for m in successful_models]
        mse_values = [m['mse'] for m in successful_models]
        names = [m['name'] for m in successful_models]
        
        ax1.scatter(params, mse_values, s=200, alpha=0.7, c=['blue', 'red'])
        for i, name in enumerate(names):
            ax1.annotate(name, (params[i], mse_values[i]), 
                        xytext=(10, 10), textcoords='offset points',
                        fontsize=12, fontweight='bold')
        
        ax1.set_xlabel('模型参数量')
        ax1.set_ylabel('测试MSE')
        ax1.set_title('参数量 vs 测试误差')
        ax1.set_xscale('log')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 参数效率比较
        efficiency = [mse / (params[i] / 1e6) for i, mse in enumerate(mse_values)]  # MSE per million parameters
        
        bars = ax2.bar(names, efficiency, color=['skyblue', 'lightcoral'], alpha=0.8)
        ax2.set_ylabel('MSE / 百万参数')
        ax2.set_title('参数效率对比 (越低越好)')
        ax2.grid(True, alpha=0.3)
        
        # 添加数值标签
        for i, (bar, eff) in enumerate(zip(bars, efficiency)):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(efficiency)*0.01,
                    f'{eff:.2e}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        save_path = os.path.join(save_dir, 'parameter_efficiency_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"参数效率分析图已保存: {save_path}")
        plt.close()

def generate_summary_table(results, save_dir):
    """生成汇总表格"""
    summary_data = []
    
    for model_name, model_data in results.items():
        if 'error' not in model_data:
            summary_data.append({
                'Model': model_name.upper(),
                'Parameters': f"{model_data.get('num_parameters', 0):,}",
                'Test MSE': f"{model_data.get('test_mse', 0):.6f}",
                'Test R²': f"{model_data.get('test_r2', 0):.4f}",
                'Final Train Loss': f"{model_data.get('training_losses', [0])[-1]:.6f}",
                'Status': '✅ 成功'
            })
        else:
            summary_data.append({
                'Model': model_name.upper(),
                'Parameters': 'N/A',
                'Test MSE': 'N/A',
                'Test R²': 'N/A',
                'Final Train Loss': 'N/A',
                'Status': '❌ 失败'
            })
    
    # 创建表格图
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # 创建表格
    table_data = []
    headers = ['模型', '参数量', '测试MSE', '测试R²', '最终训练损失', '状态']
    table_data.append(headers)
    
    for item in summary_data:
        table_data.append([
            item['Model'],
            item['Parameters'],
            item['Test MSE'],
            item['Test R²'],
            item['Final Train Loss'],
            item['Status']
        ])
    
    table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # 设置表格样式
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(table_data)):
        for j in range(len(headers)):
            if '❌' in table_data[i][j]:
                table[(i, j)].set_facecolor('#ffcccb')
            elif '✅' in table_data[i][j]:
                table[(i, j)].set_facecolor('#d4edda')
    
    plt.title('多模型训练结果汇总表', fontsize=16, fontweight='bold', pad=20)
    
    # 保存表格
    save_path = os.path.join(save_dir, 'results_summary_table.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"结果汇总表已保存: {save_path}")
    plt.close()

def main():
    """主函数"""
    # 设置路径
    results_dir = "unified_training_results"
    json_file = os.path.join(results_dir, "training_results.json")
    plots_dir = os.path.join(results_dir, "comparison_plots")
    
    # 创建图表保存目录
    os.makedirs(plots_dir, exist_ok=True)
    
    # 检查结果文件是否存在
    if not os.path.exists(json_file):
        print(f"错误: 找不到训练结果文件 {json_file}")
        return
    
    # 加载训练结果
    print(f"加载训练结果: {json_file}")
    results = load_training_results(json_file)
    
    print(f"发现 {len(results)} 个模型的训练结果")
    
    # 生成各种图表
    print("\n生成可视化图表...")
    plot_training_curves(results, plots_dir)
    plot_parameter_efficiency(results, plots_dir)
    generate_summary_table(results, plots_dir)
    
    print(f"\n所有图表已保存到: {plots_dir}")
    print("可视化分析完成!")

if __name__ == "__main__":
    main()