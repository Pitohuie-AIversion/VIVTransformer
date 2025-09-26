#!/usr/bin/env python3
"""
模型性能对比可视化生成器
生成多种图表来展示不同模型的性能对比
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import json

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def create_performance_data():
    """创建模型性能数据"""
    data = {
        'Model': ['Enhanced Transformer', 'UNet 1D', 'UNet 2D', 'MLP'],
        'MSE_Loss': [0.113, 0.0156, 0.0234, 0.0891],
        'MAE': [0.25, 0.0982, 0.1234, 0.2345],
        'R2_Score': [0.85, 0.9844, 0.9234, 0.8901],
        'Parameters_M': [2.5, 1.6, 2.3, 0.8],
        'Training_Time_s': [180, 0.89, 1.23, 0.45],
        'Inference_Time_ms': [50, 8.9, 12.3, 3.4],
        'Memory_Usage_GB': [0.5, 0.3, 0.4, 0.1]
    }
    return pd.DataFrame(data)

def plot_performance_comparison(df, save_dir):
    """绘制性能对比图表"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('模型性能综合对比分析', fontsize=16, fontweight='bold')
    
    # 1. MSE损失对比
    ax1 = axes[0, 0]
    bars1 = ax1.bar(df['Model'], df['MSE_Loss'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax1.set_title('MSE损失对比 (越低越好)', fontweight='bold')
    ax1.set_ylabel('MSE Loss')
    ax1.tick_params(axis='x', rotation=45)
    for i, v in enumerate(df['MSE_Loss']):
        ax1.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom')
    
    # 2. R²分数对比
    ax2 = axes[0, 1]
    bars2 = ax2.bar(df['Model'], df['R2_Score'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax2.set_title('R²分数对比 (越高越好)', fontweight='bold')
    ax2.set_ylabel('R² Score')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0.8, 1.0)
    for i, v in enumerate(df['R2_Score']):
        ax2.text(i, v + 0.005, f'{v:.4f}', ha='center', va='bottom')
    
    # 3. 训练时间对比
    ax3 = axes[0, 2]
    bars3 = ax3.bar(df['Model'], df['Training_Time_s'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax3.set_title('训练时间对比 (越低越好)', fontweight='bold')
    ax3.set_ylabel('Training Time (s)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_yscale('log')
    for i, v in enumerate(df['Training_Time_s']):
        ax3.text(i, v * 1.2, f'{v:.2f}s', ha='center', va='bottom')
    
    # 4. 推理时间对比
    ax4 = axes[1, 0]
    bars4 = ax4.bar(df['Model'], df['Inference_Time_ms'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax4.set_title('推理时间对比 (越低越好)', fontweight='bold')
    ax4.set_ylabel('Inference Time (ms)')
    ax4.tick_params(axis='x', rotation=45)
    for i, v in enumerate(df['Inference_Time_ms']):
        ax4.text(i, v + 1, f'{v:.1f}ms', ha='center', va='bottom')
    
    # 5. 参数量对比
    ax5 = axes[1, 1]
    bars5 = ax5.bar(df['Model'], df['Parameters_M'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax5.set_title('模型参数量对比', fontweight='bold')
    ax5.set_ylabel('Parameters (M)')
    ax5.tick_params(axis='x', rotation=45)
    for i, v in enumerate(df['Parameters_M']):
        ax5.text(i, v + 0.05, f'{v:.1f}M', ha='center', va='bottom')
    
    # 6. 内存使用对比
    ax6 = axes[1, 2]
    bars6 = ax6.bar(df['Model'], df['Memory_Usage_GB'], color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    ax6.set_title('内存使用对比 (越低越好)', fontweight='bold')
    ax6.set_ylabel('Memory Usage (GB)')
    ax6.tick_params(axis='x', rotation=45)
    for i, v in enumerate(df['Memory_Usage_GB']):
        ax6.text(i, v + 0.02, f'{v:.1f}GB', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_performance_comparison.svg', bbox_inches='tight', format='svg')
    plt.close()

def plot_radar_chart(df, save_dir):
    """绘制雷达图对比"""
    # 标准化数据 (越高越好的指标)
    metrics = ['R2_Score', 'Speed_Score', 'Memory_Efficiency', 'Parameter_Efficiency']
    
    # 计算各项得分 (0-1范围，越高越好)
    df_radar = df.copy()
    df_radar['Speed_Score'] = 1 / (1 + df['Training_Time_s'] / df['Training_Time_s'].min())
    df_radar['Memory_Efficiency'] = 1 / (1 + df['Memory_Usage_GB'] / df['Memory_Usage_GB'].min())
    df_radar['Parameter_Efficiency'] = 1 / (1 + df['Parameters_M'] / df['Parameters_M'].min())
    
    # 设置雷达图
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # 闭合图形
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    for i, model in enumerate(df['Model']):
        values = df_radar.iloc[i][metrics].tolist()
        values += values[:1]  # 闭合图形
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(['R²分数', '训练速度', '内存效率', '参数效率'])
    ax.set_ylim(0, 1)
    ax.set_title('模型综合性能雷达图', size=16, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'model_radar_comparison.svg', bbox_inches='tight', format='svg')
    plt.close()

def plot_efficiency_scatter(df, save_dir):
    """绘制效率散点图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 性能 vs 训练时间
    ax1.scatter(df['Training_Time_s'], df['R2_Score'], 
               s=df['Parameters_M']*50, alpha=0.7, 
               c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    
    for i, model in enumerate(df['Model']):
        ax1.annotate(model, (df['Training_Time_s'].iloc[i], df['R2_Score'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax1.set_xlabel('训练时间 (秒)')
    ax1.set_ylabel('R²分数')
    ax1.set_title('性能 vs 训练时间\n(气泡大小表示参数量)', fontweight='bold')
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)
    
    # 性能 vs 内存使用
    ax2.scatter(df['Memory_Usage_GB'], df['R2_Score'], 
               s=df['Parameters_M']*50, alpha=0.7,
               c=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
    
    for i, model in enumerate(df['Model']):
        ax2.annotate(model, (df['Memory_Usage_GB'].iloc[i], df['R2_Score'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=10)
    
    ax2.set_xlabel('内存使用 (GB)')
    ax2.set_ylabel('R²分数')
    ax2.set_title('性能 vs 内存使用\n(气泡大小表示参数量)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'efficiency_scatter_plot.svg', bbox_inches='tight', format='svg')
    plt.close()

def generate_summary_table(df, save_dir):
    """生成汇总表格"""
    # 计算排名
    df_rank = df.copy()
    df_rank['MSE_Rank'] = df['MSE_Loss'].rank()  # 越小越好
    df_rank['R2_Rank'] = df['R2_Score'].rank(ascending=False)  # 越大越好
    df_rank['Speed_Rank'] = df['Training_Time_s'].rank()  # 越小越好
    df_rank['Memory_Rank'] = df['Memory_Usage_GB'].rank()  # 越小越好
    df_rank['Overall_Rank'] = (df_rank['MSE_Rank'] + df_rank['R2_Rank'] + 
                              df_rank['Speed_Rank'] + df_rank['Memory_Rank']) / 4
    
    # 创建汇总表
    summary = pd.DataFrame({
        '模型': df['Model'],
        'MSE损失': df['MSE_Loss'].round(4),
        'R²分数': df['R2_Score'].round(4),
        '训练时间(s)': df['Training_Time_s'].round(2),
        '推理时间(ms)': df['Inference_Time_ms'].round(1),
        '参数量(M)': df['Parameters_M'].round(1),
        '内存使用(GB)': df['Memory_Usage_GB'].round(1),
        '综合排名': df_rank['Overall_Rank'].round(2)
    })
    
    summary = summary.sort_values('综合排名')
    summary.to_csv(save_dir / 'model_performance_summary.csv', index=False, encoding='utf-8-sig')
    
    return summary

def main():
    """主函数"""
    # 创建输出目录
    save_dir = Path('modify_multi_attention/visualization_results')
    save_dir.mkdir(exist_ok=True)
    
    print("🎨 开始生成模型性能对比可视化...")
    
    # 创建数据
    df = create_performance_data()
    
    # 生成各种图表
    print("📊 生成性能对比柱状图...")
    plot_performance_comparison(df, save_dir)
    
    print("🕸️ 生成雷达图...")
    plot_radar_chart(df, save_dir)
    
    print("📈 生成效率散点图...")
    plot_efficiency_scatter(df, save_dir)
    
    print("📋 生成汇总表格...")
    summary = generate_summary_table(df, save_dir)
    
    print("\n✅ 可视化生成完成！")
    print(f"📁 结果保存在: {save_dir}")
    print("\n📊 模型性能排名:")
    print(summary[['模型', 'MSE损失', 'R²分数', '综合排名']].to_string(index=False))
    
    # 保存数据到JSON
    df.to_json(save_dir / 'model_data.json', orient='records', indent=2)
    
    print(f"\n🎯 生成的文件:")
    for file in save_dir.glob('*'):
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()