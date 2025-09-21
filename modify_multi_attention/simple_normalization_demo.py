#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化归一化演示脚本

展示在现有项目中如何使用归一化功能

作者: AI Assistant
日期: 2025
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import logging
from pathlib import Path
import h5py

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleNormalizer:
    """简化的归一化器"""
    
    def __init__(self, method='minmax', target_range=(0, 1)):
        self.method = method
        self.target_range = target_range
        self.stats = {}
        self.fitted = False
    
    def fit(self, data):
        """拟合归一化参数"""
        if self.method == 'none':
            self.fitted = True
            return
        
        self.stats = {
            'mean': data.mean().item(),
            'std': data.std().item(),
            'min': data.min().item(),
            'max': data.max().item(),
            'median': data.median().item()
        }
        
        if self.method == 'robust':
            self.stats['q25'] = data.quantile(0.25).item()
            self.stats['q75'] = data.quantile(0.75).item()
            self.stats['iqr'] = self.stats['q75'] - self.stats['q25']
        
        self.fitted = True
        logger.info(f"归一化参数拟合完成 ({self.method}): 均值={self.stats['mean']:.6f}, "
                   f"标准差={self.stats['std']:.6f}, 范围=[{self.stats['min']:.6f}, {self.stats['max']:.6f}]")
    
    def transform(self, data):
        """应用归一化"""
        if not self.fitted or self.method == 'none':
            return data
        
        if self.method == 'minmax':
            data_range = self.stats['max'] - self.stats['min']
            if data_range == 0:
                return data
            
            # 归一化到[0, 1]
            normalized = (data - self.stats['min']) / data_range
            
            # 缩放到目标范围
            target_min, target_max = self.target_range
            normalized = normalized * (target_max - target_min) + target_min
            
            return normalized
        
        elif self.method == 'zscore':
            if self.stats['std'] == 0:
                return data
            return (data - self.stats['mean']) / self.stats['std']
        
        elif self.method == 'robust':
            if self.stats['iqr'] == 0:
                return data
            return (data - self.stats['median']) / self.stats['iqr']
        
        return data
    
    def inverse_transform(self, data):
        """反归一化"""
        if not self.fitted or self.method == 'none':
            return data
        
        if self.method == 'minmax':
            target_min, target_max = self.target_range
            # 从目标范围还原到[0, 1]
            data_01 = (data - target_min) / (target_max - target_min)
            # 从[0, 1]还原到原始范围
            return data_01 * (self.stats['max'] - self.stats['min']) + self.stats['min']
        
        elif self.method == 'zscore':
            return data * self.stats['std'] + self.stats['mean']
        
        elif self.method == 'robust':
            return data * self.stats['iqr'] + self.stats['median']
        
        return data

def load_sample_data():
    """加载示例数据"""
    # 创建模拟数据
    np.random.seed(42)
    torch.manual_seed(42)
    
    # 模拟32x32输入和128x128输出
    num_samples = 200
    input_size = 32 * 32  # 1024
    output_size = 128 * 128  # 16384
    
    # 生成具有不同分布特征的数据
    inputs = torch.randn(num_samples, input_size) * 5 + 10  # 均值10，标准差5
    outputs = torch.randn(num_samples, output_size) * 20 + 50  # 均值50，标准差20
    
    # 添加一些异常值
    inputs[0:5] += 100  # 异常值
    outputs[0:5] += 500
    
    logger.info(f"生成模拟数据: 输入形状={inputs.shape}, 输出形状={outputs.shape}")
    logger.info(f"输入数据范围: [{inputs.min():.2f}, {inputs.max():.2f}], 均值: {inputs.mean():.2f}")
    logger.info(f"输出数据范围: [{outputs.min():.2f}, {outputs.max():.2f}], 均值: {outputs.mean():.2f}")
    
    return inputs, outputs

def demonstrate_normalization_methods():
    """演示不同归一化方法"""
    # 加载数据
    inputs, outputs = load_sample_data()
    
    # 测试不同归一化方法
    methods = ['none', 'minmax', 'zscore', 'robust']
    results = {}
    
    for method in methods:
        logger.info(f"\n{'='*50}")
        logger.info(f"测试归一化方法: {method}")
        logger.info(f"{'='*50}")
        
        # 创建归一化器
        if method == 'minmax':
            normalizer = SimpleNormalizer(method=method, target_range=(0, 1))
        else:
            normalizer = SimpleNormalizer(method=method)
        
        # 拟合和转换
        normalizer.fit(inputs)
        normalized_inputs = normalizer.transform(inputs)
        
        # 记录结果
        results[method] = {
            'original_stats': {
                'mean': inputs.mean().item(),
                'std': inputs.std().item(),
                'min': inputs.min().item(),
                'max': inputs.max().item()
            },
            'normalized_stats': {
                'mean': normalized_inputs.mean().item(),
                'std': normalized_inputs.std().item(),
                'min': normalized_inputs.min().item(),
                'max': normalized_inputs.max().item()
            },
            'normalizer_stats': normalizer.stats
        }
        
        logger.info(f"原始数据: 均值={results[method]['original_stats']['mean']:.6f}, "
                   f"标准差={results[method]['original_stats']['std']:.6f}, "
                   f"范围=[{results[method]['original_stats']['min']:.6f}, {results[method]['original_stats']['max']:.6f}]")
        
        logger.info(f"归一化后: 均值={results[method]['normalized_stats']['mean']:.6f}, "
                   f"标准差={results[method]['normalized_stats']['std']:.6f}, "
                   f"范围=[{results[method]['normalized_stats']['min']:.6f}, {results[method]['normalized_stats']['max']:.6f}]")
        
        # 测试反归一化
        if method != 'none':
            recovered_inputs = normalizer.inverse_transform(normalized_inputs)
            recovery_error = torch.mean(torch.abs(inputs - recovered_inputs)).item()
            logger.info(f"反归一化误差: {recovery_error:.8f}")
    
    return results

def visualize_normalization_effects(results):
    """可视化归一化效果"""
    methods = list(results.keys())
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('归一化方法效果比较', fontsize=16, fontweight='bold')
    
    # 1. 均值比较
    original_means = [results[method]['original_stats']['mean'] for method in methods]
    normalized_means = [results[method]['normalized_stats']['mean'] for method in methods]
    
    x = np.arange(len(methods))
    width = 0.35
    
    axes[0, 0].bar(x - width/2, original_means, width, label='原始数据', alpha=0.8)
    axes[0, 0].bar(x + width/2, normalized_means, width, label='归一化后', alpha=0.8)
    axes[0, 0].set_xlabel('归一化方法')
    axes[0, 0].set_ylabel('均值')
    axes[0, 0].set_title('均值比较')
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(methods)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 标准差比较
    original_stds = [results[method]['original_stats']['std'] for method in methods]
    normalized_stds = [results[method]['normalized_stats']['std'] for method in methods]
    
    axes[0, 1].bar(x - width/2, original_stds, width, label='原始数据', alpha=0.8)
    axes[0, 1].bar(x + width/2, normalized_stds, width, label='归一化后', alpha=0.8)
    axes[0, 1].set_xlabel('归一化方法')
    axes[0, 1].set_ylabel('标准差')
    axes[0, 1].set_title('标准差比较')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(methods)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 数据范围比较
    original_ranges = [results[method]['original_stats']['max'] - results[method]['original_stats']['min'] for method in methods]
    normalized_ranges = [results[method]['normalized_stats']['max'] - results[method]['normalized_stats']['min'] for method in methods]
    
    axes[1, 0].bar(x - width/2, original_ranges, width, label='原始数据', alpha=0.8)
    axes[1, 0].bar(x + width/2, normalized_ranges, width, label='归一化后', alpha=0.8)
    axes[1, 0].set_xlabel('归一化方法')
    axes[1, 0].set_ylabel('数据范围')
    axes[1, 0].set_title('数据范围比较')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(methods)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 归一化效果总结表
    axes[1, 1].axis('off')
    
    # 创建表格数据
    table_data = []
    headers = ['方法', '原始均值', '归一化均值', '原始范围', '归一化范围']
    
    for method in methods:
        orig_mean = results[method]['original_stats']['mean']
        norm_mean = results[method]['normalized_stats']['mean']
        orig_range = results[method]['original_stats']['max'] - results[method]['original_stats']['min']
        norm_range = results[method]['normalized_stats']['max'] - results[method]['normalized_stats']['min']
        
        table_data.append([
            method,
            f"{orig_mean:.2f}",
            f"{norm_mean:.2f}",
            f"{orig_range:.2f}",
            f"{norm_range:.2f}"
        ])
    
    table = axes[1, 1].table(cellText=table_data, colLabels=headers, 
                            cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    axes[1, 1].set_title('归一化效果总结', fontweight='bold')
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path("normalization_demo_results")
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / "normalization_comparison.png", dpi=300, bbox_inches='tight')
    logger.info(f"可视化结果已保存到: {output_dir / 'normalization_comparison.png'}")
    
    plt.show()

def generate_summary_report(results):
    """生成总结报告"""
    output_dir = Path("normalization_demo_results")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "normalization_demo_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 归一化方法演示报告\n\n")
        f.write("## 概述\n\n")
        f.write("本报告展示了在深度学习模型训练中不同归一化方法的效果。\n\n")
        
        f.write("## 测试数据\n\n")
        f.write("- 输入维度: 1024 (32×32)\n")
        f.write("- 输出维度: 16384 (128×128)\n")
        f.write("- 样本数量: 200\n")
        f.write("- 数据特征: 包含正常数据和异常值\n\n")
        
        f.write("## 归一化方法比较\n\n")
        f.write("| 方法 | 原始均值 | 归一化均值 | 原始标准差 | 归一化标准差 | 原始范围 | 归一化范围 |\n")
        f.write("|------|----------|------------|------------|-------------|----------|------------|\n")
        
        for method, result in results.items():
            orig = result['original_stats']
            norm = result['normalized_stats']
            orig_range = orig['max'] - orig['min']
            norm_range = norm['max'] - norm['min']
            
            f.write(f"| {method} | {orig['mean']:.4f} | {norm['mean']:.4f} | "
                   f"{orig['std']:.4f} | {norm['std']:.4f} | "
                   f"{orig_range:.4f} | {norm_range:.4f} |\n")
        
        f.write("\n## 方法说明\n\n")
        f.write("### 1. None (不归一化)\n")
        f.write("- 保持原始数据不变\n")
        f.write("- 适用于数据已经在合适范围内的情况\n\n")
        
        f.write("### 2. Min-Max归一化\n")
        f.write("- 将数据缩放到[0, 1]范围\n")
        f.write("- 公式: (x - min) / (max - min)\n")
        f.write("- 适用于数据分布相对均匀的情况\n\n")
        
        f.write("### 3. Z-score标准化\n")
        f.write("- 将数据转换为均值0、标准差1的分布\n")
        f.write("- 公式: (x - mean) / std\n")
        f.write("- 适用于数据呈正态分布的情况\n\n")
        
        f.write("### 4. 鲁棒归一化\n")
        f.write("- 使用中位数和四分位距进行归一化\n")
        f.write("- 公式: (x - median) / IQR\n")
        f.write("- 适用于数据包含异常值的情况\n\n")
        
        f.write("## 使用建议\n\n")
        f.write("1. **数据探索**: 首先分析数据分布特征\n")
        f.write("2. **方法选择**: 根据数据特征选择合适的归一化方法\n")
        f.write("3. **效果验证**: 比较不同方法对模型性能的影响\n")
        f.write("4. **反归一化**: 在模型输出时记得进行反归一化\n\n")
        
        f.write("## 在现有项目中的集成\n\n")
        f.write("在 `run_crop_model_test.py` 中集成归一化功能的步骤:\n\n")
        f.write("1. 在配置文件中添加归一化设置\n")
        f.write("2. 在数据加载器中实现归一化处理\n")
        f.write("3. 在训练过程中应用归一化\n")
        f.write("4. 在评估时进行反归一化\n")
        f.write("5. 记录归一化统计信息用于后续分析\n\n")
    
    logger.info(f"总结报告已保存到: {report_path}")

def main():
    """主函数"""
    logger.info("开始归一化方法演示")
    
    # 演示不同归一化方法
    results = demonstrate_normalization_methods()
    
    # 可视化效果
    visualize_normalization_effects(results)
    
    # 生成报告
    generate_summary_report(results)
    
    logger.info("归一化演示完成!")

if __name__ == "__main__":
    main()