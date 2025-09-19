#!/usr/bin/env python3
"""
模型横向对比分析脚本
用于分析统一对比测试结果，生成详细的分析报告和可视化图表
"""

import yaml
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import seaborn as sns

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelComparisonAnalyzer:
    """模型对比分析器"""
    
    def __init__(self, results_file: str):
        """
        初始化分析器
        
        Args:
            results_file: 结果文件路径
        """
        self.results_file = Path(results_file)
        self.results_data = self._load_results()
        self.output_dir = self.results_file.parent / "analysis_output"
        self.output_dir.mkdir(exist_ok=True)
        
    def _load_results(self) -> List[Dict]:
        """加载结果数据"""
        if self.results_file.suffix == '.yaml':
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        elif self.results_file.suffix == '.json':
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        else:
            raise ValueError(f"不支持的文件格式: {self.results_file.suffix}")
    
    def analyze_success_rates(self) -> Dict[str, Any]:
        """分析成功率"""
        # 按模型类型统计
        model_stats = {}
        dataset_stats = {}
        
        for result in self.results_data:
            model_name = result['model_name']
            model_type = result['model_type']
            dataset_name = result['dataset_name']
            success = result['success']
            
            # 模型统计
            if model_type not in model_stats:
                model_stats[model_type] = {'total': 0, 'success': 0, 'models': set()}
            model_stats[model_type]['total'] += 1
            model_stats[model_type]['models'].add(model_name)
            if success:
                model_stats[model_type]['success'] += 1
                
            # 数据集统计
            if dataset_name not in dataset_stats:
                dataset_stats[dataset_name] = {'total': 0, 'success': 0}
            dataset_stats[dataset_name]['total'] += 1
            if success:
                dataset_stats[dataset_name]['success'] += 1
        
        # 计算成功率
        for stats in model_stats.values():
            stats['success_rate'] = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            stats['models'] = list(stats['models'])
            
        for stats in dataset_stats.values():
            stats['success_rate'] = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            
        return {
            'model_stats': model_stats,
            'dataset_stats': dataset_stats,
            'total_tests': len(self.results_data),
            'total_success': sum(1 for r in self.results_data if r['success']),
            'overall_success_rate': sum(1 for r in self.results_data if r['success']) / len(self.results_data)
        }
    
    def analyze_errors(self) -> Dict[str, Any]:
        """分析错误类型"""
        error_types = {}
        model_errors = {}
        
        for result in self.results_data:
            if not result['success'] and result['error']:
                error = result['error']
                model_name = result['model_name']
                
                # 错误分类
                if 'size' in error.lower() and 'match' in error.lower():
                    error_type = '张量尺寸不匹配'
                elif 'channel' in error.lower():
                    error_type = '通道数错误'
                elif 'argument' in error.lower():
                    error_type = '参数错误'
                elif 'dimension' in error.lower():
                    error_type = '维度错误'
                elif 'unpack' in error.lower():
                    error_type = '解包错误'
                else:
                    error_type = '其他错误'
                
                if error_type not in error_types:
                    error_types[error_type] = {'count': 0, 'models': set(), 'examples': []}
                error_types[error_type]['count'] += 1
                error_types[error_type]['models'].add(model_name)
                if len(error_types[error_type]['examples']) < 3:
                    error_types[error_type]['examples'].append(error)
                
                # 模型错误统计
                if model_name not in model_errors:
                    model_errors[model_name] = []
                model_errors[model_name].append(error_type)
        
        # 转换集合为列表
        for error_info in error_types.values():
            error_info['models'] = list(error_info['models'])
            
        return {
            'error_types': error_types,
            'model_errors': model_errors
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """分析性能指标"""
        successful_results = [r for r in self.results_data if r['success']]
        
        if not successful_results:
            return {'message': '没有成功的测试结果可供分析'}
        
        # 提取性能指标
        metrics = ['param_count', 'training_time', 'inference_time', 'memory_usage_mb', 
                  'mae', 'mse', 'r2_score', 'rmse', 'relative_error']
        
        performance_data = {}
        for metric in metrics:
            performance_data[metric] = {
                'values': [r[metric] for r in successful_results if r[metric] > 0],
                'models': [r['model_name'] for r in successful_results if r[metric] > 0]
            }
        
        return performance_data
    
    def generate_visualizations(self):
        """生成可视化图表"""
        success_analysis = self.analyze_success_rates()
        error_analysis = self.analyze_errors()
        
        # 1. 成功率对比图
        self._plot_success_rates(success_analysis)
        
        # 2. 错误类型分布图
        self._plot_error_distribution(error_analysis)
        
        # 3. 模型架构对比图
        self._plot_model_architecture_comparison()
        
        # 4. 参数量对比图
        self._plot_parameter_comparison()
        
    def _plot_success_rates(self, analysis: Dict):
        """绘制成功率对比图"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 按模型类型的成功率
        model_types = list(analysis['model_stats'].keys())
        success_rates = [analysis['model_stats'][mt]['success_rate'] * 100 for mt in model_types]
        
        bars1 = ax1.bar(model_types, success_rates, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        ax1.set_title('各模型类型成功率对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('成功率 (%)', fontsize=12)
        ax1.set_xlabel('模型类型', fontsize=12)
        ax1.set_ylim(0, 100)
        
        # 添加数值标签
        for bar, rate in zip(bars1, success_rates):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        # 按数据集的成功率
        datasets = list(analysis['dataset_stats'].keys())
        dataset_rates = [analysis['dataset_stats'][ds]['success_rate'] * 100 for ds in datasets]
        
        bars2 = ax2.bar(datasets, dataset_rates, color=['#A8E6CF', '#FFD93D', '#FF8B94', '#B4A7D6'])
        ax2.set_title('各数据集成功率对比', fontsize=14, fontweight='bold')
        ax2.set_ylabel('成功率 (%)', fontsize=12)
        ax2.set_xlabel('数据集', fontsize=12)
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='x', rotation=45)
        
        # 添加数值标签
        for bar, rate in zip(bars2, dataset_rates):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{rate:.1f}%', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'success_rates_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_error_distribution(self, analysis: Dict):
        """绘制错误类型分布图"""
        error_types = analysis['error_types']
        
        if not error_types:
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 错误类型饼图
        labels = list(error_types.keys())
        sizes = [error_types[et]['count'] for et in labels]
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        wedges, texts, autotexts = ax1.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                          colors=colors, startangle=90)
        ax1.set_title('错误类型分布', fontsize=14, fontweight='bold')
        
        # 调整文字大小
        for text in texts:
            text.set_fontsize(10)
        for autotext in autotexts:
            autotext.set_fontsize(9)
            autotext.set_color('white')
            autotext.set_weight('bold')
        
        # 错误类型柱状图
        bars = ax2.bar(range(len(labels)), sizes, color=colors)
        ax2.set_title('错误类型数量统计', fontsize=14, fontweight='bold')
        ax2.set_ylabel('错误数量', fontsize=12)
        ax2.set_xlabel('错误类型', fontsize=12)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        
        # 添加数值标签
        for bar, size in zip(bars, sizes):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    str(size), ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'error_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_model_architecture_comparison(self):
        """绘制模型架构对比图"""
        # 统计各模型类型的数量和变体
        model_counts = {}
        for result in self.results_data:
            model_type = result['model_type']
            if model_type not in model_counts:
                model_counts[model_type] = set()
            model_counts[model_type].add(result['model_name'])
        
        # 转换为计数
        types = list(model_counts.keys())
        counts = [len(model_counts[t]) for t in types]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(types, counts, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'])
        
        ax.set_title('模型架构类型及变体数量', fontsize=14, fontweight='bold')
        ax.set_ylabel('模型变体数量', fontsize=12)
        ax.set_xlabel('模型架构类型', fontsize=12)
        
        # 添加数值标签
        for bar, count in zip(bars, counts):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   str(count), ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'model_architecture_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def _plot_parameter_comparison(self):
        """绘制参数量对比图"""
        # 提取参数量数据
        param_data = []
        for result in self.results_data:
            if result['param_count'] > 0:
                param_data.append({
                    'model_name': result['model_name'],
                    'model_type': result['model_type'],
                    'param_count': result['param_count'],
                    'success': result['success']
                })
        
        if not param_data:
            return
            
        # 按模型类型分组
        df = pd.DataFrame(param_data)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 参数量分布（按模型类型）
        model_types = df['model_type'].unique()
        colors = plt.cm.Set2(np.linspace(0, 1, len(model_types)))
        
        for i, model_type in enumerate(model_types):
            type_data = df[df['model_type'] == model_type]
            ax1.scatter(type_data['model_name'], type_data['param_count'], 
                       c=[colors[i]], label=model_type, s=100, alpha=0.7)
        
        ax1.set_title('各模型参数量对比', fontsize=14, fontweight='bold')
        ax1.set_ylabel('参数量', fontsize=12)
        ax1.set_xlabel('模型名称', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 参数量箱线图（按模型类型）
        type_params = [df[df['model_type'] == mt]['param_count'].values for mt in model_types]
        box_plot = ax2.boxplot(type_params, labels=model_types, patch_artist=True)
        
        # 设置箱线图颜色
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax2.set_title('各模型类型参数量分布', fontsize=14, fontweight='bold')
        ax2.set_ylabel('参数量', fontsize=12)
        ax2.set_xlabel('模型类型', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'parameter_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
    def generate_detailed_report(self):
        """生成详细分析报告"""
        success_analysis = self.analyze_success_rates()
        error_analysis = self.analyze_errors()
        performance_analysis = self.analyze_performance()
        
        report_content = f"""
# 模型横向对比详细分析报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**数据源**: {self.results_file.name}

## 1. 总体概况

- **总测试数**: {success_analysis['total_tests']}
- **成功测试数**: {success_analysis['total_success']}
- **失败测试数**: {success_analysis['total_tests'] - success_analysis['total_success']}
- **总体成功率**: {success_analysis['overall_success_rate']:.1%}

## 2. 模型类型分析

### 2.1 各模型类型成功率

| 模型类型 | 测试数量 | 成功数量 | 成功率 | 包含模型 |
|---------|---------|---------|--------|----------|
"""
        
        for model_type, stats in success_analysis['model_stats'].items():
            models_str = ', '.join(stats['models'][:3])
            if len(stats['models']) > 3:
                models_str += f" 等{len(stats['models'])}个"
            report_content += f"| {model_type} | {stats['total']} | {stats['success']} | {stats['success_rate']:.1%} | {models_str} |\n"
        
        report_content += f"""
### 2.2 数据集适配性分析

| 数据集 | 测试数量 | 成功数量 | 成功率 |
|--------|---------|---------|--------|
"""
        
        for dataset, stats in success_analysis['dataset_stats'].items():
            report_content += f"| {dataset} | {stats['total']} | {stats['success']} | {stats['success_rate']:.1%} |\n"
        
        report_content += f"""
## 3. 错误分析

### 3.1 主要错误类型

"""
        
        for error_type, info in error_analysis['error_types'].items():
            affected_models = ', '.join(list(info['models'])[:5])
            if len(info['models']) > 5:
                affected_models += f" 等{len(info['models'])}个模型"
            
            report_content += f"""
**{error_type}** ({info['count']}次)
- 影响模型: {affected_models}
- 典型错误示例: {info['examples'][0] if info['examples'] else '无'}
"""
        
        report_content += f"""
### 3.2 模型特定问题

"""
        
        for model_name, errors in error_analysis['model_errors'].items():
            error_summary = ', '.join(set(errors))
            report_content += f"- **{model_name}**: {error_summary}\n"
        
        if 'message' not in performance_analysis:
            report_content += f"""
## 4. 性能分析

### 4.1 成功模型性能概览

基于成功运行的模型，以下是关键性能指标的分析：

"""
            for metric, data in performance_analysis.items():
                if data['values']:
                    avg_val = np.mean(data['values'])
                    min_val = np.min(data['values'])
                    max_val = np.max(data['values'])
                    report_content += f"- **{metric}**: 平均 {avg_val:.2e}, 范围 [{min_val:.2e}, {max_val:.2e}]\n"
        
        report_content += f"""
## 5. 问题总结与建议

### 5.1 主要问题

1. **张量尺寸不匹配**: 这是最常见的问题，主要原因是模型输入输出尺寸配置与数据集不匹配
2. **通道数错误**: 模型期望的输入通道数与实际数据不符
3. **参数传递错误**: 某些模型的forward方法参数定义与调用不一致
4. **维度处理错误**: 1D和2D模型在处理不同维度数据时出现问题

### 5.2 改进建议

1. **统一接口设计**: 为所有模型实现统一的输入输出接口
2. **动态尺寸适配**: 实现自动的尺寸适配机制
3. **参数验证**: 在模型创建时验证参数的有效性
4. **错误处理**: 增加更详细的错误信息和恢复机制

### 5.3 下一步工作

1. 修复识别出的接口不一致问题
2. 实现自适应的尺寸处理机制
3. 优化模型配置参数
4. 增加更多的测试用例和边界条件检查

---

**报告生成完成**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 保存报告
        report_file = self.output_dir / f"detailed_analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"详细分析报告已保存到: {report_file}")
        return report_file

def main():
    """主函数"""
    # 查找最新的结果文件
    models_dir = Path(__file__).parent
    result_files = list(models_dir.glob("unified_comparison_results_*.yaml"))
    
    if not result_files:
        print("未找到结果文件")
        return
    
    # 使用最新的结果文件
    latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
    print(f"分析文件: {latest_file}")
    
    # 创建分析器并生成报告
    analyzer = ModelComparisonAnalyzer(str(latest_file))
    
    print("生成可视化图表...")
    analyzer.generate_visualizations()
    
    print("生成详细分析报告...")
    report_file = analyzer.generate_detailed_report()
    
    print(f"\n分析完成！")
    print(f"输出目录: {analyzer.output_dir}")
    print(f"生成的文件:")
    for file in analyzer.output_dir.iterdir():
        print(f"  - {file.name}")

if __name__ == "__main__":
    main()