"""
帕累托曲线分析脚本
生成精度-资源的多维度对比图，支持Iso-Params、Iso-Compute、Iso-Latency分析

核心功能:
1. 精度 vs 参数量帕累托曲线
2. 精度 vs FLOPs帕累托曲线  
3. 精度 vs 推理延迟帕累托曲线
4. 多维度效率分析
5. 模型选择建议
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path
import logging

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

class ParetoAnalyzer:
    """帕累托分析器"""
    
    def __init__(self, results_data: Dict = None):
        """
        初始化帕累托分析器
        
        Args:
            results_data: 包含模型性能和资源消耗数据的字典
        """
        self.results_data = results_data or self._generate_sample_data()
        self.models = list(self.results_data.keys())
        
    def _generate_sample_data(self) -> Dict:
        """生成示例数据用于演示"""
        # 基于之前的分析结果生成真实的示例数据
        sample_data = {
            'MLP_Light': {
                'mse_loss': 0.045,
                'mae_loss': 0.156,
                'r2_score': 0.892,
                'params_M': 2.1,
                'flops_G': 0.5,
                'latency_ms': 4.2,
                'memory_MB': 156,
                'model_type': 'MLP'
            },
            'MLP_Standard': {
                'mse_loss': 0.038,
                'mae_loss': 0.142,
                'r2_score': 0.915,
                'params_M': 5.1,
                'flops_G': 1.0,
                'latency_ms': 8.7,
                'memory_MB': 298,
                'model_type': 'MLP'
            },
            'MLP_Large': {
                'mse_loss': 0.035,
                'mae_loss': 0.138,
                'r2_score': 0.925,
                'params_M': 12.3,
                'flops_G': 2.1,
                'latency_ms': 18.5,
                'memory_MB': 687,
                'model_type': 'MLP'
            },
            'Transformer_Light': {
                'mse_loss': 0.042,
                'mae_loss': 0.149,
                'r2_score': 0.901,
                'params_M': 2.8,
                'flops_G': 0.6,
                'latency_ms': 6.1,
                'memory_MB': 189,
                'model_type': 'Transformer'
            },
            'Transformer_Standard': {
                'mse_loss': 0.032,
                'mae_loss': 0.128,
                'r2_score': 0.935,
                'params_M': 5.2,
                'flops_G': 1.1,
                'latency_ms': 11.3,
                'memory_MB': 342,
                'model_type': 'Transformer'
            },
            'Transformer_Large': {
                'mse_loss': 0.028,
                'mae_loss': 0.121,
                'r2_score': 0.948,
                'params_M': 15.7,
                'flops_G': 2.8,
                'latency_ms': 24.6,
                'memory_MB': 892,
                'model_type': 'Transformer'
            },
            'CustomTransformer_Efficient': {
                'mse_loss': 0.036,
                'mae_loss': 0.135,
                'r2_score': 0.922,
                'params_M': 8.6,
                'flops_G': 1.2,
                'latency_ms': 9.8,
                'memory_MB': 445,
                'model_type': 'CustomTransformer'
            },
            'CustomTransformer_Enhanced': {
                'mse_loss': 0.030,
                'mae_loss': 0.125,
                'r2_score': 0.941,
                'params_M': 17.2,
                'flops_G': 2.5,
                'latency_ms': 19.2,
                'memory_MB': 756,
                'model_type': 'CustomTransformer'
            }
        }
        return sample_data
    
    def create_pareto_plots(self, save_dir: str = "./pareto_analysis", figsize: Tuple[int, int] = (20, 12)):
        """创建完整的帕累托分析图"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # 创建大图包含所有子图
        fig = plt.figure(figsize=figsize)
        
        # 定义颜色映射
        model_types = list(set([data['model_type'] for data in self.results_data.values()]))
        colors = plt.cm.Set1(np.linspace(0, 1, len(model_types)))
        color_map = dict(zip(model_types, colors))
        
        # 1. 精度 vs 参数量 (左上)
        ax1 = plt.subplot(2, 3, 1)
        self._plot_accuracy_vs_params(ax1, color_map)
        
        # 2. 精度 vs FLOPs (右上)
        ax2 = plt.subplot(2, 3, 2)
        self._plot_accuracy_vs_flops(ax2, color_map)
        
        # 3. 精度 vs 延迟 (中上)
        ax3 = plt.subplot(2, 3, 3)
        self._plot_accuracy_vs_latency(ax3, color_map)
        
        # 4. 效率综合分析 (左下)
        ax4 = plt.subplot(2, 3, 4)
        self._plot_efficiency_analysis(ax4, color_map)
        
        # 5. 资源消耗对比 (右下)
        ax5 = plt.subplot(2, 3, 5)
        self._plot_resource_comparison(ax5, color_map)
        
        # 6. 帕累托前沿 (中下)
        ax6 = plt.subplot(2, 3, 6)
        self._plot_pareto_frontier(ax6, color_map)
        
        plt.tight_layout()
        plt.savefig(save_path / 'complete_pareto_analysis.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path / 'complete_pareto_analysis.svg', bbox_inches='tight')
        plt.close()
        
        # 创建单独的高质量图
        self._create_individual_plots(save_path, color_map)
        
        logging.info(f"帕累托分析图已保存到: {save_path}")
    
    def _plot_accuracy_vs_params(self, ax, color_map):
        """精度 vs 参数量"""
        for model, data in self.results_data.items():
            ax.scatter(data['params_M'], data['r2_score'], 
                      c=[color_map[data['model_type']]], s=100, alpha=0.7,
                      label=data['model_type'] if model.endswith('_Light') else "")
            ax.annotate(model.replace('_', '\n'), 
                       (data['params_M'], data['r2_score']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('参数量 (M)')
        ax.set_ylabel('R² Score (↑)')
        ax.set_title('精度 vs 参数量\n(Iso-Params 对比)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_accuracy_vs_flops(self, ax, color_map):
        """精度 vs FLOPs"""
        for model, data in self.results_data.items():
            ax.scatter(data['flops_G'], data['r2_score'], 
                      c=[color_map[data['model_type']]], s=100, alpha=0.7)
            ax.annotate(model.replace('_', '\n'), 
                       (data['flops_G'], data['r2_score']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('FLOPs (G)')
        ax.set_ylabel('R² Score (↑)')
        ax.set_title('精度 vs 计算量\n(Iso-Compute 对比)')
        ax.grid(True, alpha=0.3)
    
    def _plot_accuracy_vs_latency(self, ax, color_map):
        """精度 vs 延迟"""
        for model, data in self.results_data.items():
            ax.scatter(data['latency_ms'], data['r2_score'], 
                      c=[color_map[data['model_type']]], s=100, alpha=0.7)
            ax.annotate(model.replace('_', '\n'), 
                       (data['latency_ms'], data['r2_score']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('推理延迟 (ms)')
        ax.set_ylabel('R² Score (↑)')
        ax.set_title('精度 vs 延迟\n(Iso-Latency 对比)')
        ax.grid(True, alpha=0.3)
    
    def _plot_efficiency_analysis(self, ax, color_map):
        """效率综合分析"""
        # 计算效率指标
        efficiency_data = []
        for model, data in self.results_data.items():
            accuracy_per_param = data['r2_score'] / data['params_M']
            accuracy_per_flop = data['r2_score'] / data['flops_G']
            accuracy_per_ms = data['r2_score'] / data['latency_ms']
            
            efficiency_data.append({
                'model': model,
                'model_type': data['model_type'],
                'acc_per_param': accuracy_per_param,
                'acc_per_flop': accuracy_per_flop,
                'acc_per_ms': accuracy_per_ms
            })
        
        # 绘制效率气泡图
        for item in efficiency_data:
            ax.scatter(item['acc_per_param'], item['acc_per_flop'], 
                      s=item['acc_per_ms']*1000, 
                      c=[color_map[item['model_type']]], alpha=0.6)
            ax.annotate(item['model'].replace('_', '\n'), 
                       (item['acc_per_param'], item['acc_per_flop']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('精度/参数量 (R²/M)')
        ax.set_ylabel('精度/FLOPs (R²/G)')
        ax.set_title('效率综合分析\n(气泡大小=精度/延迟)')
        ax.grid(True, alpha=0.3)
    
    def _plot_resource_comparison(self, ax, color_map):
        """资源消耗对比"""
        models = list(self.results_data.keys())
        params = [self.results_data[m]['params_M'] for m in models]
        flops = [self.results_data[m]['flops_G'] for m in models]
        memory = [self.results_data[m]['memory_MB']/100 for m in models]  # 缩放到合适范围
        
        x = np.arange(len(models))
        width = 0.25
        
        ax.bar(x - width, params, width, label='参数量 (M)', alpha=0.8)
        ax.bar(x, flops, width, label='FLOPs (G)', alpha=0.8)
        ax.bar(x + width, memory, width, label='显存 (100MB)', alpha=0.8)
        
        ax.set_xlabel('模型')
        ax.set_ylabel('资源消耗')
        ax.set_title('资源消耗对比')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', '\n') for m in models], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_pareto_frontier(self, ax, color_map):
        """帕累托前沿分析"""
        # 使用精度和总资源成本
        for model, data in self.results_data.items():
            # 计算归一化的总资源成本
            normalized_params = data['params_M'] / 20  # 假设最大20M参数
            normalized_flops = data['flops_G'] / 3     # 假设最大3G FLOPs
            normalized_latency = data['latency_ms'] / 30  # 假设最大30ms
            
            total_cost = (normalized_params + normalized_flops + normalized_latency) / 3
            
            ax.scatter(total_cost, data['r2_score'], 
                      c=[color_map[data['model_type']]], s=100, alpha=0.7)
            ax.annotate(model.replace('_', '\n'), 
                       (total_cost, data['r2_score']),
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('归一化总资源成本')
        ax.set_ylabel('R² Score (↑)')
        ax.set_title('帕累托前沿\n(精度 vs 总资源成本)')
        ax.grid(True, alpha=0.3)
    
    def _create_individual_plots(self, save_path: Path, color_map):
        """创建单独的高质量图"""
        # 1. 精度-参数量图
        plt.figure(figsize=(10, 8))
        self._plot_accuracy_vs_params(plt.gca(), color_map)
        plt.savefig(save_path / 'accuracy_vs_params.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path / 'accuracy_vs_params.svg', bbox_inches='tight')
        plt.close()
        
        # 2. 精度-FLOPs图
        plt.figure(figsize=(10, 8))
        self._plot_accuracy_vs_flops(plt.gca(), color_map)
        plt.savefig(save_path / 'accuracy_vs_flops.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path / 'accuracy_vs_flops.svg', bbox_inches='tight')
        plt.close()
        
        # 3. 精度-延迟图
        plt.figure(figsize=(10, 8))
        self._plot_accuracy_vs_latency(plt.gca(), color_map)
        plt.savefig(save_path / 'accuracy_vs_latency.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path / 'accuracy_vs_latency.svg', bbox_inches='tight')
        plt.close()
    
    def generate_model_recommendations(self) -> Dict[str, List[str]]:
        """生成模型选择建议"""
        recommendations = {
            'resource_constrained': [],  # 资源受限场景
            'balanced': [],              # 平衡场景
            'performance_first': []      # 性能优先场景
        }
        
        # 按不同标准排序
        models_by_efficiency = sorted(self.results_data.items(), 
                                    key=lambda x: x[1]['r2_score'] / x[1]['params_M'], 
                                    reverse=True)
        
        models_by_speed = sorted(self.results_data.items(), 
                               key=lambda x: x[1]['r2_score'] / x[1]['latency_ms'], 
                               reverse=True)
        
        models_by_accuracy = sorted(self.results_data.items(), 
                                  key=lambda x: x[1]['r2_score'], 
                                  reverse=True)
        
        # 生成建议
        recommendations['resource_constrained'] = [m[0] for m in models_by_efficiency[:3]]
        recommendations['balanced'] = [m[0] for m in models_by_speed[:3]]
        recommendations['performance_first'] = [m[0] for m in models_by_accuracy[:3]]
        
        return recommendations
    
    def save_analysis_report(self, save_path: str = "./pareto_analysis_report.json"):
        """保存分析报告"""
        recommendations = self.generate_model_recommendations()
        
        # 计算统计信息
        stats = {}
        for metric in ['params_M', 'flops_G', 'latency_ms', 'r2_score']:
            values = [data[metric] for data in self.results_data.values()]
            stats[metric] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'range_ratio': max(values) / min(values)
            }
        
        report = {
            'analysis_summary': {
                'total_models': len(self.results_data),
                'model_types': list(set([data['model_type'] for data in self.results_data.values()])),
                'statistics': stats
            },
            'recommendations': recommendations,
            'key_findings': [
                f"参数量差异: {stats['params_M']['range_ratio']:.1f}x",
                f"FLOPs差异: {stats['flops_G']['range_ratio']:.1f}x", 
                f"延迟差异: {stats['latency_ms']['range_ratio']:.1f}x",
                f"精度差异: {stats['r2_score']['range_ratio']:.3f}x"
            ]
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logging.info(f"分析报告已保存到: {save_path}")
        return report

def main():
    """主函数"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 创建帕累托分析器
    analyzer = ParetoAnalyzer()
    
    # 生成帕累托图
    analyzer.create_pareto_plots()
    
    # 生成分析报告
    report = analyzer.save_analysis_report()
    
    # 打印关键发现
    print("\n=== 帕累托分析关键发现 ===")
    for finding in report['key_findings']:
        print(f"• {finding}")
    
    print("\n=== 模型选择建议 ===")
    for scenario, models in report['recommendations'].items():
        print(f"\n{scenario}:")
        for i, model in enumerate(models, 1):
            print(f"  {i}. {model}")
    
    logging.info("帕累托分析完成！")

if __name__ == "__main__":
    main()