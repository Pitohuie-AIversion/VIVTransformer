"""
多维度模型评估工具
实现Iso-Compute、Iso-Latency、Iso-Params等多种公平对比方案
适用于稀疏到稠密重建任务的模型评估

核心功能:
1. 参数量统计 (Iso-Params)
2. FLOPs计算 (Iso-Compute) 
3. 显存使用统计
4. 推理延迟测量 (Iso-Latency)
5. 帕累托曲线生成
6. 完整评估报告
"""

import torch
import torch.nn as nn
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import yaml
import json
from collections import defaultdict
import warnings

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class ModelProfiler:
    """模型性能分析器"""
    
    def __init__(self, device: str = "cuda", warmup_iters: int = 10, measure_iters: int = 50):
        self.device = device
        self.warmup_iters = warmup_iters
        self.measure_iters = measure_iters
        self.results = {}
        
    def count_parameters(self, model: nn.Module) -> Dict[str, int]:
        """统计模型参数量"""
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'non_trainable_params': total_params - trainable_params,
            'total_params_M': total_params / 1e6,
            'trainable_params_M': trainable_params / 1e6
        }
    
    def count_flops(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """计算FLOPs (使用简化估算)"""
        try:
            # 尝试使用fvcore (如果可用)
            from fvcore.nn import FlopCountMode, flop_count
            
            model = model.to(self.device)
            dummy_input = torch.randn(input_shape).to(self.device)
            
            with torch.no_grad():
                flop_dict, _ = flop_count(model, (dummy_input,), supported_ops=None)
                total_flops = sum(flop_dict.values())
                
            return {
                'total_flops': total_flops,
                'total_flops_G': total_flops / 1e9,
                'flops_per_param': total_flops / self.count_parameters(model)['total_params']
            }
            
        except ImportError:
            # 简化估算方法
            logging.warning("fvcore未安装，使用简化FLOPs估算")
            params = self.count_parameters(model)['total_params']
            # 粗略估算: 每个参数约2个FLOPs (前向传播)
            estimated_flops = params * 2 * np.prod(input_shape[1:])  # 排除batch维度
            
            return {
                'total_flops': estimated_flops,
                'total_flops_G': estimated_flops / 1e9,
                'flops_per_param': 2.0,
                'estimation_method': 'simplified'
            }
    
    def measure_memory(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """测量显存使用"""
        if not torch.cuda.is_available():
            return {'peak_memory_MB': 0, 'model_memory_MB': 0}
            
        model = model.to(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # 测量模型本身的显存
        model_memory = torch.cuda.memory_allocated() / 1024**2
        
        # 测量前向传播的峰值显存
        dummy_input = torch.randn(input_shape).to(self.device)
        with torch.no_grad():
            _ = model(dummy_input)
            
        peak_memory = torch.cuda.max_memory_allocated() / 1024**2
        
        return {
            'peak_memory_MB': peak_memory,
            'model_memory_MB': model_memory,
            'forward_memory_MB': peak_memory - model_memory
        }
    
    def measure_latency(self, model: nn.Module, input_shape: Tuple[int, ...]) -> Dict[str, float]:
        """测量推理延迟"""
        model = model.to(self.device).eval()
        dummy_input = torch.randn(input_shape).to(self.device)
        
        # 预热
        with torch.no_grad():
            for _ in range(self.warmup_iters):
                _ = model(dummy_input)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # 测量延迟
        latencies = []
        with torch.no_grad():
            for _ in range(self.measure_iters):
                start_time = time.time()
                _ = model(dummy_input)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # 转换为毫秒
        
        latencies = np.array(latencies)
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'median_latency_ms': np.median(latencies),
            'throughput_samples_per_sec': 1000.0 / np.mean(latencies) * input_shape[0]  # 考虑batch size
        }
    
    def profile_model(self, model: nn.Module, model_name: str, input_shape: Tuple[int, ...]) -> Dict[str, Any]:
        """完整的模型性能分析"""
        logging.info(f"开始分析模型: {model_name}")
        
        results = {'model_name': model_name, 'input_shape': input_shape}
        
        # 参数量统计
        results.update(self.count_parameters(model))
        
        # FLOPs计算
        results.update(self.count_flops(model, input_shape))
        
        # 显存测量
        results.update(self.measure_memory(model, input_shape))
        
        # 延迟测量
        results.update(self.measure_latency(model, input_shape))
        
        # 计算效率指标
        results['params_per_flop'] = results['total_params'] / results['total_flops']
        results['flops_per_ms'] = results['total_flops'] / results['mean_latency_ms']
        results['memory_efficiency'] = results['total_params_M'] / results['peak_memory_MB'] if results['peak_memory_MB'] > 0 else 0
        
        self.results[model_name] = results
        logging.info(f"模型 {model_name} 分析完成")
        
        return results

class IsoConditionDesigner:
    """等量基准设计器"""
    
    def __init__(self, profiler_results: Dict[str, Dict]):
        self.results = profiler_results
        
    def design_iso_params_config(self, target_params_M: float = 5.0, tolerance: float = 0.5) -> Dict[str, List[str]]:
        """设计等参数量对比配置"""
        iso_params_groups = defaultdict(list)
        
        for model_name, metrics in self.results.items():
            params_M = metrics['total_params_M']
            if abs(params_M - target_params_M) <= tolerance:
                iso_params_groups[f"~{target_params_M:.1f}M"].append(model_name)
        
        return dict(iso_params_groups)
    
    def design_iso_compute_config(self, target_flops_G: float = 1.0, tolerance: float = 0.3) -> Dict[str, List[str]]:
        """设计等计算量对比配置"""
        iso_compute_groups = defaultdict(list)
        
        for model_name, metrics in self.results.items():
            flops_G = metrics['total_flops_G']
            if abs(flops_G - target_flops_G) <= tolerance:
                iso_compute_groups[f"~{target_flops_G:.1f}G"].append(model_name)
        
        return dict(iso_compute_groups)
    
    def design_iso_latency_config(self, target_latency_ms: float = 10.0, tolerance: float = 2.0) -> Dict[str, List[str]]:
        """设计等时延对比配置"""
        iso_latency_groups = defaultdict(list)
        
        for model_name, metrics in self.results.items():
            latency_ms = metrics['mean_latency_ms']
            if abs(latency_ms - target_latency_ms) <= tolerance:
                iso_latency_groups[f"~{target_latency_ms:.1f}ms"].append(model_name)
        
        return dict(iso_latency_groups)
    
    def find_optimal_iso_conditions(self) -> Dict[str, Dict]:
        """自动寻找最优的等量基准条件"""
        # 分析参数量分布
        params_values = [metrics['total_params_M'] for metrics in self.results.values()]
        params_median = np.median(params_values)
        
        # 分析FLOPs分布
        flops_values = [metrics['total_flops_G'] for metrics in self.results.values()]
        flops_median = np.median(flops_values)
        
        # 分析延迟分布
        latency_values = [metrics['mean_latency_ms'] for metrics in self.results.values()]
        latency_median = np.median(latency_values)
        
        return {
            'iso_params': self.design_iso_params_config(params_median, params_median * 0.2),
            'iso_compute': self.design_iso_compute_config(flops_median, flops_median * 0.3),
            'iso_latency': self.design_iso_latency_config(latency_median, latency_median * 0.2)
        }

class ParetoAnalyzer:
    """帕累托分析器"""
    
    def __init__(self, profiler_results: Dict[str, Dict], performance_results: Dict[str, Dict] = None):
        self.profiler_results = profiler_results
        self.performance_results = performance_results or {}
        
    def create_pareto_plots(self, save_dir: str = "./pareto_analysis"):
        """创建帕累托曲线图"""
        save_path = Path(save_dir)
        save_path.mkdir(exist_ok=True)
        
        # 准备数据
        models = list(self.profiler_results.keys())
        params_M = [self.profiler_results[m]['total_params_M'] for m in models]
        flops_G = [self.profiler_results[m]['total_flops_G'] for m in models]
        latency_ms = [self.profiler_results[m]['mean_latency_ms'] for m in models]
        
        # 如果有性能结果，使用真实精度；否则使用模拟数据
        if self.performance_results:
            accuracy = [self.performance_results[m].get('mse_loss', np.random.uniform(0.01, 0.1)) for m in models]
        else:
            # 模拟精度数据 (MSE loss，越小越好)
            accuracy = [0.05 + 0.03 * np.random.random() for _ in models]
            logging.warning("使用模拟精度数据，请提供真实性能结果以获得准确分析")
        
        # 创建三个帕累托图
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. 精度 vs 参数量
        axes[0].scatter(params_M, accuracy, s=100, alpha=0.7, c='blue')
        for i, model in enumerate(models):
            axes[0].annotate(model, (params_M[i], accuracy[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9)
        axes[0].set_xlabel('参数量 (M)')
        axes[0].set_ylabel('MSE Loss (↓)')
        axes[0].set_title('精度 vs 参数量')
        axes[0].grid(True, alpha=0.3)
        
        # 2. 精度 vs FLOPs
        axes[1].scatter(flops_G, accuracy, s=100, alpha=0.7, c='green')
        for i, model in enumerate(models):
            axes[1].annotate(model, (flops_G[i], accuracy[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9)
        axes[1].set_xlabel('FLOPs (G)')
        axes[1].set_ylabel('MSE Loss (↓)')
        axes[1].set_title('精度 vs 计算量')
        axes[1].grid(True, alpha=0.3)
        
        # 3. 精度 vs 延迟
        axes[2].scatter(latency_ms, accuracy, s=100, alpha=0.7, c='red')
        for i, model in enumerate(models):
            axes[2].annotate(model, (latency_ms[i], accuracy[i]), xytext=(5, 5), 
                           textcoords='offset points', fontsize=9)
        axes[2].set_xlabel('推理延迟 (ms)')
        axes[2].set_ylabel('MSE Loss (↓)')
        axes[2].set_title('精度 vs 延迟')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path / 'pareto_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(save_path / 'pareto_curves.svg', bbox_inches='tight')
        plt.close()
        
        logging.info(f"帕累托曲线已保存到: {save_path}")

class EvaluationReporter:
    """评估报告生成器"""
    
    def __init__(self, profiler_results: Dict[str, Dict], iso_conditions: Dict[str, Dict] = None):
        self.profiler_results = profiler_results
        self.iso_conditions = iso_conditions or {}
        
    def generate_summary_table(self) -> pd.DataFrame:
        """生成汇总表格"""
        data = []
        
        for model_name, metrics in self.profiler_results.items():
            row = {
                '模型': model_name,
                '参数(M)': f"{metrics['total_params_M']:.2f}",
                'FLOPs(G)': f"{metrics['total_flops_G']:.2f}",
                '显存(MB)': f"{metrics['peak_memory_MB']:.1f}",
                '延迟(ms)': f"{metrics['mean_latency_ms']:.2f}±{metrics['std_latency_ms']:.2f}",
                '吞吐(samples/s)': f"{metrics['throughput_samples_per_sec']:.1f}",
                '参数效率': f"{metrics['memory_efficiency']:.3f}",
                'FLOPs/参数': f"{metrics['flops_per_param']:.1f}"
            }
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_detailed_report(self, save_path: str = "./evaluation_report.json"):
        """保存详细报告"""
        report = {
            'profiler_results': self.profiler_results,
            'iso_conditions': self.iso_conditions,
            'summary_statistics': self._calculate_summary_stats(),
            'recommendations': self._generate_recommendations()
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logging.info(f"详细报告已保存到: {save_path}")
    
    def _calculate_summary_stats(self) -> Dict[str, Any]:
        """计算汇总统计"""
        params_values = [m['total_params_M'] for m in self.profiler_results.values()]
        flops_values = [m['total_flops_G'] for m in self.profiler_results.values()]
        latency_values = [m['mean_latency_ms'] for m in self.profiler_results.values()]
        
        return {
            'params_stats': {
                'min': min(params_values),
                'max': max(params_values),
                'mean': np.mean(params_values),
                'std': np.std(params_values),
                'ratio_max_min': max(params_values) / min(params_values)
            },
            'flops_stats': {
                'min': min(flops_values),
                'max': max(flops_values),
                'mean': np.mean(flops_values),
                'std': np.std(flops_values),
                'ratio_max_min': max(flops_values) / min(flops_values)
            },
            'latency_stats': {
                'min': min(latency_values),
                'max': max(latency_values),
                'mean': np.mean(latency_values),
                'std': np.std(latency_values),
                'ratio_max_min': max(latency_values) / min(latency_values)
            }
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成建议"""
        recommendations = []
        
        stats = self._calculate_summary_stats()
        
        if stats['params_stats']['ratio_max_min'] > 5:
            recommendations.append("参数量差异过大(>5x)，建议采用Iso-Compute或Iso-Latency对比")
        
        if stats['latency_stats']['ratio_max_min'] > 3:
            recommendations.append("延迟差异较大(>3x)，建议报告多种硬件环境下的性能")
        
        recommendations.append("建议同时报告Iso-Params、Iso-Compute、Iso-Latency三种对比结果")
        recommendations.append("建议包含置信区间或多次重复实验的统计结果")
        
        return recommendations

def main():
    """主函数 - 演示完整的多维度评估流程"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 这里应该加载实际的模型，现在用示例代替
    logging.info("开始多维度模型评估...")
    
    # 示例：创建简单的测试模型
    class SimpleModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # 创建测试模型
    models = {
        'small_mlp': SimpleModel(1024, 256, 16384),
        'medium_mlp': SimpleModel(1024, 512, 16384),
        'large_mlp': SimpleModel(1024, 1024, 16384)
    }
    
    # 初始化分析器
    profiler = ModelProfiler(device="cuda" if torch.cuda.is_available() else "cpu")
    input_shape = (32, 1024)  # batch_size=32, input_dim=1024
    
    # 分析所有模型
    for model_name, model in models.items():
        profiler.profile_model(model, model_name, input_shape)
    
    # 设计等量基准
    iso_designer = IsoConditionDesigner(profiler.results)
    iso_conditions = iso_designer.find_optimal_iso_conditions()
    
    # 生成帕累托分析
    pareto_analyzer = ParetoAnalyzer(profiler.results)
    pareto_analyzer.create_pareto_plots()
    
    # 生成评估报告
    reporter = EvaluationReporter(profiler.results, iso_conditions)
    summary_table = reporter.generate_summary_table()
    
    print("\n=== 模型性能汇总表 ===")
    print(summary_table.to_string(index=False))
    
    print("\n=== 等量基准分组 ===")
    for condition_type, groups in iso_conditions.items():
        print(f"\n{condition_type}:")
        for group_name, models_in_group in groups.items():
            print(f"  {group_name}: {models_in_group}")
    
    # 保存详细报告
    reporter.save_detailed_report()
    
    logging.info("多维度评估完成！")

if __name__ == "__main__":
    main()