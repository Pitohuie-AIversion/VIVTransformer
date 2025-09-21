"""
性能基准测试框架
验证不同硬件配置下的模型性能表现，支持统一参数量控制和硬件自适应优化
"""

import torch
import torch.nn as nn
import time
import psutil
import yaml
import logging
import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import json

from hardware_aware_deployment import HardwareProfiler, AdaptiveDeploymentManager
from unified_model_manager import UnifiedModelManager, ResourceMonitor

@dataclass
class BenchmarkMetrics:
    """基准测试指标"""
    model_name: str
    tier: str
    parameter_count: int
    
    # 性能指标
    training_time_per_epoch: float
    inference_time_per_sample: float
    memory_usage_mb: float
    gpu_memory_usage_mb: float
    cpu_utilization_percent: float
    
    # 准确性指标
    final_loss: float
    convergence_epochs: int
    
    # 硬件效率指标
    parameters_per_second: float
    memory_efficiency: float  # 参数量/内存使用量
    hardware_utilization_score: float

@dataclass
class HardwareProfile:
    """硬件配置档案"""
    cpu_cores: int
    memory_mb: int
    gpu_memory_mb: int
    gpu_name: str
    tier: str

class PerformanceBenchmark:
    """性能基准测试器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.hardware_profiler = HardwareProfiler()
        self.model_manager = UnifiedModelManager(config_path)
        self.resource_monitor = ResourceMonitor()
        
        # 测试结果存储
        self.benchmark_results: List[BenchmarkMetrics] = []
        self.hardware_profile = self._create_hardware_profile()
        
    def _create_hardware_profile(self) -> HardwareProfile:
        """创建硬件配置档案"""
        hardware_info = self.hardware_profiler.get_system_info()
        
        gpu_memory = 0
        gpu_name = "None"
        if hardware_info['gpu']['available'] and hardware_info['gpu']['devices']:
            gpu_device = hardware_info['gpu']['devices'][0]
            gpu_memory = gpu_device['memory_mb']
            gpu_name = gpu_device['name']
        
        # 确定硬件级别
        deployment_result = self.model_manager.initialize_deployment()
        tier = deployment_result['selected_tier']
        
        return HardwareProfile(
            cpu_cores=hardware_info['cpu']['cores'],
            memory_mb=hardware_info['memory']['total_mb'],
            gpu_memory_mb=gpu_memory,
            gpu_name=gpu_name,
            tier=tier
        )
    
    def run_comprehensive_benchmark(self, num_epochs: int = 10, 
                                  num_samples: int = 100) -> Dict[str, Any]:
        """运行综合基准测试"""
        self.logger.info("🚀 开始综合性能基准测试...")
        
        # 获取所有注册的模型
        model_stats = self.model_manager.get_model_statistics()
        model_names = list(model_stats['models'].keys())
        
        self.logger.info(f"📊 测试模型数量: {len(model_names)}")
        self.logger.info(f"🏷️ 硬件级别: {self.hardware_profile.tier}")
        
        # 逐个测试模型
        for model_name in model_names:
            self.logger.info(f"🔄 测试模型: {model_name}")
            
            try:
                metrics = self._benchmark_single_model(
                    model_name, num_epochs, num_samples
                )
                self.benchmark_results.append(metrics)
                
                # 卸载模型释放内存
                self.model_manager.unload_model(model_name)
                
            except Exception as e:
                self.logger.error(f"❌ 模型 {model_name} 测试失败: {e}")
                continue
        
        # 生成综合报告
        benchmark_report = self._generate_benchmark_report()
        
        self.logger.info("✅ 综合基准测试完成")
        return benchmark_report
    
    def _benchmark_single_model(self, model_name: str, num_epochs: int, 
                               num_samples: int) -> BenchmarkMetrics:
        """对单个模型进行基准测试"""
        # 加载模型
        model = self.model_manager.load_model(model_name)
        if model is None:
            raise ValueError(f"无法加载模型: {model_name}")
        
        model_info = self.model_manager.model_registry[model_name]
        
        # 创建虚拟数据
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        batch_size = 8
        input_size = 1024
        
        # 开始监控
        self.resource_monitor.start_monitoring()
        self.resource_monitor.record_snapshot('start')
        
        # 训练性能测试
        training_times = []
        losses = []
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # 创建批次数据
            inputs = torch.randn(batch_size, input_size, device=device)
            targets = torch.randn(batch_size, input_size, device=device)
            
            # 前向传播
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_time = time.time() - epoch_start
            training_times.append(epoch_time)
            losses.append(loss.item())
            
            self.resource_monitor.record_snapshot(f'epoch_{epoch}')
        
        # 推理性能测试
        model.eval()
        inference_times = []
        
        with torch.no_grad():
            for _ in range(num_samples):
                input_sample = torch.randn(1, input_size, device=device)
                
                start_time = time.time()
                _ = model(input_sample)
                inference_time = time.time() - start_time
                
                inference_times.append(inference_time)
        
        self.resource_monitor.record_snapshot('end')
        
        # 计算指标
        avg_training_time = np.mean(training_times)
        avg_inference_time = np.mean(inference_times)
        final_loss = losses[-1]
        
        # 收敛分析
        convergence_epochs = self._analyze_convergence(losses)
        
        # 资源使用分析
        peak_usage = self.resource_monitor.get_peak_usage()
        memory_usage = peak_usage.get('peak_memory_mb', 0)
        gpu_memory_usage = peak_usage.get('peak_gpu_memory_mb', 0)
        cpu_utilization = peak_usage.get('peak_cpu_percent', 0)
        
        # 效率指标计算
        parameters_per_second = model_info.parameter_count / avg_training_time if avg_training_time > 0 else 0
        memory_efficiency = model_info.parameter_count / memory_usage if memory_usage > 0 else 0
        hardware_utilization_score = self._calculate_hardware_utilization_score(
            cpu_utilization, memory_usage, gpu_memory_usage
        )
        
        return BenchmarkMetrics(
            model_name=model_name,
            tier=model_info.tier,
            parameter_count=model_info.parameter_count,
            training_time_per_epoch=avg_training_time,
            inference_time_per_sample=avg_inference_time,
            memory_usage_mb=memory_usage,
            gpu_memory_usage_mb=gpu_memory_usage,
            cpu_utilization_percent=cpu_utilization,
            final_loss=final_loss,
            convergence_epochs=convergence_epochs,
            parameters_per_second=parameters_per_second,
            memory_efficiency=memory_efficiency,
            hardware_utilization_score=hardware_utilization_score
        )
    
    def _analyze_convergence(self, losses: List[float]) -> int:
        """分析收敛情况"""
        if len(losses) < 3:
            return len(losses)
        
        # 简单的收敛检测：连续3个epoch损失变化小于1%
        for i in range(2, len(losses)):
            recent_losses = losses[i-2:i+1]
            if max(recent_losses) - min(recent_losses) < 0.01 * recent_losses[0]:
                return i + 1
        
        return len(losses)
    
    def _calculate_hardware_utilization_score(self, cpu_percent: float, 
                                            memory_mb: float, gpu_memory_mb: float) -> float:
        """计算硬件利用率评分"""
        # CPU利用率评分 (0-100)
        cpu_score = min(cpu_percent, 100)
        
        # 内存利用率评分
        memory_utilization = (memory_mb / self.hardware_profile.memory_mb) * 100
        memory_score = min(memory_utilization, 100)
        
        # GPU利用率评分
        gpu_score = 0
        if self.hardware_profile.gpu_memory_mb > 0:
            gpu_utilization = (gpu_memory_mb / self.hardware_profile.gpu_memory_mb) * 100
            gpu_score = min(gpu_utilization, 100)
        
        # 综合评分 (权重: CPU 30%, Memory 40%, GPU 30%)
        if gpu_score > 0:
            total_score = (cpu_score * 0.3 + memory_score * 0.4 + gpu_score * 0.3)
        else:
            total_score = (cpu_score * 0.4 + memory_score * 0.6)
        
        return total_score
    
    def _generate_benchmark_report(self) -> Dict[str, Any]:
        """生成基准测试报告"""
        if not self.benchmark_results:
            return {"error": "没有可用的测试结果"}
        
        # 基础统计
        total_models = len(self.benchmark_results)
        avg_training_time = np.mean([r.training_time_per_epoch for r in self.benchmark_results])
        avg_inference_time = np.mean([r.inference_time_per_sample for r in self.benchmark_results])
        avg_memory_usage = np.mean([r.memory_usage_mb for r in self.benchmark_results])
        
        # 性能排名
        performance_ranking = self._rank_models_by_performance()
        efficiency_ranking = self._rank_models_by_efficiency()
        
        # 硬件适配性分析
        hardware_compatibility = self._analyze_hardware_compatibility()
        
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hardware_profile': asdict(self.hardware_profile),
            'test_summary': {
                'total_models': total_models,
                'avg_training_time_per_epoch': avg_training_time,
                'avg_inference_time_per_sample': avg_inference_time,
                'avg_memory_usage_mb': avg_memory_usage
            },
            'performance_ranking': performance_ranking,
            'efficiency_ranking': efficiency_ranking,
            'hardware_compatibility': hardware_compatibility,
            'detailed_results': [asdict(result) for result in self.benchmark_results],
            'recommendations': self._generate_performance_recommendations()
        }
        
        return report
    
    def _rank_models_by_performance(self) -> List[Dict[str, Any]]:
        """按性能对模型排名"""
        # 综合性能评分：训练速度 + 推理速度 + 收敛性
        def performance_score(metrics: BenchmarkMetrics) -> float:
            # 训练速度评分 (越快越好)
            training_score = 1.0 / (metrics.training_time_per_epoch + 1e-6)
            
            # 推理速度评分 (越快越好)
            inference_score = 1.0 / (metrics.inference_time_per_sample + 1e-6)
            
            # 收敛性评分 (越快收敛越好)
            convergence_score = 1.0 / (metrics.convergence_epochs + 1)
            
            # 综合评分
            return (training_score * 0.4 + inference_score * 0.4 + convergence_score * 0.2)
        
        ranked_results = sorted(self.benchmark_results, 
                              key=performance_score, reverse=True)
        
        ranking = []
        for i, result in enumerate(ranked_results):
            ranking.append({
                'rank': i + 1,
                'model_name': result.model_name,
                'performance_score': performance_score(result),
                'training_time': result.training_time_per_epoch,
                'inference_time': result.inference_time_per_sample,
                'convergence_epochs': result.convergence_epochs
            })
        
        return ranking
    
    def _rank_models_by_efficiency(self) -> List[Dict[str, Any]]:
        """按效率对模型排名"""
        # 效率评分：参数效率 + 内存效率 + 硬件利用率
        def efficiency_score(metrics: BenchmarkMetrics) -> float:
            param_efficiency = metrics.parameters_per_second / 1e6  # 标准化
            memory_efficiency = metrics.memory_efficiency / 1e3  # 标准化
            hardware_score = metrics.hardware_utilization_score / 100  # 标准化
            
            return (param_efficiency * 0.4 + memory_efficiency * 0.3 + hardware_score * 0.3)
        
        ranked_results = sorted(self.benchmark_results, 
                              key=efficiency_score, reverse=True)
        
        ranking = []
        for i, result in enumerate(ranked_results):
            ranking.append({
                'rank': i + 1,
                'model_name': result.model_name,
                'efficiency_score': efficiency_score(result),
                'parameters_per_second': result.parameters_per_second,
                'memory_efficiency': result.memory_efficiency,
                'hardware_utilization_score': result.hardware_utilization_score
            })
        
        return ranking
    
    def _analyze_hardware_compatibility(self) -> Dict[str, Any]:
        """分析硬件兼容性"""
        compatibility = {
            'tier_match': self.hardware_profile.tier,
            'memory_sufficient': True,
            'gpu_utilization': 'optimal',
            'bottlenecks': []
        }
        
        # 检查内存是否充足
        max_memory_usage = max(r.memory_usage_mb for r in self.benchmark_results)
        if max_memory_usage > self.hardware_profile.memory_mb * 0.9:
            compatibility['memory_sufficient'] = False
            compatibility['bottlenecks'].append('内存不足')
        
        # 检查GPU利用率
        if self.hardware_profile.gpu_memory_mb > 0:
            avg_gpu_usage = np.mean([r.gpu_memory_usage_mb for r in self.benchmark_results])
            gpu_utilization_rate = avg_gpu_usage / self.hardware_profile.gpu_memory_mb
            
            if gpu_utilization_rate < 0.3:
                compatibility['gpu_utilization'] = 'underutilized'
                compatibility['bottlenecks'].append('GPU利用率低')
            elif gpu_utilization_rate > 0.9:
                compatibility['gpu_utilization'] = 'overutilized'
                compatibility['bottlenecks'].append('GPU内存不足')
        
        return compatibility
    
    def _generate_performance_recommendations(self) -> List[str]:
        """生成性能优化建议"""
        recommendations = []
        
        if not self.benchmark_results:
            return recommendations
        
        # 基于测试结果的建议
        avg_training_time = np.mean([r.training_time_per_epoch for r in self.benchmark_results])
        avg_memory_usage = np.mean([r.memory_usage_mb for r in self.benchmark_results])
        
        if avg_training_time > 10:  # 训练时间过长
            recommendations.append("训练时间较长，建议使用更小的模型或增加批次大小")
        
        if avg_memory_usage > self.hardware_profile.memory_mb * 0.8:
            recommendations.append("内存使用率高，建议减少批次大小或使用内存优化技术")
        
        # 基于硬件配置的建议
        if self.hardware_profile.cpu_cores < 4:
            recommendations.append("CPU核心数较少，建议使用轻量级模型")
        
        if self.hardware_profile.gpu_memory_mb == 0:
            recommendations.append("未检测到GPU，建议使用CPU优化的模型配置")
        
        # 基于模型性能的建议
        best_model = min(self.benchmark_results, key=lambda x: x.training_time_per_epoch)
        recommendations.append(f"推荐使用 {best_model.model_name}，训练效率最高")
        
        return recommendations
    
    def visualize_benchmark_results(self, output_dir: str = "results/benchmark_plots"):
        """可视化基准测试结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.benchmark_results:
            self.logger.warning("没有可用的测试结果进行可视化")
            return
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 1. 训练时间对比
        self._plot_training_time_comparison(output_dir)
        
        # 2. 内存使用对比
        self._plot_memory_usage_comparison(output_dir)
        
        # 3. 效率分析
        self._plot_efficiency_analysis(output_dir)
        
        # 4. 硬件利用率
        self._plot_hardware_utilization(output_dir)
        
        self.logger.info(f"📊 基准测试可视化图表已保存到: {output_dir}")
    
    def _plot_training_time_comparison(self, output_dir: str):
        """绘制训练时间对比图"""
        model_names = [r.model_name for r in self.benchmark_results]
        training_times = [r.training_time_per_epoch for r in self.benchmark_results]
        
        plt.figure(figsize=(12, 6))
        bars = plt.bar(model_names, training_times, color='skyblue', alpha=0.7)
        
        # 添加数值标签
        for bar, time_val in zip(bars, training_times):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{time_val:.3f}s', ha='center', va='bottom')
        
        plt.title(f'模型训练时间对比 ({self.hardware_profile.tier} 级别)', fontsize=14, fontweight='bold')
        plt.xlabel('模型名称', fontsize=12)
        plt.ylabel('每轮训练时间 (秒)', fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/training_time_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_memory_usage_comparison(self, output_dir: str):
        """绘制内存使用对比图"""
        model_names = [r.model_name for r in self.benchmark_results]
        memory_usage = [r.memory_usage_mb for r in self.benchmark_results]
        gpu_memory_usage = [r.gpu_memory_usage_mb for r in self.benchmark_results]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        plt.figure(figsize=(12, 6))
        bars1 = plt.bar(x - width/2, memory_usage, width, label='系统内存', color='lightcoral', alpha=0.7)
        bars2 = plt.bar(x + width/2, gpu_memory_usage, width, label='GPU内存', color='lightgreen', alpha=0.7)
        
        plt.title(f'模型内存使用对比 ({self.hardware_profile.tier} 级别)', fontsize=14, fontweight='bold')
        plt.xlabel('模型名称', fontsize=12)
        plt.ylabel('内存使用量 (MB)', fontsize=12)
        plt.xticks(x, model_names, rotation=45, ha='right')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/memory_usage_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_efficiency_analysis(self, output_dir: str):
        """绘制效率分析图"""
        model_names = [r.model_name for r in self.benchmark_results]
        param_efficiency = [r.parameters_per_second / 1e6 for r in self.benchmark_results]  # M params/s
        memory_efficiency = [r.memory_efficiency / 1e3 for r in self.benchmark_results]  # K params/MB
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 参数处理效率
        bars1 = ax1.bar(model_names, param_efficiency, color='gold', alpha=0.7)
        ax1.set_title('参数处理效率', fontsize=12, fontweight='bold')
        ax1.set_xlabel('模型名称', fontsize=10)
        ax1.set_ylabel('百万参数/秒', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        # 内存效率
        bars2 = ax2.bar(model_names, memory_efficiency, color='mediumpurple', alpha=0.7)
        ax2.set_title('内存效率', fontsize=12, fontweight='bold')
        ax2.set_xlabel('模型名称', fontsize=10)
        ax2.set_ylabel('千参数/MB', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.suptitle(f'模型效率分析 ({self.hardware_profile.tier} 级别)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/efficiency_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_hardware_utilization(self, output_dir: str):
        """绘制硬件利用率图"""
        model_names = [r.model_name for r in self.benchmark_results]
        cpu_utilization = [r.cpu_utilization_percent for r in self.benchmark_results]
        hardware_scores = [r.hardware_utilization_score for r in self.benchmark_results]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # CPU利用率
        bars1 = ax1.bar(model_names, cpu_utilization, color='orange', alpha=0.7)
        ax1.set_title('CPU利用率', fontsize=12, fontweight='bold')
        ax1.set_xlabel('模型名称', fontsize=10)
        ax1.set_ylabel('CPU使用率 (%)', fontsize=10)
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        ax1.set_ylim(0, 100)
        
        # 硬件利用率综合评分
        bars2 = ax2.bar(model_names, hardware_scores, color='teal', alpha=0.7)
        ax2.set_title('硬件利用率综合评分', fontsize=12, fontweight='bold')
        ax2.set_xlabel('模型名称', fontsize=10)
        ax2.set_ylabel('综合评分', fontsize=10)
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.suptitle(f'硬件利用率分析 ({self.hardware_profile.tier} 级别)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        plt.savefig(f"{output_dir}/hardware_utilization.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_benchmark_report(self, output_path: str):
        """保存基准测试报告"""
        report = self._generate_benchmark_report()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        # 同时保存JSON格式便于程序读取
        json_path = output_path.replace('.yaml', '.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"📄 基准测试报告已保存: {output_path}")

def main():
    """主函数 - 运行性能基准测试"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 配置文件路径
    config_path = "configs/unified_adaptive_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    # 创建性能基准测试器
    benchmark = PerformanceBenchmark(config_path)
    
    print("🚀 开始性能基准测试...")
    print(f"🏷️ 硬件级别: {benchmark.hardware_profile.tier}")
    print(f"💻 CPU: {benchmark.hardware_profile.cpu_cores} 核心")
    print(f"🧠 内存: {benchmark.hardware_profile.memory_mb:.0f} MB")
    print(f"🎮 GPU: {benchmark.hardware_profile.gpu_name}")
    
    # 运行基准测试
    benchmark_report = benchmark.run_comprehensive_benchmark(
        num_epochs=5,  # 减少epoch数量以加快测试
        num_samples=50  # 减少样本数量以加快测试
    )
    
    # 显示结果摘要
    if 'test_summary' in benchmark_report:
        summary = benchmark_report['test_summary']
        print(f"\n📊 测试结果摘要:")
        print(f"  测试模型数: {summary['total_models']}")
        print(f"  平均训练时间: {summary['avg_training_time_per_epoch']:.3f}s/epoch")
        print(f"  平均推理时间: {summary['avg_inference_time_per_sample']:.6f}s/sample")
        print(f"  平均内存使用: {summary['avg_memory_usage_mb']:.1f}MB")
    
    # 显示性能排名
    if 'performance_ranking' in benchmark_report:
        print(f"\n🏆 性能排名 (前3名):")
        for rank_info in benchmark_report['performance_ranking'][:3]:
            print(f"  {rank_info['rank']}. {rank_info['model_name']} "
                  f"(训练: {rank_info['training_time']:.3f}s, "
                  f"推理: {rank_info['inference_time']:.6f}s)")
    
    # 保存报告和可视化
    os.makedirs("results/benchmark", exist_ok=True)
    benchmark.save_benchmark_report("results/benchmark/performance_benchmark_report.yaml")
    benchmark.visualize_benchmark_results("results/benchmark/plots")
    
    print(f"\n✅ 基准测试完成！")
    print(f"📄 详细报告: results/benchmark/performance_benchmark_report.yaml")
    print(f"📊 可视化图表: results/benchmark/plots/")

if __name__ == "__main__":
    main()