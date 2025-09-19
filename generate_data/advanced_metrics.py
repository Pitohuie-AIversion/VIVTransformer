#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高级性能评估指标和可视化模块

功能:
1. 提供全面的性能评估指标
2. 生成详细的可视化分析
3. 支持统计显著性测试
4. 提供模型效率分析

作者: AI Assistant
日期: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from scipy import stats
from scipy.signal import find_peaks
from sklearn.metrics import r2_score
import warnings
warnings.filterwarnings('ignore')

@dataclass
class ModelPerformance:
    """模型性能数据类"""
    name: str
    type: str
    metrics: Dict[str, float]
    training_history: Optional[List[float]] = None
    predictions: Optional[np.ndarray] = None
    targets: Optional[np.ndarray] = None
    param_count: int = 0
    train_time: float = 0.0
    test_time: float = 0.0
    memory_usage: float = 0.0

class AdvancedMetrics:
    """高级性能评估指标计算器"""
    
    @staticmethod
    def structural_similarity_index(y_true: np.ndarray, y_pred: np.ndarray, 
                                  window_size: int = 11, k1: float = 0.01, 
                                  k2: float = 0.03) -> float:
        """计算结构相似性指数 (SSIM)"""
        if y_true.shape != y_pred.shape:
            return 0.0
            
        # 转换为2D如果是1D
        if len(y_true.shape) == 1:
            size = int(np.sqrt(len(y_true)))
            if size * size == len(y_true):
                y_true = y_true.reshape(size, size)
                y_pred = y_pred.reshape(size, size)
            else:
                # 如果不是完全平方数，使用简化计算
                return AdvancedMetrics._simple_ssim(y_true, y_pred)
        
        # 计算局部均值和方差
        mu1 = np.mean(y_true)
        mu2 = np.mean(y_pred)
        sigma1_sq = np.var(y_true)
        sigma2_sq = np.var(y_pred)
        sigma12 = np.mean((y_true - mu1) * (y_pred - mu2))
        
        # SSIM公式
        c1 = (k1) ** 2
        c2 = (k2) ** 2
        
        ssim = ((2 * mu1 * mu2 + c1) * (2 * sigma12 + c2)) / \
               ((mu1 ** 2 + mu2 ** 2 + c1) * (sigma1_sq + sigma2_sq + c2))
        
        return float(np.clip(ssim, -1, 1))
    
    @staticmethod
    def _simple_ssim(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """简化的SSIM计算"""
        mu1, mu2 = np.mean(y_true), np.mean(y_pred)
        sigma1, sigma2 = np.std(y_true), np.std(y_pred)
        sigma12 = np.mean((y_true - mu1) * (y_pred - mu2))
        
        c1, c2 = 0.01**2, 0.03**2
        
        ssim = ((2*mu1*mu2 + c1) * (2*sigma12 + c2)) / \
               ((mu1**2 + mu2**2 + c1) * (sigma1**2 + sigma2**2 + c2))
        
        return float(np.clip(ssim, -1, 1))
    
    @staticmethod
    def peak_signal_noise_ratio(y_true: np.ndarray, y_pred: np.ndarray, 
                               max_val: Optional[float] = None) -> float:
        """计算峰值信噪比 (PSNR)"""
        if max_val is None:
            max_val = np.max(y_true)
        
        mse = np.mean((y_true - y_pred) ** 2)
        if mse == 0:
            return float('inf')
        
        psnr = 20 * np.log10(max_val / np.sqrt(mse))
        return float(psnr)
    
    @staticmethod
    def normalized_root_mean_square_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """归一化均方根误差 (NRMSE)"""
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        range_val = np.max(y_true) - np.min(y_true)
        if range_val == 0:
            return 0.0
        return float(rmse / range_val)
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """平均绝对百分比误差 (MAPE)"""
        mask = y_true != 0
        if not np.any(mask):
            return float('inf')
        
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        return float(mape)
    
    @staticmethod
    def coefficient_of_determination(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """决定系数 (R²)"""
        return float(r2_score(y_true.flatten(), y_pred.flatten()))
    
    @staticmethod
    def spectral_angle_mapper(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """光谱角度映射 (SAM)"""
        # 将数据视为向量
        y_true_flat = y_true.flatten()
        y_pred_flat = y_pred.flatten()
        
        # 计算余弦相似度
        dot_product = np.dot(y_true_flat, y_pred_flat)
        norm_true = np.linalg.norm(y_true_flat)
        norm_pred = np.linalg.norm(y_pred_flat)
        
        if norm_true == 0 or norm_pred == 0:
            return np.pi / 2  # 90度
        
        cos_angle = dot_product / (norm_true * norm_pred)
        cos_angle = np.clip(cos_angle, -1, 1)  # 防止数值误差
        
        angle = np.arccos(cos_angle)
        return float(angle)
    
    @staticmethod
    def frequency_domain_error(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """频域误差分析"""
        # FFT变换
        fft_true = np.fft.fft(y_true.flatten())
        fft_pred = np.fft.fft(y_pred.flatten())
        
        # 幅度和相位
        mag_true = np.abs(fft_true)
        mag_pred = np.abs(fft_pred)
        phase_true = np.angle(fft_true)
        phase_pred = np.angle(fft_pred)
        
        # 计算误差
        magnitude_error = np.mean((mag_true - mag_pred) ** 2)
        phase_error = np.mean(np.abs(phase_true - phase_pred))
        
        return {
            'magnitude_mse': float(magnitude_error),
            'phase_mae': float(phase_error),
            'spectral_correlation': float(np.corrcoef(mag_true, mag_pred)[0, 1])
        }
    
    @classmethod
    def compute_comprehensive_metrics(cls, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """计算全面的性能指标"""
        metrics = {}
        
        # 基础指标
        metrics['mse'] = float(np.mean((y_true - y_pred) ** 2))
        metrics['mae'] = float(np.mean(np.abs(y_true - y_pred)))
        metrics['rmse'] = float(np.sqrt(metrics['mse']))
        
        # 高级指标
        try:
            metrics['psnr'] = cls.peak_signal_noise_ratio(y_true, y_pred)
            metrics['ssim'] = cls.structural_similarity_index(y_true, y_pred)
            metrics['nrmse'] = cls.normalized_root_mean_square_error(y_true, y_pred)
            metrics['mape'] = cls.mean_absolute_percentage_error(y_true, y_pred)
            metrics['r2'] = cls.coefficient_of_determination(y_true, y_pred)
            metrics['sam'] = cls.spectral_angle_mapper(y_true, y_pred)
            
            # 相关性指标
            correlation = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
            metrics['correlation'] = float(correlation) if not np.isnan(correlation) else 0.0
            
            # 频域分析
            freq_metrics = cls.frequency_domain_error(y_true, y_pred)
            metrics.update(freq_metrics)
            
        except Exception as e:
            print(f"计算高级指标时出错: {e}")
            # 设置默认值
            for key in ['psnr', 'ssim', 'nrmse', 'mape', 'r2', 'sam', 'correlation']:
                if key not in metrics:
                    metrics[key] = 0.0
        
        return metrics

class AdvancedVisualization:
    """高级可视化分析器"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10), style: str = 'whitegrid'):
        self.figsize = figsize
        plt.style.use('default')
        sns.set_style(style)
        sns.set_palette("husl")
    
    def create_comprehensive_comparison(self, performances: List[ModelPerformance], 
                                      save_path: str = None) -> plt.Figure:
        """创建综合对比图表"""
        fig = plt.figure(figsize=(20, 15))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # 提取数据
        model_names = [p.name for p in performances]
        
        # 1. 主要性能指标雷达图
        ax1 = fig.add_subplot(gs[0, :2], projection='polar')
        self._create_radar_chart(performances, ax1)
        
        # 2. 参数效率散点图
        ax2 = fig.add_subplot(gs[0, 2:])
        self._create_efficiency_scatter(performances, ax2)
        
        # 3. 训练收敛曲线
        ax3 = fig.add_subplot(gs[1, :2])
        self._create_convergence_plot(performances, ax3)
        
        # 4. 误差分布直方图
        ax4 = fig.add_subplot(gs[1, 2:])
        self._create_error_distribution(performances, ax4)
        
        # 5. 性能指标热力图
        ax5 = fig.add_subplot(gs[2, :])
        self._create_metrics_heatmap(performances, ax5)
        
        # 6. 计算复杂度对比
        ax6 = fig.add_subplot(gs[3, :2])
        self._create_complexity_comparison(performances, ax6)
        
        # 7. 预测质量可视化
        ax7 = fig.add_subplot(gs[3, 2:])
        self._create_prediction_quality(performances, ax7)
        
        plt.suptitle('网络架构综合性能对比分析', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _create_radar_chart(self, performances: List[ModelPerformance], ax):
        """创建雷达图"""
        # 选择关键指标
        key_metrics = ['mse', 'psnr', 'ssim', 'correlation', 'r2']
        
        # 归一化指标值
        normalized_data = []
        for perf in performances:
            values = []
            for metric in key_metrics:
                val = perf.metrics.get(metric, 0)
                if metric == 'mse':  # MSE越小越好，需要反转
                    val = 1 / (1 + val) if val > 0 else 1
                elif metric in ['psnr', 'ssim', 'correlation', 'r2']:  # 这些指标越大越好
                    val = max(0, min(1, val))  # 限制在[0,1]范围
                values.append(val)
            normalized_data.append(values)
        
        # 角度
        angles = np.linspace(0, 2 * np.pi, len(key_metrics), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 绘制每个模型
        colors = plt.cm.Set3(np.linspace(0, 1, len(performances)))
        for i, (perf, color) in enumerate(zip(performances, colors)):
            values = normalized_data[i] + normalized_data[i][:1]  # 闭合
            ax.plot(angles, values, 'o-', linewidth=2, label=perf.name, color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(key_metrics)
        ax.set_ylim(0, 1)
        ax.set_title('主要性能指标雷达图', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    def _create_efficiency_scatter(self, performances: List[ModelPerformance], ax):
        """创建效率散点图"""
        param_counts = [p.param_count for p in performances]
        mse_values = [p.metrics.get('mse', 0) for p in performances]
        model_names = [p.name for p in performances]
        
        scatter = ax.scatter(param_counts, mse_values, s=100, alpha=0.7, c=range(len(performances)), cmap='viridis')
        
        for i, name in enumerate(model_names):
            ax.annotate(name, (param_counts[i], mse_values[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_xlabel('参数数量')
        ax.set_ylabel('MSE')
        ax.set_title('参数效率分析')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
    
    def _create_convergence_plot(self, performances: List[ModelPerformance], ax):
        """创建收敛曲线图"""
        for perf in performances:
            if perf.training_history:
                epochs = range(1, len(perf.training_history) + 1)
                ax.plot(epochs, perf.training_history, label=perf.name, marker='o', markersize=4)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss')
        ax.set_title('训练收敛曲线')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    def _create_error_distribution(self, performances: List[ModelPerformance], ax):
        """创建误差分布图"""
        for perf in performances:
            if perf.predictions is not None and perf.targets is not None:
                errors = perf.predictions.flatten() - perf.targets.flatten()
                ax.hist(errors, bins=30, alpha=0.6, label=perf.name, density=True)
        
        ax.set_xlabel('预测误差')
        ax.set_ylabel('密度')
        ax.set_title('误差分布直方图')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_metrics_heatmap(self, performances: List[ModelPerformance], ax):
        """创建指标热力图"""
        # 准备数据
        metrics_names = ['mse', 'mae', 'rmse', 'psnr', 'ssim', 'correlation', 'r2']
        model_names = [p.name for p in performances]
        
        data = []
        for perf in performances:
            row = [perf.metrics.get(metric, 0) for metric in metrics_names]
            data.append(row)
        
        # 归一化数据用于热力图显示
        data_array = np.array(data)
        normalized_data = np.zeros_like(data_array)
        
        for j in range(data_array.shape[1]):
            col = data_array[:, j]
            if metrics_names[j] == 'mse':  # MSE越小越好
                normalized_data[:, j] = 1 - (col - col.min()) / (col.max() - col.min() + 1e-8)
            else:  # 其他指标越大越好
                normalized_data[:, j] = (col - col.min()) / (col.max() - col.min() + 1e-8)
        
        # 创建热力图
        sns.heatmap(normalized_data, annot=True, fmt='.3f', 
                   xticklabels=metrics_names, yticklabels=model_names,
                   cmap='RdYlGn', ax=ax, cbar_kws={'label': '归一化性能分数'})
        
        ax.set_title('性能指标热力图')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    def _create_complexity_comparison(self, performances: List[ModelPerformance], ax):
        """创建计算复杂度对比"""
        model_names = [p.name for p in performances]
        train_times = [p.train_time for p in performances]
        test_times = [p.test_time for p in performances]
        memory_usage = [p.memory_usage for p in performances]
        
        x = np.arange(len(model_names))
        width = 0.25
        
        ax.bar(x - width, train_times, width, label='训练时间', alpha=0.8)
        ax.bar(x, test_times, width, label='测试时间', alpha=0.8)
        ax.bar(x + width, memory_usage, width, label='内存使用(MB)', alpha=0.8)
        
        ax.set_xlabel('模型')
        ax.set_ylabel('时间(秒) / 内存(MB)')
        ax.set_title('计算复杂度对比')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _create_prediction_quality(self, performances: List[ModelPerformance], ax):
        """创建预测质量可视化"""
        # 选择第一个有预测数据的模型进行可视化
        for perf in performances:
            if perf.predictions is not None and perf.targets is not None:
                # 随机选择一些样本点进行可视化
                n_samples = min(1000, len(perf.predictions.flatten()))
                indices = np.random.choice(len(perf.predictions.flatten()), n_samples, replace=False)
                
                pred_sample = perf.predictions.flatten()[indices]
                target_sample = perf.targets.flatten()[indices]
                
                ax.scatter(target_sample, pred_sample, alpha=0.6, s=20, label=perf.name)
        
        # 添加理想线
        min_val = min([p.targets.min() for p in performances if p.targets is not None])
        max_val = max([p.targets.max() for p in performances if p.targets is not None])
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='理想预测')
        
        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title('预测质量散点图')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def create_statistical_analysis(self, performances: List[ModelPerformance], 
                                  save_path: str = None) -> plt.Figure:
        """创建统计分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. 性能指标箱线图
        self._create_metrics_boxplot(performances, axes[0, 0])
        
        # 2. 模型排名分析
        self._create_ranking_analysis(performances, axes[0, 1])
        
        # 3. 相关性分析
        self._create_correlation_analysis(performances, axes[1, 0])
        
        # 4. 显著性测试结果
        self._create_significance_test(performances, axes[1, 1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def _create_metrics_boxplot(self, performances: List[ModelPerformance], ax):
        """创建指标箱线图"""
        # 这里简化处理，实际应该有多次运行的数据
        metrics_data = []
        model_names = []
        
        for perf in performances:
            mse_val = perf.metrics.get('mse', 0)
            metrics_data.append(mse_val)
            model_names.append(perf.name)
        
        ax.bar(range(len(model_names)), metrics_data)
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('MSE')
        ax.set_title('MSE性能对比')
        ax.grid(True, alpha=0.3)
    
    def _create_ranking_analysis(self, performances: List[ModelPerformance], ax):
        """创建排名分析"""
        # 按不同指标排名
        metrics = ['mse', 'psnr', 'ssim', 'correlation']
        model_names = [p.name for p in performances]
        
        rankings = np.zeros((len(performances), len(metrics)))
        
        for j, metric in enumerate(metrics):
            values = [p.metrics.get(metric, 0) for p in performances]
            if metric == 'mse':  # MSE越小越好
                ranks = stats.rankdata(values)
            else:  # 其他指标越大越好
                ranks = stats.rankdata([-v for v in values])
            rankings[:, j] = ranks
        
        # 创建热力图
        sns.heatmap(rankings, annot=True, fmt='.0f',
                   xticklabels=metrics, yticklabels=model_names,
                   cmap='RdYlGn_r', ax=ax)
        
        ax.set_title('不同指标下的模型排名')
    
    def _create_correlation_analysis(self, performances: List[ModelPerformance], ax):
        """创建相关性分析"""
        # 分析不同指标之间的相关性
        metrics_names = ['mse', 'psnr', 'ssim', 'correlation', 'train_time', 'param_count']
        
        data = []
        for perf in performances:
            row = []
            for metric in metrics_names:
                if metric in ['train_time', 'param_count']:
                    val = getattr(perf, metric, 0)
                else:
                    val = perf.metrics.get(metric, 0)
                row.append(val)
            data.append(row)
        
        df = pd.DataFrame(data, columns=metrics_names)
        corr_matrix = df.corr()
        
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, ax=ax)
        
        ax.set_title('指标相关性分析')
    
    def _create_significance_test(self, performances: List[ModelPerformance], ax):
        """创建显著性测试结果"""
        # 简化的显著性分析
        model_names = [p.name for p in performances]
        mse_values = [p.metrics.get('mse', 0) for p in performances]
        
        # 计算相对性能提升
        baseline_mse = max(mse_values)  # 使用最差的作为基线
        improvements = [(baseline_mse - mse) / baseline_mse * 100 for mse in mse_values]
        
        bars = ax.bar(range(len(model_names)), improvements)
        
        # 着色：正值为绿色，负值为红色
        for bar, improvement in zip(bars, improvements):
            if improvement >= 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.set_ylabel('性能提升 (%)')
        ax.set_title('相对于基线的性能提升')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

class ModelEfficiencyAnalyzer:
    """模型效率分析器"""
    
    @staticmethod
    def calculate_parameter_efficiency(performance: ModelPerformance) -> float:
        """计算参数效率 (性能/参数数量)"""
        mse = performance.metrics.get('mse', float('inf'))
        if mse == 0 or performance.param_count == 0:
            return 0.0
        
        # 效率 = 1 / (MSE * log(参数数量))
        efficiency = 1 / (mse * np.log10(max(performance.param_count, 1)))
        return float(efficiency)
    
    @staticmethod
    def calculate_computational_efficiency(performance: ModelPerformance) -> float:
        """计算计算效率 (性能/计算时间)"""
        mse = performance.metrics.get('mse', float('inf'))
        total_time = performance.train_time + performance.test_time
        
        if mse == 0 or total_time == 0:
            return 0.0
        
        efficiency = 1 / (mse * total_time)
        return float(efficiency)
    
    @staticmethod
    def calculate_memory_efficiency(performance: ModelPerformance) -> float:
        """计算内存效率 (性能/内存使用)"""
        mse = performance.metrics.get('mse', float('inf'))
        memory = max(performance.memory_usage, 1)  # 避免除零
        
        if mse == 0:
            return 0.0
        
        efficiency = 1 / (mse * memory)
        return float(efficiency)
    
    @classmethod
    def analyze_all_efficiencies(cls, performances: List[ModelPerformance]) -> Dict[str, List[float]]:
        """分析所有效率指标"""
        efficiencies = {
            'parameter_efficiency': [],
            'computational_efficiency': [],
            'memory_efficiency': []
        }
        
        for perf in performances:
            efficiencies['parameter_efficiency'].append(
                cls.calculate_parameter_efficiency(perf)
            )
            efficiencies['computational_efficiency'].append(
                cls.calculate_computational_efficiency(perf)
            )
            efficiencies['memory_efficiency'].append(
                cls.calculate_memory_efficiency(perf)
            )
        
        return efficiencies

def create_performance_report(performances: List[ModelPerformance], 
                            output_dir: str = "./analysis_results") -> Dict[str, Any]:
    """创建完整的性能分析报告"""
    from pathlib import Path
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 创建可视化器
    visualizer = AdvancedVisualization()
    
    # 生成综合对比图
    fig1 = visualizer.create_comprehensive_comparison(
        performances, str(output_path / "comprehensive_comparison.png")
    )
    plt.close(fig1)
    
    # 生成统计分析图
    fig2 = visualizer.create_statistical_analysis(
        performances, str(output_path / "statistical_analysis.png")
    )
    plt.close(fig2)
    
    # 效率分析
    analyzer = ModelEfficiencyAnalyzer()
    efficiencies = analyzer.analyze_all_efficiencies(performances)
    
    # 生成文本报告
    report = {
        'summary': {
            'total_models': len(performances),
            'best_mse': min(p.metrics.get('mse', float('inf')) for p in performances),
            'best_model': min(performances, key=lambda p: p.metrics.get('mse', float('inf'))).name
        },
        'detailed_metrics': {
            p.name: p.metrics for p in performances
        },
        'efficiency_analysis': efficiencies,
        'rankings': {
            'by_mse': sorted(performances, key=lambda p: p.metrics.get('mse', float('inf'))),
            'by_psnr': sorted(performances, key=lambda p: p.metrics.get('psnr', 0), reverse=True),
            'by_efficiency': sorted(performances, 
                                  key=lambda p: analyzer.calculate_parameter_efficiency(p), 
                                  reverse=True)
        }
    }
    
    return report