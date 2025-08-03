#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SVD损失函数版本对比分析脚本

功能:
1. 对比原版和增强版SVD损失函数的性能
2. 分析计算时间、内存使用和数值稳定性
3. 测试不同配置下的表现差异
4. 生成详细的对比报告

作者: AI Assistant
日期: 2025
"""

import sys
import torch
import numpy as np
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

# 添加路径
project_root = Path(__file__).parent.parent
modify_multi_attention_path = project_root / 'modify_multi_attention'
sys.path.insert(0, str(modify_multi_attention_path))

try:
    from utils.svd10_loss import TotalLossWithSVD, get_svd_modes, svd_topk_losses
    from utils.enhanced_svd_loss import (
        EnhancedTotalLossWithSVD, create_enhanced_svd_loss,
        get_global_svd_stats, reset_global_svd_stats
    )
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    print("请确保路径配置正确")
    sys.exit(1)

class SVDComparator:
    """SVD损失函数对比器"""
    
    def __init__(self, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.results = {
            'original': {},
            'enhanced': {},
            'comparison': {}
        }
    
    def generate_test_data(self, batch_size: int, height: int, width: int, 
                          data_type: str = 'normal') -> Tuple[torch.Tensor, torch.Tensor]:
        """生成测试数据"""
        if data_type == 'normal':
            pred = torch.randn(batch_size, height, width, device=self.device, requires_grad=True)
            target = torch.randn(batch_size, height, width, device=self.device)
        elif data_type == 'extreme':
            pred = torch.randn(batch_size, height, width, device=self.device, requires_grad=True) * 1000
            target = torch.randn(batch_size, height, width, device=self.device) * 1000
        elif data_type == 'small':
            pred = torch.randn(batch_size, height, width, device=self.device, requires_grad=True) * 0.001
            target = torch.randn(batch_size, height, width, device=self.device) * 0.001
        elif data_type == 'mixed_precision':
            pred = torch.randn(batch_size, height, width, device=self.device, dtype=torch.float16, requires_grad=True)
            target = torch.randn(batch_size, height, width, device=self.device, dtype=torch.float16)
        else:
            raise ValueError(f"未知的数据类型: {data_type}")
        
        return pred, target
    
    def benchmark_original_svd(self, test_cases: List[Dict]) -> Dict:
        """基准测试原版SVD损失函数"""
        print("\n=== 测试原版SVD损失函数 ===")
        
        results = {
            'computation_times': [],
            'memory_usage': [],
            'loss_values': [],
            'gradient_norms': [],
            'errors': [],
            'test_cases': []
        }
        
        for i, case in enumerate(test_cases):
            print(f"\n--- 测试用例 {i+1}: {case['name']} ---")
            
            try:
                # 创建原版损失函数
                criterion = TotalLossWithSVD(
                    base_weight=case.get('base_weight', 0.5),
                    svd_weights=case.get('svd_weights', [0.05] * 10),
                    topk=case.get('topk', 10)
                )
                
                # 生成测试数据
                pred, target = self.generate_test_data(
                    case['batch_size'], case['height'], case['width'], case.get('data_type', 'normal')
                )
                
                # 记录内存使用
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    initial_memory = torch.cuda.memory_allocated()
                
                # 计算时间
                start_time = time.time()
                
                # 前向传播
                loss = criterion(pred, target)
                
                # 反向传播
                loss.backward()
                
                end_time = time.time()
                computation_time = (end_time - start_time) * 1000  # ms
                
                # 记录结果
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() - initial_memory
                    memory_mb = peak_memory / 1024 / 1024
                else:
                    memory_mb = 0
                
                gradient_norm = pred.grad.norm().item() if pred.grad is not None else 0
                
                results['computation_times'].append(computation_time)
                results['memory_usage'].append(memory_mb)
                results['loss_values'].append(loss.item())
                results['gradient_norms'].append(gradient_norm)
                results['errors'].append(None)
                results['test_cases'].append(case['name'])
                
                print(f"   ✅ 成功 - 时间: {computation_time:.2f}ms, 内存: {memory_mb:.2f}MB, 损失: {loss.item():.6f}")
                
            except Exception as e:
                print(f"   ❌ 失败 - {str(e)}")
                results['computation_times'].append(float('inf'))
                results['memory_usage'].append(float('inf'))
                results['loss_values'].append(float('nan'))
                results['gradient_norms'].append(float('nan'))
                results['errors'].append(str(e))
                results['test_cases'].append(case['name'])
        
        return results
    
    def benchmark_enhanced_svd(self, test_cases: List[Dict]) -> Dict:
        """基准测试增强版SVD损失函数"""
        print("\n=== 测试增强版SVD损失函数 ===")
        
        results = {
            'computation_times': [],
            'memory_usage': [],
            'loss_values': [],
            'gradient_norms': [],
            'errors': [],
            'test_cases': [],
            'svd_stats': [],
            'weight_adaptations': []
        }
        
        for i, case in enumerate(test_cases):
            print(f"\n--- 测试用例 {i+1}: {case['name']} ---")
            
            try:
                # 重置监控器
                reset_global_svd_stats()
                
                # 创建增强版损失函数
                criterion = create_enhanced_svd_loss(
                    base_weight=case.get('base_weight', 0.5),
                    svd_weights=case.get('svd_weights', [0.05] * 10),
                    topk=case.get('topk', 10),
                    mixed_precision=case.get('mixed_precision', True),
                    adaptive_weights=case.get('adaptive_weights', True),
                    monitoring=True,
                    fallback_level=case.get('fallback_level', 2)
                )
                
                # 生成测试数据
                pred, target = self.generate_test_data(
                    case['batch_size'], case['height'], case['width'], case.get('data_type', 'normal')
                )
                
                # 记录内存使用
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                    initial_memory = torch.cuda.memory_allocated()
                
                # 计算时间
                start_time = time.time()
                
                # 前向传播
                loss = criterion(pred, target)
                
                # 反向传播
                loss.backward()
                
                end_time = time.time()
                computation_time = (end_time - start_time) * 1000  # ms
                
                # 记录结果
                if torch.cuda.is_available():
                    peak_memory = torch.cuda.max_memory_allocated() - initial_memory
                    memory_mb = peak_memory / 1024 / 1024
                else:
                    memory_mb = 0
                
                gradient_norm = pred.grad.norm().item() if pred.grad is not None else 0
                
                # 获取SVD统计信息
                svd_stats = criterion.get_performance_stats()
                weight_info = criterion.get_weight_info()
                
                results['computation_times'].append(computation_time)
                results['memory_usage'].append(memory_mb)
                results['loss_values'].append(loss.item())
                results['gradient_norms'].append(gradient_norm)
                results['errors'].append(None)
                results['test_cases'].append(case['name'])
                results['svd_stats'].append(svd_stats)
                results['weight_adaptations'].append(len(weight_info.get('adaptation_history', [])))
                
                print(f"   ✅ 成功 - 时间: {computation_time:.2f}ms, 内存: {memory_mb:.2f}MB, 损失: {loss.item():.6f}")
                print(f"      SVD时间: {svd_stats.get('avg_svd_time_ms', 0):.2f}ms, "
                      f"Fallback: {len(svd_stats.get('fallback_usage', {}))}, "
                      f"NaN率: {svd_stats.get('nan_inf_rate', 0):.2%}")
                
            except Exception as e:
                print(f"   ❌ 失败 - {str(e)}")
                results['computation_times'].append(float('inf'))
                results['memory_usage'].append(float('inf'))
                results['loss_values'].append(float('nan'))
                results['gradient_norms'].append(float('nan'))
                results['errors'].append(str(e))
                results['test_cases'].append(case['name'])
                results['svd_stats'].append({})
                results['weight_adaptations'].append(0)
        
        return results
    
    def analyze_results(self, original_results: Dict, enhanced_results: Dict) -> Dict:
        """分析对比结果"""
        print("\n=== 结果分析 ===")
        
        analysis = {
            'performance_improvement': {},
            'stability_improvement': {},
            'feature_comparison': {},
            'recommendations': []
        }
        
        # 性能对比
        orig_times = [t for t in original_results['computation_times'] if t != float('inf')]
        enh_times = [t for t in enhanced_results['computation_times'] if t != float('inf')]
        
        if orig_times and enh_times:
            avg_orig_time = np.mean(orig_times)
            avg_enh_time = np.mean(enh_times)
            time_improvement = (avg_orig_time - avg_enh_time) / avg_orig_time * 100
            
            analysis['performance_improvement']['computation_time'] = {
                'original_avg_ms': avg_orig_time,
                'enhanced_avg_ms': avg_enh_time,
                'improvement_percent': time_improvement
            }
            
            print(f"计算时间对比:")
            print(f"  原版平均: {avg_orig_time:.2f}ms")
            print(f"  增强版平均: {avg_enh_time:.2f}ms")
            print(f"  改进: {time_improvement:+.1f}%")
        
        # 内存使用对比
        orig_memory = [m for m in original_results['memory_usage'] if m != float('inf')]
        enh_memory = [m for m in enhanced_results['memory_usage'] if m != float('inf')]
        
        if orig_memory and enh_memory:
            avg_orig_memory = np.mean(orig_memory)
            avg_enh_memory = np.mean(enh_memory)
            memory_improvement = (avg_orig_memory - avg_enh_memory) / avg_orig_memory * 100
            
            analysis['performance_improvement']['memory_usage'] = {
                'original_avg_mb': avg_orig_memory,
                'enhanced_avg_mb': avg_enh_memory,
                'improvement_percent': memory_improvement
            }
            
            print(f"\n内存使用对比:")
            print(f"  原版平均: {avg_orig_memory:.2f}MB")
            print(f"  增强版平均: {avg_enh_memory:.2f}MB")
            print(f"  改进: {memory_improvement:+.1f}%")
        
        # 稳定性对比
        orig_errors = sum(1 for e in original_results['errors'] if e is not None)
        enh_errors = sum(1 for e in enhanced_results['errors'] if e is not None)
        total_cases = len(original_results['errors'])
        
        analysis['stability_improvement'] = {
            'original_error_rate': orig_errors / total_cases,
            'enhanced_error_rate': enh_errors / total_cases,
            'error_reduction': (orig_errors - enh_errors) / max(1, orig_errors) * 100
        }
        
        print(f"\n稳定性对比:")
        print(f"  原版错误率: {orig_errors}/{total_cases} ({orig_errors/total_cases:.1%})")
        print(f"  增强版错误率: {enh_errors}/{total_cases} ({enh_errors/total_cases:.1%})")
        if orig_errors > 0:
            error_reduction = (orig_errors - enh_errors) / orig_errors * 100
            print(f"  错误减少: {error_reduction:.1f}%")
        
        # 功能对比
        analysis['feature_comparison'] = {
            'mixed_precision_support': {'original': False, 'enhanced': True},
            'adaptive_weights': {'original': False, 'enhanced': True},
            'performance_monitoring': {'original': False, 'enhanced': True},
            'advanced_error_handling': {'original': False, 'enhanced': True},
            'configurable_fallback': {'original': False, 'enhanced': True}
        }
        
        # 生成建议
        recommendations = []
        
        if time_improvement > 10:
            recommendations.append("增强版在计算性能上有显著提升，建议使用")
        elif time_improvement < -10:
            recommendations.append("增强版计算开销较大，在性能敏感场景下需谨慎使用")
        
        if orig_errors > enh_errors:
            recommendations.append("增强版在数值稳定性上有明显改进，推荐用于复杂数据")
        
        if any(enhanced_results['svd_stats']):
            recommendations.append("增强版提供详细的性能监控，有助于调试和优化")
        
        if any(enhanced_results['weight_adaptations']):
            recommendations.append("增强版支持自适应权重，可以提高训练效果")
        
        analysis['recommendations'] = recommendations
        
        print(f"\n建议:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return analysis
    
    def generate_visualization(self, original_results: Dict, enhanced_results: Dict, save_path: str = None):
        """生成可视化对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SVD损失函数版本对比', fontsize=16, fontweight='bold')
        
        test_cases = original_results['test_cases']
        x_pos = np.arange(len(test_cases))
        
        # 计算时间对比
        ax1 = axes[0, 0]
        orig_times = [t if t != float('inf') else 0 for t in original_results['computation_times']]
        enh_times = [t if t != float('inf') else 0 for t in enhanced_results['computation_times']]
        
        width = 0.35
        ax1.bar(x_pos - width/2, orig_times, width, label='原版', alpha=0.8, color='skyblue')
        ax1.bar(x_pos + width/2, enh_times, width, label='增强版', alpha=0.8, color='lightcoral')
        ax1.set_xlabel('测试用例')
        ax1.set_ylabel('计算时间 (ms)')
        ax1.set_title('计算时间对比')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([case[:10] + '...' if len(case) > 10 else case for case in test_cases], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 内存使用对比
        ax2 = axes[0, 1]
        orig_memory = [m if m != float('inf') else 0 for m in original_results['memory_usage']]
        enh_memory = [m if m != float('inf') else 0 for m in enhanced_results['memory_usage']]
        
        ax2.bar(x_pos - width/2, orig_memory, width, label='原版', alpha=0.8, color='skyblue')
        ax2.bar(x_pos + width/2, enh_memory, width, label='增强版', alpha=0.8, color='lightcoral')
        ax2.set_xlabel('测试用例')
        ax2.set_ylabel('内存使用 (MB)')
        ax2.set_title('内存使用对比')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([case[:10] + '...' if len(case) > 10 else case for case in test_cases], rotation=45)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 错误率对比
        ax3 = axes[1, 0]
        orig_error_rate = [1 if e is not None else 0 for e in original_results['errors']]
        enh_error_rate = [1 if e is not None else 0 for e in enhanced_results['errors']]
        
        ax3.bar(x_pos - width/2, orig_error_rate, width, label='原版', alpha=0.8, color='skyblue')
        ax3.bar(x_pos + width/2, enh_error_rate, width, label='增强版', alpha=0.8, color='lightcoral')
        ax3.set_xlabel('测试用例')
        ax3.set_ylabel('错误发生 (0/1)')
        ax3.set_title('错误率对比')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels([case[:10] + '...' if len(case) > 10 else case for case in test_cases], rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 功能特性对比
        ax4 = axes[1, 1]
        features = ['混合精度', '自适应权重', '性能监控', '高级错误处理', '可配置Fallback']
        original_support = [0, 0, 0, 0, 0]  # 原版都不支持
        enhanced_support = [1, 1, 1, 1, 1]  # 增强版都支持
        
        y_pos = np.arange(len(features))
        ax4.barh(y_pos - 0.2, original_support, 0.4, label='原版', alpha=0.8, color='skyblue')
        ax4.barh(y_pos + 0.2, enhanced_support, 0.4, label='增强版', alpha=0.8, color='lightcoral')
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(features)
        ax4.set_xlabel('支持程度')
        ax4.set_title('功能特性对比')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\n📊 可视化图表已保存: {save_path}")
        
        plt.show()
    
    def save_detailed_report(self, original_results: Dict, enhanced_results: Dict, 
                           analysis: Dict, save_path: str):
        """保存详细报告"""
        report = {
            'metadata': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'device': str(self.device),
                'pytorch_version': torch.__version__,
                'cuda_available': torch.cuda.is_available()
            },
            'original_results': original_results,
            'enhanced_results': enhanced_results,
            'analysis': analysis
        }
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        
        print(f"\n📄 详细报告已保存: {save_path}")
    
    def run_comparison(self, test_cases: List[Dict] = None, save_dir: str = './results'):
        """运行完整对比测试"""
        if test_cases is None:
            test_cases = self.get_default_test_cases()
        
        # 创建保存目录
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        print(f"🔍 开始SVD损失函数版本对比测试")
        print(f"设备: {self.device}")
        print(f"测试用例数量: {len(test_cases)}")
        
        # 运行测试
        original_results = self.benchmark_original_svd(test_cases)
        enhanced_results = self.benchmark_enhanced_svd(test_cases)
        
        # 分析结果
        analysis = self.analyze_results(original_results, enhanced_results)
        
        # 生成可视化
        viz_path = save_dir / 'svd_comparison_visualization.png'
        self.generate_visualization(original_results, enhanced_results, str(viz_path))
        
        # 保存详细报告
        report_path = save_dir / 'svd_comparison_report.json'
        self.save_detailed_report(original_results, enhanced_results, analysis, str(report_path))
        
        # 保存结果到实例
        self.results['original'] = original_results
        self.results['enhanced'] = enhanced_results
        self.results['comparison'] = analysis
        
        print(f"\n🎉 对比测试完成！")
        print(f"结果保存在: {save_dir}")
        
        return analysis
    
    def get_default_test_cases(self) -> List[Dict]:
        """获取默认测试用例"""
        return [
            {
                'name': '小批次正常数据',
                'batch_size': 2,
                'height': 16,
                'width': 16,
                'data_type': 'normal'
            },
            {
                'name': '大批次正常数据',
                'batch_size': 8,
                'height': 32,
                'width': 32,
                'data_type': 'normal'
            },
            {
                'name': '极值数据',
                'batch_size': 4,
                'height': 24,
                'width': 24,
                'data_type': 'extreme'
            },
            {
                'name': '小数值数据',
                'batch_size': 4,
                'height': 24,
                'width': 24,
                'data_type': 'small'
            },
            {
                'name': '高分辨率数据',
                'batch_size': 2,
                'height': 64,
                'width': 64,
                'data_type': 'normal'
            }
        ]
        
        # 如果支持CUDA，添加混合精度测试
        if torch.cuda.is_available():
            test_cases.append({
                'name': '混合精度数据',
                'batch_size': 4,
                'height': 32,
                'width': 32,
                'data_type': 'mixed_precision'
            })
        
        return test_cases

def main():
    """主函数"""
    print("🔍 SVD损失函数版本对比分析")
    
    # 创建对比器
    comparator = SVDComparator()
    
    # 运行对比测试
    analysis = comparator.run_comparison()
    
    # 显示总结
    print("\n=== 对比总结 ===")
    perf = analysis.get('performance_improvement', {})
    stab = analysis.get('stability_improvement', {})
    
    if 'computation_time' in perf:
        time_imp = perf['computation_time']['improvement_percent']
        print(f"⏱️ 计算性能: {time_imp:+.1f}%")
    
    if 'memory_usage' in perf:
        mem_imp = perf['memory_usage']['improvement_percent']
        print(f"💾 内存使用: {mem_imp:+.1f}%")
    
    if 'error_reduction' in stab:
        err_red = stab['error_reduction']
        print(f"🛡️ 错误减少: {err_red:.1f}%")
    
    print("\n📋 主要改进:")
    for rec in analysis.get('recommendations', []):
        print(f"  • {rec}")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 测试被用户中断")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()