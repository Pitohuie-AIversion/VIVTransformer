"""模型复杂度分析工具"""

import torch
import torch.nn as nn
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from optimization_utils import get_model_complexity, MemoryProfiler

# 导入所有模型创建函数
from enhanced_fno import create_enhanced_fno1d, create_enhanced_fno2d
from enhanced_mlp import create_enhanced_mlp1d, create_enhanced_mlp2d
from enhanced_pinn import create_enhanced_pinn1d, create_enhanced_pinn2d
from enhanced_unet import create_enhanced_unet1d, create_enhanced_unet2d, create_enhanced_unet3d

class ModelComplexityAnalyzer:
    """模型复杂度分析器"""
    
    def __init__(self, device='auto'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"分析设备: {self.device}")
        
        # 分析结果存储
        self.analysis_results = {}
        
        # 模型配置
        self.model_configs = {
            'FNO1d': {
                'creator': create_enhanced_fno1d,
                'input_shape': (1, 32, 1),
                'params': {'input_resolution': 32, 'output_resolution': 128, 'num_channels': 1, 'modes': 16, 'width': 64}
            },
            'FNO2d': {
                'creator': create_enhanced_fno2d,
                'input_shape': (1, 32, 32, 1),
                'params': {'input_resolution': (32, 32), 'output_resolution': (128, 128), 'num_channels': 1, 'modes1': 12, 'modes2': 12, 'width': 20}
            },
            'MLP1d': {
                'creator': create_enhanced_mlp1d,
                'input_shape': (1, 32, 1),
                'params': {'input_resolution': 32, 'output_resolution': 128, 'input_channels': 1, 'hidden_dim': 256, 'num_layers': 6}
            },
            'MLP2d': {
                'creator': create_enhanced_mlp2d,
                'input_shape': (1, 32, 32, 1),
                'params': {'input_resolution': (32, 32), 'output_resolution': (128, 128), 'input_channels': 1, 'hidden_dim': 256, 'num_layers': 6}
            },
            'PINN1d': {
                'creator': create_enhanced_pinn1d,
                'input_shape': (1, 32, 1),
                'params': {'input_resolution': 32, 'output_resolution': 128, 'output_dim': 1, 'hidden_dim': 256, 'num_layers': 6}
            },
            'PINN2d': {
                'creator': create_enhanced_pinn2d,
                'input_shape': (1, 32, 32, 1),
                'params': {'input_resolution': (32, 32), 'output_resolution': (128, 128), 'output_dim': 1, 'hidden_dim': 256, 'num_layers': 6}
            },
            'UNet1d': {
                'creator': create_enhanced_unet1d,
                'input_shape': (1, 32, 1),
                'params': {'input_resolution': 32, 'output_resolution': 128, 'in_channels': 1, 'out_channels': 1, 'init_features': 32}
            },
            'UNet2d': {
                'creator': create_enhanced_unet2d,
                'input_shape': (1, 32, 32, 1),
                'params': {'input_resolution': (32, 32), 'output_resolution': (128, 128), 'in_channels': 1, 'out_channels': 1, 'init_features': 32}
            },
            'UNet3d': {
                'creator': create_enhanced_unet3d,
                'input_shape': (1, 16, 32, 32, 1),
                'params': {'input_resolution': (16, 32, 32), 'output_resolution': (32, 128, 128), 'in_channels': 1, 'out_channels': 1, 'init_features': 16}
            }
        }
    
    def analyze_layer_complexity(self, model: nn.Module) -> Dict[str, Any]:
        """分析每层的复杂度"""
        layer_stats = defaultdict(lambda: {'count': 0, 'params': 0, 'memory': 0})
        
        for name, module in model.named_modules():
            module_type = type(module).__name__
            
            # 计算参数数量
            params = sum(p.numel() for p in module.parameters() if p.requires_grad)
            
            # 估算内存使用（参数 + 缓冲区）
            memory = sum(p.numel() * p.element_size() for p in module.parameters())
            memory += sum(b.numel() * b.element_size() for b in module.buffers())
            
            layer_stats[module_type]['count'] += 1
            layer_stats[module_type]['params'] += params
            layer_stats[module_type]['memory'] += memory
        
        return dict(layer_stats)
    
    def analyze_computational_graph(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """分析计算图复杂度"""
        model.eval()
        
        # 记录每层的计算量
        layer_flops = {}
        layer_memory = {}
        
        def flop_memory_hook(name):
            def hook(module, input, output):
                # 计算FLOPs
                flops = 0
                if isinstance(module, nn.Linear):
                    flops = input[0].numel() * module.out_features
                elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    kernel_size = np.prod(module.kernel_size)
                    output_elements = output.numel()
                    flops = output_elements * kernel_size * module.in_channels
                elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm)):
                    flops = 2 * input[0].numel()
                elif isinstance(module, (nn.ReLU, nn.GELU, nn.Tanh, nn.Sigmoid)):
                    flops = input[0].numel()
                
                layer_flops[name] = flops
                
                # 计算内存使用
                input_memory = sum(inp.numel() * inp.element_size() for inp in input if isinstance(inp, torch.Tensor))
                output_memory = output.numel() * output.element_size() if isinstance(output, torch.Tensor) else 0
                layer_memory[name] = {'input': input_memory, 'output': output_memory}
            
            return hook
        
        # 注册钩子
        hooks = []
        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 只对叶子节点注册
                hooks.append(module.register_forward_hook(flop_memory_hook(name)))
        
        # 前向传播
        with torch.no_grad():
            _ = model(input_tensor)
        
        # 移除钩子
        for hook in hooks:
            hook.remove()
        
        return {
            'layer_flops': layer_flops,
            'layer_memory': layer_memory,
            'total_flops': sum(layer_flops.values()),
            'total_memory': sum(mem['input'] + mem['output'] for mem in layer_memory.values())
        }
    
    def benchmark_scaling(self, model_name: str, input_sizes: List[Tuple[int, ...]]) -> Dict[str, List[float]]:
        """测试模型在不同输入尺寸下的性能"""
        config = self.model_configs[model_name]
        
        times = []
        memories = []
        flops = []
        
        for input_size in input_sizes:
            try:
                # 调整模型参数以适应不同输入尺寸
                adjusted_params = config['params'].copy()
                
                # 根据输入尺寸调整分辨率参数
                if len(input_size) == 3:  # 1D
                    adjusted_params['input_resolution'] = input_size[1]
                elif len(input_size) == 4:  # 2D
                    adjusted_params['input_resolution'] = input_size[1:3]
                elif len(input_size) == 5:  # 3D
                    adjusted_params['input_resolution'] = input_size[1:4]
                
                # 创建模型
                model = config['creator'](**adjusted_params).to(self.device)
                input_tensor = torch.randn(*input_size).to(self.device)
                
                # 测量时间
                model.eval()
                with torch.no_grad():
                    # 预热
                    for _ in range(3):
                        _ = model(input_tensor)
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    start_time = time.time()
                    for _ in range(10):
                        _ = model(input_tensor)
                    
                    if self.device.type == 'cuda':
                        torch.cuda.synchronize()
                    
                    end_time = time.time()
                    avg_time = (end_time - start_time) / 10
                
                # 测量内存
                with MemoryProfiler(self.device) as profiler:
                    with torch.no_grad():
                        _ = model(input_tensor)
                
                # 计算FLOPs
                graph_analysis = self.analyze_computational_graph(model, input_tensor)
                
                times.append(avg_time * 1000)  # 转换为毫秒
                memories.append(profiler.get_peak_memory_mb())
                flops.append(graph_analysis['total_flops'])
                
                # 清理内存
                del model, input_tensor
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"跳过输入尺寸 {input_size}: {e}")
                times.append(float('nan'))
                memories.append(float('nan'))
                flops.append(float('nan'))
        
        return {
            'input_sizes': input_sizes,
            'inference_times_ms': times,
            'peak_memories_mb': memories,
            'flops': flops
        }
    
    def analyze_all_models(self) -> Dict[str, Dict[str, Any]]:
        """分析所有模型"""
        print("开始全面复杂度分析...")
        
        for model_name, config in self.model_configs.items():
            print(f"\n分析模型: {model_name}")
            
            try:
                # 创建模型
                model = config['creator'](**config['params']).to(self.device)
                input_tensor = torch.randn(*config['input_shape']).to(self.device)
                
                # 基础复杂度分析
                complexity = get_model_complexity(model, config['input_shape'], self.device)
                
                # 层级分析
                layer_analysis = self.analyze_layer_complexity(model)
                
                # 计算图分析
                graph_analysis = self.analyze_computational_graph(model, input_tensor)
                
                # 扩展性分析（不同输入尺寸）
                if model_name.endswith('1d'):
                    test_sizes = [(1, 16, 1), (1, 32, 1), (1, 64, 1), (1, 128, 1)]
                elif model_name.endswith('2d'):
                    test_sizes = [(1, 16, 16, 1), (1, 32, 32, 1), (1, 64, 64, 1)]
                elif model_name.endswith('3d'):
                    test_sizes = [(1, 8, 16, 16, 1), (1, 16, 32, 32, 1)]
                else:
                    test_sizes = [config['input_shape']]
                
                scaling_analysis = self.benchmark_scaling(model_name, test_sizes)
                
                # 汇总结果
                self.analysis_results[model_name] = {
                    'basic_complexity': complexity,
                    'layer_analysis': layer_analysis,
                    'graph_analysis': graph_analysis,
                    'scaling_analysis': scaling_analysis,
                    'model_info': {
                        'input_shape': config['input_shape'],
                        'parameters': config['params']
                    }
                }
                
                print(f"✅ {model_name} 分析完成")
                
                # 清理内存
                del model, input_tensor
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"❌ {model_name} 分析失败: {e}")
                self.analysis_results[model_name] = {'error': str(e)}
        
        return self.analysis_results
    
    def generate_comparison_report(self) -> str:
        """生成比较报告"""
        if not self.analysis_results:
            return "没有分析结果可用"
        
        report = []
        report.append("="*80)
        report.append("模型复杂度分析报告")
        report.append("="*80)
        
        # 成功分析的模型
        successful_models = {k: v for k, v in self.analysis_results.items() if 'error' not in v}
        
        if not successful_models:
            report.append("所有模型分析都失败了")
            return "\n".join(report)
        
        # 基础统计表
        report.append("\n基础模型统计:")
        report.append("-"*80)
        report.append(f"{'模型':<12} {'参数数':<12} {'模型大小(MB)':<15} {'推理时间(ms)':<15} {'内存(MB)':<12}")
        report.append("-"*80)
        
        for model_name, results in successful_models.items():
            basic = results['basic_complexity']
            report.append(
                f"{model_name:<12} "
                f"{basic['total_parameters']:<12,} "
                f"{basic['model_size_mb']:<15.2f} "
                f"{basic['avg_inference_time_s']*1000:<15.2f} "
                f"{basic['peak_memory_mb']:<12.2f}"
            )
        
        # FLOPs比较
        report.append("\n\nFLOPs分析:")
        report.append("-"*60)
        report.append(f"{'模型':<12} {'FLOPs':<15} {'FLOPs/参数':<15}")
        report.append("-"*60)
        
        for model_name, results in successful_models.items():
            basic = results['basic_complexity']
            flops = basic['flops_estimate']
            params = basic['total_parameters']
            flops_per_param = flops / params if params > 0 else 0
            
            report.append(
                f"{model_name:<12} "
                f"{flops:<15,} "
                f"{flops_per_param:<15.2f}"
            )
        
        # 层级分析摘要
        report.append("\n\n层级分析摘要:")
        report.append("-"*60)
        
        for model_name, results in successful_models.items():
            if 'layer_analysis' in results:
                report.append(f"\n{model_name}:")
                layer_stats = results['layer_analysis']
                
                # 按参数数量排序
                sorted_layers = sorted(layer_stats.items(), 
                                     key=lambda x: x[1]['params'], reverse=True)
                
                for layer_type, stats in sorted_layers[:5]:  # 只显示前5个
                    if stats['params'] > 0:
                        report.append(
                            f"  {layer_type}: {stats['count']}个, "
                            f"{stats['params']:,}参数, "
                            f"{stats['memory']/1024/1024:.2f}MB"
                        )
        
        # 性能排名
        report.append("\n\n性能排名:")
        report.append("-"*60)
        
        # 按推理速度排序
        speed_ranking = sorted(successful_models.items(), 
                             key=lambda x: x[1]['basic_complexity']['avg_inference_time_s'])
        report.append("推理速度排名 (快到慢):")
        for i, (model_name, results) in enumerate(speed_ranking, 1):
            time_ms = results['basic_complexity']['avg_inference_time_s'] * 1000
            report.append(f"  {i}. {model_name}: {time_ms:.2f}ms")
        
        # 按内存使用排序
        memory_ranking = sorted(successful_models.items(), 
                              key=lambda x: x[1]['basic_complexity']['peak_memory_mb'])
        report.append("\n内存使用排名 (少到多):")
        for i, (model_name, results) in enumerate(memory_ranking, 1):
            memory_mb = results['basic_complexity']['peak_memory_mb']
            report.append(f"  {i}. {model_name}: {memory_mb:.2f}MB")
        
        # 按参数效率排序
        param_ranking = sorted(successful_models.items(), 
                             key=lambda x: x[1]['basic_complexity']['total_parameters'])
        report.append("\n参数数量排名 (少到多):")
        for i, (model_name, results) in enumerate(param_ranking, 1):
            params = results['basic_complexity']['total_parameters']
            report.append(f"  {i}. {model_name}: {params:,}")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "complexity_analysis_results.json"):
        """保存分析结果"""
        # 转换numpy类型为Python原生类型以便JSON序列化
        def convert_for_json(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_for_json(item) for item in obj]
            else:
                return obj
        
        json_results = convert_for_json(self.analysis_results)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n分析结果已保存到: {filename}")
    
    def save_report(self, filename: str = "complexity_analysis_report.txt"):
        """保存分析报告"""
        report = self.generate_comparison_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"分析报告已保存到: {filename}")

def main():
    """主函数"""
    # 创建分析器
    analyzer = ModelComplexityAnalyzer()
    
    # 运行全面分析
    results = analyzer.analyze_all_models()
    
    # 生成并显示报告
    report = analyzer.generate_comparison_report()
    print(report)
    
    # 保存结果
    analyzer.save_results("complexity_analysis_results.json")
    analyzer.save_report("complexity_analysis_report.txt")
    
    return results

if __name__ == "__main__":
    main()