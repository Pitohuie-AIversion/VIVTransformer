"""
统一模型管理器
支持参数量标准化和硬件资源优化的模型管理系统
"""

import torch
import torch.nn as nn
import yaml
import logging
import os
import time
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

from hardware_aware_deployment import HardwareProfiler, AdaptiveDeploymentManager

@dataclass
class ModelInfo:
    """模型信息数据类"""
    name: str
    type: str
    config: Dict[str, Any]
    parameter_count: int
    estimated_memory_mb: float
    tier: str
    status: str = "initialized"  # initialized, loaded, trained, evaluated

class ParameterCounter:
    """参数量计算器"""
    
    @staticmethod
    def count_parameters(model: nn.Module) -> int:
        """计算模型参数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def estimate_memory_usage(model: nn.Module, input_shape: Tuple[int, ...], 
                            batch_size: int = 1, dtype: torch.dtype = torch.float32) -> float:
        """估算模型内存使用量 (MB)"""
        # 参数内存
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters())
        
        # 激活内存 (粗略估算)
        input_memory = np.prod(input_shape) * batch_size * 4  # float32
        activation_memory = input_memory * 10  # 经验值，激活约为输入的10倍
        
        # 梯度内存
        grad_memory = param_memory
        
        total_memory = (param_memory + activation_memory + grad_memory) / (1024 * 1024)
        return total_memory

class ResourceMonitor:
    """资源监控器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.monitoring_data = defaultdict(list)
        
    def start_monitoring(self):
        """开始监控"""
        self.start_time = time.time()
        self.initial_memory = psutil.virtual_memory().used
        if torch.cuda.is_available():
            self.initial_gpu_memory = torch.cuda.memory_allocated()
        
    def record_snapshot(self, stage: str):
        """记录资源快照"""
        current_time = time.time() - self.start_time
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        snapshot = {
            'timestamp': current_time,
            'stage': stage,
            'cpu_percent': cpu_percent,
            'memory_used_mb': memory_info.used / (1024 * 1024),
            'memory_percent': memory_info.percent
        }
        
        if torch.cuda.is_available():
            snapshot['gpu_memory_mb'] = torch.cuda.memory_allocated() / (1024 * 1024)
            snapshot['gpu_memory_cached_mb'] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        self.monitoring_data[stage].append(snapshot)
        
    def get_peak_usage(self) -> Dict[str, float]:
        """获取峰值使用量"""
        all_snapshots = []
        for stage_data in self.monitoring_data.values():
            all_snapshots.extend(stage_data)
        
        if not all_snapshots:
            return {}
        
        peak_cpu = max(s['cpu_percent'] for s in all_snapshots)
        peak_memory = max(s['memory_used_mb'] for s in all_snapshots)
        
        result = {
            'peak_cpu_percent': peak_cpu,
            'peak_memory_mb': peak_memory
        }
        
        if torch.cuda.is_available():
            peak_gpu = max(s.get('gpu_memory_mb', 0) for s in all_snapshots)
            result['peak_gpu_memory_mb'] = peak_gpu
        
        return result
    
    def cleanup_memory(self):
        """清理内存"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class UnifiedModelManager:
    """统一模型管理器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
        # 初始化组件
        self.hardware_profiler = HardwareProfiler()
        self.deployment_manager = AdaptiveDeploymentManager(config_path)
        self.parameter_counter = ParameterCounter()
        self.resource_monitor = ResourceMonitor()
        
        # 模型注册表
        self.model_registry: Dict[str, ModelInfo] = {}
        self.loaded_models: Dict[str, nn.Module] = {}
        
        # 硬件信息
        self.hardware_info = self.hardware_profiler.get_system_info()
        self.selected_tier = None
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def initialize_deployment(self) -> Dict[str, Any]:
        """初始化部署配置"""
        self.logger.info("🚀 初始化统一模型部署...")
        
        # 硬件感知部署
        deployment_result = self.deployment_manager.analyze_and_deploy()
        self.selected_tier = deployment_result['selected_tier']
        
        # 注册模型
        self._register_models(deployment_result['deployment_config'])
        
        self.logger.info(f"✅ 部署初始化完成，选择级别: {self.selected_tier}")
        self.logger.info(f"📊 注册模型数量: {len(self.model_registry)}")
        
        return deployment_result
    
    def _register_models(self, deployment_config: Dict[str, Any]):
        """注册模型到管理器"""
        models_config = deployment_config.get('models', {})
        
        for model_name, model_config in models_config.items():
            if not model_config.get('enabled', True):
                continue
                
            # 估算参数量和内存使用
            estimated_params = self._estimate_model_parameters(model_config)
            estimated_memory = self._estimate_model_memory(model_config)
            
            model_info = ModelInfo(
                name=model_name,
                type=model_config.get('type', 'unknown'),
                config=model_config,
                parameter_count=estimated_params,
                estimated_memory_mb=estimated_memory,
                tier=self.selected_tier
            )
            
            self.model_registry[model_name] = model_info
            self.logger.info(f"📝 注册模型: {model_name} ({estimated_params:,} 参数, ~{estimated_memory:.1f}MB)")
    
    def _estimate_model_parameters(self, model_config: Dict[str, Any]) -> int:
        """估算模型参数量"""
        model_type = model_config.get('type', '')
        
        if model_type == 'transformer':
            d_model = model_config.get('d_model', 192)
            num_layers = model_config.get('num_layers', 3)
            num_heads = model_config.get('num_heads', 6)
            
            # Transformer参数量估算公式
            # 每层: 4 * d_model^2 (self-attention + FFN)
            params_per_layer = 4 * d_model * d_model
            total_params = params_per_layer * num_layers
            
        elif model_type == 'enhanced_mlp':
            hidden_dim = model_config.get('hidden_dim', 48)
            num_layers = model_config.get('num_layers', 19)
            
            # MLP参数量估算
            params_per_layer = hidden_dim * hidden_dim
            total_params = params_per_layer * num_layers
            
        elif model_type == 'enhanced_fno1d':
            width = model_config.get('width', 48)
            modes = model_config.get('modes', 16)
            num_layers = model_config.get('num_layers', 3)
            
            # FNO参数量估算
            params_per_layer = width * width + modes * width
            total_params = params_per_layer * num_layers
            
        else:
            # 默认估算
            total_params = 1000000  # 1M参数
        
        return int(total_params)
    
    def _estimate_model_memory(self, model_config: Dict[str, Any]) -> float:
        """估算模型内存使用量"""
        estimated_params = self._estimate_model_parameters(model_config)
        
        # 基础内存: 参数 + 梯度 + 优化器状态
        param_memory = estimated_params * 4 / (1024 * 1024)  # float32, MB
        grad_memory = param_memory
        optimizer_memory = param_memory * 2  # Adam优化器
        
        # 激活内存 (基于输入分辨率)
        input_res = self.config.get('data', {}).get('input_resolution', [32, 32])
        batch_size = self.config.get('data', {}).get('batch_size', 8)
        activation_memory = np.prod(input_res) * batch_size * 4 / (1024 * 1024)
        
        total_memory = param_memory + grad_memory + optimizer_memory + activation_memory
        return total_memory
    
    def load_model(self, model_name: str) -> Optional[nn.Module]:
        """加载指定模型"""
        if model_name not in self.model_registry:
            self.logger.error(f"模型 {model_name} 未注册")
            return None
        
        if model_name in self.loaded_models:
            self.logger.info(f"模型 {model_name} 已加载")
            return self.loaded_models[model_name]
        
        self.resource_monitor.start_monitoring()
        self.resource_monitor.record_snapshot('before_load')
        
        try:
            # 这里应该根据模型类型加载实际模型
            # 为了演示，我们创建一个简单的模型
            model_info = self.model_registry[model_name]
            model = self._create_model(model_info)
            
            # 验证参数量
            actual_params = self.parameter_counter.count_parameters(model)
            model_info.parameter_count = actual_params
            
            self.loaded_models[model_name] = model
            model_info.status = "loaded"
            
            self.resource_monitor.record_snapshot('after_load')
            
            self.logger.info(f"✅ 模型 {model_name} 加载成功 ({actual_params:,} 参数)")
            
            return model
            
        except Exception as e:
            self.logger.error(f"❌ 模型 {model_name} 加载失败: {e}")
            return None
    
    def _create_model(self, model_info: ModelInfo) -> nn.Module:
        """创建模型实例 (简化版本)"""
        # 这里应该根据实际的模型工厂创建模型
        # 为了演示，创建一个简单的线性模型
        
        if model_info.type == 'transformer':
            d_model = model_info.config.get('d_model', 192)
            return nn.Sequential(
                nn.Linear(1024, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
                nn.ReLU(),
                nn.Linear(d_model, 1024)
            )
        elif model_info.type == 'enhanced_mlp':
            hidden_dim = model_info.config.get('hidden_dim', 48)
            num_layers = model_info.config.get('num_layers', 19)
            
            layers = [nn.Linear(1024, hidden_dim), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
            layers.append(nn.Linear(hidden_dim, 1024))
            
            return nn.Sequential(*layers)
        else:
            # 默认模型
            return nn.Sequential(
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Linear(512, 1024)
            )
    
    def unload_model(self, model_name: str):
        """卸载模型"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            self.model_registry[model_name].status = "initialized"
            self.resource_monitor.cleanup_memory()
            self.logger.info(f"🗑️ 模型 {model_name} 已卸载")
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """获取模型统计信息"""
        stats = {
            'total_models': len(self.model_registry),
            'loaded_models': len(self.loaded_models),
            'tier': self.selected_tier,
            'models': {}
        }
        
        total_params = 0
        total_memory = 0
        
        for name, info in self.model_registry.items():
            model_stats = {
                'type': info.type,
                'parameters': info.parameter_count,
                'estimated_memory_mb': info.estimated_memory_mb,
                'status': info.status,
                'tier': info.tier
            }
            
            stats['models'][name] = model_stats
            total_params += info.parameter_count
            total_memory += info.estimated_memory_mb
        
        stats['total_parameters'] = total_params
        stats['total_estimated_memory_mb'] = total_memory
        
        return stats
    
    def optimize_resource_usage(self) -> Dict[str, Any]:
        """优化资源使用"""
        self.logger.info("🔧 开始资源优化...")
        
        # 获取当前资源使用情况
        current_memory = psutil.virtual_memory()
        memory_usage_percent = current_memory.percent
        
        optimization_actions = []
        
        # 如果内存使用率过高，卸载部分模型
        if memory_usage_percent > 85:
            loaded_models = list(self.loaded_models.keys())
            for model_name in loaded_models[:-1]:  # 保留最后一个模型
                self.unload_model(model_name)
                optimization_actions.append(f"卸载模型: {model_name}")
        
        # 清理内存
        self.resource_monitor.cleanup_memory()
        optimization_actions.append("执行内存清理")
        
        # 获取优化后的资源使用情况
        after_memory = psutil.virtual_memory()
        
        optimization_result = {
            'before_memory_percent': memory_usage_percent,
            'after_memory_percent': after_memory.percent,
            'memory_saved_mb': (current_memory.used - after_memory.used) / (1024 * 1024),
            'actions': optimization_actions
        }
        
        self.logger.info(f"✅ 资源优化完成，内存使用率: {memory_usage_percent:.1f}% → {after_memory.percent:.1f}%")
        
        return optimization_result
    
    def validate_parameter_uniformity(self) -> Dict[str, Any]:
        """验证参数量统一性"""
        self.logger.info("🔍 验证参数量统一性...")
        
        tier_config = self.config.get('parameter_tiers', {}).get(self.selected_tier, {})
        target_range = tier_config.get('target_params', '1M-2M')
        
        # 解析目标范围
        range_parts = target_range.replace('M', '').split('-')
        min_params = float(range_parts[0]) * 1_000_000
        max_params = float(range_parts[1]) * 1_000_000
        
        validation_results = {
            'tier': self.selected_tier,
            'target_range': target_range,
            'min_params': int(min_params),
            'max_params': int(max_params),
            'models': {},
            'compliant_models': 0,
            'non_compliant_models': 0
        }
        
        for name, info in self.model_registry.items():
            is_compliant = min_params <= info.parameter_count <= max_params
            
            validation_results['models'][name] = {
                'parameter_count': info.parameter_count,
                'is_compliant': is_compliant,
                'deviation_percent': ((info.parameter_count - (min_params + max_params) / 2) / 
                                    ((max_params - min_params) / 2)) * 100
            }
            
            if is_compliant:
                validation_results['compliant_models'] += 1
            else:
                validation_results['non_compliant_models'] += 1
        
        compliance_rate = (validation_results['compliant_models'] / 
                          len(self.model_registry) * 100) if self.model_registry else 0
        validation_results['compliance_rate'] = compliance_rate
        
        self.logger.info(f"📊 参数量统一性验证完成，合规率: {compliance_rate:.1f}%")
        
        return validation_results
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """生成部署报告"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'hardware_info': self.hardware_info,
            'selected_tier': self.selected_tier,
            'model_statistics': self.get_model_statistics(),
            'parameter_validation': self.validate_parameter_uniformity(),
            'resource_usage': self.resource_monitor.get_peak_usage(),
            'recommendations': self._generate_deployment_recommendations()
        }
        
        return report
    
    def _generate_deployment_recommendations(self) -> List[str]:
        """生成部署建议"""
        recommendations = []
        
        # 基于硬件信息的建议
        memory_mb = self.hardware_info['memory']['total_mb']
        cpu_cores = self.hardware_info['cpu']['cores']
        
        if memory_mb < 8192:
            recommendations.append("内存较小，建议使用small级别模型或减少批次大小")
        
        if cpu_cores < 4:
            recommendations.append("CPU核心数较少，建议减少数据加载器工作进程数")
        
        # 基于模型统计的建议
        stats = self.get_model_statistics()
        if stats['total_estimated_memory_mb'] > memory_mb * 0.8:
            recommendations.append("模型总内存需求较高，建议分批训练或使用更小的模型")
        
        # 基于参数量验证的建议
        validation = self.validate_parameter_uniformity()
        if validation['compliance_rate'] < 80:
            recommendations.append("部分模型参数量不符合统一标准，建议调整模型配置")
        
        return recommendations
    
    def save_deployment_report(self, output_path: str):
        """保存部署报告"""
        report = self.generate_deployment_report()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(report, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        self.logger.info(f"📄 部署报告已保存: {output_path}")

def main():
    """主函数 - 演示统一模型管理"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 配置文件路径
    config_path = "configs/unified_adaptive_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    # 创建统一模型管理器
    manager = UnifiedModelManager(config_path)
    
    # 初始化部署
    print("🚀 初始化统一模型部署...")
    deployment_result = manager.initialize_deployment()
    
    # 显示模型统计
    print("\n📊 模型统计信息:")
    stats = manager.get_model_statistics()
    print(f"  总模型数: {stats['total_models']}")
    print(f"  参数量级别: {stats['tier']}")
    print(f"  总参数量: {stats['total_parameters']:,}")
    print(f"  预估内存: {stats['total_estimated_memory_mb']:.1f}MB")
    
    # 验证参数量统一性
    print("\n🔍 参数量统一性验证:")
    validation = manager.validate_parameter_uniformity()
    print(f"  目标范围: {validation['target_range']}")
    print(f"  合规率: {validation['compliance_rate']:.1f}%")
    
    # 加载一个模型进行测试
    if stats['total_models'] > 0:
        model_name = list(manager.model_registry.keys())[0]
        print(f"\n🔄 加载模型: {model_name}")
        model = manager.load_model(model_name)
        if model:
            print(f"✅ 模型加载成功")
    
    # 生成部署报告
    report_path = "results/unified_deployment_report.yaml"
    manager.save_deployment_report(report_path)
    print(f"\n📄 部署报告已生成: {report_path}")

if __name__ == "__main__":
    main()