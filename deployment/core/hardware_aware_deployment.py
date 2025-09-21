"""
硬件感知的模型部署策略
根据硬件性能自动选择合适的参数量级别，实现统一的模型部署和管理
"""

import psutil
import torch
import yaml
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import subprocess
import platform

class HardwareProfiler:
    """硬件性能分析器"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统硬件信息"""
        info = {
            'cpu': self._get_cpu_info(),
            'memory': self._get_memory_info(),
            'gpu': self._get_gpu_info(),
            'platform': self._get_platform_info()
        }
        return info
    
    def _get_cpu_info(self) -> Dict[str, Any]:
        """获取CPU信息"""
        return {
            'cores': psutil.cpu_count(logical=False),
            'logical_cores': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq().max if psutil.cpu_freq() else 0,
            'usage_percent': psutil.cpu_percent(interval=1),
            'architecture': platform.machine()
        }
    
    def _get_memory_info(self) -> Dict[str, Any]:
        """获取内存信息"""
        memory = psutil.virtual_memory()
        return {
            'total_mb': memory.total // (1024 * 1024),
            'available_mb': memory.available // (1024 * 1024),
            'used_mb': memory.used // (1024 * 1024),
            'percent': memory.percent
        }
    
    def _get_gpu_info(self) -> Dict[str, Any]:
        """获取GPU信息"""
        gpu_info = {
            'available': torch.cuda.is_available(),
            'count': 0,
            'devices': []
        }
        
        if torch.cuda.is_available():
            gpu_info['count'] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                device_props = torch.cuda.get_device_properties(i)
                memory_mb = device_props.total_memory // (1024 * 1024)
                gpu_info['devices'].append({
                    'id': i,
                    'name': device_props.name,
                    'memory_mb': memory_mb,
                    'compute_capability': f"{device_props.major}.{device_props.minor}"
                })
        
        return gpu_info
    
    def _get_platform_info(self) -> Dict[str, str]:
        """获取平台信息"""
        return {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'processor': platform.processor()
        }

class TierSelector:
    """参数量级别选择器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = logging.getLogger(__name__)
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def select_tier(self, hardware_info: Dict[str, Any]) -> str:
        """根据硬件信息选择参数量级别"""
        if not self.config.get('hardware_detection', {}).get('enabled', False):
            return 'medium'  # 默认级别
            
        if not self.config.get('hardware_detection', {}).get('auto_select_tier', False):
            return 'medium'  # 默认级别
            
        # 获取阈值配置
        memory_thresholds = self.config['hardware_detection']['memory_threshold']
        cpu_thresholds = self.config['hardware_detection']['cpu_threshold']
        gpu_thresholds = self.config['hardware_detection']['gpu_threshold']
        
        # 获取硬件性能指标
        available_memory = hardware_info['memory']['available_mb']
        cpu_cores = hardware_info['cpu']['cores']
        gpu_memory = 0
        if hardware_info['gpu']['available'] and hardware_info['gpu']['devices']:
            gpu_memory = max(device['memory_mb'] for device in hardware_info['gpu']['devices'])
        
        # 计算各维度的级别分数
        memory_score = self._calculate_tier_score(available_memory, memory_thresholds)
        cpu_score = self._calculate_tier_score(cpu_cores, cpu_thresholds)
        gpu_score = self._calculate_tier_score(gpu_memory, gpu_thresholds) if gpu_memory > 0 else 1
        
        # 综合评分 (取最小值确保稳定性)
        final_score = min(memory_score, cpu_score, gpu_score)
        
        if final_score >= 3:
            tier = 'large'
        elif final_score >= 2:
            tier = 'medium'
        else:
            tier = 'small'
            
        self.logger.info(f"硬件评分 - 内存: {memory_score}, CPU: {cpu_score}, GPU: {gpu_score}")
        self.logger.info(f"最终评分: {final_score}, 选择级别: {tier}")
        
        return tier
    
    def _calculate_tier_score(self, value: float, thresholds: Dict[str, float]) -> int:
        """计算单个维度的级别分数"""
        if value >= thresholds['high']:
            return 3  # large
        elif value >= thresholds['medium']:
            return 2  # medium
        elif value >= thresholds['low']:
            return 1  # small
        else:
            return 0  # 资源不足

class AdaptiveDeploymentManager:
    """自适应部署管理器"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.profiler = HardwareProfiler()
        self.tier_selector = TierSelector(config_path)
        self.logger = logging.getLogger(__name__)
        
    def analyze_and_deploy(self) -> Dict[str, Any]:
        """分析硬件并部署合适的配置"""
        # 1. 硬件性能分析
        hardware_info = self.profiler.get_system_info()
        self.logger.info("硬件信息分析完成")
        
        # 2. 选择参数量级别
        selected_tier = self.tier_selector.select_tier(hardware_info)
        self.logger.info(f"选择参数量级别: {selected_tier}")
        
        # 3. 生成部署配置
        deployment_config = self._generate_deployment_config(hardware_info, selected_tier)
        
        # 4. 应用自适应资源配置
        adaptive_config = self._apply_adaptive_resources(hardware_info, deployment_config)
        
        return {
            'hardware_info': hardware_info,
            'selected_tier': selected_tier,
            'deployment_config': deployment_config,
            'adaptive_config': adaptive_config,
            'recommendations': self._generate_recommendations(hardware_info, selected_tier)
        }
    
    def _generate_deployment_config(self, hardware_info: Dict[str, Any], tier: str) -> Dict[str, Any]:
        """生成部署配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        # 选择对应级别的模型配置
        tier_key = f"{tier}_tier"
        if tier_key not in base_config['models']:
            self.logger.warning(f"未找到级别 {tier} 的配置，使用medium级别")
            tier_key = "medium_tier"
        
        # 构建部署配置
        deployment_config = {
            'data': base_config['data'].copy(),
            'training': base_config['training'].copy(),
            'loss': base_config['loss'].copy(),
            'attention': base_config['attention'].copy(),
            'models': base_config['models'][tier_key].copy(),
            'device': base_config['device'],
            'random_seed': base_config['random_seed'],
            'output': base_config['output'].copy(),
            'experiment': base_config['experiment'].copy()
        }
        
        # 更新实验信息
        deployment_config['experiment']['tier'] = tier
        deployment_config['experiment']['hardware_tier'] = tier
        deployment_config['output']['results_dir'] = f"./results/unified_adaptive/{tier}_tier"
        
        return deployment_config
    
    def _apply_adaptive_resources(self, hardware_info: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
        """应用自适应资源配置"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            base_config = yaml.safe_load(f)
        
        adaptive_resources = base_config.get('adaptive_resources', {})
        if not adaptive_resources.get('enabled', False):
            return config
        
        # 自适应批次大小
        if adaptive_resources.get('batch_size', {}).get('enabled', False):
            batch_config = adaptive_resources['batch_size']
            optimal_batch_size = self._calculate_optimal_batch_size(
                hardware_info, batch_config
            )
            config['data']['batch_size'] = optimal_batch_size
        
        # 自适应数据加载器
        if adaptive_resources.get('dataloader', {}).get('enabled', False):
            dataloader_config = adaptive_resources['dataloader']
            optimal_workers = self._calculate_optimal_workers(
                hardware_info, dataloader_config
            )
            config['data']['num_workers'] = optimal_workers
        
        return config
    
    def _calculate_optimal_batch_size(self, hardware_info: Dict[str, Any], batch_config: Dict[str, Any]) -> int:
        """计算最优批次大小"""
        base_size = batch_config.get('base_size', 8)
        min_size = batch_config.get('min_size', 2)
        max_size = batch_config.get('max_size', 64)
        memory_factor = batch_config.get('memory_factor', 0.8)
        
        available_memory = hardware_info['memory']['available_mb']
        
        # 基于可用内存调整批次大小
        if available_memory < 4096:  # < 4GB
            optimal_size = min(base_size // 2, max_size)
        elif available_memory < 8192:  # < 8GB
            optimal_size = base_size
        elif available_memory < 16384:  # < 16GB
            optimal_size = min(base_size * 2, max_size)
        else:  # >= 16GB
            optimal_size = min(base_size * 4, max_size)
        
        return max(min_size, min(optimal_size, max_size))
    
    def _calculate_optimal_workers(self, hardware_info: Dict[str, Any], dataloader_config: Dict[str, Any]) -> int:
        """计算最优工作进程数"""
        base_workers = dataloader_config.get('base_workers', 4)
        min_workers = dataloader_config.get('min_workers', 1)
        max_workers = dataloader_config.get('max_workers', 16)
        cpu_factor = dataloader_config.get('cpu_factor', 0.75)
        
        cpu_cores = hardware_info['cpu']['cores']
        
        # 基于CPU核心数调整工作进程数
        optimal_workers = int(cpu_cores * cpu_factor)
        
        return max(min_workers, min(optimal_workers, max_workers))
    
    def _generate_recommendations(self, hardware_info: Dict[str, Any], tier: str) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        # 内存建议
        memory_percent = hardware_info['memory']['percent']
        if memory_percent > 80:
            recommendations.append("内存使用率较高，建议关闭其他应用程序或降低批次大小")
        
        # CPU建议
        cpu_cores = hardware_info['cpu']['cores']
        if cpu_cores < 4:
            recommendations.append("CPU核心数较少，建议减少数据加载器工作进程数")
        
        # GPU建议
        if not hardware_info['gpu']['available']:
            recommendations.append("未检测到GPU，将使用CPU进行训练，建议使用较小的模型")
        elif hardware_info['gpu']['devices']:
            gpu_memory = max(device['memory_mb'] for device in hardware_info['gpu']['devices'])
            if gpu_memory < 4096:
                recommendations.append("GPU显存较小，建议使用small级别的模型配置")
        
        # 级别建议
        if tier == 'small':
            recommendations.append("当前硬件配置适合轻量级模型，建议优先使用transformer和mlp_small")
        elif tier == 'large':
            recommendations.append("当前硬件配置强大，可以使用所有大型模型进行对比实验")
        
        return recommendations
    
    def save_deployment_config(self, deployment_result: Dict[str, Any], output_path: str) -> str:
        """保存部署配置到文件"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存完整的部署配置
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(deployment_result['deployment_config'], f, 
                     default_flow_style=False, allow_unicode=True, indent=2)
        
        # 保存硬件信息和建议
        info_path = output_path.replace('.yaml', '_info.yaml')
        info_data = {
            'hardware_info': deployment_result['hardware_info'],
            'selected_tier': deployment_result['selected_tier'],
            'recommendations': deployment_result['recommendations'],
            'adaptive_config': deployment_result['adaptive_config']
        }
        
        with open(info_path, 'w', encoding='utf-8') as f:
            yaml.dump(info_data, f, default_flow_style=False, allow_unicode=True, indent=2)
        
        self.logger.info(f"部署配置已保存到: {output_path}")
        self.logger.info(f"硬件信息已保存到: {info_path}")
        
        return output_path

def main():
    """主函数 - 演示硬件感知部署"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 配置文件路径
    config_path = "configs/unified_adaptive_config.yaml"
    
    if not os.path.exists(config_path):
        print(f"配置文件不存在: {config_path}")
        return
    
    # 创建部署管理器
    deployment_manager = AdaptiveDeploymentManager(config_path)
    
    # 分析硬件并生成部署配置
    print("🔍 正在分析硬件配置...")
    deployment_result = deployment_manager.analyze_and_deploy()
    
    # 显示结果
    print(f"\n📊 硬件分析结果:")
    hardware_info = deployment_result['hardware_info']
    print(f"  CPU: {hardware_info['cpu']['cores']}核心 ({hardware_info['cpu']['logical_cores']}逻辑核心)")
    print(f"  内存: {hardware_info['memory']['total_mb']:.0f}MB (可用: {hardware_info['memory']['available_mb']:.0f}MB)")
    
    if hardware_info['gpu']['available']:
        for gpu in hardware_info['gpu']['devices']:
            print(f"  GPU: {gpu['name']} ({gpu['memory_mb']:.0f}MB)")
    else:
        print("  GPU: 未检测到")
    
    print(f"\n🎯 选择的参数量级别: {deployment_result['selected_tier']}")
    
    print(f"\n📝 优化建议:")
    for rec in deployment_result['recommendations']:
        print(f"  • {rec}")
    
    # 保存配置
    output_path = f"configs/deployed_{deployment_result['selected_tier']}_config.yaml"
    deployment_manager.save_deployment_config(deployment_result, output_path)
    
    print(f"\n✅ 部署配置已生成: {output_path}")
    print(f"💡 使用方法: python run_crop_model_test.py --config {output_path}")

if __name__ == "__main__":
    main()