#!/usr/bin/env python3
"""
SVD 损失权重计算器 - 基于模态分析结果生成权重建议

使用方法:
    python weight_calculator.py --report_path /path/to/modality_analysis_report.json
    
功能:
    - 基于能量谱分布计算SVD各模态权重
    - 提供多种权重分配策略
    - 输出配置文件格式的权重配置
    - 提供SVD损失函数代码示例
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml


class SVDWeightCalculator:
    """SVD权重计算器"""
    
    def __init__(self, report_path: str):
        """
        初始化权重计算器
        
        Args:
            report_path: 模态分析报告JSON文件路径
        """
        self.report_path = Path(report_path)
        self.report_data = self._load_report()
        
    def _load_report(self) -> Dict:
        """加载模态分析报告"""
        if not self.report_path.exists():
            raise FileNotFoundError(f"分析报告文件不存在: {self.report_path}")
        
        with open(self.report_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_energy_ratios(self) -> Tuple[List[float], List[float]]:
        """获取输入和输出的能量比例，兼容不同JSON结构"""
        data = self.report_data
        
        # Debug: 打印数据结构以便诊断
        print("JSON 键:", list(data.keys()) if isinstance(data, dict) else "Non-dict data")
        
        # 尝试多种可能的路径
        input_energy = None
        output_energy = None
        
        # 路径1: 顶级键
        if isinstance(data, dict) and input_energy is None and 'input_energy_ratios' in data:
            input_energy = data['input_energy_ratios']
            output_energy = data.get('output_energy_ratios', [])
        
        # 路径2: modality_data 嵌套
        if isinstance(data, dict) and input_energy is None and 'modality_data' in data:
            md = data['modality_data']
            if isinstance(md, dict) and 'input_energy_ratios' in md:
                input_energy = md['input_energy_ratios']
                output_energy = md.get('output_energy_ratios', [])
        
        # 路径3: modality_analysis 嵌套（本项目实际结构）
        if isinstance(data, dict) and input_energy is None and 'modality_analysis' in data:
            ma = data['modality_analysis']
            if isinstance(ma, dict):
                # 3a: 直接位于 modality_analysis 下
                if 'input_energy_ratios' in ma and 'output_energy_ratios' in ma:
                    input_energy = ma['input_energy_ratios']
                    output_energy = ma['output_energy_ratios']
                # 3b: 位于 projection_info 下
                elif 'projection_info' in ma and isinstance(ma['projection_info'], dict):
                    pi = ma['projection_info']
                    if 'input_energy_ratios' in pi and 'output_energy_ratios' in pi:
                        input_energy = pi['input_energy_ratios']
                        output_energy = pi['output_energy_ratios']
        
        # 路径4: 其他候选命名
        if isinstance(data, dict) and input_energy is None and 'modality' in data:
            modality = data['modality']
            if isinstance(modality, dict):
                input_energy = modality.get('input_energy_ratios')
                output_energy = modality.get('output_energy_ratios')
        
        # 路径5: svd_projection_result 嵌套
        if isinstance(data, dict) and input_energy is None and 'svd_projection_result' in data:
            result = data['svd_projection_result']
            if isinstance(result, dict):
                input_energy = result.get('input_energy_ratios')
                output_energy = result.get('output_energy_ratios')
        
        if input_energy is None or output_energy is None:
            # 展示完整的数据结构以便调试
            print("完整数据结构预览:")
            def print_dict_structure(d, prefix="", max_depth=3):
                if max_depth <= 0:
                    return
                if isinstance(d, dict):
                    for k, v in list(d.items())[:20]:  # 显示更多键
                        if isinstance(v, (dict, list)):
                            print(f"{prefix}{k}: {type(v).__name__}")
                            if isinstance(v, dict):
                                print_dict_structure(v, prefix + "  ", max_depth - 1)
                        else:
                            print(f"{prefix}{k}: {type(v).__name__} = {str(v)[:100]}")
            print_dict_structure(data)
            raise KeyError("未在报告中找到 input_energy_ratios/output_energy_ratios 关键字段")
        
        return list(input_energy), list(output_energy)
    
    def calculate_energy_based_weights(self, topk: int = 10, 
                                     strategy: str = 'averaged',
                                     normalize: bool = True) -> List[float]:
        """
        基于能量分布计算权重
        
        Args:
            topk: SVD模态数量
            strategy: 权重策略 ('averaged', 'input_priority', 'output_priority', 'geometric_mean')
            normalize: 是否归一化权重
            
        Returns:
            SVD各模态的权重列表
        """
        input_energy, output_energy = self.get_energy_ratios()
        effective_modes = min(len(input_energy), len(output_energy), topk)
        
        # 扩展能量比例到指定模态数（超出部分填0）
        input_extended = input_energy[:effective_modes] + [0.0] * (topk - effective_modes)
        output_extended = output_energy[:effective_modes] + [0.0] * (topk - effective_modes)
        
        if strategy == 'averaged':
            # 平均策略：取输入输出能量的算术平均
            weights = [(i + o) / 2 for i, o in zip(input_extended, output_extended)]
        elif strategy == 'input_priority':
            # 输入优先：以输入能量为主，输出为辅
            weights = [0.7 * i + 0.3 * o for i, o in zip(input_extended, output_extended)]
        elif strategy == 'output_priority':
            # 输出优先：以输出能量为主，输入为辅
            weights = [0.3 * i + 0.7 * o for i, o in zip(input_extended, output_extended)]
        elif strategy == 'geometric_mean':
            # 几何平均：对能量取几何平均（避免一个为0的情况）
            weights = [np.sqrt(max(i, 1e-8) * max(o, 1e-8)) for i, o in zip(input_extended, output_extended)]
        else:
            raise ValueError(f"未知策略: {strategy}")
        
        if normalize and sum(weights) > 0:
            total = sum(weights)
            weights = [w / total for w in weights]
        
        return weights
    
    def calculate_adaptive_weights(self, base_weight: float = 0.6, 
                                 svd_weight_total: float = 0.4,
                                 topk: int = 10,
                                 strategy: str = 'averaged') -> Tuple[float, List[float]]:
        """
        计算适应性权重配置
        
        Args:
            base_weight: 基础MSE损失权重
            svd_weight_total: SVD损失总权重
            topk: SVD模态数量
            strategy: 权重策略
            
        Returns:
            (基础权重, SVD权重列表)
        """
        # 获取基于能量的相对权重
        relative_weights = self.calculate_energy_based_weights(topk, strategy, normalize=True)
        
        # 分配总SVD权重
        svd_weights = [w * svd_weight_total for w in relative_weights]
        
        return base_weight, svd_weights
    
    def generate_weight_recommendations(self) -> Dict:
        """生成多种权重配置建议"""
        input_energy, output_energy = self.get_energy_ratios()
        effective_modes = len(input_energy)
        
        recommendations = {
            'analysis_summary': {
                'effective_modes': effective_modes,
                'input_energy_ratios': input_energy,
                'output_energy_ratios': output_energy,
                'dominant_mode_energy': {
                    'input': input_energy[0] if input_energy else 0,
                    'output': output_energy[0] if output_energy else 0
                }
            },
            'configurations': {}
        }
        
        # 配置1: 保守配置（基础权重高）
        base_weight_1, svd_weights_1 = self.calculate_adaptive_weights(
            base_weight=0.7, svd_weight_total=0.3, topk=min(5, effective_modes), strategy='averaged'
        )
        recommendations['configurations']['conservative'] = {
            'description': '保守配置 - 基础MSE权重较高，适合训练初期',
            'base_weight': base_weight_1,
            'svd_weights': svd_weights_1,
            'topk': min(5, effective_modes),
            'total_weight': base_weight_1 + sum(svd_weights_1)
        }
        
        # 配置2: 平衡配置（权重相等）
        base_weight_2, svd_weights_2 = self.calculate_adaptive_weights(
            base_weight=0.5, svd_weight_total=0.5, topk=min(10, effective_modes), strategy='averaged'
        )
        recommendations['configurations']['balanced'] = {
            'description': '平衡配置 - 基础MSE与SVD权重相等',
            'base_weight': base_weight_2,
            'svd_weights': svd_weights_2,
            'topk': min(10, effective_modes),
            'total_weight': base_weight_2 + sum(svd_weights_2)
        }
        
        # 配置3: 激进配置（SVD权重高）
        base_weight_3, svd_weights_3 = self.calculate_adaptive_weights(
            base_weight=0.3, svd_weight_total=0.7, topk=effective_modes, strategy='averaged'
        )
        recommendations['configurations']['aggressive'] = {
            'description': '激进配置 - SVD权重较高，适合模态结构明确的数据',
            'base_weight': base_weight_3,
            'svd_weights': svd_weights_3,
            'topk': effective_modes,
            'total_weight': base_weight_3 + sum(svd_weights_3)
        }
        
        # 配置4: 基于数据特性的自适应配置
        # 如果第一模态能量占比极高（>95%），降低SVD权重
        dominant_energy = max(input_energy[0] if input_energy else 0, 
                            output_energy[0] if output_energy else 0)
        
        if dominant_energy > 0.95:
            adaptive_base = 0.8
            adaptive_svd_total = 0.2
            adaptive_strategy = 'geometric_mean'  # 使用几何平均避免过度依赖主模态
        else:
            adaptive_base = 0.4
            adaptive_svd_total = 0.6
            adaptive_strategy = 'averaged'
        
        base_weight_4, svd_weights_4 = self.calculate_adaptive_weights(
            base_weight=adaptive_base, svd_weight_total=adaptive_svd_total, 
            topk=min(8, effective_modes), strategy=adaptive_strategy
        )
        recommendations['configurations']['data_adaptive'] = {
            'description': f'数据自适应配置 - 基于主模态能量比例({dominant_energy:.3f})调整',
            'base_weight': base_weight_4,
            'svd_weights': svd_weights_4,
            'topk': min(8, effective_modes),
            'total_weight': base_weight_4 + sum(svd_weights_4),
            'strategy_used': adaptive_strategy
        }
        
        return recommendations
    
    def export_yaml_config(self, config_name: str, output_path: Optional[str] = None) -> str:
        """导出YAML格式的配置文件"""
        recommendations = self.generate_weight_recommendations()
        
        if config_name not in recommendations['configurations']:
            raise ValueError(f"配置名称 '{config_name}' 不存在。可用配置: {list(recommendations['configurations'].keys())}")
        
        config = recommendations['configurations'][config_name]
        
        yaml_config = {
            'loss': {
                'svd_loss_enabled': True,
                'base_weight': config['base_weight'],
                'svd_weights': config['svd_weights'],
                'topk': config['topk']
            },
            'svd_loss': {
                'enabled': True,
                'enhanced': True,
                'mixed_precision': True,
                'adaptive_weights': False,
                'fallback_level': 2,
                'monitoring': True,
                'topk': config['topk']
            },
            'comment': f"权重配置: {config['description']}"
        }
        
        if output_path is None:
            output_path = f"svd_loss_config_{config_name}.yaml"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(yaml_config, f, default_flow_style=False, allow_unicode=True)
        
        return output_path
    
    def generate_code_example(self, config_name: str) -> str:
        """生成使用指定配置的代码示例"""
        recommendations = self.generate_weight_recommendations()
        config = recommendations['configurations'][config_name]
        
        code_template = f'''# SVD损失函数配置 - {config['description']}
from modify_multi_attention.utils.enhanced_svd_loss import create_enhanced_svd_loss

# 方法1: 使用增强版SVD损失（推荐）
criterion = create_enhanced_svd_loss(
    base_weight={config['base_weight']:.3f},
    svd_weights={config['svd_weights']},
    topk={config['topk']},
    mixed_precision=True,
    adaptive_weights=False,
    monitoring=True,
    fallback_level=2
)

# 方法2: 使用基础版SVD损失
from modify_multi_attention.utils.loss import TotalLossWithSVD

criterion_basic = TotalLossWithSVD(
    base_weight={config['base_weight']:.3f},
    svd_weights={config['svd_weights']},
    topk={config['topk']}
)

# 训练中使用
def train_step(model, data_loader, optimizer, criterion):
    for batch_idx, (input_data, target_data) in enumerate(data_loader):
        optimizer.zero_grad()
        
        # 前向传播
        output = model(input_data)
        
        # 计算损失
        loss = criterion(output, target_data)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f'Batch {{batch_idx}}, Loss: {{loss.item():.6f}}')

# 权重信息查看
criterion.print_weight_info()
'''
        return code_template
    
    def print_recommendations(self):
        """打印权重建议"""
        recommendations = self.generate_weight_recommendations()
        
        print("=" * 60)
        print("SVD 损失权重配置建议")
        print("=" * 60)
        
        summary = recommendations['analysis_summary']
        print(f"数据分析摘要:")
        print(f"  - 有效模态数: {summary['effective_modes']}")
        print(f"  - 主模态能量占比 (输入): {summary['dominant_mode_energy']['input']:.3f}")
        print(f"  - 主模态能量占比 (输出): {summary['dominant_mode_energy']['output']:.3f}")
        print()
        
        for config_name, config in recommendations['configurations'].items():
            print(f"【{config_name.upper()}】{config['description']}")
            print(f"  - 基础权重: {config['base_weight']:.3f}")
            print(f"  - SVD权重: {[f'{w:.4f}' for w in config['svd_weights'][:5]]}...")
            print(f"  - TopK模态: {config['topk']}")
            print(f"  - 权重总和: {config['total_weight']:.3f}")
            print()


def main():
    parser = argparse.ArgumentParser(description='SVD损失权重计算器')
    parser.add_argument('--report_path', type=str, required=True,
                       help='模态分析报告JSON文件路径')
    parser.add_argument('--config', type=str, default='data_adaptive',
                       choices=['conservative', 'balanced', 'aggressive', 'data_adaptive'],
                       help='选择权重配置策略')
    parser.add_argument('--export_yaml', action='store_true',
                       help='导出YAML配置文件')
    parser.add_argument('--export_code', action='store_true',
                       help='导出代码示例')
    parser.add_argument('--output_dir', type=str, default='.',
                       help='输出目录')
    
    args = parser.parse_args()
    
    try:
        calculator = SVDWeightCalculator(args.report_path)
        
        # 打印建议
        calculator.print_recommendations()
        
        # 导出YAML配置
        if args.export_yaml:
            yaml_path = calculator.export_yaml_config(
                args.config, 
                f"{args.output_dir}/svd_loss_config_{args.config}.yaml"
            )
            print(f"[OK] YAML配置已导出: {yaml_path}")
        
        # 导出代码示例
        if args.export_code:
            code_example = calculator.generate_code_example(args.config)
            code_path = f"{args.output_dir}/svd_loss_example_{args.config}.py"
            with open(code_path, 'w', encoding='utf-8') as f:
                f.write(code_example)
            print(f"[OK] 代码示例已导出: {code_path}")
            
    except Exception as e:
        print(f"[ERROR] 错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())