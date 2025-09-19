#!/usr/bin/env python3
"""测试结果可靠性和公平性验证脚本"""

import yaml
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
from datetime import datetime

class FairnessValidator:
    """公平性验证器"""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.results = self.load_results()
        self.successful_tests = [r for r in self.results if r.get('success', False)]
        
    def load_results(self) -> List[Dict[str, Any]]:
        """加载测试结果"""
        if self.results_file.endswith('.yaml'):
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        else:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        
        return data.get('results', [])
    
    def check_data_consistency(self) -> Dict[str, Any]:
        """检查数据一致性"""
        print("\n=== 数据一致性检查 ===")
        
        # 检查每个模型是否在相同数据集上测试
        model_datasets = {}
        dataset_models = {}
        
        for result in self.successful_tests:
            model = result['model_name']
            dataset = result['dataset_name']
            
            if model not in model_datasets:
                model_datasets[model] = set()
            model_datasets[model].add(dataset)
            
            if dataset not in dataset_models:
                dataset_models[dataset] = set()
            dataset_models[dataset].add(model)
        
        # 检查数据集覆盖一致性
        all_datasets = set()
        for datasets in model_datasets.values():
            all_datasets.update(datasets)
        
        consistency_report = {
            'total_datasets': len(all_datasets),
            'datasets': list(all_datasets),
            'model_coverage': {},
            'dataset_coverage': {},
            'missing_combinations': []
        }
        
        # 检查每个模型的数据集覆盖
        for model, datasets in model_datasets.items():
            coverage = len(datasets) / len(all_datasets)
            consistency_report['model_coverage'][model] = {
                'datasets': list(datasets),
                'coverage_ratio': coverage,
                'missing_datasets': list(all_datasets - datasets)
            }
            
            # 记录缺失的组合
            for missing_dataset in all_datasets - datasets:
                consistency_report['missing_combinations'].append(f"{model} + {missing_dataset}")
        
        # 检查每个数据集的模型覆盖
        for dataset, models in dataset_models.items():
            consistency_report['dataset_coverage'][dataset] = {
                'models': list(models),
                'model_count': len(models)
            }
        
        print(f"总数据集数: {consistency_report['total_datasets']}")
        print(f"缺失的模型-数据集组合数: {len(consistency_report['missing_combinations'])}")
        
        return consistency_report
    
    def check_parameter_fairness(self) -> Dict[str, Any]:
        """检查参数量公平性"""
        print("\n=== 参数量公平性检查 ===")
        
        # 按模型类型分组
        model_types = {}
        for result in self.successful_tests:
            model_type = result['model_type']
            if model_type not in model_types:
                model_types[model_type] = []
            model_types[model_type].append(result)
        
        fairness_report = {
            'parameter_ranges': {},
            'type_statistics': {},
            'outliers': []
        }
        
        for model_type, results in model_types.items():
            params = [r['parameter_count'] for r in results]
            
            stats = {
                'count': len(params),
                'min': min(params),
                'max': max(params),
                'mean': np.mean(params),
                'std': np.std(params),
                'range_ratio': max(params) / min(params) if min(params) > 0 else float('inf')
            }
            
            fairness_report['type_statistics'][model_type] = stats
            fairness_report['parameter_ranges'][model_type] = f"{min(params):,} - {max(params):,}"
            
            # 检查异常值
            if stats['range_ratio'] > 10:  # 参数量差异超过10倍
                fairness_report['outliers'].append({
                    'type': model_type,
                    'issue': 'large_parameter_range',
                    'ratio': stats['range_ratio']
                })
        
        print("各模型类型参数量范围:")
        for model_type, range_str in fairness_report['parameter_ranges'].items():
            print(f"  {model_type}: {range_str}")
        
        return fairness_report
    
    def check_training_fairness(self) -> Dict[str, Any]:
        """检查训练配置公平性"""
        print("\n=== 训练配置公平性检查 ===")
        
        training_times = [r['training_time'] for r in self.successful_tests]
        
        fairness_report = {
            'training_time_stats': {
                'min': min(training_times),
                'max': max(training_times),
                'mean': np.mean(training_times),
                'std': np.std(training_times),
                'range_ratio': max(training_times) / min(training_times) if min(training_times) > 0 else float('inf')
            },
            'time_outliers': []
        }
        
        # 检查训练时间异常值
        mean_time = np.mean(training_times)
        std_time = np.std(training_times)
        
        for result in self.successful_tests:
            time = result['training_time']
            z_score = abs(time - mean_time) / std_time if std_time > 0 else 0
            
            if z_score > 2:  # 超过2个标准差
                fairness_report['time_outliers'].append({
                    'model': result['model_name'],
                    'dataset': result['dataset_name'],
                    'time': time,
                    'z_score': z_score
                })
        
        print(f"训练时间范围: {min(training_times):.3f}s - {max(training_times):.3f}s")
        print(f"训练时间异常值数量: {len(fairness_report['time_outliers'])}")
        
        return fairness_report
    
    def check_performance_reliability(self) -> Dict[str, Any]:
        """检查性能指标可靠性"""
        print("\n=== 性能指标可靠性检查 ===")
        
        # 检查R²值的合理性
        r2_values = [r['r2_score'] for r in self.successful_tests if 'r2_score' in r]
        mse_values = [r['mse'] for r in self.successful_tests]
        
        reliability_report = {
            'r2_statistics': {
                'count': len(r2_values),
                'min': min(r2_values) if r2_values else None,
                'max': max(r2_values) if r2_values else None,
                'mean': np.mean(r2_values) if r2_values else None,
                'negative_count': sum(1 for r2 in r2_values if r2 < 0)
            },
            'mse_statistics': {
                'count': len(mse_values),
                'min': min(mse_values),
                'max': max(mse_values),
                'mean': np.mean(mse_values),
                'zero_count': sum(1 for mse in mse_values if mse == 0)
            },
            'suspicious_results': []
        }
        
        # 检查可疑结果
        for result in self.successful_tests:
            issues = []
            
            # 检查MSE为0的情况
            if result['mse'] == 0:
                issues.append('MSE为0，可能存在过拟合或数据泄露')
            
            # 检查R²过高的情况
            if 'r2_score' in result and result['r2_score'] > 0.99:
                issues.append('R²过高，可能存在过拟合')
            
            # 检查R²为负的情况
            if 'r2_score' in result and result['r2_score'] < -1:
                issues.append('R²过低，模型性能极差')
            
            if issues:
                reliability_report['suspicious_results'].append({
                    'model': result['model_name'],
                    'dataset': result['dataset_name'],
                    'issues': issues,
                    'mse': result['mse'],
                    'r2': result.get('r2_score', 'N/A')
                })
        
        print(f"R²负值数量: {reliability_report['r2_statistics']['negative_count']}")
        print(f"MSE为0数量: {reliability_report['mse_statistics']['zero_count']}")
        print(f"可疑结果数量: {len(reliability_report['suspicious_results'])}")
        
        return reliability_report
    
    def generate_validation_report(self) -> str:
        """生成验证报告"""
        print("\n=== 生成验证报告 ===")
        
        # 执行所有检查
        consistency = self.check_data_consistency()
        parameter_fairness = self.check_parameter_fairness()
        training_fairness = self.check_training_fairness()
        performance_reliability = self.check_performance_reliability()
        
        # 生成报告
        report = f"""
# 测试结果可靠性和公平性验证报告

**验证时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**测试结果文件**: {self.results_file}
**成功测试数**: {len(self.successful_tests)}
**总测试数**: {len(self.results)}

## 1. 数据一致性评估

### 1.1 数据集覆盖情况
- 总数据集数: {consistency['total_datasets']}
- 缺失的模型-数据集组合: {len(consistency['missing_combinations'])}

### 1.2 模型覆盖详情
"""
        
        for model, info in consistency['model_coverage'].items():
            report += f"\n**{model}**:\n"
            report += f"- 覆盖率: {info['coverage_ratio']:.1%}\n"
            report += f"- 测试数据集: {', '.join(info['datasets'])}\n"
            if info['missing_datasets']:
                report += f"- 缺失数据集: {', '.join(info['missing_datasets'])}\n"
        
        report += f"""

## 2. 参数量公平性评估

### 2.1 各模型类型参数量范围
"""
        
        for model_type, range_str in parameter_fairness['parameter_ranges'].items():
            stats = parameter_fairness['type_statistics'][model_type]
            report += f"\n**{model_type}**: {range_str} (范围比: {stats['range_ratio']:.1f}x)\n"
        
        if parameter_fairness['outliers']:
            report += "\n### 2.2 参数量异常值\n"
            for outlier in parameter_fairness['outliers']:
                report += f"- {outlier['type']}: 参数量差异 {outlier['ratio']:.1f}倍\n"
        
        report += f"""

## 3. 训练配置公平性评估

### 3.1 训练时间统计
- 最短训练时间: {training_fairness['training_time_stats']['min']:.3f}s
- 最长训练时间: {training_fairness['training_time_stats']['max']:.3f}s
- 平均训练时间: {training_fairness['training_time_stats']['mean']:.3f}s
- 时间范围比: {training_fairness['training_time_stats']['range_ratio']:.1f}x
"""
        
        if training_fairness['time_outliers']:
            report += "\n### 3.2 训练时间异常值\n"
            for outlier in training_fairness['time_outliers']:
                report += f"- {outlier['model']} on {outlier['dataset']}: {outlier['time']:.3f}s (Z-score: {outlier['z_score']:.2f})\n"
        
        report += f"""

## 4. 性能指标可靠性评估

### 4.1 R²统计
- R²范围: {performance_reliability['r2_statistics']['min']:.3f} - {performance_reliability['r2_statistics']['max']:.3f}
- R²平均值: {performance_reliability['r2_statistics']['mean']:.3f}
- 负R²数量: {performance_reliability['r2_statistics']['negative_count']}

### 4.2 MSE统计
- MSE范围: {performance_reliability['mse_statistics']['min']:.6f} - {performance_reliability['mse_statistics']['max']:.6f}
- MSE平均值: {performance_reliability['mse_statistics']['mean']:.6f}
- MSE为0数量: {performance_reliability['mse_statistics']['zero_count']}
"""
        
        if performance_reliability['suspicious_results']:
            report += "\n### 4.3 可疑结果\n"
            for suspicious in performance_reliability['suspicious_results']:
                report += f"\n**{suspicious['model']} on {suspicious['dataset']}**:\n"
                for issue in suspicious['issues']:
                    report += f"- {issue}\n"
                report += f"- MSE: {suspicious['mse']:.6f}, R²: {suspicious['r2']}\n"
        
        # 总体评估
        total_issues = (
            len(consistency['missing_combinations']) +
            len(parameter_fairness['outliers']) +
            len(training_fairness['time_outliers']) +
            len(performance_reliability['suspicious_results'])
        )
        
        if total_issues == 0:
            fairness_level = "优秀"
            recommendation = "测试结果具有高度可靠性和公平性，可以直接用于模型对比分析。"
        elif total_issues <= 5:
            fairness_level = "良好"
            recommendation = "测试结果基本可靠，建议关注上述问题并在分析时加以说明。"
        elif total_issues <= 10:
            fairness_level = "一般"
            recommendation = "测试结果存在一些问题，建议修复主要问题后重新测试。"
        else:
            fairness_level = "较差"
            recommendation = "测试结果存在较多问题，强烈建议重新设计测试方案。"
        
        report += f"""

## 5. 总体评估

### 5.1 公平性等级
**{fairness_level}** (发现 {total_issues} 个问题)

### 5.2 建议
{recommendation}

### 5.3 改进建议
1. 确保所有模型在相同数据集上进行测试
2. 控制模型参数量在合理范围内
3. 统一训练配置和超参数
4. 检查异常的性能指标
5. 增加多次运行的统计分析
"""
        
        return report
    
    def save_validation_report(self, report: str):
        """保存验证报告"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_file = Path(self.results_file).parent / f"validation_report_{timestamp}.md"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n验证报告已保存到: {report_file}")
        return report_file

def parse_text_report(report_file: str) -> List[Dict[str, Any]]:
    """解析文本报告文件"""
    results = []
    
    with open(report_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析成功的测试结果
    lines = content.split('\n')
    current_dataset = None
    
    for i, line in enumerate(lines):
        line = line.strip()
        
        # 检测数据集名称
        if '数据集' in line and '结果:' in line:
            # 提取数据集名称，如 "数据集 time_series_long 结果:"
            parts = line.split()
            if len(parts) >= 2:
                current_dataset = parts[1]
        
        # 解析模型结果行
        elif 'MSE:' in line and 'MAE:' in line and 'R²:' in line and current_dataset:
            # 查找前一行获取模型信息
            if i > 0:
                prev_line = lines[i-1].strip()
                # 解析模型行，如 "1. pinn_2d (pinn)"
                if '. ' in prev_line and '(' in prev_line and ')' in prev_line:
                    model_part = prev_line.split('. ', 1)[1]  # 去掉序号
                    model_name = model_part.split(' (')[0]  # 提取模型名
                    model_type = model_part.split('(')[1].split(')')[0]  # 提取模型类型
                    
                    # 解析指标
                    try:
                        # 使用正则表达式或字符串分割解析指标
                        import re
                        
                        mse_match = re.search(r'MSE: ([0-9.]+)', line)
                        mae_match = re.search(r'MAE: ([0-9.]+)', line)
                        r2_match = re.search(r'R²: ([0-9.-]+)', line)
                        
                        if mse_match and mae_match and r2_match:
                            # 查找下一行获取参数量和训练时间
                            param_count = 1000000  # 默认值
                            training_time = 10.0   # 默认值
                            
                            if i + 1 < len(lines):
                                next_line = lines[i+1].strip()
                                param_match = re.search(r'参数量: ([0-9,]+)', next_line)
                                time_match = re.search(r'训练时间: ([0-9.]+)s', next_line)
                                
                                if param_match:
                                    param_count = int(param_match.group(1).replace(',', ''))
                                if time_match:
                                    training_time = float(time_match.group(1))
                            
                            result = {
                                'model_name': model_name,
                                'dataset_name': current_dataset,
                                'model_type': model_type,
                                'mse': float(mse_match.group(1)),
                                'mae': float(mae_match.group(1)),
                                'r2_score': float(r2_match.group(1)),
                                'parameter_count': param_count,
                                'training_time': training_time,
                                'success': True
                            }
                            
                            results.append(result)
                    except (ValueError, AttributeError) as e:
                        print(f"解析错误: {e}, 行: {line}")
                        continue
    
    return results

def main():
    """主函数"""
    # 查找最新的文本报告文件
    results_dir = Path(__file__).parent
    txt_files = list(results_dir.glob("unified_comparison_report_*.txt"))
    
    if not txt_files:
        print("未找到测试结果文件")
        return
    
    # 使用最新的结果文件
    latest_file = max(txt_files, key=lambda x: x.stat().st_mtime)
    print(f"使用结果文件: {latest_file}")
    
    # 解析文本报告
    results = parse_text_report(str(latest_file))
    
    if not results:
        print("未能解析到有效的测试结果")
        return
    
    # 创建临时验证器
    class SimpleValidator:
        def __init__(self, results):
            self.results = results
            self.successful_tests = [r for r in results if r.get('success', False)]
            
        def generate_simple_report(self):
            print(f"\n=== 简化验证报告 ===")
            print(f"成功测试数: {len(self.successful_tests)}")
            print(f"总测试数: {len(self.results)}")
            
            # 统计模型和数据集
            models = set(r['model_name'] for r in self.successful_tests)
            datasets = set(r['dataset_name'] for r in self.successful_tests)
            
            print(f"\n模型数量: {len(models)}")
            print(f"数据集数量: {len(datasets)}")
            print(f"模型列表: {', '.join(sorted(models))}")
            print(f"数据集列表: {', '.join(sorted(datasets))}")
            
            # 检查覆盖率
            expected_combinations = len(models) * len(datasets)
            actual_combinations = len(self.successful_tests)
            coverage = actual_combinations / expected_combinations if expected_combinations > 0 else 0
            
            print(f"\n预期组合数: {expected_combinations}")
            print(f"实际成功组合数: {actual_combinations}")
            print(f"覆盖率: {coverage:.1%}")
            
            # 性能统计
            mse_values = [r['mse'] for r in self.successful_tests]
            r2_values = [r['r2_score'] for r in self.successful_tests]
            
            print(f"\nMSE范围: {min(mse_values):.6f} - {max(mse_values):.6f}")
            print(f"R²范围: {min(r2_values):.3f} - {max(r2_values):.3f}")
            print(f"负R²数量: {sum(1 for r2 in r2_values if r2 < 0)}")
            
            # 生成简单报告
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_content = f"""
# 测试结果验证报告（简化版）

**验证时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**成功测试数**: {len(self.successful_tests)}
**总测试数**: {len(self.results)}
**覆盖率**: {coverage:.1%}

## 基本统计

- 模型数量: {len(models)}
- 数据集数量: {len(datasets)}
- 预期组合数: {expected_combinations}
- 实际成功组合数: {actual_combinations}

## 性能指标范围

- MSE: {min(mse_values):.6f} - {max(mse_values):.6f}
- R²: {min(r2_values):.3f} - {max(r2_values):.3f}
- 负R²数量: {sum(1 for r2 in r2_values if r2 < 0)}

## 评估结论

{'测试覆盖率良好，结果基本可靠。' if coverage > 0.8 else '测试覆盖率不足，建议补充测试。'}
"""
            
            report_file = results_dir / f"simple_validation_report_{timestamp}.md"
            with open(report_file, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            print(f"\n简化验证报告已保存到: {report_file}")
            return report_file
    
    # 创建验证器并生成报告
    validator = SimpleValidator(results)
    validator.generate_simple_report()
    
    print("\n验证完成！")

if __name__ == "__main__":
    main()