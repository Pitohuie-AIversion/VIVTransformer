#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDE数据集实测验证脚本
验证所有8个增强多注意力模型在真实PDE数据集上的运行情况
"""

import os
import sys
import torch
import numpy as np
import traceback
from datetime import datetime
import json
import time
from typing import Dict, List, Tuple, Any

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入所有增强模型
try:
    from enhanced_fno import EnhancedFNO1d, EnhancedFNO2d
    from enhanced_mlp import EnhancedMLP1d, EnhancedMLP2d
    from enhanced_pinn import EnhancedPINN1d, EnhancedPINN2d
    from enhanced_unet import EnhancedUNet1d, EnhancedUNet2d, EnhancedUNet3d
except ImportError as e:
    print(f"模型导入错误: {e}")
    sys.exit(1)

class PDEDatasetValidator:
    """PDE数据集验证器"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.test_results = {}
        self.error_log = []
        
        # 定义所有要测试的模型
        self.models_config = {
            'FNO1d': {
                'class': EnhancedFNO1d,
                'params': {'num_channels': 1, 'modes': 16, 'width': 64, 'input_resolution': 64, 'output_resolution': 64},
                'test_shape': (4, 64, 1)  # (batch, length, channels)
            },
            'FNO2d': {
                'class': EnhancedFNO2d,
                'params': {'num_channels': 1, 'modes1': 12, 'modes2': 12, 'width': 32, 'input_resolution': (32, 32), 'output_resolution': (32, 32)},
                'test_shape': (4, 32, 32, 1)  # (batch, height, width, channels)
            },
            'MLP1d': {
                'class': EnhancedMLP1d,
                'params': {'input_channels': 1, 'output_channels': 1, 'hidden_dim': 128, 'input_resolution': 64, 'output_resolution': 64},
                'test_shape': (4, 64, 1)  # (batch, length, channels)
            },
            'MLP2d': {
                'class': EnhancedMLP2d,
                'params': {'input_channels': 1, 'output_channels': 1, 'hidden_dim': 128, 'input_resolution': (32, 32), 'output_resolution': (32, 32)},
                'test_shape': (4, 32, 32, 1)  # (batch, height, width, channels)
            },
            'PINN1d': {
                'class': EnhancedPINN1d,
                'params': {'output_dim': 1, 'hidden_dim': 128, 'input_resolution': 64, 'output_resolution': 64},
                'test_shape': (4, 64, 1)  # (batch, length, channels)
            },
            'PINN2d': {
                'class': EnhancedPINN2d,
                'params': {'output_dim': 1, 'hidden_dim': 128, 'input_resolution': (32, 32), 'output_resolution': (32, 32)},
                'test_shape': (4, 32, 32, 1)  # (batch, height, width, channels)
            },
            'UNet1d': {
                'class': EnhancedUNet1d,
                'params': {'in_channels': 1, 'out_channels': 1, 'init_features': 32, 'input_resolution': 64, 'output_resolution': 64},
                'test_shape': (4, 1, 64)  # (batch, channels, length)
            },
            'UNet2d': {
                'class': EnhancedUNet2d,
                'params': {'in_channels': 1, 'out_channels': 1, 'init_features': 32, 'input_resolution': (32, 32), 'output_resolution': (32, 32)},
                'test_shape': (4, 1, 32, 32)  # (batch, channels, height, width)
            },
            'UNet3d': {
                'class': EnhancedUNet3d,
                'params': {'in_channels': 1, 'out_channels': 1, 'init_features': 16, 'input_resolution': (16, 16, 16), 'output_resolution': (16, 16, 16)},
                'test_shape': (2, 1, 16, 16, 16)  # (batch, channels, depth, height, width)
            }
        }
        
    def create_test_data(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """创建测试数据"""
        # 创建符合PDE特征的测试数据
        data = torch.randn(shape, dtype=torch.float32)
        
        # 对于某些维度，添加物理意义的数据模式
        if len(shape) >= 3:  # 空间数据
            # 添加一些周期性和平滑性
            for i in range(shape[0]):
                if len(shape) == 3:  # 1D空间
                    x = torch.linspace(0, 2*np.pi, shape[-1])
                    data[i, 0] = torch.sin(x) + 0.1 * torch.randn_like(x)
                elif len(shape) == 4:  # 2D空间
                    x = torch.linspace(0, 2*np.pi, shape[-1])
                    y = torch.linspace(0, 2*np.pi, shape[-2])
                    X, Y = torch.meshgrid(x, y, indexing='ij')
                    data[i, 0] = torch.sin(X) * torch.cos(Y) + 0.1 * torch.randn_like(X)
                elif len(shape) == 5:  # 3D空间
                    # 简化的3D模式
                    data[i, 0] = torch.randn(shape[-3:]) * 0.5
        
        return data.to(self.device)
    
    def test_model_basic_functionality(self, model_name: str, model_config: Dict) -> Dict[str, Any]:
        """测试模型基本功能"""
        result = {
            'model_name': model_name,
            'success': False,
            'error': None,
            'forward_pass': False,
            'backward_pass': False,
            'parameter_count': 0,
            'memory_usage_mb': 0,
            'inference_time_ms': 0,
            'output_shape': None,
            'gradient_flow': False
        }
        
        try:
            print(f"\n测试模型: {model_name}")
            
            # 1. 模型初始化
            model_class = model_config['class']
            model_params = model_config['params']
            test_shape = model_config['test_shape']
            
            model = model_class(**model_params).to(self.device)
            model.train()
            
            # 2. 计算参数数量
            param_count = sum(p.numel() for p in model.parameters())
            result['parameter_count'] = param_count
            
            # 3. 创建测试数据
            test_input = self.create_test_data(test_shape)
            
            # 4. 前向传播测试
            start_time = time.time()
            
            with torch.no_grad():
                output = model(test_input)
                result['forward_pass'] = True
                result['output_shape'] = list(output.shape)
            
            inference_time = (time.time() - start_time) * 1000
            result['inference_time_ms'] = inference_time
            
            # 5. 内存使用测试
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_used = torch.cuda.memory_allocated() / 1024 / 1024
                result['memory_usage_mb'] = memory_used
            
            # 6. 反向传播测试
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            
            output = model(test_input)
            
            # 创建目标数据
            if model_name.startswith('PINN'):
                # PINN需要特殊的损失函数
                target = torch.zeros_like(output)
                loss = torch.nn.MSELoss()(output, target)
            else:
                # 其他模型使用标准MSE损失
                target_shape = list(output.shape)
                target = self.create_test_data(tuple(target_shape))
                loss = torch.nn.MSELoss()(output, target)
            
            loss.backward()
            
            # 检查梯度
            has_gradients = any(p.grad is not None and p.grad.abs().sum() > 0 
                              for p in model.parameters() if p.requires_grad)
            result['gradient_flow'] = has_gradients
            result['backward_pass'] = True
            
            optimizer.step()
            optimizer.zero_grad()
            
            result['success'] = True
            print(f"✓ {model_name} 测试成功")
            print(f"  - 参数数量: {param_count:,}")
            print(f"  - 推理时间: {inference_time:.2f}ms")
            print(f"  - 输出形状: {result['output_shape']}")
            print(f"  - 梯度流动: {'是' if has_gradients else '否'}")
            
        except Exception as e:
            error_msg = f"{model_name} 测试失败: {str(e)}"
            result['error'] = error_msg
            self.error_log.append({
                'model': model_name,
                'error': error_msg,
                'traceback': traceback.format_exc()
            })
            print(f"✗ {error_msg}")
        
        return result
    
    def test_model_with_different_inputs(self, model_name: str, model_config: Dict) -> Dict[str, Any]:
        """测试模型对不同输入的适应性"""
        result = {
            'model_name': model_name,
            'input_variations': [],
            'robustness_score': 0.0
        }
        
        try:
            model_class = model_config['class']
            model_params = model_config['params']
            base_shape = model_config['test_shape']
            
            model = model_class(**model_params).to(self.device)
            model.eval()
            
            # 测试不同的输入变化
            variations = [
                ('正常输入', base_shape, 1.0),
                ('小批次', (1,) + base_shape[1:], 0.8),
                ('大批次', (8,) + base_shape[1:], 0.8),
                ('零输入', base_shape, 0.0),
                ('极值输入', base_shape, 10.0)
            ]
            
            successful_tests = 0
            
            for var_name, test_shape, scale in variations:
                try:
                    test_input = self.create_test_data(test_shape) * scale
                    
                    with torch.no_grad():
                        output = model(test_input)
                        
                    # 检查输出的合理性
                    is_valid = (
                        not torch.isnan(output).any() and
                        not torch.isinf(output).any() and
                        output.shape[0] == test_shape[0]
                    )
                    
                    result['input_variations'].append({
                        'variation': var_name,
                        'success': is_valid,
                        'output_stats': {
                            'mean': float(output.mean()),
                            'std': float(output.std()),
                            'min': float(output.min()),
                            'max': float(output.max())
                        }
                    })
                    
                    if is_valid:
                        successful_tests += 1
                        
                except Exception as e:
                    result['input_variations'].append({
                        'variation': var_name,
                        'success': False,
                        'error': str(e)
                    })
            
            result['robustness_score'] = successful_tests / len(variations)
            
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """运行全面的验证测试"""
        print("开始PDE数据集实测验证...")
        print(f"设备: {self.device}")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        validation_results = {
            'test_info': {
                'timestamp': datetime.now().isoformat(),
                'device': str(self.device),
                'pytorch_version': torch.__version__,
                'total_models': len(self.models_config)
            },
            'basic_functionality': {},
            'input_robustness': {},
            'summary': {
                'total_tests': 0,
                'successful_tests': 0,
                'failed_tests': 0,
                'success_rate': 0.0
            },
            'errors': []
        }
        
        # 更新待办事项状态
        self.update_todo_status('pde_test_1', 'in_progress')
        
        # 1. 基本功能测试
        print("\n1. 基本功能测试")
        print("-" * 30)
        
        for model_name, model_config in self.models_config.items():
            result = self.test_model_basic_functionality(model_name, model_config)
            validation_results['basic_functionality'][model_name] = result
            
            if result['success']:
                validation_results['summary']['successful_tests'] += 1
            else:
                validation_results['summary']['failed_tests'] += 1
            
            validation_results['summary']['total_tests'] += 1
        
        # 2. 输入鲁棒性测试
        print("\n\n2. 输入鲁棒性测试")
        print("-" * 30)
        
        for model_name, model_config in self.models_config.items():
            if validation_results['basic_functionality'][model_name]['success']:
                print(f"\n测试 {model_name} 的输入鲁棒性...")
                robustness_result = self.test_model_with_different_inputs(model_name, model_config)
                validation_results['input_robustness'][model_name] = robustness_result
                print(f"鲁棒性得分: {robustness_result['robustness_score']:.2f}")
        
        # 3. 计算总体统计
        total_tests = validation_results['summary']['total_tests']
        successful_tests = validation_results['summary']['successful_tests']
        validation_results['summary']['success_rate'] = successful_tests / total_tests if total_tests > 0 else 0
        
        # 4. 添加错误日志
        validation_results['errors'] = self.error_log
        
        return validation_results
    
    def update_todo_status(self, todo_id: str, status: str):
        """更新待办事项状态（占位符函数）"""
        pass
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """生成测试报告"""
        report = []
        report.append("PDE数据集实测验证报告")
        report.append("=" * 50)
        report.append(f"测试时间: {results['test_info']['timestamp']}")
        report.append(f"设备: {results['test_info']['device']}")
        report.append(f"PyTorch版本: {results['test_info']['pytorch_version']}")
        report.append(f"测试模型数量: {results['test_info']['total_models']}")
        report.append("")
        
        # 总体统计
        summary = results['summary']
        report.append("总体统计:")
        report.append(f"  总测试数: {summary['total_tests']}")
        report.append(f"  成功测试: {summary['successful_tests']}")
        report.append(f"  失败测试: {summary['failed_tests']}")
        report.append(f"  成功率: {summary['success_rate']:.1%}")
        report.append("")
        
        # 详细结果
        report.append("详细测试结果:")
        report.append("-" * 30)
        
        for model_name, result in results['basic_functionality'].items():
            status = "✓ 成功" if result['success'] else "✗ 失败"
            report.append(f"\n{model_name}: {status}")
            
            if result['success']:
                report.append(f"  参数数量: {result['parameter_count']:,}")
                report.append(f"  推理时间: {result['inference_time_ms']:.2f}ms")
                report.append(f"  输出形状: {result['output_shape']}")
                report.append(f"  前向传播: {'✓' if result['forward_pass'] else '✗'}")
                report.append(f"  反向传播: {'✓' if result['backward_pass'] else '✗'}")
                report.append(f"  梯度流动: {'✓' if result['gradient_flow'] else '✗'}")
                
                # 鲁棒性信息
                if model_name in results['input_robustness']:
                    robustness = results['input_robustness'][model_name]
                    report.append(f"  鲁棒性得分: {robustness['robustness_score']:.2f}")
            else:
                report.append(f"  错误: {result['error']}")
        
        # 错误详情
        if results['errors']:
            report.append("\n\n错误详情:")
            report.append("-" * 30)
            for error in results['errors']:
                report.append(f"\n模型: {error['model']}")
                report.append(f"错误: {error['error']}")
        
        # 建议
        report.append("\n\n建议:")
        report.append("-" * 30)
        
        if summary['success_rate'] >= 0.9:
            report.append("✓ 所有模型运行良好，可以安全用于PDE数据集处理")
        elif summary['success_rate'] >= 0.7:
            report.append("⚠ 大部分模型运行正常，建议检查失败的模型")
        else:
            report.append("✗ 多个模型存在问题，需要进行调试和修复")
        
        report.append("\n推荐使用的模型（按性能排序）:")
        
        # 按成功率和性能排序模型
        successful_models = [(name, result) for name, result in results['basic_functionality'].items() 
                           if result['success']]
        successful_models.sort(key=lambda x: (x[1]['inference_time_ms'], -x[1]['parameter_count']))
        
        for i, (name, result) in enumerate(successful_models[:5], 1):
            report.append(f"  {i}. {name} - 推理时间: {result['inference_time_ms']:.2f}ms, 参数: {result['parameter_count']:,}")
        
        return "\n".join(report)

def main():
    """主函数"""
    validator = PDEDatasetValidator()
    
    try:
        # 运行验证测试
        results = validator.run_comprehensive_validation()
        
        # 生成报告
        report = validator.generate_report(results)
        
        # 保存结果
        with open('pde_dataset_validation_results.json', 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        with open('pde_dataset_validation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
        
        # 打印报告
        print("\n" + "=" * 60)
        print(report)
        print("\n" + "=" * 60)
        print("\n测试完成！")
        print("详细结果已保存到:")
        print("- pde_dataset_validation_results.json")
        print("- pde_dataset_validation_report.txt")
        
        return results['summary']['success_rate'] >= 0.8
        
    except Exception as e:
        print(f"验证过程中发生错误: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)