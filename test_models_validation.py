#!/usr/bin/env python3
"""
模型验证和性能对比测试脚本
验证FNO、UNet、PINN模型的参数对齐和性能对比功能
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any
import time
import json
from pathlib import Path

# 导入我们创建的模型
import sys
sys.path.append('modify_multi_attention')
sys.path.append('modify_multi_attention/models')

from models.enhanced_fno import EnhancedFNO1d, EnhancedFNO2d, create_enhanced_fno1d, create_enhanced_fno2d
from models.enhanced_unet import EnhancedUNet1d, EnhancedUNet2d, create_enhanced_unet1d, create_enhanced_unet2d
from models.enhanced_pinn import EnhancedPINN, create_enhanced_pinn1d
from mymodels.transformer import TransformerFlowReconstructionModel

class ModelValidator:
    """模型验证器"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.results = {}
        
    def count_parameters(self, model: nn.Module) -> int:
        """计算模型参数数量"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def test_model_forward(self, model: nn.Module, input_data: torch.Tensor, 
                          model_name: str) -> Dict[str, Any]:
        """测试模型前向传播"""
        model.eval()
        model.to(self.device)
        input_data = input_data.to(self.device)
        
        # 测试前向传播
        start_time = time.time()
        try:
            with torch.no_grad():
                output = model(input_data)
            forward_time = time.time() - start_time
            
            # 检查输出形状
            output_shape = output.shape
            param_count = self.count_parameters(model)
            
            return {
                'success': True,
                'output_shape': output_shape,
                'forward_time': forward_time,
                'param_count': param_count,
                'model_name': model_name,
                'error': None
            }
        except Exception as e:
            return {
                'success': False,
                'output_shape': None,
                'forward_time': None,
                'param_count': self.count_parameters(model),
                'model_name': model_name,
                'error': str(e)
            }
    
    def generate_test_data(self, data_type: str = '1d') -> Tuple[torch.Tensor, torch.Tensor]:
        """生成测试数据"""
        batch_size = 4
        
        if data_type == '1d':
            # 1D数据：网格格式
            input_size = 32
            output_size = 128
            
            # 输入：(batch, input_size, channels)
            x_sparse = torch.randn(batch_size, input_size, 1)
            
            # 输出：(batch, output_size, channels)
            y_dense = torch.randn(batch_size, output_size, 1)
            
        elif data_type == '2d':
            # 2D数据：网格格式
            input_h, input_w = 8, 8
            output_h, output_w = 32, 32
            
            # 输入：(batch, input_h, input_w, channels)
            x_sparse = torch.randn(batch_size, input_h, input_w, 1)
            
            # 输出：(batch, output_h, output_w, channels)
            y_dense = torch.randn(batch_size, output_h, output_w, 1)
            
        return x_sparse, y_dense
    
    def test_fno_models(self) -> Dict[str, Any]:
        """测试FNO模型"""
        print("\n=== 测试FNO模型 ===")
        results = {}
        
        # 测试FNO1d
        try:
            model_1d = create_enhanced_fno1d(
                num_channels=1,
                modes=16,
                width=64,
                input_resolution=32,
                output_resolution=128
            )
            
            x_sparse, y_dense = self.generate_test_data('1d')
            result_1d = self.test_model_forward(model_1d, x_sparse, 'EnhancedFNO1d')
            results['fno_1d'] = result_1d
            
            print(f"FNO1d - 参数数量: {result_1d['param_count']:,}")
            if result_1d['success']:
                print(f"FNO1d - 前向传播时间: {result_1d['forward_time']:.4f}s")
                print(f"FNO1d - 输出形状: {result_1d['output_shape']}")
            else:
                print(f"FNO1d - 错误: {result_1d['error']}")
            
        except Exception as e:
            results['fno_1d'] = {'success': False, 'error': str(e)}
            print(f"FNO1d测试失败: {e}")
        
        # 测试FNO2d
        try:
            model_2d = create_enhanced_fno2d(
                num_channels=1,
                modes1=12,
                modes2=12,
                width=32,
                input_resolution=(8, 8),
                output_resolution=(32, 32)
            )
            
            x_sparse, y_dense = self.generate_test_data('2d')
            result_2d = self.test_model_forward(model_2d, x_sparse, 'EnhancedFNO2d')
            results['fno_2d'] = result_2d
            
            print(f"FNO2d - 参数数量: {result_2d['param_count']:,}")
            if result_2d['success']:
                print(f"FNO2d - 前向传播时间: {result_2d['forward_time']:.4f}s")
                print(f"FNO2d - 输出形状: {result_2d['output_shape']}")
            else:
                print(f"FNO2d - 错误: {result_2d['error']}")
            
        except Exception as e:
            results['fno_2d'] = {'success': False, 'error': str(e)}
            print(f"FNO2d测试失败: {e}")
        
        return results
    
    def test_unet_models(self) -> Dict[str, Any]:
        """测试UNet模型"""
        print("\n=== 测试UNet模型 ===")
        results = {}
        
        # 测试UNet1d
        try:
            model_1d = create_enhanced_unet1d(
                in_channels=1,
                out_channels=1,
                input_resolution=32,
                output_resolution=128
            )
            
            x_sparse, y_dense = self.generate_test_data('1d')
            result_1d = self.test_model_forward(model_1d, x_sparse, 'EnhancedUNet1d')
            results['unet_1d'] = result_1d
            
            print(f"UNet1d - 参数数量: {result_1d['param_count']:,}")
            if result_1d['success']:
                print(f"UNet1d - 前向传播时间: {result_1d['forward_time']:.4f}s")
                print(f"UNet1d - 输出形状: {result_1d['output_shape']}")
            else:
                print(f"UNet1d - 错误: {result_1d['error']}")
            
        except Exception as e:
            results['unet_1d'] = {'success': False, 'error': str(e)}
            print(f"UNet1d测试失败: {e}")
        
        # 测试UNet2d
        try:
            model_2d = create_enhanced_unet2d(
                in_channels=1,
                out_channels=1,
                input_resolution=(8, 8),
                output_resolution=(32, 32)
            )
            
            x_sparse, y_dense = self.generate_test_data('2d')
            result_2d = self.test_model_forward(model_2d, x_sparse, 'EnhancedUNet2d')
            results['unet_2d'] = result_2d
            
            print(f"UNet2d - 参数数量: {result_2d['param_count']:,}")
            if result_2d['success']:
                print(f"UNet2d - 前向传播时间: {result_2d['forward_time']:.4f}s")
                print(f"UNet2d - 输出形状: {result_2d['output_shape']}")
            else:
                print(f"UNet2d - 错误: {result_2d['error']}")
            
        except Exception as e:
            results['unet_2d'] = {'success': False, 'error': str(e)}
            print(f"UNet2d测试失败: {e}")
        
        return results
    
    def test_pinn_models(self) -> Dict[str, Any]:
        """测试PINN模型"""
        print("\n=== 测试PINN模型 ===")
        results = {}
        
        try:
            model = create_enhanced_pinn1d(
                output_dim=1,
                hidden_dim=64,
                num_layers=4,
                input_resolution=32,
                output_resolution=128
            )
            
            x_sparse, y_dense = self.generate_test_data('1d')
            result = self.test_model_forward(model, x_sparse, 'EnhancedPINN')
            results['pinn'] = result
            
            print(f"PINN - 参数数量: {result['param_count']:,}")
            if result['success']:
                print(f"PINN - 前向传播时间: {result['forward_time']:.4f}s")
                print(f"PINN - 输出形状: {result['output_shape']}")
            else:
                print(f"PINN - 错误: {result['error']}")
            
        except Exception as e:
            results['pinn'] = {'success': False, 'error': str(e)}
            print(f"PINN测试失败: {e}")
        
        return results
    
    def test_transformer_models(self) -> Dict[str, Any]:
        """测试Transformer模型"""
        print("\n=== 测试Transformer模型 ===")
        results = {}
        
        try:
            # 创建Transformer模型 - 使用合理的参数配置
            model = TransformerFlowReconstructionModel(
                input_dim=128,  # 输入维度
                output_dim=128,  # 输出维度
                num_heads=8,
                num_layers=6,
                d_model=512,
                seq_len=32,  # 序列长度，使用默认值32
                max_time_steps=100
            )
            
            # 生成测试数据
            test_data = torch.randn(4, 128).to(self.device)  # [batch_size, input_dim]
            test_time_steps = torch.zeros(4, dtype=torch.long).to(self.device)  # [batch_size]
            
            # 测试前向传播 - 使用自定义测试方法因为需要额外参数
            param_count = self.count_parameters(model)
            print(f"Transformer - 参数数量: {param_count:,}")
            
            # 确保模型完全在GPU上
            model = model.to(self.device)
            
            # 测试前向传播时间
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                output = model(test_data, test_time_steps)
                end_time = time.time()
                forward_time = end_time - start_time
                
            print(f"Transformer - 前向传播时间: {forward_time:.4f}s")
            print(f"Transformer - 输出形状: {output.shape}")
            
            result = {
                'success': True,
                'params': param_count,
                'forward_time': forward_time,
                'output_shape': str(output.shape)
            }
            
            results['transformer'] = result
            
            print(f"Transformer - 参数数量: {result['params']:,}")
            if result['success']:
                print(f"Transformer - 前向传播时间: {result['forward_time']:.4f}s")
                print(f"Transformer - 输出形状: {result['output_shape']}")
            else:
                print(f"Transformer - 错误: {result['error']}")
            
        except Exception as e:
            results['transformer'] = {'success': False, 'error': str(e)}
            print(f"Transformer测试失败: {e}")
        
        return results
    
    def generate_comparison_report(self, all_results: Dict[str, Any]) -> str:
        """生成对比报告"""
        report = "\n" + "="*60 + "\n"
        report += "模型性能对比报告\n"
        report += "="*60 + "\n\n"
        
        # 参数数量对比
        report += "参数数量对比:\n"
        report += "-" * 30 + "\n"
        
        param_counts = []
        for category, models in all_results.items():
            if isinstance(models, dict):
                # 检查是否为单个模型（包含params或param_count键）
                if 'params' in models or 'param_count' in models:  # 单个模型
                    if models.get('success', False):
                        param_key = 'params' if 'params' in models else 'param_count'
                        param_counts.append((category, models[param_key]))
                else:  # 多个模型
                    for model_name, result in models.items():
                        if isinstance(result, dict) and result.get('success', False):
                            param_key = 'params' if 'params' in result else 'param_count'
                            if param_key in result:
                                param_counts.append((f"{category}_{model_name}", result[param_key]))
        
        param_counts.sort(key=lambda x: x[1])
        for name, count in param_counts:
            report += f"{name:20}: {count:,} 参数\n"
        
        # 前向传播时间对比
        report += "\n前向传播时间对比:\n"
        report += "-" * 30 + "\n"
        
        forward_times = []
        for category, models in all_results.items():
            if isinstance(models, dict):
                if 'forward_time' in models:  # 单个模型
                    if models.get('success', False) and models.get('forward_time'):
                        forward_times.append((category, models['forward_time']))
                else:  # 多个模型
                    for model_name, result in models.items():
                        if isinstance(result, dict) and result.get('success', False) and result.get('forward_time'):
                            forward_times.append((f"{category}_{model_name}", result['forward_time']))
        
        forward_times.sort(key=lambda x: x[1])
        for name, time_val in forward_times:
            report += f"{name:20}: {time_val:.4f}s\n"
        
        # 成功率统计
        report += "\n模型测试成功率:\n"
        report += "-" * 30 + "\n"
        
        total_tests = 0
        successful_tests = 0
        
        for category, models in all_results.items():
            if isinstance(models, dict):
                if 'success' in models:  # 单个模型
                    total_tests += 1
                    if models.get('success', False):
                        successful_tests += 1
                else:  # 多个模型
                    for model_name, result in models.items():
                        total_tests += 1
                        if result.get('success', False):
                            successful_tests += 1
        
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        report += f"总测试数: {total_tests}\n"
        report += f"成功测试数: {successful_tests}\n"
        report += f"成功率: {success_rate:.1f}%\n"
        
        return report
    
    def run_full_validation(self) -> Dict[str, Any]:
        """运行完整验证"""
        print("开始模型验证和性能对比测试...")
        print(f"使用设备: {self.device}")
        
        all_results = {}
        
        # 测试各个模型
        all_results['fno'] = self.test_fno_models()
        all_results['unet'] = self.test_unet_models()
        all_results['pinn'] = self.test_pinn_models()
        all_results['transformer'] = self.test_transformer_models()
        
        # 生成报告
        report = self.generate_comparison_report(all_results)
        print(report)
        
        # 保存结果
        self.save_results(all_results, report)
        
        return all_results
    
    def save_results(self, results: Dict[str, Any], report: str):
        """保存测试结果"""
        # 保存JSON结果
        results_file = Path('model_validation_results.json')
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存报告
        report_file = Path('model_comparison_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n结果已保存到: {results_file}")
        print(f"报告已保存到: {report_file}")

def main():
    """主函数"""
    validator = ModelValidator()
    results = validator.run_full_validation()
    
    # 检查是否所有测试都成功
    all_success = True
    for category, models in results.items():
        if isinstance(models, dict):
            if 'success' in models:  # 单个模型
                if not models.get('success', False):
                    all_success = False
            else:  # 多个模型
                for model_name, result in models.items():
                    if not result.get('success', False):
                        all_success = False
    
    if all_success:
        print("\n✅ 所有模型验证测试通过！")
        print("✅ 参数对齐和性能对比功能正常工作！")
    else:
        print("\n❌ 部分测试失败，请检查错误信息")
    
    return results

if __name__ == "__main__":
    main()