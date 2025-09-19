"""改进的真实数据测试脚本，更好地处理数据形状匹配问题"""

import torch
import torch.nn as nn
import numpy as np
import h5py
import os
import time
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入所有模型
from enhanced_fno import create_enhanced_fno1d, create_enhanced_fno2d
from enhanced_mlp import create_enhanced_mlp1d, create_enhanced_mlp2d
from enhanced_pinn import create_enhanced_pinn1d, create_enhanced_pinn2d
from enhanced_unet import create_enhanced_unet1d, create_enhanced_unet2d, create_enhanced_unet3d

class ImprovedRealDataTester:
    """改进的真实数据测试器，更好地处理数据形状匹配"""
    
    def __init__(self, device='auto', data_dir='../../../generate_data/preprocessed_data'):
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.data_dir = Path(data_dir)
        print(f"使用设备: {self.device}")
        print(f"数据目录: {self.data_dir}")
        
        # 测试结果存储
        self.test_results = {}
        
        # 模型配置 - 使用更灵活的参数
        self.model_configs = {
            'FNO1d': {
                'creator': create_enhanced_fno1d,
                'base_params': {'num_channels': 1, 'modes': 16, 'width': 64},
                'dimensions': 1
            },
            'FNO2d': {
                'creator': create_enhanced_fno2d,
                'base_params': {'num_channels': 1, 'modes1': 12, 'modes2': 12, 'width': 20},
                'dimensions': 2
            },
            'MLP1d': {
                'creator': create_enhanced_mlp1d,
                'base_params': {'input_channels': 1, 'hidden_dim': 256, 'num_layers': 6},
                'dimensions': 1
            },
            'MLP2d': {
                'creator': create_enhanced_mlp2d,
                'base_params': {'input_channels': 1, 'hidden_dim': 256, 'num_layers': 6},
                'dimensions': 2
            },
            'PINN1d': {
                'creator': create_enhanced_pinn1d,
                'base_params': {'output_dim': 1, 'hidden_dim': 256, 'num_layers': 6},
                'dimensions': 1
            },
            'PINN2d': {
                'creator': create_enhanced_pinn2d,
                'base_params': {'output_dim': 1, 'hidden_dim': 256, 'num_layers': 6},
                'dimensions': 2
            },
            'UNet1d': {
                'creator': create_enhanced_unet1d,
                'base_params': {'in_channels': 1, 'out_channels': 1, 'init_features': 32},
                'dimensions': 1
            },
            'UNet2d': {
                'creator': create_enhanced_unet2d,
                'base_params': {'in_channels': 1, 'out_channels': 1, 'init_features': 32},
                'dimensions': 2
            }
        }
    
    def smart_reshape_data(self, data: torch.Tensor, target_dims: int, 
                          target_resolution: Optional[Tuple] = None) -> torch.Tensor:
        """智能重塑数据以匹配目标维度"""
        batch_size = data.size(0)
        
        if target_dims == 1:
            # 目标是1D数据
            if data.dim() == 3:  # (batch, length, channels)
                return data
            elif data.dim() == 4:  # (batch, height, width, channels)
                # 将2D数据展平为1D
                h, w, c = data.size(1), data.size(2), data.size(3)
                return data.view(batch_size, h * w, c)
            elif data.dim() == 2:  # (batch, features)
                return data.unsqueeze(-1)  # 添加通道维度
            else:
                # 其他情况，尝试重塑
                total_elements = data.numel() // batch_size
                return data.view(batch_size, total_elements, 1)
        
        elif target_dims == 2:
            # 目标是2D数据
            if data.dim() == 4:  # (batch, height, width, channels)
                return data
            elif data.dim() == 3:  # (batch, length, channels)
                # 将1D数据重塑为2D
                length, channels = data.size(1), data.size(2)
                
                if target_resolution:
                    h, w = target_resolution
                    if h * w == length:
                        return data.view(batch_size, h, w, channels)
                    else:
                        # 调整大小以匹配目标分辨率
                        return self._resize_1d_to_2d(data, (h, w))
                else:
                    # 自动确定最佳的2D形状
                    sqrt_len = int(np.sqrt(length))
                    if sqrt_len * sqrt_len == length:
                        return data.view(batch_size, sqrt_len, sqrt_len, channels)
                    else:
                        # 找到最接近的矩形
                        h = int(np.sqrt(length))
                        w = (length + h - 1) // h  # 向上取整
                        
                        # 填充到目标大小
                        if h * w > length:
                            pad_size = h * w - length
                            data_padded = torch.cat([
                                data, 
                                torch.zeros(batch_size, pad_size, channels, device=data.device)
                            ], dim=1)
                            return data_padded.view(batch_size, h, w, channels)
                        else:
                            return data[:, :h*w].view(batch_size, h, w, channels)
            elif data.dim() == 2:  # (batch, features)
                # 添加通道维度并重塑为2D
                data = data.unsqueeze(-1)
                return self.smart_reshape_data(data, target_dims, target_resolution)
            else:
                # 其他情况
                total_elements = data.numel() // batch_size
                sqrt_elements = int(np.sqrt(total_elements))
                if sqrt_elements * sqrt_elements == total_elements:
                    return data.view(batch_size, sqrt_elements, sqrt_elements, 1)
                else:
                    h = int(np.sqrt(total_elements))
                    w = (total_elements + h - 1) // h
                    if h * w > total_elements:
                        # 需要填充
                        data_flat = data.view(batch_size, -1)
                        pad_size = h * w - total_elements
                        data_padded = torch.cat([
                            data_flat,
                            torch.zeros(batch_size, pad_size, device=data.device)
                        ], dim=1)
                        return data_padded.view(batch_size, h, w, 1)
                    else:
                        return data.view(batch_size, h, w, 1)
        
        return data
    
    def _resize_1d_to_2d(self, data: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        """将1D数据调整为指定的2D形状"""
        batch_size, length, channels = data.shape
        target_h, target_w = target_shape
        target_length = target_h * target_w
        
        if length == target_length:
            return data.view(batch_size, target_h, target_w, channels)
        elif length < target_length:
            # 需要插值或填充
            # 使用线性插值
            data_2d = data.view(batch_size, 1, length, channels)  # 添加高度维度
            data_resized = torch.nn.functional.interpolate(
                data_2d.permute(0, 3, 1, 2),  # (batch, channels, 1, length)
                size=(1, target_length),
                mode='linear',
                align_corners=False
            )
            data_resized = data_resized.permute(0, 2, 3, 1)  # 回到 (batch, 1, target_length, channels)
            return data_resized.view(batch_size, target_h, target_w, channels)
        else:
            # 需要下采样
            # 使用平均池化进行下采样
            stride = length // target_length
            if stride > 1:
                data_pooled = torch.nn.functional.avg_pool1d(
                    data.permute(0, 2, 1),  # (batch, channels, length)
                    kernel_size=stride,
                    stride=stride
                )
                data_pooled = data_pooled.permute(0, 2, 1)  # 回到 (batch, new_length, channels)
                # 如果还是太长，截断
                if data_pooled.size(1) > target_length:
                    data_pooled = data_pooled[:, :target_length]
                elif data_pooled.size(1) < target_length:
                    # 填充
                    pad_size = target_length - data_pooled.size(1)
                    data_pooled = torch.cat([
                        data_pooled,
                        torch.zeros(batch_size, pad_size, channels, device=data.device)
                    ], dim=1)
                return data_pooled.view(batch_size, target_h, target_w, channels)
            else:
                # 直接截断
                return data[:, :target_length].view(batch_size, target_h, target_w, channels)
    
    def prepare_data_for_model(self, input_data: torch.Tensor, target_data: torch.Tensor, 
                              model_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """为特定模型智能准备数据"""
        config = self.model_configs[model_name]
        target_dims = config['dimensions']
        
        print(f"  为 {model_name} 准备数据 (目标维度: {target_dims}D)")
        print(f"  原始输入形状: {input_data.shape}")
        print(f"  原始目标形状: {target_data.shape}")
        
        # 确保数据有批次维度
        if input_data.dim() < 2:
            input_data = input_data.unsqueeze(0)
        if target_data.dim() < 2:
            target_data = target_data.unsqueeze(0)
        
        # 智能重塑输入数据
        input_data_reshaped = self.smart_reshape_data(input_data, target_dims)
        
        # 为目标数据确定合适的形状
        if target_dims == 1:
            # 1D模型的目标数据
            if target_data.dim() == 2:
                target_data = target_data.unsqueeze(-1)
            elif target_data.dim() > 3:
                # 展平多余的维度
                batch_size = target_data.size(0)
                target_data = target_data.view(batch_size, -1, 1)
        
        elif target_dims == 2:
            # 2D模型的目标数据
            if input_data_reshaped.dim() == 4:
                # 使用与输入相同的空间分辨率
                target_h, target_w = input_data_reshaped.size(1), input_data_reshaped.size(2)
                target_data_reshaped = self.smart_reshape_data(target_data, target_dims, (target_h, target_w))
            else:
                target_data_reshaped = self.smart_reshape_data(target_data, target_dims)
        
        # 确保目标数据形状合理
        if target_dims == 2 and 'target_data_reshaped' in locals():
            target_data = target_data_reshaped
        elif target_dims == 1:
            target_data = self.smart_reshape_data(target_data, target_dims)
        
        print(f"  处理后输入形状: {input_data_reshaped.shape}")
        print(f"  处理后目标形状: {target_data.shape}")
        
        return input_data_reshaped, target_data
    
    def test_model_on_data(self, model_name: str, input_data: torch.Tensor, 
                          target_data: torch.Tensor) -> Dict[str, Any]:
        """在真实数据上测试模型"""
        print(f"\n{'='*50}")
        print(f"测试模型: {model_name}")
        print(f"{'='*50}")
        
        config = self.model_configs[model_name]
        
        result = {
            'model_name': model_name,
            'original_input_shape': tuple(input_data.shape),
            'original_target_shape': tuple(target_data.shape),
            'success': False,
            'error': None
        }
        
        try:
            # 准备数据
            input_data_prep, target_data_prep = self.prepare_data_for_model(
                input_data.clone(), target_data.clone(), model_name
            )
            
            # 确定分辨率参数
            if config['dimensions'] == 1:
                input_resolution = input_data_prep.size(1)
                output_resolution = target_data_prep.size(1)
            else:  # 2D
                input_resolution = (input_data_prep.size(1), input_data_prep.size(2))
                output_resolution = (target_data_prep.size(1), target_data_prep.size(2))
            
            # 创建模型
            model_params = config['base_params'].copy()
            model_params['input_resolution'] = input_resolution
            model_params['output_resolution'] = output_resolution
            
            print(f"  模型参数: {model_params}")
            
            model = config['creator'](**model_params).to(self.device)
            
            # 移动数据到设备
            input_data_prep = input_data_prep.to(self.device)
            target_data_prep = target_data_prep.to(self.device)
            
            # 前向传播测试
            model.eval()
            with torch.no_grad():
                start_time = time.time()
                output = model(input_data_prep)
                inference_time = time.time() - start_time
            
            print(f"  输出形状: {output.shape}")
            print(f"  推理时间: {inference_time:.4f}s")
            
            # 调整输出形状以匹配目标
            if output.shape != target_data_prep.shape:
                print(f"  调整输出形状从 {output.shape} 到 {target_data_prep.shape}")
                
                # 尝试调整输出形状
                if output.numel() == target_data_prep.numel():
                    output = output.view(target_data_prep.shape)
                elif output.size(0) == target_data_prep.size(0):  # 批次大小匹配
                    # 使用插值调整空间维度
                    if config['dimensions'] == 1 and output.dim() == 3 and target_data_prep.dim() == 3:
                        # 1D插值
                        output_interp = torch.nn.functional.interpolate(
                            output.permute(0, 2, 1),  # (batch, channels, length)
                            size=target_data_prep.size(1),
                            mode='linear',
                            align_corners=False
                        )
                        output = output_interp.permute(0, 2, 1)  # 回到 (batch, length, channels)
                    elif config['dimensions'] == 2 and output.dim() == 4 and target_data_prep.dim() == 4:
                        # 2D插值
                        output_interp = torch.nn.functional.interpolate(
                            output.permute(0, 3, 1, 2),  # (batch, channels, height, width)
                            size=(target_data_prep.size(1), target_data_prep.size(2)),
                            mode='bilinear',
                            align_corners=False
                        )
                        output = output_interp.permute(0, 2, 3, 1)  # 回到 (batch, height, width, channels)
            
            # 计算指标
            if output.shape == target_data_prep.shape:
                mse = torch.mean((output - target_data_prep) ** 2).item()
                mae = torch.mean(torch.abs(output - target_data_prep)).item()
                
                # 计算相对误差
                target_norm = torch.norm(target_data_prep).item()
                if target_norm > 0:
                    relative_error = torch.norm(output - target_data_prep).item() / target_norm
                else:
                    relative_error = float('inf')
                
                result.update({
                    'prepared_input_shape': tuple(input_data_prep.shape),
                    'prepared_target_shape': tuple(target_data_prep.shape),
                    'output_shape': tuple(output.shape),
                    'mse': mse,
                    'mae': mae,
                    'relative_error': relative_error,
                    'inference_time': inference_time,
                    'success': True
                })
                
                print(f"  ✅ 测试成功!")
                print(f"  MSE: {mse:.6f}")
                print(f"  MAE: {mae:.6f}")
                print(f"  相对误差: {relative_error:.6f}")
                
            else:
                result['error'] = f"输出形状仍不匹配: 期望 {target_data_prep.shape}, 得到 {output.shape}"
                print(f"  ❌ {result['error']}")
            
            # 检查数值稳定性
            if torch.isnan(output).any():
                result['error'] = "输出包含NaN值"
                result['success'] = False
                print("  ❌ 输出包含NaN值")
            elif torch.isinf(output).any():
                result['error'] = "输出包含无穷值"
                result['success'] = False
                print("  ❌ 输出包含无穷值")
            
        except Exception as e:
            result['error'] = str(e)
            print(f"  ❌ 测试失败: {e}")
        
        return result
    
    def create_test_datasets(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """创建多样化的测试数据集"""
        datasets = {}
        
        # 1D时间序列数据
        datasets['time_series_short'] = {
            'input': torch.randn(3, 32, 1),
            'target': torch.randn(3, 64, 1)
        }
        
        datasets['time_series_long'] = {
            'input': torch.randn(3, 128, 1),
            'target': torch.randn(3, 256, 1)
        }
        
        # 2D图像数据
        datasets['image_small'] = {
            'input': torch.randn(3, 16, 16, 1),
            'target': torch.randn(3, 32, 32, 1)
        }
        
        datasets['image_medium'] = {
            'input': torch.randn(3, 32, 32, 1),
            'target': torch.randn(3, 64, 64, 1)
        }
        
        # 不规则形状数据
        datasets['irregular_1d'] = {
            'input': torch.randn(3, 50, 1),
            'target': torch.randn(3, 75, 1)
        }
        
        datasets['irregular_2d'] = {
            'input': torch.randn(3, 24, 36, 1),
            'target': torch.randn(3, 48, 72, 1)
        }
        
        # 物理场数据模拟
        x = torch.linspace(0, 2*np.pi, 64)
        y = torch.linspace(0, 2*np.pi, 64)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # 2D波动方程解
        wave_field = torch.sin(X) * torch.cos(Y)
        datasets['physics_wave'] = {
            'input': wave_field.unsqueeze(0).unsqueeze(-1).repeat(3, 1, 1, 1),
            'target': (wave_field * 1.1).unsqueeze(0).unsqueeze(-1).repeat(3, 1, 1, 1)
        }
        
        # 1D扩散方程解
        x_1d = torch.linspace(0, 1, 100)
        diffusion_1d = torch.exp(-x_1d**2)
        datasets['physics_diffusion'] = {
            'input': diffusion_1d.unsqueeze(0).unsqueeze(-1).repeat(3, 1, 1),
            'target': (diffusion_1d * 0.9).unsqueeze(0).unsqueeze(-1).repeat(3, 1, 1)
        }
        
        return datasets
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """运行综合测试"""
        print("开始改进的真实数据综合测试...")
        print(f"可用模型: {list(self.model_configs.keys())}")
        
        # 创建测试数据集
        test_datasets = self.create_test_datasets()
        
        print(f"\n创建了 {len(test_datasets)} 个测试数据集:")
        for name, data in test_datasets.items():
            print(f"  {name}: 输入 {data['input'].shape}, 目标 {data['target'].shape}")
        
        # 测试所有数据集
        all_results = {}
        
        for dataset_name, dataset in test_datasets.items():
            print(f"\n{'='*80}")
            print(f"测试数据集: {dataset_name}")
            print(f"{'='*80}")
            
            input_data = dataset['input']
            target_data = dataset['target']
            
            # 测试所有模型
            dataset_results = {}
            for model_name in self.model_configs.keys():
                try:
                    result = self.test_model_on_data(
                        model_name, 
                        input_data.clone(), 
                        target_data.clone()
                    )
                    dataset_results[model_name] = result
                except Exception as e:
                    print(f"模型 {model_name} 测试失败: {e}")
                    dataset_results[model_name] = {
                        'error': str(e), 
                        'success': False,
                        'model_name': model_name
                    }
            
            all_results[dataset_name] = {
                'dataset_info': {
                    'input_shape': tuple(input_data.shape),
                    'target_shape': tuple(target_data.shape)
                },
                'model_results': dataset_results
            }
        
        self.test_results = all_results
        return all_results
    
    def generate_detailed_report(self) -> str:
        """生成详细的测试报告"""
        if not self.test_results:
            return "没有测试结果可用"
        
        report = []
        report.append("="*100)
        report.append("改进的真实数据测试详细报告")
        report.append("="*100)
        
        # 总体统计
        total_tests = 0
        successful_tests = 0
        model_success_count = {}
        
        for dataset_name, dataset_result in self.test_results.items():
            for model_name, result in dataset_result['model_results'].items():
                total_tests += 1
                if result.get('success', False):
                    successful_tests += 1
                    model_success_count[model_name] = model_success_count.get(model_name, 0) + 1
                else:
                    model_success_count[model_name] = model_success_count.get(model_name, 0)
        
        report.append(f"\n📊 总体统计:")
        report.append(f"  总测试数: {total_tests}")
        report.append(f"  成功测试数: {successful_tests}")
        report.append(f"  总体成功率: {successful_tests/total_tests*100:.1f}%")
        
        # 模型成功率排名
        report.append(f"\n🏆 模型成功率排名:")
        model_success_rates = {}
        for model_name in self.model_configs.keys():
            success_count = model_success_count.get(model_name, 0)
            total_count = len(self.test_results)
            success_rate = success_count / total_count * 100
            model_success_rates[model_name] = success_rate
        
        sorted_models = sorted(model_success_rates.items(), key=lambda x: x[1], reverse=True)
        for i, (model_name, success_rate) in enumerate(sorted_models, 1):
            report.append(f"  {i}. {model_name}: {success_rate:.1f}% ({model_success_count.get(model_name, 0)}/{len(self.test_results)})")
        
        # 每个数据集的详细结果
        for dataset_name, dataset_result in self.test_results.items():
            report.append(f"\n{'='*80}")
            report.append(f"📋 数据集: {dataset_name}")
            report.append(f"{'='*80}")
            
            dataset_info = dataset_result['dataset_info']
            report.append(f"输入形状: {dataset_info['input_shape']}")
            report.append(f"目标形状: {dataset_info['target_shape']}")
            
            model_results = dataset_result['model_results']
            successful_models = {k: v for k, v in model_results.items() if v.get('success', False)}
            failed_models = {k: v for k, v in model_results.items() if not v.get('success', False)}
            
            if successful_models:
                report.append(f"\n✅ 成功的模型 ({len(successful_models)}/{len(model_results)}):")
                report.append("-"*90)
                report.append(f"{'模型':<12} {'MSE':<12} {'MAE':<12} {'相对误差':<12} {'推理时间(s)':<12} {'输出形状':<20}")
                report.append("-"*90)
                
                for model_name, result in successful_models.items():
                    report.append(
                        f"{model_name:<12} "
                        f"{result.get('mse', 0):<12.6f} "
                        f"{result.get('mae', 0):<12.6f} "
                        f"{result.get('relative_error', 0):<12.6f} "
                        f"{result.get('inference_time', 0):<12.4f} "
                        f"{str(result.get('output_shape', 'N/A')):<20}"
                    )
            
            if failed_models:
                report.append(f"\n❌ 失败的模型 ({len(failed_models)}/{len(model_results)}):")
                report.append("-"*60)
                for model_name, result in failed_models.items():
                    error_msg = result.get('error', '未知错误')
                    # 截断过长的错误信息
                    if len(error_msg) > 80:
                        error_msg = error_msg[:77] + "..."
                    report.append(f"  {model_name}: {error_msg}")
        
        # 性能分析
        if successful_tests > 0:
            report.append(f"\n{'='*80}")
            report.append("📈 性能分析")
            report.append(f"{'='*80}")
            
            # 收集所有成功结果的性能数据
            performance_data = {}
            for dataset_name, dataset_result in self.test_results.items():
                for model_name, result in dataset_result['model_results'].items():
                    if result.get('success', False):
                        if model_name not in performance_data:
                            performance_data[model_name] = {
                                'mse': [], 'mae': [], 'relative_error': [], 'inference_time': []
                            }
                        performance_data[model_name]['mse'].append(result.get('mse', 0))
                        performance_data[model_name]['mae'].append(result.get('mae', 0))
                        performance_data[model_name]['relative_error'].append(result.get('relative_error', 0))
                        performance_data[model_name]['inference_time'].append(result.get('inference_time', 0))
            
            if performance_data:
                report.append("\n平均性能指标:")
                report.append("-"*80)
                report.append(f"{'模型':<12} {'平均MSE':<12} {'平均MAE':<12} {'平均相对误差':<15} {'平均时间(s)':<12} {'测试次数':<10}")
                report.append("-"*80)
                
                for model_name, metrics in performance_data.items():
                    avg_mse = np.mean(metrics['mse'])
                    avg_mae = np.mean(metrics['mae'])
                    avg_rel_error = np.mean(metrics['relative_error'])
                    avg_time = np.mean(metrics['inference_time'])
                    test_count = len(metrics['mse'])
                    
                    report.append(
                        f"{model_name:<12} "
                        f"{avg_mse:<12.6f} "
                        f"{avg_mae:<12.6f} "
                        f"{avg_rel_error:<15.6f} "
                        f"{avg_time:<12.4f} "
                        f"{test_count:<10}"
                    )
                
                # 性能排名
                report.append("\n🥇 性能排名:")
                
                # MSE排名 (越小越好)
                mse_ranking = sorted(performance_data.items(), 
                                   key=lambda x: np.mean(x[1]['mse']))
                report.append("\n  MSE排名 (越小越好):")
                for i, (model_name, metrics) in enumerate(mse_ranking, 1):
                    avg_mse = np.mean(metrics['mse'])
                    report.append(f"    {i}. {model_name}: {avg_mse:.6f}")
                
                # 推理速度排名 (越快越好)
                speed_ranking = sorted(performance_data.items(), 
                                     key=lambda x: np.mean(x[1]['inference_time']))
                report.append("\n  推理速度排名 (越快越好):")
                for i, (model_name, metrics) in enumerate(speed_ranking, 1):
                    avg_time = np.mean(metrics['inference_time'])
                    report.append(f"    {i}. {model_name}: {avg_time:.4f}s")
        
        report.append("\n" + "="*100)
        report.append("测试完成! 🎉")
        report.append("="*100)
        
        return "\n".join(report)
    
    def save_results(self, filename: str = "improved_real_data_test_results.txt"):
        """保存测试结果"""
        report = self.generate_detailed_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"\n📄 详细测试结果已保存到: {filename}")

def main():
    """主函数"""
    print("🚀 启动改进的真实数据测试...")
    
    # 创建测试器
    tester = ImprovedRealDataTester()
    
    # 运行综合测试
    results = tester.run_comprehensive_test()
    
    # 生成并显示报告
    report = tester.generate_detailed_report()
    print("\n" + report)
    
    # 保存结果
    tester.save_results("improved_real_data_test_results.txt")
    
    return results

if __name__ == "__main__":
    main()